package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const authCookieName = "vllm_manager_session"

type dashboardAuth struct {
	password     string
	sessionToken string
}

func newDashboardAuth() dashboardAuth {
	password := strings.TrimSpace(os.Getenv("VLLM_MANAGER_PASSWORD"))
	if password == "" {
		return dashboardAuth{}
	}
	return dashboardAuth{
		password:     password,
		sessionToken: randomSessionToken(),
	}
}

func randomSessionToken() string {
	var data [32]byte
	if _, err := rand.Read(data[:]); err != nil {
		log.Fatalf("failed to initialize dashboard auth session: %v", err)
	}
	return hex.EncodeToString(data[:])
}

func (a dashboardAuth) required() bool {
	return a.password != ""
}

func (a dashboardAuth) authenticated(r *http.Request) bool {
	if !a.required() {
		return true
	}
	cookie, err := r.Cookie(authCookieName)
	if err != nil {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(cookie.Value), []byte(a.sessionToken)) == 1
}

func (a dashboardAuth) require(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !a.authenticated(r) {
			writeJSONError(w, http.StatusUnauthorized, "authentication required")
			return
		}
		next(w, r)
	}
}

func (a dashboardAuth) statusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, map[string]bool{
		"required":      a.required(),
		"authenticated": a.authenticated(r),
	})
}

func (a dashboardAuth) loginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !a.required() {
		writeJSON(w, map[string]bool{"required": false, "authenticated": true})
		return
	}

	var req struct {
		Password string `json:"password"`
	}
	if !decodeJSONBody(w, r, &req) {
		return
	}
	if subtle.ConstantTimeCompare([]byte(req.Password), []byte(a.password)) != 1 {
		writeJSONError(w, http.StatusUnauthorized, "invalid password")
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     authCookieName,
		Value:    a.sessionToken,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
		MaxAge:   int((24 * time.Hour).Seconds()),
	})
	writeJSON(w, map[string]bool{"required": true, "authenticated": true})
}

func (a dashboardAuth) logoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     authCookieName,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
		MaxAge:   -1,
	})
	writeJSON(w, map[string]bool{"authenticated": false})
}
