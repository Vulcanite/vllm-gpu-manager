package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	authCookieName       = "vllm_manager_session"
	sessionTTL           = 24 * time.Hour
	loginRateLimitWindow = time.Minute
	loginRateLimitMax    = 5
)

type dashboardAuth struct {
	password string
	mu       sync.Mutex
	sessions map[string]time.Time   // token -> expiry
	loginAt  map[string][]time.Time // client IP -> failed attempt times
}

func newDashboardAuth() *dashboardAuth {
	password := strings.TrimSpace(os.Getenv("VLLM_MANAGER_PASSWORD"))
	if password == "" {
		return &dashboardAuth{}
	}
	return &dashboardAuth{
		password: password,
		sessions: make(map[string]time.Time),
		loginAt:  make(map[string][]time.Time),
	}
}

func randomSessionToken() string {
	var data [32]byte
	if _, err := rand.Read(data[:]); err != nil {
		log.Fatalf("failed to initialize dashboard auth session: %v", err)
	}
	return hex.EncodeToString(data[:])
}

func (a *dashboardAuth) required() bool {
	return a != nil && a.password != ""
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(strings.TrimSpace(r.RemoteAddr))
	if err != nil {
		return strings.TrimSpace(r.RemoteAddr)
	}
	return host
}

func (a *dashboardAuth) loginAttemptsExceeded(ip string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := time.Now()
	cutoff := now.Add(-loginRateLimitWindow)
	attempts := a.loginAt[ip]
	kept := attempts[:0]
	for _, at := range attempts {
		if at.After(cutoff) {
			kept = append(kept, at)
		}
	}
	a.loginAt[ip] = kept
	return len(kept) >= loginRateLimitMax
}

func (a *dashboardAuth) recordLoginFailure(ip string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.loginAt[ip] = append(a.loginAt[ip], time.Now())
}

func (a *dashboardAuth) lookupSession(token string) bool {
	if token == "" {
		return false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	expiresAt, ok := a.sessions[token]
	if !ok {
		return false
	}
	if time.Now().After(expiresAt) {
		delete(a.sessions, token)
		return false
	}
	return true
}

func (a *dashboardAuth) createSession() (string, time.Time) {
	token := randomSessionToken()
	expiresAt := time.Now().Add(sessionTTL)
	a.mu.Lock()
	a.sessions[token] = expiresAt
	a.mu.Unlock()
	return token, expiresAt
}

func (a *dashboardAuth) deleteSession(token string) {
	if token == "" {
		return
	}
	a.mu.Lock()
	delete(a.sessions, token)
	a.mu.Unlock()
}

func (a *dashboardAuth) authenticated(r *http.Request) bool {
	if !a.required() {
		return true
	}
	cookie, err := r.Cookie(authCookieName)
	if err != nil {
		return false
	}
	return a.lookupSession(cookie.Value)
}

func (a *dashboardAuth) require(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !a.authenticated(r) {
			writeJSONError(w, http.StatusUnauthorized, "authentication required")
			return
		}
		next(w, r)
	}
}

func (a *dashboardAuth) statusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, map[string]bool{
		"required":      a.required(),
		"authenticated": a.authenticated(r),
	})
}

func (a *dashboardAuth) loginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !a.required() {
		writeJSON(w, map[string]bool{"required": false, "authenticated": true})
		return
	}

	ip := clientIP(r)
	if a.loginAttemptsExceeded(ip) {
		writeJSONError(w, http.StatusTooManyRequests, "too many login attempts")
		return
	}

	var req struct {
		Password string `json:"password"`
	}
	if !decodeJSONBody(w, r, &req) {
		return
	}
	if subtle.ConstantTimeCompare([]byte(req.Password), []byte(a.password)) != 1 {
		a.recordLoginFailure(ip)
		writeJSONError(w, http.StatusUnauthorized, "Invalid Password")
		return
	}

	if cookie, err := r.Cookie(authCookieName); err == nil {
		a.deleteSession(cookie.Value)
	}
	token, _ := a.createSession()
	http.SetCookie(w, &http.Cookie{
		Name:     authCookieName,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
		MaxAge:   int(sessionTTL.Seconds()),
	})
	writeJSON(w, map[string]bool{"required": true, "authenticated": true})
}

func (a *dashboardAuth) logoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if cookie, err := r.Cookie(authCookieName); err == nil {
		a.deleteSession(cookie.Value)
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
