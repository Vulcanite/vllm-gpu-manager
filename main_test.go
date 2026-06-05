package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDashboardAuthDisabledAllowsRequests(t *testing.T) {
	auth := dashboardAuth{}

	req := httptest.NewRequest(http.MethodGet, "/api/gpus", nil)
	rr := httptest.NewRecorder()

	auth.require(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status=%d, want %d", rr.Code, http.StatusNoContent)
	}
}

func TestDashboardAuthRequiresSessionCookie(t *testing.T) {
	auth := dashboardAuth{password: "secret", sessionToken: "session-token"}

	req := httptest.NewRequest(http.MethodGet, "/api/gpus", nil)
	rr := httptest.NewRecorder()

	auth.require(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d, want %d", rr.Code, http.StatusUnauthorized)
	}
}

func TestDashboardAuthLoginSetsUsableCookie(t *testing.T) {
	auth := dashboardAuth{password: "secret", sessionToken: "session-token"}

	loginReq := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(`{"password":"secret"}`))
	loginReq.Header.Set("Content-Type", "application/json")
	loginRR := httptest.NewRecorder()
	auth.loginHandler(loginRR, loginReq)

	if loginRR.Code != http.StatusOK {
		t.Fatalf("login status=%d, want %d", loginRR.Code, http.StatusOK)
	}

	cookies := loginRR.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("cookies=%d, want 1", len(cookies))
	}

	req := httptest.NewRequest(http.MethodGet, "/api/gpus", nil)
	req.AddCookie(cookies[0])
	rr := httptest.NewRecorder()
	auth.require(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("authenticated status=%d, want %d", rr.Code, http.StatusNoContent)
	}
}

func TestNormalizeVLLMAPIKeyTrimsValue(t *testing.T) {
	key, err := normalizeVLLMAPIKey("  token-secret  ")
	if err != nil {
		t.Fatalf("normalizeVLLMAPIKey() error=%v", err)
	}
	if key != "token-secret" {
		t.Fatalf("normalizeVLLMAPIKey()=%q, want token-secret", key)
	}
}

func TestNormalizeVLLMAPIKeyRejectsWhitespace(t *testing.T) {
	_, err := normalizeVLLMAPIKey("token secret")
	if err == nil {
		t.Fatal("normalizeVLLMAPIKey() error=nil, want error")
	}
	appErr, ok := err.(appError)
	if !ok {
		t.Fatalf("normalizeVLLMAPIKey() error type=%T, want appError", err)
	}
	if appErr.Status != http.StatusBadRequest {
		t.Fatalf("status=%d, want %d", appErr.Status, http.StatusBadRequest)
	}
}
