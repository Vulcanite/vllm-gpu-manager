package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
)

func resetSettingsCache(t *testing.T, settings Settings, loaded bool) {
	t.Helper()
	settingsCacheMu.Lock()
	previous := settingsCache
	previousLoaded := settingsCacheLoaded
	settingsCache = settings
	settingsCacheLoaded = loaded
	settingsCacheMu.Unlock()
	t.Cleanup(func() {
		settingsCacheMu.Lock()
		settingsCache = previous
		settingsCacheLoaded = previousLoaded
		settingsCacheMu.Unlock()
	})
}

func TestShouldRunDashboard(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want bool
	}{
		{
			name: "explicit dashboard flag",
			args: []string{"--dashboard"},
			want: true,
		},
		{
			name: "single dash dashboard flag",
			args: []string{"-dashboard"},
			want: true,
		},
		{
			name: "dashboard flag with port",
			args: []string{"--dashboard", "--port", "9090"},
			want: true,
		},
		{
			name: "model start port does not imply dashboard",
			args: []string{"model", "start", "Qwen/Qwen2.5-7B-Instruct", "--port", "8000", "--gpus", "1"},
			want: false,
		},
		{
			name: "root port alone does not imply dashboard",
			args: []string{"--port", "9090"},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldRunDashboard(tt.args); got != tt.want {
				t.Fatalf("shouldRunDashboard(%v)=%v, want %v", tt.args, got, tt.want)
			}
		})
	}
}

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

func TestSaveSettingsWritesPrivateFileAndUpdatesCache(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	resetSettingsCache(t, Settings{}, false)

	cacheDir := t.TempDir()
	if err := saveSettings(Settings{HFCache: cacheDir, HFToken: "hf-secret", VLLMAPIKey: "api-secret"}); err != nil {
		t.Fatalf("saveSettings() error=%v", err)
	}

	info, err := os.Stat(settingsFile)
	if err != nil {
		t.Fatalf("settings file stat error=%v", err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("settings mode=%#o, want 0600", got)
	}
	settings := getSettings()
	if settings.HFCache != cacheDir || settings.HFToken != "hf-secret" || settings.VLLMAPIKey != "api-secret" {
		t.Fatalf("settings=%+v, want saved values", settings)
	}
}

func TestGetSettingsReturnsCopiedRoutingTable(t *testing.T) {
	resetSettingsCache(t, Settings{HFCache: t.TempDir(), RoutingTable: map[string]int{"model-a": 8000}}, true)

	settings := getSettings()
	settings.RoutingTable["model-b"] = 8001

	settings = getSettings()
	if _, ok := settings.RoutingTable["model-b"]; ok {
		t.Fatalf("getSettings returned shared routing table: %+v", settings.RoutingTable)
	}
}

func TestUpdateSettingsUpdatesAndClearsVLLMAPIKey(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir, HFToken: "hf-old", VLLMAPIKey: "old-key"}, true)
	t.Chdir(t.TempDir())

	updated, err := updateSettings(SettingsUpdate{HFCache: cacheDir, VLLMAPIKey: " new-key "})
	if err != nil {
		t.Fatalf("updateSettings() error=%v", err)
	}
	if updated.VLLMAPIKey != "new-key" {
		t.Fatalf("VLLMAPIKey=%q, want new-key", updated.VLLMAPIKey)
	}

	updated, err = updateSettings(SettingsUpdate{HFCache: cacheDir, ClearVLLMAPIKey: true})
	if err != nil {
		t.Fatalf("clear updateSettings() error=%v", err)
	}
	if updated.VLLMAPIKey != "" {
		t.Fatalf("VLLMAPIKey=%q, want cleared", updated.VLLMAPIKey)
	}
}

func TestUpdateSettingsUpdatesServingEnvironmentToggles(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	t.Chdir(t.TempDir())

	updated, err := updateSettings(SettingsUpdate{
		HFCache:                 cacheDir,
		EnableCUDACompatibility: true,
		SetCUDACompatibility:    true,
		HFHubOffline:            true,
		SetHFHubOffline:         true,
	})
	if err != nil {
		t.Fatalf("updateSettings() error=%v", err)
	}
	if !updated.EnableCUDACompatibility || !updated.HFHubOffline {
		t.Fatalf("settings=%+v, want serving environment toggles enabled", updated)
	}

	updated, err = updateSettings(SettingsUpdate{HFCache: cacheDir})
	if err != nil {
		t.Fatalf("preserve updateSettings() error=%v", err)
	}
	if !updated.EnableCUDACompatibility || !updated.HFHubOffline {
		t.Fatalf("settings=%+v, want omitted toggles preserved", updated)
	}

	updated, err = updateSettings(SettingsUpdate{
		HFCache:                 cacheDir,
		EnableCUDACompatibility: false,
		SetCUDACompatibility:    true,
		HFHubOffline:            false,
		SetHFHubOffline:         true,
	})
	if err != nil {
		t.Fatalf("disable updateSettings() error=%v", err)
	}
	if updated.EnableCUDACompatibility || updated.HFHubOffline {
		t.Fatalf("settings=%+v, want serving environment toggles disabled", updated)
	}
}

func TestIsAllowedOrigin(t *testing.T) {
	if !isAllowedOrigin("http://example.test:9090", "example.test:9090") {
		t.Fatal("same host origin was rejected")
	}
	if !isAllowedOrigin("http://localhost:5173", "example.test:9090") {
		t.Fatal("localhost origin was rejected")
	}
	if isAllowedOrigin("http://evil.test", "example.test:9090") {
		t.Fatal("untrusted origin was allowed")
	}
}

func TestCorsMiddlewareRejectsBadOrigin(t *testing.T) {
	handler := corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	req.Host = "example.test:9090"
	req.Header.Set("Origin", "http://evil.test")
	rr := httptest.NewRecorder()

	handler(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("status=%d, want %d", rr.Code, http.StatusForbidden)
	}
}

func TestInferenceProxyRewritesAuthorizationForBackend(t *testing.T) {
	backendAuth := make(chan string, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendAuth <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	idx := strings.LastIndex(backend.URL, ":")
	if idx == -1 {
		t.Fatalf("backend URL %q does not contain port", backend.URL)
	}
	port := backend.URL[idx+1:]
	resetSettingsCache(t, Settings{
		HFCache:         t.TempDir(),
		VLLMAPIKey:      "vllm-secret",
		InferenceAPIKey: "inference-secret",
		RoutingTable:    map[string]int{"test-model": mustAtoi(t, port)},
	}, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	req.Header.Set("Authorization", "Bearer inference-secret")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if got := <-backendAuth; got != "Bearer vllm-secret" {
		t.Fatalf("backend Authorization=%q, want vLLM key", got)
	}
}

func mustAtoi(t *testing.T, value string) int {
	t.Helper()
	parsed, err := strconv.Atoi(value)
	if err != nil {
		t.Fatalf("parse int %q: %v", value, err)
	}
	return parsed
}

func TestSettingsEndpointReturnsStatus(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir, HFToken: "hf-secret", VLLMAPIKey: "api-secret", EnableCUDACompatibility: true, HFHubOffline: true}, true)
	mux, _ := newDashboardMux(true)
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	rr := httptest.NewRecorder()

	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	var status SettingsStatus
	if err := json.NewDecoder(rr.Body).Decode(&status); err != nil {
		t.Fatalf("decode settings status: %v", err)
	}
	if status.HFCache != cacheDir || !status.HFTokenSet || !status.VLLMAPIKeySet || !status.EnableCUDACompatibility || !status.HFHubOffline {
		t.Fatalf("status=%+v, want cache and secret flags", status)
	}
	if status.HFTokenSuffix != "cret" || status.VLLMAPIKeySuffix != "cret" {
		t.Fatalf("status suffixes=%q/%q, want last four characters", status.HFTokenSuffix, status.VLLMAPIKeySuffix)
	}
}

func TestSPAFallbackServesIndex(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	mux, _ := newDashboardMux(true)
	req := httptest.NewRequest(http.MethodGet, "/settings", nil)
	rr := httptest.NewRecorder()

	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d", rr.Code, http.StatusOK)
	}
	if !strings.Contains(rr.Body.String(), "<div id=\"root\">") {
		t.Fatal("SPA fallback did not serve index.html")
	}
}
