package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"vllm-manager/api"
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

func stubUpstreamAPIKeyLookup(t *testing.T, fn func(int) (string, error)) {
	t.Helper()
	original := getUpstreamAPIKeyByPort
	getUpstreamAPIKeyByPort = fn
	upstreamAPIKeyCacheMu.Lock()
	upstreamAPIKeyCache = map[int]upstreamAPIKeyCacheEntry{}
	upstreamAPIKeyCacheMu.Unlock()
	t.Cleanup(func() {
		getUpstreamAPIKeyByPort = original
		upstreamAPIKeyCacheMu.Lock()
		upstreamAPIKeyCache = map[int]upstreamAPIKeyCacheEntry{}
		upstreamAPIKeyCacheMu.Unlock()
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

func TestParseDashboardFlagsProxyDefaults(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantPort  string
		wantHost  string
		wantProxy bool
	}{
		{
			name:      "proxy enabled by default",
			args:      []string{"--dashboard"},
			wantPort:  defaultDashboardPort,
			wantHost:  "127.0.0.1",
			wantProxy: true,
		},
		{
			name:      "custom port keeps proxy enabled",
			args:      []string{"--dashboard", "--port", "9443"},
			wantPort:  "9443",
			wantHost:  "127.0.0.1",
			wantProxy: true,
		},
		{
			name:      "custom host overrides loopback default",
			args:      []string{"--dashboard", "--host", "0.0.0.0"},
			wantPort:  defaultDashboardPort,
			wantHost:  "0.0.0.0",
			wantProxy: true,
		},
		{
			name:      "no proxy disables gateway",
			args:      []string{"--dashboard", "--no-proxy"},
			wantPort:  defaultDashboardPort,
			wantHost:  "127.0.0.1",
			wantProxy: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			values, proxyEnabled, err := parseDashboardFlags(tt.args, io.Discard)
			if err != nil {
				t.Fatalf("parseDashboardFlags() error=%v", err)
			}
			if values.value != tt.wantPort {
				t.Fatalf("port=%q, want %q", values.value, tt.wantPort)
			}
			if values.host != tt.wantHost {
				t.Fatalf("host=%q, want %q", values.host, tt.wantHost)
			}
			if proxyEnabled != tt.wantProxy {
				t.Fatalf("proxyEnabled=%v, want %v", proxyEnabled, tt.wantProxy)
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

func TestUpdateSettingsPreservesPerServerKeysAndHeartbeat(t *testing.T) {
	cacheDir := t.TempDir()
	heartbeat := &Heartbeat{PID: os.Getpid(), LastSeen: time.Now()}
	resetSettingsCache(t, Settings{
		HFCache:       cacheDir,
		RoutingTable:  map[string]int{"model-a": 8000},
		ServerAPIKeys: map[string]string{"vllm-model-a-8000": "server-key"},
		Heartbeat:     heartbeat,
	}, true)
	t.Chdir(t.TempDir())

	updated, err := updateSettings(SettingsUpdate{HFCache: cacheDir, HFToken: "hf-new"})
	if err != nil {
		t.Fatalf("updateSettings() error=%v", err)
	}
	if updated.ServerAPIKeys["vllm-model-a-8000"] != "server-key" {
		t.Fatalf("ServerAPIKeys=%+v, want preserved per-server key", updated.ServerAPIKeys)
	}
	if updated.Heartbeat == nil || updated.Heartbeat.PID != os.Getpid() {
		t.Fatalf("Heartbeat=%+v, want preserved heartbeat", updated.Heartbeat)
	}
}

func TestBuildRoutingTableFromActiveModelsPrunesStaleEntries(t *testing.T) {
	table := buildRoutingTableFromActiveModels([]api.ActiveModel{
		{Model: "live-model", Port: "8000", State: "running"},
		{Model: "stopped-model", Port: "8001", State: "exited"},
		{Model: "unknown-port", Port: "unknown", State: "running"},
		{Model: "bad-port", Port: "0", State: "running"},
	})

	if len(table) != 1 {
		t.Fatalf("routing table=%+v, want only live route", table)
	}
	if got := table["live-model"]; got != 8000 {
		t.Fatalf("live-model port=%d, want 8000", got)
	}
}

func TestUpsertRoutePreservesDiskRoutesWhenCacheIsStale(t *testing.T) {
	t.Chdir(t.TempDir())
	cacheDir := t.TempDir()
	raw, err := json.Marshal(Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"disk-model": 8000},
	})
	if err != nil {
		t.Fatalf("marshal settings: %v", err)
	}
	if err := os.WriteFile(settingsFile, raw, 0600); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	resetSettingsCache(t, Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"stale-cache-model": 7000},
	}, true)

	upsertRoute("new-model", 8001)

	var saved Settings
	data, err := os.ReadFile(settingsFile)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatalf("unmarshal settings: %v", err)
	}
	if got := saved.RoutingTable["disk-model"]; got != 8000 {
		t.Fatalf("disk-model route=%d, want preserved 8000 in %+v", got, saved.RoutingTable)
	}
	if got := saved.RoutingTable["new-model"]; got != 8001 {
		t.Fatalf("new-model route=%d, want 8001 in %+v", got, saved.RoutingTable)
	}
	if _, ok := saved.RoutingTable["stale-cache-model"]; ok {
		t.Fatalf("stale cache route was written back: %+v", saved.RoutingTable)
	}
}

func TestWriteHeartbeatPreservesDiskRoutesWhenCacheIsStale(t *testing.T) {
	t.Chdir(t.TempDir())
	cacheDir := t.TempDir()
	raw, err := json.Marshal(Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"disk-model": 8000},
	})
	if err != nil {
		t.Fatalf("marshal settings: %v", err)
	}
	if err := os.WriteFile(settingsFile, raw, 0600); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	resetSettingsCache(t, Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"stale-cache-model": 7000},
	}, true)

	now := time.Now()
	if err := writeHeartbeat(os.Getpid(), now); err != nil {
		t.Fatalf("writeHeartbeat() error=%v", err)
	}

	var saved Settings
	data, err := os.ReadFile(settingsFile)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatalf("unmarshal settings: %v", err)
	}
	if got := saved.RoutingTable["disk-model"]; got != 8000 {
		t.Fatalf("disk-model route=%d, want preserved 8000 in %+v", got, saved.RoutingTable)
	}
	if _, ok := saved.RoutingTable["stale-cache-model"]; ok {
		t.Fatalf("stale cache route was written back: %+v", saved.RoutingTable)
	}
	if saved.Heartbeat == nil || saved.Heartbeat.PID != os.Getpid() {
		t.Fatalf("heartbeat=%+v, want current pid", saved.Heartbeat)
	}
}

func TestIncrementProxyUserRequestCountPreservesDiskRoutesWhenCacheIsStale(t *testing.T) {
	t.Chdir(t.TempDir())
	cacheDir := t.TempDir()
	raw, err := json.Marshal(Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"disk-model": 8000},
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "alice-key", RequestCount: 4}},
	})
	if err != nil {
		t.Fatalf("marshal settings: %v", err)
	}
	if err := os.WriteFile(settingsFile, raw, 0600); err != nil {
		t.Fatalf("write settings: %v", err)
	}
	resetSettingsCache(t, Settings{
		HFCache:      cacheDir,
		RoutingTable: map[string]int{"stale-cache-model": 7000},
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "alice-key", RequestCount: 1}},
	}, true)

	incrementProxyUserRequestCount("alice")

	var saved Settings
	data, err := os.ReadFile(settingsFile)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		t.Fatalf("unmarshal settings: %v", err)
	}
	if got := saved.RoutingTable["disk-model"]; got != 8000 {
		t.Fatalf("disk-model route=%d, want preserved 8000 in %+v", got, saved.RoutingTable)
	}
	if _, ok := saved.RoutingTable["stale-cache-model"]; ok {
		t.Fatalf("stale cache route was written back: %+v", saved.RoutingTable)
	}
	if len(saved.ProxyUsers) != 1 || saved.ProxyUsers[0].RequestCount != 5 {
		t.Fatalf("proxy users=%+v, want alice request count incremented from disk value to 5", saved.ProxyUsers)
	}
}

func TestValidateRoutingTableForActiveModels(t *testing.T) {
	active := []api.ActiveModel{{Model: "Qwen/Qwen2.5-7B-Instruct", Port: "8000", State: "running"}}

	validated, err := validateRoutingTableForActiveModels(map[string]int{
		" Qwen/Qwen2.5-7B-Instruct ": 8000,
	}, active)
	if err != nil {
		t.Fatalf("validateRoutingTableForActiveModels() error=%v", err)
	}
	if len(validated) != 1 || validated["Qwen/Qwen2.5-7B-Instruct"] != 8000 {
		t.Fatalf("validated=%+v, want trimmed managed route", validated)
	}

	cases := []struct {
		name  string
		table map[string]int
	}{
		{name: "invalid model", table: map[string]int{"bad model": 8000}},
		{name: "invalid port", table: map[string]int{"Qwen/Qwen2.5-7B-Instruct": 0}},
		{name: "unmanaged port", table: map[string]int{"Qwen/Qwen2.5-7B-Instruct": 8001}},
		{name: "model mismatch", table: map[string]int{"other/model": 8000}},
		{name: "duplicate port", table: map[string]int{"Qwen/Qwen2.5-7B-Instruct": 8000, "other/model": 8000}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := validateRoutingTableForActiveModels(tc.table, active); err == nil {
				t.Fatalf("validateRoutingTableForActiveModels(%+v) error=nil, want error", tc.table)
			}
		})
	}
}

func TestUpdateSettingsUpdatesServingEnvironmentToggles(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	t.Chdir(t.TempDir())

	trueVal := true
	falseVal := false

	imageVal := "vllm/vllm-openai:v0.4.3"
	updated, err := updateSettings(SettingsUpdate{
		HFCache:                 cacheDir,
		EnableCUDACompatibility: &trueVal,
		HFHubOffline:            &trueVal,
		VLLMImage:               &imageVal,
	})
	if err != nil {
		t.Fatalf("updateSettings() error=%v", err)
	}
	if !updated.EnableCUDACompatibility || !updated.HFHubOffline || updated.VLLMImage != imageVal {
		t.Fatalf("settings=%+v, want serving environment toggles enabled and vllm image set", updated)
	}

	updated, err = updateSettings(SettingsUpdate{HFCache: cacheDir})
	if err != nil {
		t.Fatalf("preserve updateSettings() error=%v", err)
	}
	if !updated.EnableCUDACompatibility || !updated.HFHubOffline || updated.VLLMImage != imageVal {
		t.Fatalf("settings=%+v, want omitted toggles and image preserved", updated)
	}

	updated, err = updateSettings(SettingsUpdate{
		HFCache:                 cacheDir,
		EnableCUDACompatibility: &falseVal,
		HFHubOffline:            &falseVal,
	})
	if err != nil {
		t.Fatalf("disable updateSettings() error=%v", err)
	}
	if updated.EnableCUDACompatibility || updated.HFHubOffline {
		t.Fatalf("settings=%+v, want serving environment toggles disabled", updated)
	}
	invalidImage := "invalid space name"
	_, err = updateSettings(SettingsUpdate{
		HFCache:   cacheDir,
		VLLMImage: &invalidImage,
	})
	if err == nil {
		t.Fatalf("expected error when setting invalid Docker image reference, got nil")
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

func TestInferenceProxyStripsAuthorizationForBackend(t *testing.T) {
	stubUpstreamAPIKeyLookup(t, func(port int) (string, error) {
		return "", nil
	})
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
		HFCache:      t.TempDir(),
		VLLMAPIKey:   "vllm-secret",
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "proxy-secret"}},
		RoutingTable: map[string]int{"test-model": mustAtoi(t, port)},
	}, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	req.Header.Set("Authorization", "Bearer proxy-secret")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if got := <-backendAuth; got != "" {
		t.Fatalf("backend Authorization=%q, want stripped header", got)
	}
}

func TestInferenceProxyForwardsLegacyContainerAPIKey(t *testing.T) {
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
	portNum := mustAtoi(t, port)
	stubUpstreamAPIKeyLookup(t, func(port int) (string, error) {
		if port != portNum {
			t.Fatalf("lookup port=%d, want %d", port, portNum)
		}
		return "legacy-upstream-secret", nil
	})
	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "proxy-secret"}},
		RoutingTable: map[string]int{"test-model": portNum},
	}, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	req.Header.Set("Authorization", "Bearer proxy-secret")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	if got := <-backendAuth; got != "Bearer legacy-upstream-secret" {
		t.Fatalf("backend Authorization=%q, want legacy upstream key", got)
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
	resetSettingsCache(t, Settings{HFCache: cacheDir, HFToken: "hf-secret", VLLMAPIKey: "api-secret", EnableCUDACompatibility: true, HFHubOffline: true, VLLMImage: "vllm/vllm-openai:v0.4.3"}, true)
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
	if status.HFCache != cacheDir || !status.HFTokenSet || !status.VLLMAPIKeySet || !status.EnableCUDACompatibility || !status.HFHubOffline || status.VLLMImage != "vllm/vllm-openai:v0.4.3" {
		t.Fatalf("status=%+v, want cache, secret flags, and vllm image", status)
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

func TestModelsInfoEndpointReportsDownloadedState(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	cacheDir := t.TempDir()

	// Pre-populate the HF hub with a model directory so IsModelDownloaded
	// reports a hit.
	hub := cacheDir + "/hub"
	snapDir := hub + "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc"
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		t.Fatalf("mkdir hub: %v", err)
	}
	if err := os.WriteFile(snapDir+"/weights.bin", []byte("hello"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	mux, _ := newDashboardMux(true)

	// Stub the network call.
	original := fetchHFModelInfoForTest
	fetchHFModelInfoForTest = func(model string) (HFModelInfo, error) {
		return HFModelInfo{Model: model, SizeBytes: 2048, Downloads: 1, Likes: 0}, nil
	}
	t.Cleanup(func() { fetchHFModelInfoForTest = original })

	// Hit case.
	req := httptest.NewRequest(http.MethodGet, "/api/models/info?model=Qwen/Qwen2.5-7B-Instruct", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	var payload map[string]interface{}
	if err := json.NewDecoder(rr.Body).Decode(&payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if payload["downloaded"] != true {
		t.Fatalf("downloaded=%v, want true on hit", payload["downloaded"])
	}
	if cached, _ := payload["cached_bytes"].(float64); cached <= 0 {
		t.Fatalf("cached_bytes=%v, want >0 on hit", payload["cached_bytes"])
	}

	// Miss case.
	req = httptest.NewRequest(http.MethodGet, "/api/models/info?model=meta-llama/Missing", nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	payload = nil
	if err := json.NewDecoder(rr.Body).Decode(&payload); err != nil {
		t.Fatalf("decode miss: %v", err)
	}
	if payload["downloaded"] != false {
		t.Fatalf("downloaded=%v, want false on miss", payload["downloaded"])
	}
	if cached, _ := payload["cached_bytes"].(float64); cached != 0 {
		t.Fatalf("cached_bytes=%v, want 0 on miss", payload["cached_bytes"])
	}
}

func TestRedactSensitive(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "equals-form api key",
			in:   "starting container with --api-key=supersecret-1234 done",
			want: "starting container with --api-key=<redacted> done",
		},
		{
			name: "space-form api key",
			in:   "docker run --api-key supersecret-1234 image",
			want: "docker run --api-key <redacted> image",
		},
		{
			name: "HF_TOKEN env",
			in:   "spawning with HF_TOKEN=hf_abcdef exited",
			want: "spawning with HF_TOKEN=<redacted> exited",
		},
		{
			name: "VLLM_API_KEY env",
			in:   "error: VLLM_API_KEY=vvv_secret returned 401",
			want: "error: VLLM_API_KEY=<redacted> returned 401",
		},
		{
			name: "bearer header",
			in:   "curl -H \"Authorization: Bearer token-xyz\" http://localhost",
			want: "curl -H Authorization: <redacted> http://localhost",
		},
		{
			name: "inference api key equals form",
			in:   "config inference_api_key=inf_key_12345 was set",
			want: "config inference_api_key=<redacted> was set",
		},
		{
			name: "inference key dash form",
			in:   "args: --inference-key inf_key_12345 -h",
			want: "args: --inference-key <redacted> -h",
		},
		{
			name: "negative no secrets",
			in:   "container failed to bind to port 8080",
			want: "container failed to bind to port 8080",
		},
		{
			name: "preserves newlines",
			in:   "first line HF_TOKEN=h1\nsecond line",
			want: "first line HF_TOKEN=<redacted>\nsecond line",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := redactSensitive(tc.in); got != tc.want {
				t.Fatalf("redactSensitive(%q)=%q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestWriteAndReadHeartbeat(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	resetSettingsCache(t, Settings{HFCache: t.TempDir(), RoutingTable: map[string]int{}}, true)

	now := time.Now()
	if err := writeHeartbeat(os.Getpid(), now); err != nil {
		t.Fatalf("writeHeartbeat() error=%v", err)
	}
	info, err := os.Stat(settingsFile)
	if err != nil {
		t.Fatalf("stat settings: %v", err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("settings mode=%#o, want 0600", got)
	}

	raw, err := os.ReadFile(settingsFile)
	if err != nil {
		t.Fatalf("read settings: %v", err)
	}
	var s Settings
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Heartbeat == nil {
		t.Fatal("heartbeat missing after write")
	}
	if s.Heartbeat.PID != os.Getpid() {
		t.Fatalf("heartbeat.PID=%d, want %d", s.Heartbeat.PID, os.Getpid())
	}
}

func TestDashboardIsLiveDetectsFreshHeartbeat(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache: t.TempDir(),
		Heartbeat: &Heartbeat{
			PID:      os.Getpid(),
			LastSeen: time.Now(),
		},
	}, true)
	if !dashboardIsLive() {
		t.Fatal("dashboardIsLive()=false, want true for fresh heartbeat + own PID")
	}
}

func TestDashboardIsLiveIgnoresStaleHeartbeat(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache: t.TempDir(),
		Heartbeat: &Heartbeat{
			PID:      os.Getpid(),
			LastSeen: time.Now().Add(-2 * heartbeatFreshnessWindow),
		},
	}, true)
	if dashboardIsLive() {
		t.Fatal("dashboardIsLive()=true, want false for stale heartbeat")
	}
}

func TestDashboardIsLiveIgnoresMissingHeartbeat(t *testing.T) {
	resetSettingsCache(t, Settings{HFCache: t.TempDir()}, true)
	if dashboardIsLive() {
		t.Fatal("dashboardIsLive()=true, want false when no heartbeat is set")
	}
}

func TestGuardMutationAllowsWhenDashboardNotRunning(t *testing.T) {
	prev := dashboardLiveForTest
	dashboardLiveForTest = func() bool { return false }
	t.Cleanup(func() { dashboardLiveForTest = prev })

	if err := guardMutation(false, "model start"); err != nil {
		t.Fatalf("guardMutation()=%v, want nil", err)
	}
}

func TestGuardMutationRefusesWhenDashboardLive(t *testing.T) {
	prev := dashboardLiveForTest
	dashboardLiveForTest = func() bool { return true }
	t.Cleanup(func() { dashboardLiveForTest = prev })

	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{
		HFCache: cacheDir,
		Heartbeat: &Heartbeat{
			PID:      os.Getpid(),
			LastSeen: time.Now(),
		},
	}, true)
	t.Chdir(t.TempDir())

	err := guardMutation(false, "model start")
	if err == nil {
		t.Fatal("guardMutation()=nil, want error")
	}
	if !strings.Contains(err.Error(), "dashboard is running") {
		t.Fatalf("error=%q, want dashboard running message", err.Error())
	}

	if err := guardMutation(true, "model start"); err != nil {
		t.Fatalf("guardMutation(force=true)=%v, want nil", err)
	}
}

func TestSecretSuffix(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "empty", in: "", want: ""},
		{name: "exactly four returns whole", in: "abcd", want: "abcd"},
		{name: "long returns last four", in: "abcdefgh", want: "efgh"},
		{name: "realistic token tail", in: "hf_xxxxxxxxxxxxcret", want: "cret"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := secretSuffix(c.in); got != c.want {
				t.Fatalf("secretSuffix(%q)=%q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestModelDownloadRefusesAlreadyDownloaded(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	t.Chdir(t.TempDir())

	// Place a model directory on disk.
	hub := cacheDir + "/hub"
	snapDir := hub + "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc"
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		t.Fatalf("mkdir hub: %v", err)
	}
	if err := os.WriteFile(snapDir+"/weights.bin", []byte("hello"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// Capture stdout/stderr so the error message doesn't pollute test
	// output. We only care about the error itself.
	oldOut, oldErr := os.Stdout, os.Stderr
	devNull, _ := os.Open(os.DevNull)
	os.Stdout = devNull
	os.Stderr = devNull
	err := cliModelDownload([]string{"Qwen/Qwen2.5-7B-Instruct"}, cliOptions{})
	os.Stdout, os.Stderr = oldOut, oldErr
	devNull.Close()

	if err == nil {
		t.Fatal("cliModelDownload()=nil, want error for already-downloaded model")
	}
	if !strings.Contains(err.Error(), "already downloaded") {
		t.Fatalf("error=%q, want already downloaded", err.Error())
	}
	if !strings.Contains(err.Error(), "cache delete") {
		t.Fatalf("error=%q, want it to suggest `cache delete`", err.Error())
	}
}

func TestModelDownloadPassesThroughWhenAbsent(t *testing.T) {
	// Pre-condition: no model on disk. Stub the actual download so this
	// verifies the CLI pre-flight without touching Docker.
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	t.Chdir(t.TempDir())
	originalDownload := downloadModelForCLI
	downloadModelForCLI = func(model string, hfCache string, hfToken string) (string, error) {
		if model != "meta-llama/Missing" {
			t.Fatalf("model=%q, want meta-llama/Missing", model)
		}
		if hfCache != cacheDir {
			t.Fatalf("hfCache=%q, want %q", hfCache, cacheDir)
		}
		return "download-container", nil
	}
	t.Cleanup(func() { downloadModelForCLI = originalDownload })

	oldOut, oldErr := os.Stdout, os.Stderr
	devNull, _ := os.Open(os.DevNull)
	os.Stdout = devNull
	os.Stderr = devNull
	err := cliModelDownload([]string{"meta-llama/Missing"}, cliOptions{})
	os.Stdout, os.Stderr = oldOut, oldErr
	devNull.Close()

	if err != nil {
		t.Fatalf("cliModelDownload(absent) error=%v", err)
	}
}

func TestModelInfoReportsDownloadedState(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	t.Chdir(t.TempDir())

	// Place a model directory on disk in the HF hub layout so
	// IsModelDownloaded finds it.
	hub := cacheDir + "/hub"
	snapDir := hub + "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc"
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		t.Fatalf("mkdir hub: %v", err)
	}
	if err := os.WriteFile(snapDir+"/weights.bin", []byte("model-payload"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// Stub the network call so the test never hits Hugging Face.
	original := fetchHFModelInfoForTest
	fetchHFModelInfoForTest = func(model string) (HFModelInfo, error) {
		return HFModelInfo{Model: model, SizeBytes: 1024, Downloads: 7, Likes: 3}, nil
	}
	t.Cleanup(func() { fetchHFModelInfoForTest = original })

	// Hit case: model is on disk, JSON should report downloaded=true with
	// the on-disk size and no cache warning.
	got, err := captureStdout(t, func() error {
		return cliModelInfo([]string{"Qwen/Qwen2.5-7B-Instruct"}, cliOptions{JSON: true})
	})
	if err != nil {
		t.Fatalf("cliModelInfo() error=%v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal([]byte(got), &payload); err != nil {
		t.Fatalf("decode stdout: %v\nbody=%s", err, got)
	}
	if payload["downloaded"] != true {
		t.Fatalf("downloaded=%v, want true (hit)", payload["downloaded"])
	}
	if cached, _ := payload["cached_bytes"].(float64); cached <= 0 {
		t.Fatalf("cached_bytes=%v, want >0 on hit", payload["cached_bytes"])
	}
	if payload["cache_warning"] != "" {
		t.Fatalf("cache_warning=%q, want empty on hit", payload["cache_warning"])
	}

	// Miss case: model not on disk, JSON should report downloaded=false and
	// cached_bytes=0.
	got, err = captureStdout(t, func() error {
		return cliModelInfo([]string{"meta-llama/Missing"}, cliOptions{JSON: true})
	})
	if err != nil {
		t.Fatalf("cliModelInfo(miss) error=%v", err)
	}
	payload = nil
	if err := json.Unmarshal([]byte(got), &payload); err != nil {
		t.Fatalf("decode stdout: %v\nbody=%s", err, got)
	}
	if payload["downloaded"] != false {
		t.Fatalf("downloaded=%v, want false (miss)", payload["downloaded"])
	}
	if cached, _ := payload["cached_bytes"].(float64); cached != 0 {
		t.Fatalf("cached_bytes=%v, want 0 on miss", payload["cached_bytes"])
	}
}

// captureStdout runs fn, returns whatever it wrote to stdout, and restores
// the original writer. Used by tests that print through the CLI's pretty
// printers without forcing the JSON path.
func captureStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	err = fn()
	w.Close()
	os.Stdout = old
	if err != nil {
		return "", err
	}
	buf, err := io.ReadAll(r)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func TestModelsDownloadEndpointReturns409WhenAlreadyDownloaded(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	cacheDir := t.TempDir()

	hub := cacheDir + "/hub"
	snapDir := hub + "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc"
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		t.Fatalf("mkdir hub: %v", err)
	}
	if err := os.WriteFile(snapDir+"/weights.bin", []byte("hello"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	resetSettingsCache(t, Settings{HFCache: cacheDir}, true)
	mux, _ := newDashboardMux(true)

	body := strings.NewReader(`{"model":"Qwen/Qwen2.5-7B-Instruct"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/models/download", body)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusConflict {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusConflict, rr.Body.String())
	}
	if got := rr.Header().Get("X-Cache-State"); got != "downloaded" {
		t.Fatalf("X-Cache-State=%q, want downloaded", got)
	}
	if got := rr.Header().Get("X-Cached-Bytes"); got == "" || got == "0" {
		t.Fatalf("X-Cached-Bytes=%q, want positive", got)
	}
	if !strings.Contains(rr.Body.String(), "already downloaded") {
		t.Fatalf("body=%q, want it to mention 'already downloaded'", rr.Body.String())
	}
}

func TestUpdateSettingsRejectsNonExistentTag(t *testing.T) {
	cacheDir := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cacheDir, HFHubOffline: false}, true)
	t.Chdir(t.TempDir())

	nonexistentImage := "vllm/vllm-openai:nonexistent-tag-12345"
	_, err := updateSettings(SettingsUpdate{
		HFCache:   cacheDir,
		VLLMImage: &nonexistentImage,
	})
	if err == nil {
		t.Fatalf("expected error when setting nonexistent Docker tag, got nil")
	}
	if !strings.Contains(err.Error(), "does not exist on Docker Hub") {
		t.Fatalf("unexpected error: %v", err)
	}
}
