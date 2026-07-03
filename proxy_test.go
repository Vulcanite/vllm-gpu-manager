package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLookupServerAPIKeyByPort(t *testing.T) {
	keys := map[string]string{
		"vllm-llama-3-8080":     "key-a",
		"vllm-llama-3-8081":     "key-b",
		"vllm-mistral-9000":     "key-c",
		"vllm-no-port":          "key-d",
		"vllm-llama-3-8080-bad": "key-e", // should not match
	}

	cases := []struct {
		name string
		port int
		want string
	}{
		{"matching suffix", 8080, "key-a"},
		{"another match", 9000, "key-c"},
		{"no match", 9999, ""},
		{"empty map and zero port", 0, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := lookupServerAPIKeyByPort(keys, tc.port)
			if got != tc.want {
				t.Fatalf("port=%d got %q, want %q", tc.port, got, tc.want)
			}
		})
	}
}

func TestLookupServerAPIKeyByPortIgnoresEmptyValues(t *testing.T) {
	keys := map[string]string{
		"vllm-empty-8080": "",
	}
	if got := lookupServerAPIKeyByPort(keys, 8080); got != "" {
		t.Fatalf("expected empty result for empty key, got %q", got)
	}
}

func TestLookupServerAPIKeyByPortNilMap(t *testing.T) {
	if got := lookupServerAPIKeyByPort(nil, 8080); got != "" {
		t.Fatalf("nil map should return empty string, got %q", got)
	}
}

func TestInferenceProxyRejectsMissingInvalidOrAbsentUserKeys(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "alice-key"}},
		RoutingTable: map[string]int{"test-model": 8080},
	}, true)

	for _, tt := range []struct {
		name string
		auth string
	}{
		{name: "missing auth"},
		{name: "invalid auth", auth: "Bearer wrong-key"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
			if tt.auth != "" {
				req.Header.Set("Authorization", tt.auth)
			}
			rr := httptest.NewRecorder()

			inferenceProxyHandler()(rr, req)

			if rr.Code != http.StatusUnauthorized {
				t.Fatalf("status=%d, want %d", rr.Code, http.StatusUnauthorized)
			}
		})
	}

	resetSettingsCache(t, Settings{HFCache: t.TempDir(), RoutingTable: map[string]int{"test-model": 8080}}, true)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer alice-key")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d, want %d when no proxy users exist", rr.Code, http.StatusUnauthorized)
	}
}

func TestV1ModelsRequiresAuthAndDoesNotIncrementRequestCount(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "alice-key", RequestCount: 7}},
		RoutingTable: map[string]int{"exact/model": 8080},
	}, true)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer alice-key")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(rr.Body).Decode(&payload); err != nil {
		t.Fatalf("decode models response: %v", err)
	}
	if len(payload.Data) != 1 || payload.Data[0].ID != "exact/model" {
		t.Fatalf("models=%+v, want exact routed model name", payload.Data)
	}
	if got := getSettings().ProxyUsers[0].RequestCount; got != 7 {
		t.Fatalf("request count=%d, want unchanged 7", got)
	}
}

func TestInferenceProxyIncrementsOnlyMatchingUserAndRevokedKeyFails(t *testing.T) {
	t.Chdir(t.TempDir())
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()
	port := backend.URL[strings.LastIndex(backend.URL, ":")+1:]

	resetSettingsCache(t, Settings{
		HFCache: t.TempDir(),
		ProxyUsers: []ProxyUser{
			{Username: "alice", APIKey: "alice-key", RequestCount: 2},
			{Username: "bob", APIKey: "bob-key", RequestCount: 4},
		},
		RoutingTable: map[string]int{"test-model": mustAtoi(t, port)},
	}, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	req.Header.Set("Authorization", "Bearer bob-key")
	rr := httptest.NewRecorder()
	inferenceProxyHandler()(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}

	settings := getSettings()
	counts := map[string]int64{}
	for _, user := range settings.ProxyUsers {
		counts[user.Username] = user.RequestCount
	}
	if counts["alice"] != 2 || counts["bob"] != 5 {
		t.Fatalf("counts=%+v, want alice unchanged and bob incremented", counts)
	}

	if err := revokeProxyUser("bob"); err != nil {
		t.Fatalf("revokeProxyUser() error=%v", err)
	}
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	req.Header.Set("Authorization", "Bearer bob-key")
	rr = httptest.NewRecorder()
	inferenceProxyHandler()(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d, want %d for revoked key", rr.Code, http.StatusUnauthorized)
	}
}

func TestInferenceProxyUnknownModelDoesNotIncrement(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "alice-key", RequestCount: 3}},
		RoutingTable: map[string]int{"known-model": 8080},
	}, true)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"unknown-model"}`))
	req.Header.Set("Authorization", "Bearer alice-key")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("status=%d, want %d", rr.Code, http.StatusNotFound)
	}
	if got := getSettings().ProxyUsers[0].RequestCount; got != 3 {
		t.Fatalf("request count=%d, want unchanged 3", got)
	}
}

func TestCreateProxyUserValidationAndReveal(t *testing.T) {
	t.Chdir(t.TempDir())
	resetSettingsCache(t, Settings{HFCache: t.TempDir()}, true)

	user, err := createProxyUser(" Alice_1 ")
	if err != nil {
		t.Fatalf("createProxyUser() error=%v", err)
	}
	if user.Username != "Alice_1" {
		t.Fatalf("username=%q, want trimmed Alice_1", user.Username)
	}
	if !strings.HasPrefix(user.APIKey, "vllm_") || len(user.APIKey) != 45 {
		t.Fatalf("api key=%q, want vllm_ plus 40 chars", user.APIKey)
	}
	key, err := revealProxyUserKey("alice_1")
	if err != nil {
		t.Fatalf("revealProxyUserKey() error=%v", err)
	}
	if key != user.APIKey {
		t.Fatalf("revealed key=%q, want generated key", key)
	}

	for _, tt := range []struct {
		name     string
		username string
		wantCode int
	}{
		{name: "empty", username: " ", wantCode: http.StatusBadRequest},
		{name: "invalid character", username: "bad user", wantCode: http.StatusBadRequest},
		{name: "duplicate case insensitive", username: "alice_1", wantCode: http.StatusConflict},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := createProxyUser(tt.username)
			if err == nil {
				t.Fatal("createProxyUser() error=nil, want error")
			}
			appErr, ok := err.(appError)
			if !ok {
				t.Fatalf("error type=%T, want appError", err)
			}
			if appErr.Status != tt.wantCode {
				t.Fatalf("status=%d, want %d", appErr.Status, tt.wantCode)
			}
		})
	}
}
