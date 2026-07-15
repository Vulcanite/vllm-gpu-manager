package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vllm-manager/api"
)

func TestAppendAuditEventSanitizesSecretsAndReadsNewestFirst(t *testing.T) {
	t.Chdir(t.TempDir())

	appendAuditEvent(auditEvent{
		Action:  "model_deploy",
		Result:  "failed",
		Model:   "first/model",
		Port:    8080,
		Error:   "boom",
		Params:  map[string]interface{}{"model": "first/model", "api_key": "secret", "hf_token": "token"},
		Summary: "first",
	})
	appendAuditEvent(auditEvent{
		Action:  "model_deploy",
		Result:  "success",
		Model:   "second/model",
		Port:    8082,
		Params:  map[string]interface{}{"model": "second/model", "server_api_key": "secret", "trust_remote_code": true, "tokenizer_mode": "auto"},
		Summary: "second",
	})

	events, err := readAuditEvents(10)
	if err != nil {
		t.Fatalf("readAuditEvents() error=%v", err)
	}
	if len(events) != 2 {
		t.Fatalf("events=%d, want 2", len(events))
	}
	if events[0].Model != "second/model" || events[1].Model != "first/model" {
		t.Fatalf("events order=%+v, want newest first", events)
	}
	if _, ok := events[0].Params["server_api_key"]; ok {
		t.Fatalf("server_api_key was not redacted: %+v", events[0].Params)
	}
	if _, ok := events[1].Params["api_key"]; ok {
		t.Fatalf("api_key was not redacted: %+v", events[1].Params)
	}
	if _, ok := events[1].Params["hf_token"]; ok {
		t.Fatalf("hf_token was not redacted: %+v", events[1].Params)
	}
	if got := events[0].Params["trust_remote_code"]; got != true {
		t.Fatalf("trust_remote_code=%v, want true", got)
	}
	if got := events[0].Params["tokenizer_mode"]; got != "auto" {
		t.Fatalf("tokenizer_mode=%v, want auto", got)
	}
}

func TestAuditLogsEndpointReturnsEvents(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	t.Chdir(t.TempDir())
	resetSettingsCache(t, Settings{HFCache: t.TempDir()}, true)

	appendAuditEvent(auditEvent{Action: "proxy_api_key_created", Result: "success", Username: "alice", Summary: "Created proxy API key for alice"})
	mux, _ := newDashboardMuxWithOptions(dashboardOptions{ProxyEnabled: true})

	req := httptest.NewRequest(http.MethodGet, "/api/audit-logs?limit=5", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	var events []auditEvent
	if err := json.NewDecoder(rr.Body).Decode(&events); err != nil {
		t.Fatalf("decode audit response: %v", err)
	}
	if len(events) != 1 || events[0].Username != "alice" {
		t.Fatalf("events=%+v, want alice audit event", events)
	}
}

func TestAuditContainerDisplayNamePrefersContainerName(t *testing.T) {
	model := api.ActiveModel{ContainerName: "vllm-qwen3-8084"}
	if got := auditContainerDisplayName("abcdef123456", model); got != "vllm-qwen3-8084" {
		t.Fatalf("display name=%q, want container name", got)
	}
	if got := auditContainerDisplayName("abcdef123456", api.ActiveModel{}); got != "abcdef123456" {
		t.Fatalf("display name=%q, want container id fallback", got)
	}
}

func TestProxyUserHandlerWritesAuditEventWithoutSecret(t *testing.T) {
	t.Setenv("VLLM_MANAGER_PASSWORD", "")
	t.Chdir(t.TempDir())
	resetSettingsCache(t, Settings{HFCache: t.TempDir()}, true)
	mux, _ := newDashboardMuxWithOptions(dashboardOptions{ProxyEnabled: true})

	req := httptest.NewRequest(http.MethodPost, "/api/proxy-users", strings.NewReader(`{"username":"alice"}`))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusOK, rr.Body.String())
	}
	events, err := readAuditEvents(10)
	if err != nil {
		t.Fatalf("readAuditEvents() error=%v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events=%d, want 1", len(events))
	}
	if events[0].Action != "proxy_api_key_created" || events[0].Username != "alice" {
		t.Fatalf("event=%+v, want proxy key created for alice", events[0])
	}
	raw := rr.Body.String()
	if strings.Contains(events[0].Summary, "vllm_") || strings.Contains(events[0].Error, "vllm_") {
		t.Fatalf("audit event leaked generated key: %+v; response was %s", events[0], raw)
	}
}
