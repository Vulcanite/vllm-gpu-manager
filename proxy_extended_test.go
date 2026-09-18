package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestInferenceProxyRejectsNonV1Paths(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:    t.TempDir(),
		ProxyUsers: []ProxyUser{{Username: "alice", APIKey: "test-key"}},
	}, true)

	req := httptest.NewRequest(http.MethodGet, "/invalid/path", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Errorf("status = %d, want %d", rr.Code, http.StatusForbidden)
	}
}

func TestInferenceProxyUnmappedModelReturns404(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "alice", APIKey: "test-key"}},
		RoutingTable: map[string]int{"mapped-model": 8080},
	}, true)

	body := `{"model": "unmapped-model", "messages": [{"role": "user", "content": "hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d", rr.Code, http.StatusNotFound)
	}
}

func TestInferenceProxyForwardsWithUpstreamAuth(t *testing.T) {
	receivedAuth := ""
	receivedBody := ""

	// Spin up a fake upstream vLLM server
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		receivedBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id": "chatcmpl-123", "choices": []}`))
	}))
	defer upstreamServer.Close()

	// Extract port from upstreamServer URL
	parts := strings.Split(upstreamServer.URL, ":")
	upstreamPort, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		t.Fatalf("failed to parse upstream port: %v", err)
	}

	resetSettingsCache(t, Settings{
		HFCache:      t.TempDir(),
		ProxyUsers:   []ProxyUser{{Username: "bob", APIKey: "client-proxy-key"}},
		VLLMAPIKey:   "upstream-vllm-token",
		RoutingTable: map[string]int{"test-model": upstreamPort},
	}, true)

	stubUpstreamAPIKeyLookup(t, func(port int) (string, error) {
		return "upstream-vllm-token", nil
	})

	requestPayload := `{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(requestPayload))
	req.Header.Set("Authorization", "Bearer client-proxy-key")
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("proxy status = %d, want %d; body = %s", rr.Code, http.StatusOK, rr.Body.String())
	}

	if receivedAuth != "Bearer upstream-vllm-token" {
		t.Errorf("upstream received auth %q, want %q", receivedAuth, "Bearer upstream-vllm-token")
	}

	if !strings.Contains(receivedBody, "test-model") {
		t.Errorf("upstream received body %q, want payload with test-model", receivedBody)
	}
}

func TestHandleV1ModelsOutputStructure(t *testing.T) {
	table := map[string]int{
		"model-a": 8000,
		"model-b": 8001,
	}

	rr := httptest.NewRecorder()
	handleV1Models(rr, table)

	if rr.Code != http.StatusOK {
		t.Fatalf("handleV1Models status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode /v1/models response: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("response object = %q, want 'list'", resp.Object)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("response data length = %d, want 2", len(resp.Data))
	}
}

func TestInferenceProxyRejectsOversizedBody(t *testing.T) {
	resetSettingsCache(t, Settings{
		HFCache:           t.TempDir(),
		ProxyUsers:        []ProxyUser{{Username: "alice", APIKey: "alice-key", RequestCount: 3}},
		RoutingTable:      map[string]int{"test-model": 8080},
		ProxyMaxBodyBytes: 64,
	}, true)

	body := `{"model":"test-model","prompt":"` + strings.Repeat("x", 128) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer alice-key")
	rr := httptest.NewRecorder()
	inferenceProxyHandler()(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want %d: %s", rr.Code, http.StatusRequestEntityTooLarge, rr.Body.String())
	}
	if got := getSettings().ProxyUsers[0].RequestCount; got != 3 {
		t.Fatalf("request count=%d, want unchanged 3 after rejected body", got)
	}
}
