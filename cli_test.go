package main

import (
	"bytes"
	"strings"
	"testing"

	"vllm-manager/api"
)

func TestPrintActiveModelSections(t *testing.T) {
	var out bytes.Buffer
	printActiveModelSections(&out, []api.ActiveModel{{
		ContainerID:   "a6acf889e92248b7c6aa745dd2f97c288f1e6297b511cbe60670609dbed94bad",
		ContainerName: "vllm-Qwen2.5-VL-72B-8080",
		Model:         "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
		State:         "running",
		IsReady:       true,
		APIUrl:        "http://localhost:8080/v1",
		GPUIDs:        "0,1",
		GPUs:          "2",
		MemUtil:       "0.75",
		MaxLen:        "128000",
		MaxNumSeqs:    "64",
		KVCacheDType:  "auto",
		ToolParser:    "hermes",
	}})

	rendered := out.String()
	for _, want := range []string{
		"## vllm-Qwen2.5-VL-72B-8080",
		"- Model: Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
		"- Status: running (ready: yes)",
		"- Serve: http://localhost:8080/v1",
		"- GPUs: 0,1, tensor parallel 2, memory utilization 0.75",
		"- Runtime: max length 128000, max seqs 64, KV cache auto, parser hermes",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered output missing %q:\n%s", want, rendered)
		}
	}
	if strings.Contains(rendered, "|") {
		t.Fatalf("active model output should be section-based, got table-like output:\n%s", rendered)
	}
}

func TestGroupCommandsShowScopedHelp(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "gpu", args: []string{"gpu"}, want: "Available Commands:\n  list"},
		{name: "model", args: []string{"model"}, want: "Available Commands:\n  active"},
		{name: "cache", args: []string{"cache"}, want: "Available Commands:\n  delete"},
		{name: "settings", args: []string{"settings"}, want: "Available Commands:\n  get"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			var errOut bytes.Buffer
			cmd := newCLICommand(&out, &errOut)
			cmd.SetArgs(tt.args)

			if err := cmd.Execute(); err != nil {
				t.Fatalf("expected scoped help without error, got %v", err)
			}
			if got := errOut.String(); got != "" {
				t.Fatalf("expected no stderr output, got %q", got)
			}
			if rendered := out.String(); !strings.Contains(rendered, tt.want) {
				t.Fatalf("help output missing %q:\n%s", tt.want, rendered)
			}
		})
	}
}

func TestRootHelpIncludesDashboardMode(t *testing.T) {
	var out bytes.Buffer
	var errOut bytes.Buffer
	cmd := newCLICommand(&out, &errOut)
	cmd.SetArgs([]string{"--help"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("help returned error: %v", err)
	}
	rendered := out.String()
	for _, want := range []string{"--dashboard", "--port", "--no-proxy", "gpu", "model", "settings"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("help output missing %q:\n%s", want, rendered)
		}
	}
	if strings.Contains(rendered, "      --proxy ") || strings.Contains(rendered, "\n      --proxy\t") {
		t.Fatalf("help output should not advertise --proxy:\n%s", rendered)
	}
}

func TestCLISettingsGetField(t *testing.T) {
	cache := t.TempDir()
	resetSettingsCache(t, Settings{HFCache: cache}, true)

	got, err := captureStdout(t, func() error {
		return cliSettingsGet(cliOptions{}, "hf_cache")
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(got) != cache {
		t.Fatalf("hf_cache=%q, want %q", strings.TrimSpace(got), cache)
	}
	if err := cliSettingsGet(cliOptions{}, "unknown"); err == nil {
		t.Fatal("unknown field should fail")
	}
}
