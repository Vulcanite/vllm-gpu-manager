package api

import (
	"strings"
	"testing"
)

func TestCheckDownloadPreflightRejectsGGUFModelName(t *testing.T) {
	errMsg := "fetch should not be called for obvious GGUF names"
	prev := fetchHFModelInfoForTest
	fetchHFModelInfoForTest = func(model string, hfToken string) (HFModelInfo, error) {
		t.Fatal(errMsg)
		return HFModelInfo{}, nil
	}
	t.Cleanup(func() { fetchHFModelInfoForTest = prev })

	_, err := CheckDownloadPreflight("TheBloke/Qwen2.5-7B-Instruct-GGUF", t.TempDir(), "")
	if err == nil {
		t.Fatal("CheckDownloadPreflight() error=nil, want GGUF rejection")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "gguf") {
		t.Fatalf("error=%q, want GGUF detail", err.Error())
	}
}

func TestCheckDownloadPreflightRejectsGGUFTag(t *testing.T) {
	prev := fetchHFModelInfoForTest
	fetchHFModelInfoForTest = func(model string, hfToken string) (HFModelInfo, error) {
		return HFModelInfo{Model: model, SizeBytes: 1024, Tags: []string{"gguf"}}, nil
	}
	t.Cleanup(func() { fetchHFModelInfoForTest = prev })

	_, err := CheckDownloadPreflight("org/model", t.TempDir(), "")
	if err == nil {
		t.Fatal("CheckDownloadPreflight() error=nil, want GGUF rejection")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "gguf") {
		t.Fatalf("error=%q, want GGUF detail", err.Error())
	}
}

func TestQuantizationFromStringsDetectsGGUF(t *testing.T) {
	if got := quantizationFromStrings("quantization:GGUF"); got != "GGUF" {
		t.Fatalf("quantizationFromStrings()=%q, want GGUF", got)
	}
}
