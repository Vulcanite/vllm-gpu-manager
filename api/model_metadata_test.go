package api

import (
	"strings"
	"testing"
)

func TestCheckDownloadPreflightRejectsGGUFModelName(t *testing.T) {
	fetch := func() (HFModelInfo, error) {
		t.Fatal("fetch should not be called for obvious GGUF names")
		return HFModelInfo{}, nil
	}

	_, err := CheckDownloadPreflight("TheBloke/Qwen2.5-7B-Instruct-GGUF", t.TempDir(), fetch)
	if err == nil {
		t.Fatal("CheckDownloadPreflight() error=nil, want GGUF rejection")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "gguf") {
		t.Fatalf("error=%q, want GGUF detail", err.Error())
	}
}

func TestCheckDownloadPreflightRejectsGGUFTag(t *testing.T) {
	info := KnownHFInfo(HFModelInfo{Model: "org/model", SizeBytes: 1024, Tags: []string{"gguf"}}, nil)

	_, err := CheckDownloadPreflight("org/model", t.TempDir(), info)
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

func TestValidateModelID(t *testing.T) {
	for _, ok := range []string{"Qwen/Qwen2.5-0.5B-Instruct", "gpt2", "meta-llama/Llama-3.1-8B", "org_1/m.v2"} {
		if err := ValidateModelID(ok); err != nil {
			t.Errorf("ValidateModelID(%q) = %v, want nil", ok, err)
		}
	}
	for _, bad := range []string{"", "a%zz", "org/m\x7f", "org/model name", "a/b/c", "../etc", "org/..", "org--x/y", "-org/x", "org/", "/x", "org/x?y"} {
		if err := ValidateModelID(bad); err == nil || !IsValidationError(err) {
			t.Errorf("ValidateModelID(%q) = %v, want validation error", bad, err)
		}
	}
}

func TestFetchHFModelInfoRejectsBadIDWithoutPanicking(t *testing.T) {
	// "a%zz" used to make http.NewRequest return nil and crash.
	if _, err := FetchHFModelInfo("a%zz", ""); err == nil || !IsValidationError(err) {
		t.Fatalf("FetchHFModelInfo(a%%zz) err=%v, want validation error", err)
	}
}

func TestDownloadExcludesOnlyWhenSafetensorsPresent(t *testing.T) {
	files := []string{"config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx", "original/consolidated.00.pth", "tokenizer.json"}
	excludes := downloadExcludes(files)
	var kept []string
	for _, f := range files {
		if !excluded(f, excludes) {
			kept = append(kept, f)
		}
	}
	if strings.Join(kept, ",") != "config.json,model.safetensors,tokenizer.json" {
		t.Fatalf("kept=%v", kept)
	}
	if got := downloadExcludes([]string{"config.json", "pytorch_model.bin"}); got != nil {
		t.Fatalf("bin-only repo excludes=%v, want nil so the weights still download", got)
	}
}
