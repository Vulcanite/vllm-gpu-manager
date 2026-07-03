package api

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCalculateDirectorySize(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.bin"), []byte("12345"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "nested", "b.bin"), []byte("123"), 0600); err != nil {
		t.Fatal(err)
	}

	size, err := calculateDirectorySize(dir)
	if err != nil {
		t.Fatalf("calculateDirectorySize() error=%v", err)
	}
	if size != 8 {
		t.Fatalf("size=%d, want 8", size)
	}
}

func TestGetDownloadedModelSizeRejectsMissingModel(t *testing.T) {
	_, _, err := GetDownloadedModelSize(t.TempDir(), "missing/model")
	if err == nil {
		t.Fatal("GetDownloadedModelSize() error=nil, want missing model error")
	}
	if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestDownloadContainerForModelNormalizesCase(t *testing.T) {
	snapshot := []managedVLLMContainer{
		{
			name: "vllm-download-qwen-2.5-7b-1234",
			labels: map[string]string{
				containerModelLabel: "Qwen/Qwen2.5-7B-Instruct",
			},
		},
		{
			name:   "vllm-qwen-9000",
			labels: map[string]string{containerModelLabel: "Qwen/Qwen2.5-7B-Instruct"},
		},
	}

	// Different case in user input must still match a download container.
	if got := downloadContainerForModel(snapshot, "qwen/qwen2.5-7b-instruct"); got != "vllm-download-qwen-2.5-7b-1234" {
		t.Fatalf("downloadContainerForModel()=%q, want the lowercase-normalized match", got)
	}
	if got := downloadContainerForModel(snapshot, "Qwen/Qwen2.5-7B-Instruct"); got != "vllm-download-qwen-2.5-7b-1234" {
		t.Fatalf("downloadContainerForModel()=%q, want case-insensitive match", got)
	}
	// A non-matching model must not be reported as a download.
	if got := downloadContainerForModel(snapshot, "different/model"); got != "" {
		t.Fatalf("downloadContainerForModel()=%q, want empty for unmatched model", got)
	}
	// An empty model must not be treated as a match.
	if got := downloadContainerForModel(snapshot, ""); got != "" {
		t.Fatalf("downloadContainerForModel(empty)=%q, want empty", got)
	}
}

func TestIsModelDownloaded(t *testing.T) {
	cache := t.TempDir()
	// Hub layout: <hfCache>/hub/models--<owner>--<repo>/snapshots/...
	hub := filepath.Join(cache, "hub")
	if err := os.MkdirAll(filepath.Join(hub, "models--Qwen--Qwen2.5-7B-Instruct", "snapshots", "abc"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hub, "models--Qwen--Qwen2.5-7B-Instruct", "snapshots", "abc", "weights.bin"), []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	// Hit: model directory exists, with on-disk bytes > 0.
	ok, size, err := IsModelDownloaded("Qwen/Qwen2.5-7B-Instruct", cache)
	if err != nil {
		t.Fatalf("IsModelDownloaded() error=%v", err)
	}
	if !ok {
		t.Fatal("IsModelDownloaded()=false for present model, want true")
	}
	if size == 0 {
		t.Fatal("IsModelDownloaded() size=0, want >0 for present weights")
	}

	// Miss: model not on disk, must not error.
	ok, size, err = IsModelDownloaded("meta-llama/Missing", cache)
	if err != nil {
		t.Fatalf("IsModelDownloaded() error=%v, want nil for absent model", err)
	}
	if ok {
		t.Fatal("IsModelDownloaded()=true for absent model, want false")
	}
	if size != 0 {
		t.Fatalf("IsModelDownloaded() size=%d, want 0 for absent model", size)
	}

	// Empty model is a no-op miss, never an error.
	ok, _, err = IsModelDownloaded("", cache)
	if err != nil || ok {
		t.Fatalf("IsModelDownloaded(empty)=ok=%v err=%v, want false/nil", ok, err)
	}

	// Path that has a hub dir but no model-* entries: miss, no error.
	emptyCache := t.TempDir()
	if err := os.MkdirAll(filepath.Join(emptyCache, "hub"), 0755); err != nil {
		t.Fatal(err)
	}
	ok, _, err = IsModelDownloaded("foo/bar", emptyCache)
	if err != nil || ok {
		t.Fatalf("IsModelDownloaded()=ok=%v err=%v on empty hub, want false/nil", ok, err)
	}

	// Empty cache path: must return a non-nil error so the caller can flag
	// the misconfiguration in the status payload.
	if _, _, err := IsModelDownloaded("foo/bar", ""); err == nil {
		t.Fatal("IsModelDownloaded()=nil error for empty cache path, want error")
	}
}
