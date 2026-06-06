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
