package api

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCheckModelFitsInVRAM(t *testing.T) {
	// Setup mock HF cache
	hfCache := t.TempDir()
	model := "test-model"
	modelDir := filepath.Join(hfCache, "hub", "models--test-model")
	os.MkdirAll(modelDir, 0755)

	// Create a sparse file to mock 10GB model weights
	dummySizeFile := filepath.Join(modelDir, "dummy.bin")
	f, err := os.Create(dummySizeFile)
	if err != nil {
		t.Fatalf("failed to create dummy file: %v", err)
	}
	// 10 GiB = 10 * 1024 * 1024 * 1024
	f.Truncate(10 * 1024 * 1024 * 1024)
	f.Close()

	// Create config.json to mock KV cache estimate parameters
	configPath := filepath.Join(modelDir, "config.json")
	configJSON := `{
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"hidden_size": 4096,
		"num_key_value_heads": 8,
		"max_position_embeddings": 8192
	}`
	os.WriteFile(configPath, []byte(configJSON), 0644)

	tests := []struct {
		name      string
		gpus      []GPUInfo
		gpusCount int
		parallel  string
		memUtil   string
		maxLen    string
		kvDType   string
		wantErr   string
	}{
		{
			name:      "Fits within VRAM",
			gpus:      []GPUInfo{{ID: 0, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}}, // 24GB
			gpusCount: 1,
			parallel:  ParallelismTensor,
			memUtil:   "0.9", // budget ~21.6GB. Weights ~10.8GB. Fits.
			maxLen:    "4096",
			kvDType:   "auto",
			wantErr:   "",
		},
		{
			name:      "Fails: weights don't fit in current free",
			gpus:      []GPUInfo{{ID: 0, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 8 * 1024 * 1024 * 1024}}, // 8GB free
			gpusCount: 1,
			parallel:  ParallelismTensor,
			memUtil:   "0.9",
			maxLen:    "4096",
			kvDType:   "auto",
			wantErr:   "only has 8.00 GiB currently free",
		},
		{
			name:      "Fails: max_model_len exceeds model limit",
			gpus:      []GPUInfo{{ID: 0, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}},
			gpusCount: 1,
			parallel:  ParallelismTensor,
			memUtil:   "0.9",
			maxLen:    "16384",
			kvDType:   "auto",
			wantErr:   "exceeds the selected model's default maximum context length",
		},
		{
			name:      "Fails: KV heads not divisible by tensor parallel size",
			gpus:      []GPUInfo{{ID: 0, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}, {ID: 1, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}, {ID: 2, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}},
			gpusCount: 3, // 8 KV heads % 3 != 0
			parallel:  ParallelismTensor,
			memUtil:   "0.9",
			maxLen:    "4096",
			kvDType:   "auto",
			wantErr:   "not divisible by tensor_parallel_size (3)",
		},
		{
			name:      "Pipeline mode skips tensor KV-head divisibility",
			gpus:      []GPUInfo{{ID: 0, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}, {ID: 1, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}, {ID: 2, MemoryTotal: 24 * 1024 * 1024 * 1024, MemoryFree: 24 * 1024 * 1024 * 1024}},
			gpusCount: 3,
			parallel:  ParallelismPipeline,
			memUtil:   "0.9",
			maxLen:    "4096",
			kvDType:   "auto",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckModelFitsInVRAM(model, hfCache, tt.gpus, tt.gpusCount, tt.parallel, tt.memUtil, tt.maxLen, tt.kvDType)
			if tt.wantErr == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("expected error containing %q, got %v", tt.wantErr, err)
				}
			}
		})
	}
}
