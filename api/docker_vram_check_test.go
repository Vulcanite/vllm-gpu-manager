package api

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func writeVRAMTestModel(t *testing.T, hfCache string, model string, config string, weightBytes int) {
	t.Helper()
	root := filepath.Join(HFHubPath(hfCache), modelCacheDirName(model))
	snapshot := filepath.Join(root, "snapshots", "test-revision")
	if err := os.MkdirAll(filepath.Join(root, "refs"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(snapshot, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "refs", "main"), []byte("test-revision"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "config.json"), []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snapshot, "weights.bin"), make([]byte, weightBytes), 0600); err != nil {
		t.Fatal(err)
	}
	clearDownloadedModelSizeCache(root)
}

func tinyVRAMConfig(maxLen int) string {
	return `{
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"hidden_size": 64,
		"head_dim": 16,
		"torch_dtype": "float16",
		"max_position_embeddings": ` + strconv.Itoa(maxLen) + `
	}`
}

func deepSeekV4VRAMConfig(maxLen int) string {
	return `{
		"model_type": "deepseek_v4",
		"num_hidden_layers": 43,
		"num_attention_heads": 64,
		"num_key_value_heads": 1,
		"hidden_size": 4096,
		"head_dim": 512,
		"torch_dtype": "bfloat16",
		"max_position_embeddings": ` + strconv.Itoa(maxLen) + `
	}`
}

func TestCheckModelFitsInVRAMAllowsSmallModel(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/tiny"
	writeVRAMTestModel(t, hfCache, model, tinyVRAMConfig(1024), 1024)

	err := CheckModelFitsInVRAM(model, hfCache, []GPUInfo{{
		ID:          0,
		MemoryTotal: 16 * 1024 * 1024 * 1024,
		MemoryFree:  16 * 1024 * 1024 * 1024,
	}}, 1, ParallelismTensor, "0.9", "128", "auto")
	if err != nil {
		t.Fatalf("CheckModelFitsInVRAM() error=%v", err)
	}
}

func TestCheckModelFitsInVRAMRejectsMaxLenAboveModelLimit(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/tiny"
	writeVRAMTestModel(t, hfCache, model, tinyVRAMConfig(128), 1024)

	err := CheckModelFitsInVRAM(model, hfCache, []GPUInfo{{
		ID:          0,
		MemoryTotal: 16 * 1024 * 1024 * 1024,
		MemoryFree:  16 * 1024 * 1024 * 1024,
	}}, 1, ParallelismTensor, "0.9", "256", "auto")
	if err == nil {
		t.Fatal("CheckModelFitsInVRAM() error=nil, want max len validation error")
	}
	if !strings.Contains(err.Error(), "exceeds the selected model's default maximum context length") {
		t.Fatalf("error=%q, want max length detail", err.Error())
	}
}

func TestCheckModelFitsInVRAMRejectsMaxLenAboveModelLimitWithPartialConfig(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/partial-config"
	writeVRAMTestModel(t, hfCache, model, `{"max_position_embeddings": 128}`, 1024)

	err := CheckModelFitsInVRAM(model, hfCache, []GPUInfo{{
		ID:          0,
		MemoryTotal: 16 * 1024 * 1024 * 1024,
		MemoryFree:  16 * 1024 * 1024 * 1024,
	}}, 1, ParallelismTensor, "0.9", "256", "auto")
	if err == nil {
		t.Fatal("CheckModelFitsInVRAM() error=nil, want max len validation error")
	}
	if !strings.Contains(err.Error(), "exceeds the selected model's default maximum context length") {
		t.Fatalf("error=%q, want max length detail", err.Error())
	}
}

func TestCheckModelFitsInVRAMRejectsInsufficientFreeMemory(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/tiny"
	writeVRAMTestModel(t, hfCache, model, tinyVRAMConfig(1024), 2*1024*1024)

	err := CheckModelFitsInVRAM(model, hfCache, []GPUInfo{{
		ID:          0,
		MemoryTotal: 16 * 1024 * 1024 * 1024,
		MemoryFree:  1024 * 1024,
	}}, 1, ParallelismTensor, "0.9", "128", "auto")
	if err == nil {
		t.Fatal("CheckModelFitsInVRAM() error=nil, want free memory validation error")
	}
	if !strings.Contains(err.Error(), "currently free") {
		t.Fatalf("error=%q, want free memory detail", err.Error())
	}
}

func TestCheckModelFitsInVRAMAllowsDeepSeekV4MLAKVHeadsWithTensorParallel(t *testing.T) {
	hfCache := t.TempDir()
	model := "deepseek-ai/DeepSeek-V4-Flash-DSpark"
	writeVRAMTestModel(t, hfCache, model, deepSeekV4VRAMConfig(1048576), 1024)

	err := CheckModelFitsInVRAM(model, hfCache, []GPUInfo{
		{ID: 0, MemoryTotal: 16 * 1024 * 1024 * 1024, MemoryFree: 16 * 1024 * 1024 * 1024},
		{ID: 1, MemoryTotal: 16 * 1024 * 1024 * 1024, MemoryFree: 16 * 1024 * 1024 * 1024},
	}, 2, ParallelismTensor, "0.95", "32768", "fp8")
	if err != nil {
		t.Fatalf("CheckModelFitsInVRAM() error=%v", err)
	}
}

func TestGetModelKVCacheEstimateDetectsMultimodalMetadata(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/vision-model"
	writeVRAMTestModel(t, hfCache, model, `{
		"model_type": "qwen2_5_vl",
		"architectures": ["Qwen2_5_VLForConditionalGeneration"],
		"vision_config": {"hidden_size": 1024},
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"hidden_size": 64,
		"head_dim": 16,
		"torch_dtype": "float16",
		"max_position_embeddings": 1024
	}`, 1024)

	estimate, err := GetModelKVCacheEstimate(hfCache, model)
	if err != nil {
		t.Fatalf("GetModelKVCacheEstimate() error=%v", err)
	}
	if !estimate.IsMultimodal {
		t.Fatal("IsMultimodal=false, want true from cached config metadata")
	}
}

func TestGetModelKVCacheEstimateDoesNotGuessMultimodalFromName(t *testing.T) {
	hfCache := t.TempDir()
	model := "test/model-with-vision-in-name"
	writeVRAMTestModel(t, hfCache, model, tinyVRAMConfig(1024), 1024)

	estimate, err := GetModelKVCacheEstimate(hfCache, model)
	if err != nil {
		t.Fatalf("GetModelKVCacheEstimate() error=%v", err)
	}
	if estimate.IsMultimodal {
		t.Fatal("IsMultimodal=true without multimodal config metadata")
	}
}
