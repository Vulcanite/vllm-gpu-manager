package api

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseImportedCommandValidationErrors(t *testing.T) {
	tests := []struct {
		name    string
		command string
		wantErr string
	}{
		{
			name:    "empty command",
			command: "   ",
			wantErr: "command is required",
		},
		{
			name:    "no model or subcommands",
			command: "docker run -it vllm/vllm-openai:latest",
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseImportedCommand(tt.command)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
				}
			}
		})
	}
}

func TestParseImportedCommandAdvancedVLLMFlags(t *testing.T) {
	command := `vllm serve meta-llama/Llama-3.3-70B-Instruct \
		--port 8000 \
		--tensor-parallel-size 4 \
		--tool-call-parser hermes \
		--tokenizer-mode mistral \
		--reasoning-parser deepseek_r1 \
		--enable-expert-parallel \
		--moe-backend marlin \
		--max-model-len 16384 \
		--max-num-seqs 128 \
		--gpu-memory-utilization 0.88 \
		--kv-cache-dtype fp8_e4m3 \
		--dtype bfloat16 \
		--chat-template /path/to/template.jinja \
		--max-seq-len-to-capture 8192`

	parsed, err := ParseImportedCommand(command)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if parsed.Fields.Model != "meta-llama/Llama-3.3-70B-Instruct" {
		t.Errorf("model = %q, want %q", parsed.Fields.Model, "meta-llama/Llama-3.3-70B-Instruct")
	}
	if parsed.Fields.Port != 8000 {
		t.Errorf("port = %d, want 8000", parsed.Fields.Port)
	}
	if parsed.Fields.GPUs != 4 {
		t.Errorf("gpus = %d, want 4", parsed.Fields.GPUs)
	}
	if parsed.Fields.ToolParser != "hermes" {
		t.Errorf("tool_parser = %q, want hermes", parsed.Fields.ToolParser)
	}
	if parsed.Fields.TokenizerMode != "mistral" {
		t.Errorf("tokenizer_mode = %q, want mistral", parsed.Fields.TokenizerMode)
	}
	if parsed.Fields.ReasoningParser != "deepseek_r1" {
		t.Errorf("reasoning_parser = %q, want deepseek_r1", parsed.Fields.ReasoningParser)
	}
	if !parsed.Fields.ExpertParallel {
		t.Errorf("enable_expert_parallel = false, want true")
	}
	if parsed.Fields.MoeBackend != "marlin" {
		t.Errorf("moe_backend = %q, want marlin", parsed.Fields.MoeBackend)
	}
	if parsed.Fields.MaxLen != "16384" {
		t.Errorf("max_len = %q, want 16384", parsed.Fields.MaxLen)
	}
	if parsed.Fields.MaxNumSeqs != "128" {
		t.Errorf("max_num_seqs = %q, want 128", parsed.Fields.MaxNumSeqs)
	}
	if parsed.Fields.MemUtil != "0.88" {
		t.Errorf("mem_util = %q, want 0.88", parsed.Fields.MemUtil)
	}
	if parsed.Fields.KVCacheDType != "fp8_e4m3" {
		t.Errorf("kv_cache_dtype = %q, want fp8_e4m3", parsed.Fields.KVCacheDType)
	}
	if parsed.Fields.DType != "bfloat16" {
		t.Errorf("dtype = %q, want bfloat16", parsed.Fields.DType)
	}

	expectedExtra := []string{
		"--chat-template", "/path/to/template.jinja",
		"--max-seq-len-to-capture", "8192",
	}
	if !reflect.DeepEqual(parsed.ExtraVLLMArgs, expectedExtra) {
		t.Errorf("extra args = %v, want %v", parsed.ExtraVLLMArgs, expectedExtra)
	}
}

func TestParseImportedDockerFlagsClassification(t *testing.T) {
	command := `docker run -d --rm --name my-server \
		-p 8080:8080 -v /cache:/hf_cache -e HF_HOME=/hf_cache \
		--network host --device /dev/nvidia0 --mount type=bind,source=/a,target=/b \
		vllm/vllm-openai:v0.23.0 Qwen/Qwen2.5-7B-Instruct`

	parsed, err := ParseImportedCommand(command)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if parsed.Fields.Model != "Qwen/Qwen2.5-7B-Instruct" {
		t.Errorf("model = %q, want %q", parsed.Fields.Model, "Qwen/Qwen2.5-7B-Instruct")
	}

	// Verify blocked flags
	for _, blocked := range []string{"--network", "--device", "--mount"} {
		if !containsString(parsed.Unsupported, blocked) {
			t.Errorf("expected unsupported to contain %q, got %v", blocked, parsed.Unsupported)
		}
	}

	// Verify ignored flags
	ignoredCombined := strings.Join(parsed.Ignored, " ")
	for _, ignored := range []string{"--name", "-v", "-e"} {
		if !strings.Contains(ignoredCombined, ignored) {
			t.Errorf("expected ignored to contain %q, got %v", ignored, parsed.Ignored)
		}
	}
}
