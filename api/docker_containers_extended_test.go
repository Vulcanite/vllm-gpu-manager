package api

import (
	"testing"
)

func TestBuildVLLMServeArgsComprehensive(t *testing.T) {
	spec := LaunchSpec{
		Model:                     "Qwen/Qwen2.5-Coder-32B-Instruct",
		ServedModelName:           "qwen-coder",
		Port:                      8080,
		GPUs:                      2,
		GPUIDs:                    "0,1",
		MemUtil:                   "0.85",
		MaxLen:                    "16384",
		MaxNumSeqs:                "128",
		CPUOffloadGB:              "8",
		DType:                     "bfloat16",
		ParallelismMode:           ParallelismTensor,
		KVCacheDType:              "fp8",
		ToolParser:                "hermes",
		TokenizerMode:             "mistral",
		ReasoningParser:           "deepseek_r1",
		TrustRemoteCode:           true,
		EnablePrefixCaching:       true,
		BlockSize:                 "16",
		MoeBackend:                "marlin",
		ExpertParallel:            true,
		DisableFlashInferAutotune: true,
		SpeculativeConfig:         `{"method":"eagle"}`,
		ExtraVLLMArgs:             []string{"--chat-template", "/tmp/template.jinja"},
	}

	args := buildVLLMServeArgs(spec, "sk-secret-key")

	expectedTokens := []string{
		"--model", "Qwen/Qwen2.5-Coder-32B-Instruct",
		"--served-model-name", "qwen-coder",
		"--port", "8080",
		"--tensor-parallel-size", "2",
		"--gpu-memory-utilization", "0.85",
		"--max-model-len", "16384",
		"--max-num-seqs", "128",
		"--cpu-offload-gb", "8",
		"--dtype", "bfloat16",
		"--kv-cache-dtype", "fp8",
		"--tool-call-parser", "hermes",
		"--tokenizer-mode", "mistral",
		"--reasoning-parser", "deepseek_r1",
		"--trust-remote-code",
		"--enable-prefix-caching",
		"--block-size", "16",
		"--moe-backend", "marlin",
		"--enable-expert-parallel",
		"--no-enable-flashinfer-autotune",
		"--speculative-config", `{"method":"eagle"}`,
		"--api-key", "sk-secret-key",
		"--chat-template", "/tmp/template.jinja",
	}

	for _, token := range expectedTokens {
		if !containsArg(args, token) {
			t.Errorf("expected buildVLLMServeArgs to contain %q, but got args: %v", token, args)
		}
	}
}

func TestBuildVLLMServeArgsPipelineMode(t *testing.T) {
	spec := LaunchSpec{
		Model:           "meta-llama/Llama-3.1-8B",
		Port:            9000,
		GPUs:            4,
		ParallelismMode: ParallelismPipeline,
	}

	args := buildVLLMServeArgs(spec, "")

	if !containsArg(args, "--pipeline-parallel-size") {
		t.Errorf("pipeline parallel flag missing in args: %v", args)
	}
	if containsArg(args, "--tensor-parallel-size") {
		t.Errorf("tensor parallel flag should not be present in pipeline mode: %v", args)
	}
}

func TestSanitizeDockerNamePart(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"},
		{"deepseek-ai/DeepSeek-R1", "deepseek-ai-DeepSeek-R1"},
		{"invalid @#$ name!", "invalid-name"},
		{"", "model"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := SanitizeDockerNamePart(tt.input)
			if got != tt.want {
				t.Errorf("SanitizeDockerNamePart(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestNormalizeHostBindIP(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"127.0.0.1", "127.0.0.1", false},
		{"0.0.0.0", "0.0.0.0", false},
		{"::1", "::1", false},
		{"", "0.0.0.0", false},
		{"invalid-ip", "", true},
		{"999.999.999.999", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := normalizeHostBindIP(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("normalizeHostBindIP(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if !tt.wantErr && got.String() != tt.want {
				t.Errorf("normalizeHostBindIP(%q) = %q, want %q", tt.input, got.String(), tt.want)
			}
		})
	}
}

func TestNormalizeContainerExposure(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"direct", ContainerExposureDirect, false},
		{"proxy", ContainerExposureProxy, false},
		{"DIRECT", ContainerExposureDirect, false},
		{"", ContainerExposureDirect, false},
		{"unsupported", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := normalizeContainerExposure(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("normalizeContainerExposure(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("normalizeContainerExposure(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func containsArg(args []string, target string) bool {
	for _, arg := range args {
		if arg == target {
			return true
		}
	}
	return false
}
