package api

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseImportedDockerRunCommand(t *testing.T) {
	command := `docker run --gpus device=0,1 \
  --privileged --ipc=host -p 127.0.0.1:8084:8084 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --served-model-name DeepSeek-V4 \
  --cpu-offload-gb 16 \
  --no-enable-flashinfer-autotune \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'`

	got, err := ParseImportedCommand(command)
	if err != nil {
		t.Fatalf("ParseImportedCommand() error=%v", err)
	}
	if got.Image != "vllm/vllm-openai:latest" {
		t.Fatalf("image=%q", got.Image)
	}
	if got.Fields.Model != "deepseek-ai/DeepSeek-V4-Flash" {
		t.Fatalf("model=%q", got.Fields.Model)
	}
	if got.Fields.Port != 8084 || got.Fields.GPUs != 2 || got.Fields.GPUIDs != "0,1" {
		t.Fatalf("fields=%+v, want port 8084 and GPUs 0,1", got.Fields)
	}
	if !got.Fields.TrustRemoteCode || got.Fields.KVCacheDType != "fp8" || got.Fields.BlockSize != "256" {
		t.Fatalf("fields=%+v, want imported vLLM flags", got.Fields)
	}
	if got.Fields.SpeculativeConfig != `{"method":"dspark","num_speculative_tokens":7}` {
		t.Fatalf("speculative_config=%q", got.Fields.SpeculativeConfig)
	}
	if got.Fields.ServedModelName != "DeepSeek-V4" {
		t.Fatalf("served_model_name=%q", got.Fields.ServedModelName)
	}
	if got.Fields.CPUOffloadGB != "16" {
		t.Fatalf("cpu_offload_gb=%q", got.Fields.CPUOffloadGB)
	}
	if !reflect.DeepEqual(got.ExtraVLLMArgs, []string{"--no-enable-flashinfer-autotune", "--enable-chunked-prefill"}) {
		t.Fatalf("extra=%v", got.ExtraVLLMArgs)
	}
	if !containsString(got.Unsupported, "--privileged") {
		t.Fatalf("unsupported=%v, want --privileged", got.Unsupported)
	}
}

func TestParseImportedVLLMServeCommand(t *testing.T) {
	got, err := ParseImportedCommand(`vllm serve Qwen/Qwen3 --port 9000 --tensor-parallel-size=2 --max-num-batched-tokens 8192`)
	if err != nil {
		t.Fatalf("ParseImportedCommand() error=%v", err)
	}
	if got.Fields.Model != "Qwen/Qwen3" || got.Fields.Port != 9000 || got.Fields.GPUs != 2 {
		t.Fatalf("fields=%+v", got.Fields)
	}
	if !reflect.DeepEqual(got.ExtraVLLMArgs, []string{"--max-num-batched-tokens", "8192"}) {
		t.Fatalf("extra=%v", got.ExtraVLLMArgs)
	}
}

func TestParseImportedCommandRejectsBadQuoting(t *testing.T) {
	_, err := ParseImportedCommand(`vllm serve "Qwen/Qwen3`)
	if err == nil {
		t.Fatal("ParseImportedCommand() error=nil, want quote error")
	}
	if !strings.Contains(err.Error(), "unterminated quote") {
		t.Fatalf("error=%q", err.Error())
	}
}

func TestNormalizeExtraVLLMArgs(t *testing.T) {
	got, err := NormalizeExtraVLLMArgs([]string{"--disable-log-requests", "--language-model-only", "--no-enable-flashinfer-autotune"})
	if err != nil {
		t.Fatalf("NormalizeExtraVLLMArgs() error=%v", err)
	}
	if !reflect.DeepEqual(got, []string{"--disable-log-requests", "--language-model-only", "--no-enable-flashinfer-autotune"}) {
		t.Fatalf("NormalizeExtraVLLMArgs()=%v", got)
	}

	if _, err := NormalizeExtraVLLMArgs([]string{"--served-model-name=alias"}); err == nil {
		t.Fatal("NormalizeExtraVLLMArgs accepted managed --served-model-name")
	}
	if _, err := NormalizeExtraVLLMArgs([]string{"--cpu-offload-gb", "16"}); err == nil {
		t.Fatal("NormalizeExtraVLLMArgs accepted managed --cpu-offload-gb")
	}
	if _, err := NormalizeExtraVLLMArgs([]string{"--model", "Qwen/Qwen3"}); err == nil {
		t.Fatal("NormalizeExtraVLLMArgs accepted managed --model")
	}
	if _, err := NormalizeExtraVLLMArgs([]string{"--unknown-vllm-flag"}); err == nil {
		t.Fatal("NormalizeExtraVLLMArgs accepted unknown flag")
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
