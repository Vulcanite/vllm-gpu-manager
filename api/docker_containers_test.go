package api

import (
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/moby/moby/api/types/container"
)

func stubGPUInfo(t *testing.T, fn func() ([]GPUInfo, error)) {
	t.Helper()
	original := getGPUInfo
	getGPUInfo = fn
	t.Cleanup(func() {
		getGPUInfo = original
	})
}

func TestValidateGPUSelectionRejectsDiscoveryFailure(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return nil, errors.New("nvidia-smi unavailable")
	})

	_, _, err := validateGPUSelection(1, "all")
	if err == nil {
		t.Fatal("validateGPUSelection returned nil error, want discovery failure")
	}
	if !strings.Contains(err.Error(), "GPU detection failed") {
		t.Fatalf("error=%q, want GPU detection failure", err.Error())
	}
}

func TestValidateGPUSelectionRejectsNoDetectedGPUs(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return []GPUInfo{}, nil
	})

	_, _, err := validateGPUSelection(1, "all")
	if err == nil {
		t.Fatal("validateGPUSelection returned nil error, want no-GPU validation error")
	}
	if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
	if !strings.Contains(err.Error(), "no GPUs detected") {
		t.Fatalf("error=%q, want no GPUs detected", err.Error())
	}
}

func TestValidateGPUSelectionAllUsesDetectedDevices(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return []GPUInfo{{ID: 1}, {ID: 0}}, nil
	})

	gpuFlag, selected, err := validateGPUSelection(2, "all")
	if err != nil {
		t.Fatalf("validateGPUSelection returned error: %v", err)
	}
	if gpuFlag != "device=0,1" {
		t.Fatalf("gpuFlag=%q, want device=0,1", gpuFlag)
	}
	if !reflect.DeepEqual(selected, []GPUInfo{{ID: 0}, {ID: 1}}) {
		t.Fatalf("selected=%v, want GPU IDs 0 and 1", selected)
	}
}

func TestIsManagedContainerRequiresLabels(t *testing.T) {
	tests := []struct {
		name   string
		labels map[string]string
		want   bool
	}{
		{
			name:   "managed label",
			labels: map[string]string{managedContainerLabel: "true"},
			want:   true,
		},
		{
			name:   "legacy app label",
			labels: map[string]string{"app": "vllm-manager"},
			want:   true,
		},
		{
			name:   "vllm name without labels is not enough",
			labels: map[string]string{},
			want:   false,
		},
		{
			name:   "nil labels",
			labels: nil,
			want:   false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isManagedContainer(test.labels); got != test.want {
				t.Fatalf("isManagedContainer()=%v, want %v", got, test.want)
			}
		})
	}
}

func TestIsManagedServerContainerRejectsDownloads(t *testing.T) {
	if !isManagedServerContainer(map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindServer}) {
		t.Fatal("managed server container was rejected")
	}
	if !isManagedServerContainer(map[string]string{"app": "vllm-manager"}) {
		t.Fatal("legacy managed server container without kind was rejected")
	}
	if isManagedServerContainer(map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindDownload}) {
		t.Fatal("managed download container was accepted as server container")
	}
}

func TestBuildActiveModelParsesFlashInferAutotuneFlag(t *testing.T) {
	model, ok := buildActiveModel(managedVLLMContainer{
		id:     "container-id",
		name:   "vllm-test-8080",
		state:  "running",
		labels: map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindServer},
		metadata: container.InspectResponse{
			Config: &container.Config{
				Cmd: []string{
					"--model", "test/model",
					"--port", "8080",
					"--trust-remote-code",
					"--no-enable-flashinfer-autotune",
				},
			},
		},
	})
	if !ok {
		t.Fatal("buildActiveModel rejected managed server container")
	}
	if !model.TrustRemoteCode {
		t.Fatal("TrustRemoteCode=false, want true")
	}
	if !model.DisableFlashInferAutotune {
		t.Fatal("DisableFlashInferAutotune=false, want true")
	}
}

func TestBuildActiveModelParsesAdvancedEditFields(t *testing.T) {
	model, ok := buildActiveModel(managedVLLMContainer{
		id:     "container-id",
		name:   "vllm-test-8080",
		state:  "exited",
		labels: map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindServer},
		metadata: container.InspectResponse{
			Config: &container.Config{
				Cmd: []string{
					"--model", "test/model",
					"--port", "8080",
					"--moe-backend", "flashinfer",
					"--enable-expert-parallel",
					"--speculative-config", `{"method":"dspark"}`,
					"--served-model-name", "alias",
					"--enable-chunked-prefill",
				},
			},
		},
	})
	if !ok {
		t.Fatal("buildActiveModel rejected managed server container")
	}
	if model.MoeBackend != "flashinfer" {
		t.Fatalf("MoeBackend=%q, want flashinfer", model.MoeBackend)
	}
	if !model.ExpertParallel {
		t.Fatal("ExpertParallel=false, want true")
	}
	if model.SpeculativeConfig != `{"method":"dspark"}` {
		t.Fatalf("SpeculativeConfig=%q", model.SpeculativeConfig)
	}
	if model.ServedModelName != "alias" {
		t.Fatalf("ServedModelName=%q, want alias", model.ServedModelName)
	}
	wantExtra := []string{"--enable-chunked-prefill"}
	if !reflect.DeepEqual(model.ExtraVLLMArgs, wantExtra) {
		t.Fatalf("ExtraVLLMArgs=%v, want %v", model.ExtraVLLMArgs, wantExtra)
	}
}

func TestBuildActiveModelUsesLaunchSpecLabelsForEditMetadata(t *testing.T) {
	spec := LaunchSpec{
		Model:                     "label/model",
		ServedModelName:           "alias",
		Port:                      9090,
		GPUs:                      2,
		GPUIDs:                    "0,1",
		MemUtil:                   "0.85",
		MaxLen:                    "32768",
		MaxNumSeqs:                "8",
		CPUOffloadGB:              "16",
		DType:                     "bfloat16",
		ParallelismMode:           ParallelismPipeline,
		KVCacheDType:              "fp8",
		ToolParser:                "deepseek_v4",
		TokenizerMode:             "deepseek_v4",
		ReasoningParser:           "deepseek_v4",
		TrustRemoteCode:           true,
		BlockSize:                 "256",
		MoeBackend:                "flashinfer_cutlass",
		ExpertParallel:            true,
		DisableFlashInferAutotune: true,
		SpeculativeConfig:         `{"method":"dspark"}`,
		BindIP:                    "127.0.0.1",
		ExtraVLLMArgs:             []string{"--enable-chunked-prefill"},
	}
	specBytes, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("marshal launch spec: %v", err)
	}

	model, ok := buildActiveModel(managedVLLMContainer{
		id:    "container-id",
		name:  "vllm-test-9090",
		state: "exited",
		labels: map[string]string{
			managedContainerLabel:           "true",
			containerKindLabel:              containerKindServer,
			containerExposureLabel:          ContainerExposureProxy,
			containerLaunchSpecVersionLabel: LaunchSpecVersion,
			containerLaunchSpecLabel:        string(specBytes),
		},
		metadata: container.InspectResponse{
			Config: &container.Config{
				Cmd: []string{
					"--model", "arg/model",
					"--port", "8080",
					"--tensor-parallel-size", "1",
					"--served-model-name", "from-args",
				},
			},
		},
	})
	if !ok {
		t.Fatal("buildActiveModel rejected managed server container")
	}
	if model.LaunchSpec == nil {
		t.Fatal("LaunchSpec=nil, want parsed launch spec")
	}
	if model.Exposure != ContainerExposureProxy {
		t.Fatalf("Exposure=%q, want proxy", model.Exposure)
	}
	if model.Model != spec.Model || model.Port != "9090" || model.GPUs != "2" || model.CPUOffloadGB != "16" || model.DType != "bfloat16" || model.ParallelismMode != ParallelismPipeline {
		t.Fatalf("model fields=%q/%q/%q/%q, want launch spec values", model.Model, model.Port, model.GPUs, model.CPUOffloadGB)
	}
	if model.ServedModelName != spec.ServedModelName {
		t.Fatalf("ServedModelName=%q, want %q", model.ServedModelName, spec.ServedModelName)
	}
	if !reflect.DeepEqual(model.LaunchSpec.ExtraVLLMArgs, spec.ExtraVLLMArgs) {
		t.Fatalf("LaunchSpec.ExtraVLLMArgs=%v, want %v", model.LaunchSpec.ExtraVLLMArgs, spec.ExtraVLLMArgs)
	}
	if !reflect.DeepEqual(model.ExtraVLLMArgs, spec.ExtraVLLMArgs) {
		t.Fatalf("ExtraVLLMArgs=%v, want launch spec extras only", model.ExtraVLLMArgs)
	}
}

func TestServerContainerLabelsIncludeExposureAndLaunchSpec(t *testing.T) {
	spec := LaunchSpec{
		Model:           "test/model",
		ServedModelName: "alias",
		Port:            8082,
		GPUs:            2,
		GPUIDs:          "0,1",
		MemUtil:         "0.9",
		MaxLen:          "32768",
		CPUOffloadGB:    "8",
		DType:           "float16",
		ParallelismMode: ParallelismTensor,
		KVCacheDType:    "auto",
		TokenizerMode:   "auto",
		MoeBackend:      "auto",
		BindIP:          "127.0.0.1",
		ExtraVLLMArgs:   []string{"--enable-chunked-prefill"},
	}
	labels, err := serverContainerLabels(spec.Model, ContainerExposureProxy, spec)
	if err != nil {
		t.Fatalf("serverContainerLabels returned error: %v", err)
	}
	if labels[containerExposureLabel] != ContainerExposureProxy {
		t.Fatalf("exposure label=%q, want proxy", labels[containerExposureLabel])
	}
	if labels[containerLaunchSpecVersionLabel] != LaunchSpecVersion {
		t.Fatalf("launch spec version=%q, want %q", labels[containerLaunchSpecVersionLabel], LaunchSpecVersion)
	}
	var got LaunchSpec
	if err := json.Unmarshal([]byte(labels[containerLaunchSpecLabel]), &got); err != nil {
		t.Fatalf("launch spec label is not valid JSON: %v", err)
	}
	if !reflect.DeepEqual(got, spec) {
		t.Fatalf("launch spec=%+v, want %+v", got, spec)
	}
	if strings.Contains(labels[containerLaunchSpecLabel], "api_key") || strings.Contains(labels[containerLaunchSpecLabel], "secret") {
		t.Fatalf("launch spec label appears to contain secret material: %s", labels[containerLaunchSpecLabel])
	}
}

func TestLaunchSpecWithoutPrefixCachingFieldDefaultsEnabled(t *testing.T) {
	labels := map[string]string{
		containerLaunchSpecVersionLabel: LaunchSpecVersion,
		containerLaunchSpecLabel:        `{"model":"test/model","port":8080}`,
	}
	spec, ok := launchSpecFromLabels(labels)
	if !ok || !spec.EnablePrefixCaching {
		t.Fatalf("launchSpecFromLabels()=(%+v, %t), want prefix caching enabled", spec, ok)
	}
}

func TestBuildVLLMServeArgsUsesSingleParallelismMode(t *testing.T) {
	tensorArgs := buildVLLMServeArgs(LaunchSpec{
		Model:           "test/model",
		Port:            8080,
		GPUs:            2,
		MemUtil:         "0.9",
		DType:           "auto",
		ParallelismMode: ParallelismTensor,
	}, "")
	if !hasArgPair(tensorArgs, "--tensor-parallel-size", "2") || slices.Contains(tensorArgs, "--pipeline-parallel-size") {
		t.Fatalf("tensor args=%v", tensorArgs)
	}
	if slices.Contains(tensorArgs, "--dtype") || slices.Contains(tensorArgs, "--block-size") {
		t.Fatalf("default GUI args unexpectedly emit dtype or block size: %v", tensorArgs)
	}

	pipelineArgs := buildVLLMServeArgs(LaunchSpec{
		Model:           "test/model",
		Port:            8080,
		GPUs:            4,
		MemUtil:         "0.9",
		DType:           "float32",
		ParallelismMode: ParallelismPipeline,
		ExtraVLLMArgs:   []string{"--block-size", "256"},
	}, "")
	if !hasArgPair(pipelineArgs, "--pipeline-parallel-size", "4") || slices.Contains(pipelineArgs, "--tensor-parallel-size") {
		t.Fatalf("pipeline args=%v", pipelineArgs)
	}
	if !hasArgPair(pipelineArgs, "--dtype", "float32") || !hasArgPair(pipelineArgs, "--block-size", "256") {
		t.Fatalf("pipeline args did not preserve dtype and legacy block-size extra: %v", pipelineArgs)
	}
}

func TestBuildVLLMServeArgsHonorsPrefixCaching(t *testing.T) {
	enabled := buildVLLMServeArgs(LaunchSpec{Model: "test/model", Port: 8080, GPUs: 1, MemUtil: "0.9", EnablePrefixCaching: true}, "")
	disabled := buildVLLMServeArgs(LaunchSpec{Model: "test/model", Port: 8080, GPUs: 1, MemUtil: "0.9"}, "")
	if !slices.Contains(enabled, "--enable-prefix-caching") || slices.Contains(enabled, "--no-enable-prefix-caching") {
		t.Fatalf("enabled args=%v", enabled)
	}
	if !slices.Contains(disabled, "--no-enable-prefix-caching") || slices.Contains(disabled, "--enable-prefix-caching") {
		t.Fatalf("disabled args=%v", disabled)
	}
}

func hasArgPair(args []string, flag string, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func TestNormalizeKVCacheDType(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  string
	}{
		{name: "empty defaults to auto", value: "", want: "auto"},
		{name: "auto", value: "auto", want: "auto"},
		{name: "fp8", value: "fp8", want: "fp8"},
		{name: "fp8 e5m2", value: "fp8_e5m2", want: "fp8_e5m2"},
		{name: "fp8 e4m3", value: "fp8_e4m3", want: "fp8_e4m3"},
		{name: "trims and lowercases", value: " FP8_E4M3 ", want: "fp8_e4m3"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeKVCacheDType(test.value)
			if err != nil {
				t.Fatalf("normalizeKVCacheDType returned error: %v", err)
			}
			if got != test.want {
				t.Fatalf("normalizeKVCacheDType()=%q, want %q", got, test.want)
			}
		})
	}
}

func TestNormalizeKVCacheDTypeRejectsInvalidValue(t *testing.T) {
	if _, err := normalizeKVCacheDType("bf16"); err == nil {
		t.Fatal("normalizeKVCacheDType returned nil error for invalid dtype")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestNormalizeDTypeAliases(t *testing.T) {
	for _, test := range []struct {
		value string
		want  string
	}{
		{value: "", want: "auto"},
		{value: "auto", want: "auto"},
		{value: "bfloat16", want: "bfloat16"},
		{value: "half", want: "float16"},
		{value: "float16", want: "float16"},
		{value: "float", want: "float32"},
		{value: "float32", want: "float32"},
	} {
		got, err := normalizeDType(test.value)
		if err != nil || got != test.want {
			t.Fatalf("normalizeDType(%q)=%q, %v; want %q", test.value, got, err, test.want)
		}
	}
	if _, err := normalizeDType("int8"); err == nil {
		t.Fatal("normalizeDType accepted unsupported dtype")
	}
}

func TestNormalizeParallelismMode(t *testing.T) {
	if got, err := normalizeParallelismMode("", 1); err != nil || got != ParallelismTensor {
		t.Fatalf("normalizeParallelismMode(empty)=%q, %v", got, err)
	}
	if got, err := normalizeParallelismMode(ParallelismPipeline, 2); err != nil || got != ParallelismPipeline {
		t.Fatalf("normalizeParallelismMode(pipeline)=%q, %v", got, err)
	}
	if _, err := normalizeParallelismMode(ParallelismPipeline, 1); err == nil {
		t.Fatal("normalizeParallelismMode accepted single-GPU pipeline mode")
	}
	if _, err := normalizeParallelismMode("hybrid", 4); err == nil {
		t.Fatal("normalizeParallelismMode accepted unsupported mode")
	}
}

func TestStartModelRejectsIneligibleParallelismSettingsBeforeDocker(t *testing.T) {
	tests := []struct {
		name           string
		gpus           int
		parallelism    string
		moeBackend     string
		expertParallel bool
		want           string
	}{
		{
			name:        "single GPU pipeline",
			gpus:        1,
			parallelism: ParallelismPipeline,
			moeBackend:  "auto",
			want:        "pipeline parallelism requires at least 2 GPUs",
		},
		{
			name:           "single GPU expert parallel",
			gpus:           1,
			parallelism:    ParallelismTensor,
			moeBackend:     "auto",
			expertParallel: true,
			want:           "expert parallelism requires at least 2 GPUs",
		},
		{
			name:        "MoE backend without expert parallel",
			gpus:        2,
			parallelism: ParallelismTensor,
			moeBackend:  "cutlass",
			want:        "non-default MoE backend requires expert parallelism",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := StartModel(
				"test/model", "", 8080, test.gpus, "all", "0.9", "32768", "256", "0",
				"auto", test.parallelism, "auto", "", "auto", "", false, true, "", test.moeBackend,
				test.expertParallel, false, "", t.TempDir(), "", "", false, false, "127.0.0.1",
				ContainerExposureDirect, "", nil,
			)
			if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("StartModel() error=%v, want validation error containing %q", err, test.want)
			}
		})
	}
}

func TestKVCacheDTypeElementBytes(t *testing.T) {
	if got := kvCacheDTypeElementBytes("auto", 2); got != 2 {
		t.Fatalf("auto bytes=%d, want 2", got)
	}
	if got := kvCacheDTypeElementBytes("fp8_e5m2", 2); got != 1 {
		t.Fatalf("fp8 bytes=%d, want 1", got)
	}
}

func TestKVBytesPerTokenPerGPUUsesCeilDiv(t *testing.T) {
	estimate := modelKVCacheEstimate{
		Layers:        1,
		KeyValueHeads: 3,
		HeadDim:       10,
		DTypeBytes:    2,
	}

	got, err := estimate.bytesPerTokenPerGPU(2)
	if err != nil {
		t.Fatalf("bytesPerTokenPerGPU returned error: %v", err)
	}
	const want = uint64(80) // 2 * 1 layer * ceil(3/2) KV heads * 10 head dim * 2 bytes
	if got != want {
		t.Fatalf("bytesPerTokenPerGPU=%d, want %d", got, want)
	}
}

func TestKVBytesPerTokenPerGPURejectsOverflow(t *testing.T) {
	estimate := modelKVCacheEstimate{
		Layers:        1 << 30,
		KeyValueHeads: 1 << 30,
		HeadDim:       1 << 30,
		DTypeBytes:    16,
	}

	if _, err := estimate.bytesPerTokenPerGPU(1); err == nil {
		t.Fatal("bytesPerTokenPerGPU returned nil error for overflowing estimate")
	} else if !strings.Contains(err.Error(), "increase tensor_parallel_size") {
		t.Fatalf("error=%q, want actionable tensor_parallel_size guidance", err.Error())
	}
}

func TestCheckedMulUint64RejectsOverflow(t *testing.T) {
	if _, ok := checkedMulUint64(^uint64(0), 2); ok {
		t.Fatal("checkedMulUint64 accepted overflowing multiplication")
	}

	got, ok := checkedMulUint64(2, 3, 5)
	if !ok {
		t.Fatal("checkedMulUint64 rejected valid multiplication")
	}
	if got != 30 {
		t.Fatalf("checkedMulUint64=%d, want 30", got)
	}
}

func TestHealthCheckActiveModelsUpdatesReadyAndDefaultMaxLen(t *testing.T) {
	original := activeModelHealthProbe
	type probeArgs struct {
		containerID   string
		port          string
		currentMaxLen string
		apiKey        string
	}
	argsCh := make(chan probeArgs, 1)
	activeModelHealthProbe = func(containerID string, port string, currentMaxLen string, apiKey string) (bool, string) {
		argsCh <- probeArgs{containerID: containerID, port: port, currentMaxLen: currentMaxLen, apiKey: apiKey}
		return true, "4096"
	}
	t.Cleanup(func() {
		activeModelHealthProbe = original
	})

	models := []ActiveModel{{
		ContainerID: "health-check-test-container",
		Port:        "12345",
		State:       "running",
		MaxLen:      "default",
	}}

	healthCheckActiveModels(models, "fallback-key")

	args := <-argsCh
	if args != (probeArgs{containerID: "health-check-test-container", port: "12345", currentMaxLen: "default", apiKey: "fallback-key"}) {
		t.Fatalf("probe args=%+v, want test container, port, max len, fallback key", args)
	}

	if !models[0].IsReady {
		t.Fatal("model was not marked ready")
	}
	if models[0].MaxLen != "4096" {
		t.Fatalf("MaxLen=%q, want 4096", models[0].MaxLen)
	}
}

func TestNormalizeModelKey(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "lowercases and trims", in: "  Qwen/Qwen2.5-7B-Instruct  ", want: "qwen/qwen2.5-7b-instruct"},
		{name: "keeps slashes and dots", in: "meta-llama/Llama-3.1-8B", want: "meta-llama/llama-3.1-8b"},
		{name: "empty", in: "  ", want: ""},
		{name: "leading slash invalid", in: "/foo", want: ""},
		{name: "rejects spaces", in: "foo bar", want: ""},
		{name: "rejects punctuation", in: "foo!", want: ""},
		{name: "too long", in: "a" + strings.Repeat("b", 200), want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeModelKey(tc.in); got != tc.want {
				t.Fatalf("normalizeModelKey(%q)=%q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestNormalizeModelKeyForMatchIsLoose(t *testing.T) {
	// The matcher does not enforce the strict character class, so historical
	// Docker labels with characters we no longer accept on input still match.
	if got := NormalizeModelKeyForMatch("Qwen/Qwen2.5 7B"); got != "qwen/qwen2.5 7b" {
		t.Fatalf("NormalizeModelKeyForMatch()=%q, want lowercased trim", got)
	}
	if got := NormalizeModelKeyForMatch(""); got != "" {
		t.Fatalf("NormalizeModelKeyForMatch(empty)=%q, want empty", got)
	}
}

func TestAllowedToolParsersRejectsUnknown(t *testing.T) {
	for _, known := range []string{
		"hermes", "llama3_json", "mistral", "granite", "granite4", "granite-20b-fc",
		"internlm", "jamba", "pythonic", "python_tool", "jinja", "guidance",
		"lm_format_enforcer", "xlam", "qwen2", "qwen3_coder", "qwen3_xml", "llama4_pythonic",
		"deepseek_v3", "deepseek_v31", "deepseek_v4", "openai", "kimi_k2", "hunyuan_a13b",
		"cohere_command3", "longcat", "glm45", "glm47", "functiongemma",
		"olmo3", "gigachat3", "apertus",
	} {
		if _, ok := allowedToolParsers[known]; !ok {
			t.Fatalf("expected %q to be in the allowlist", known)
		}
	}
	for _, unknown := range []string{"hermes; rm -rf /", "../../etc/passwd", "drop-table", ""} {
		if _, ok := allowedToolParsers[unknown]; ok && unknown != "" {
			t.Fatalf("expected %q to be rejected", unknown)
		}
	}
}

func TestAllowedToolParserNamesSortedAndUnique(t *testing.T) {
	names := allowedToolParserNames()
	if len(names) != len(allowedToolParsers) {
		t.Fatalf("len(names)=%d, want %d", len(names), len(allowedToolParsers))
	}
	for i := 1; i < len(names); i++ {
		if names[i-1] >= names[i] {
			t.Fatalf("names not strictly sorted at %d: %q >= %q", i, names[i-1], names[i])
		}
	}
}

func TestNormalizeTokenizerMode(t *testing.T) {
	for _, test := range []struct {
		value string
		want  string
	}{
		{value: "", want: "auto"},
		{value: " auto ", want: "auto"},
		{value: "hf", want: "hf"},
		{value: "slow", want: "slow"},
		{value: "MISTRAL", want: "mistral"},
		{value: "deepseek_v32", want: "deepseek_v32"},
		{value: "deepseek_v4", want: "deepseek_v4"},
	} {
		got, err := normalizeTokenizerMode(test.value)
		if err != nil {
			t.Fatalf("normalizeTokenizerMode(%q) returned error: %v", test.value, err)
		}
		if got != test.want {
			t.Fatalf("normalizeTokenizerMode(%q)=%q, want %q", test.value, got, test.want)
		}
	}

	if _, err := normalizeTokenizerMode("unsafe"); err == nil {
		t.Fatal("normalizeTokenizerMode accepted unsupported value")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestNormalizeReasoningParser(t *testing.T) {
	if got, err := normalizeReasoningParser(""); err != nil || got != "" {
		t.Fatalf("normalizeReasoningParser(empty)=%q, %v; want empty nil", got, err)
	}
	if got, err := normalizeReasoningParser("deepseek_v4"); err != nil || got != "deepseek_v4" {
		t.Fatalf("normalizeReasoningParser(deepseek_v4)=%q, %v; want deepseek_v4 nil", got, err)
	}
	if _, err := normalizeReasoningParser("../../etc/passwd"); err == nil {
		t.Fatal("normalizeReasoningParser accepted unsupported value")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestNormalizeBlockSize(t *testing.T) {
	for _, test := range []struct {
		value string
		want  string
	}{
		{value: "", want: ""},
		{value: "16", want: "16"},
		{value: "128", want: "128"},
	} {
		got, err := normalizeBlockSize(test.value)
		if err != nil {
			t.Fatalf("normalizeBlockSize(%q) returned error: %v", test.value, err)
		}
		if got != test.want {
			t.Fatalf("normalizeBlockSize(%q)=%q, want %q", test.value, got, test.want)
		}
	}

	for _, value := range []string{"0", "-2", "3", "8", "abc"} {
		if _, err := normalizeBlockSize(value); err == nil {
			t.Fatalf("normalizeBlockSize(%q) accepted invalid value", value)
		} else if !IsValidationError(err) {
			t.Fatalf("error type=%T, want ValidationError", err)
		}
	}
}

func TestNormalizeMoeBackend(t *testing.T) {
	for _, test := range []struct {
		value string
		want  string
	}{
		{value: "", want: "auto"},
		{value: "auto", want: "auto"},
		{value: "FlashInfer_Cutlass", want: "flashinfer_cutlass"},
		{value: "deep_gemm_mega_moe", want: "deep_gemm_mega_moe"},
	} {
		got, err := normalizeMoeBackend(test.value)
		if err != nil {
			t.Fatalf("normalizeMoeBackend(%q) returned error: %v", test.value, err)
		}
		if got != test.want {
			t.Fatalf("normalizeMoeBackend(%q)=%q, want %q", test.value, got, test.want)
		}
	}

	if _, err := normalizeMoeBackend("../../etc/passwd"); err == nil {
		t.Fatal("normalizeMoeBackend accepted unsupported value")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestNormalizeSpeculativeConfig(t *testing.T) {
	got, err := normalizeSpeculativeConfig(`{
		"model": "draft",
		"num_speculative_tokens": 4
	}`)
	if err != nil {
		t.Fatalf("normalizeSpeculativeConfig returned error: %v", err)
	}
	if got != `{"model":"draft","num_speculative_tokens":4}` {
		t.Fatalf("normalizeSpeculativeConfig compacted to %q", got)
	}

	if got, err := normalizeSpeculativeConfig(""); err != nil || got != "" {
		t.Fatalf("normalizeSpeculativeConfig(empty)=%q, %v; want empty nil", got, err)
	}

	if _, err := normalizeSpeculativeConfig("{invalid"); err == nil {
		t.Fatal("normalizeSpeculativeConfig accepted invalid JSON")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
}

func TestNormalizeCPUOffloadGB(t *testing.T) {
	tests := []struct {
		value string
		want  int
	}{
		{value: "", want: 0},
		{value: "0", want: 0},
		{value: "16", want: 16},
		{value: " 32 ", want: 32},
	}
	for _, test := range tests {
		got, err := normalizeCPUOffloadGB(test.value)
		if err != nil {
			t.Fatalf("normalizeCPUOffloadGB(%q) error=%v", test.value, err)
		}
		if got != test.want {
			t.Fatalf("normalizeCPUOffloadGB(%q)=%d, want %d", test.value, got, test.want)
		}
	}

	for _, value := range []string{"-1", "1.5", "bad"} {
		if _, err := normalizeCPUOffloadGB(value); err == nil {
			t.Fatalf("normalizeCPUOffloadGB(%q) accepted invalid value", value)
		}
	}
}

func TestValidateCPUOffloadRAM(t *testing.T) {
	ram := SystemRAM{Free: 40 * bytesPerGiB}
	if err := validateCPUOffloadRAM(16, 2, ram); err != nil {
		t.Fatalf("validateCPUOffloadRAM returned error: %v", err)
	}
	if err := validateCPUOffloadRAM(17, 2, ram); err == nil {
		t.Fatal("validateCPUOffloadRAM accepted insufficient RAM")
	} else if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
	if err := validateCPUOffloadRAM(0, 2, SystemRAM{}); err != nil {
		t.Fatalf("validateCPUOffloadRAM disabled returned error: %v", err)
	}
}

func TestCPUOffloadUlimits(t *testing.T) {
	if got := cpuOffloadUlimits(0); got != nil {
		t.Fatalf("cpuOffloadUlimits(0)=%v, want nil", got)
	}
	got := cpuOffloadUlimits(1)
	if len(got) != 1 || got[0].Name != "memlock" || got[0].Soft != -1 || got[0].Hard != -1 {
		t.Fatalf("cpuOffloadUlimits(1)=%+v, want memlock -1:-1", got)
	}
}

func TestStartModelRejectsGlobalAndPerServerAPIKey(t *testing.T) {
	err := StartModel("any-model", "", 8080, 1, "all", "0.9", "32768", "", "0", "auto", ParallelismTensor, "auto", "", "auto", "", false, true, "", "auto", false, false, "", "/tmp/cache", "", "global-key", false, false, "0.0.0.0", ContainerExposureDirect, "per-server-key", nil)
	if err == nil {
		t.Fatal("StartModel accepted both global and per-server key; want validation error")
	}
	if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("error=%q, want 'mutually exclusive' message", err.Error())
	}
}

func TestStartModelRejectsWhitespaceInPerServerAPIKey(t *testing.T) {
	err := StartModel("any-model", "", 8080, 1, "all", "0.9", "32768", "", "0", "auto", ParallelismTensor, "auto", "", "auto", "", false, true, "", "auto", false, false, "", "/tmp/cache", "", "", false, false, "0.0.0.0", ContainerExposureDirect, "key with space", nil)
	if err == nil {
		t.Fatal("StartModel accepted whitespace in per-server key; want validation error")
	}
	if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
	if !strings.Contains(err.Error(), "per-server API key") {
		t.Fatalf("error=%q, want 'per-server API key' message", err.Error())
	}
}
