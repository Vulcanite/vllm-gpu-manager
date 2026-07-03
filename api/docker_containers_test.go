package api

import (
	"errors"
	"reflect"
	"strings"
	"testing"
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
		"deepseek_v3", "deepseek_v31", "openai", "kimi_k2", "hunyuan_a13b",
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

func TestStartModelRejectsGlobalAndPerServerAPIKey(t *testing.T) {
	err := StartModel("any-model", 8080, 1, "all", "0.9", "32768", "", "auto", "", "/tmp/cache", "", "global-key", false, false, "0.0.0.0", "per-server-key")
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
	err := StartModel("any-model", 8080, 1, "all", "0.9", "32768", "", "auto", "", "/tmp/cache", "", "", false, false, "0.0.0.0", "key with space")
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
