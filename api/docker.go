package api

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

type ActiveModel struct {
	ContainerID               string      `json:"container_id"`
	ContainerName             string      `json:"container_name"`
	Model                     string      `json:"model"`
	ServedModelName           string      `json:"served_model_name"`
	Port                      string      `json:"port"`
	GPUs                      string      `json:"gpus"`
	GPUIDs                    string      `json:"gpu_ids"`
	MemUtil                   string      `json:"mem_util"`
	MaxLen                    string      `json:"max_len"`
	MaxNumSeqs                string      `json:"max_num_seqs"`
	CPUOffloadGB              string      `json:"cpu_offload_gb"`
	DType                     string      `json:"dtype"`
	ParallelismMode           string      `json:"parallelism_mode"`
	KVCacheDType              string      `json:"kv_cache_dtype"`
	ToolParser                string      `json:"tool_parser"`
	TokenizerMode             string      `json:"tokenizer_mode"`
	ReasoningParser           string      `json:"reasoning_parser"`
	TrustRemoteCode           bool        `json:"trust_remote_code"`
	EnablePrefixCaching       bool        `json:"enable_prefix_caching"`
	DisableFlashInferAutotune bool        `json:"disable_flashinfer_autotune"`
	BlockSize                 string      `json:"block_size"`
	MoeBackend                string      `json:"moe_backend"`
	ExpertParallel            bool        `json:"enable_expert_parallel"`
	SpeculativeConfig         string      `json:"speculative_config"`
	ExtraVLLMArgs             []string    `json:"extra_vllm_args"`
	BindIP                    string      `json:"bind_ip"`
	MetricsTarget             string      `json:"metrics_target"`
	APIUrl                    string      `json:"api_url"`
	Exposure                  string      `json:"exposure,omitempty"`
	LaunchSpecVersion         string      `json:"launch_spec_version,omitempty"`
	LaunchSpec                *LaunchSpec `json:"launch_spec,omitempty"`
	IsReady                   bool        `json:"is_ready"`
	Arch                      string      `json:"arch"`
	State                     string      `json:"state"`
	APIKey                    string      `json:"-"`
	// HasServerAPIKey is populated by the dashboard handler so the UI can
	// distinguish a per-server key from the global one without exposing the
	// secret value.
	HasServerAPIKey bool   `json:"has_server_api_key,omitempty"`
	APIKeyMode      string `json:"api_key_mode,omitempty"`
}

type LaunchSpec struct {
	Model                     string   `json:"model"`
	ServedModelName           string   `json:"served_model_name"`
	Port                      int      `json:"port"`
	GPUs                      int      `json:"gpus"`
	GPUIDs                    string   `json:"gpu_ids"`
	MemUtil                   string   `json:"mem_util"`
	MaxLen                    string   `json:"max_len"`
	MaxNumSeqs                string   `json:"max_num_seqs"`
	CPUOffloadGB              string   `json:"cpu_offload_gb"`
	DType                     string   `json:"dtype"`
	ParallelismMode           string   `json:"parallelism_mode"`
	KVCacheDType              string   `json:"kv_cache_dtype"`
	ToolParser                string   `json:"tool_parser"`
	TokenizerMode             string   `json:"tokenizer_mode"`
	ReasoningParser           string   `json:"reasoning_parser"`
	TrustRemoteCode           bool     `json:"trust_remote_code"`
	EnablePrefixCaching       bool     `json:"enable_prefix_caching"`
	BlockSize                 string   `json:"block_size"`
	MoeBackend                string   `json:"moe_backend"`
	ExpertParallel            bool     `json:"enable_expert_parallel"`
	DisableFlashInferAutotune bool     `json:"disable_flashinfer_autotune"`
	SpeculativeConfig         string   `json:"speculative_config"`
	BindIP                    string   `json:"bind_ip"`
	ExtraVLLMArgs             []string `json:"extra_vllm_args"`
}

type DownloadedModel struct {
	Name                 string `json:"name"`
	Size                 string `json:"size"`
	SizeBytes            uint64 `json:"size_bytes"`
	KVCacheBytesPerToken uint64 `json:"kv_cache_bytes_per_token,omitempty"`
	KVCacheLayers        int    `json:"kv_cache_layers,omitempty"`
	KVCacheKeyValueHeads int    `json:"kv_cache_key_value_heads,omitempty"`
	KVCacheHeadDim       int    `json:"kv_cache_head_dim,omitempty"`
	KVCacheDTypeBytes    int    `json:"kv_cache_dtype_bytes,omitempty"`
	DefaultMaxModelLen   int    `json:"default_max_model_len,omitempty"`
	IsMultimodal         bool   `json:"is_multimodal"`
}

type ValidationError struct {
	Detail string
}

func (e ValidationError) Error() string {
	return e.Detail
}

func validationError(format string, args ...interface{}) error {
	return ValidationError{Detail: fmt.Sprintf(format, args...)}
}

func IsValidationError(err error) bool {
	var validationErr ValidationError
	return errors.As(err, &validationErr)
}

const (
	managedContainerLabel           = "vllm-manager.managed"
	containerKindLabel              = "vllm-manager.kind"
	containerModelLabel             = "vllm-manager.model"
	containerExposureLabel          = "vllm-manager.exposure"
	containerLaunchSpecVersionLabel = "vllm-manager.launch_spec_version"
	containerLaunchSpecLabel        = "vllm-manager.launch_spec"
	containerKindServer             = "server"
	containerKindDownload           = "download"
	ContainerExposureProxy          = "proxy"
	ContainerExposureDirect         = "direct"
	ParallelismTensor               = "tensor"
	ParallelismPipeline             = "pipeline"
	LaunchSpecVersion               = "1"
)

const DefaultVLLMImage = "vllm/vllm-openai:v0.23.0"

var (
	VLLMImage      = DefaultVLLMImage
	ManagerVersion = "dev"
)

var invalidDockerNameChars = regexp.MustCompile(`[^a-zA-Z0-9_.-]+`)

var modelOperationLocks sync.Map

func NormalizeHFHome(hfCache string) string {
	cleaned := filepath.Clean(strings.TrimSpace(hfCache))
	if cleaned == "." {
		return ""
	}
	if filepath.Base(cleaned) == "hub" {
		return filepath.Dir(cleaned)
	}
	return cleaned
}

func HFHubPath(hfCache string) string {
	return filepath.Join(NormalizeHFHome(hfCache), "hub")
}

func modelCacheDirName(model string) string {
	return "models--" + strings.ReplaceAll(strings.TrimSpace(model), "/", "--")
}

func modelOperationLockKey(hfCache string, model string) string {
	return filepath.Clean(HFHubPath(hfCache)) + "\x00" + strings.TrimSpace(model)
}

func getModelOperationLock(hfCache string, model string) *sync.Mutex {
	lock, _ := modelOperationLocks.LoadOrStore(modelOperationLockKey(hfCache, model), &sync.Mutex{})
	return lock.(*sync.Mutex)
}

// SanitizeDockerNamePart returns a Docker-name-safe fragment for model-derived
// container names.
func SanitizeDockerNamePart(value string) string {
	cleaned := invalidDockerNameChars.ReplaceAllString(value, "-")
	cleaned = strings.Trim(cleaned, "-_.")
	if cleaned == "" {
		return "model"
	}
	return cleaned
}

func FormatBytes(bytes uint64) string {
	if bytes == 0 {
		return "0 B"
	}

	// Binary (base-2) units: KiB = 1024 bytes, MiB = 1024 KiB, etc.
	units := []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
	value := float64(bytes)
	unitIndex := 0
	for value >= 1024 && unitIndex < len(units)-1 {
		value /= 1024
		unitIndex++
	}
	return fmt.Sprintf("%.2f %s", value, units[unitIndex])
}
