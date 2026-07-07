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
	ContainerID     string `json:"container_id"`
	ContainerName   string `json:"container_name"`
	Model           string `json:"model"`
	Port            string `json:"port"`
	GPUs            string `json:"gpus"`
	GPUIDs          string `json:"gpu_ids"`
	MemUtil         string `json:"mem_util"`
	MaxLen          string `json:"max_len"`
	MaxNumSeqs      string `json:"max_num_seqs"`
	KVCacheDType    string `json:"kv_cache_dtype"`
	ToolParser      string `json:"tool_parser"`
	TokenizerMode   string `json:"tokenizer_mode"`
	ReasoningParser string `json:"reasoning_parser"`
	TrustRemoteCode bool   `json:"trust_remote_code"`
	BlockSize       string `json:"block_size"`
	BindIP          string `json:"bind_ip"`
	MetricsTarget   string `json:"metrics_target"`
	APIUrl          string `json:"api_url"`
	IsReady         bool   `json:"is_ready"`
	Arch            string `json:"arch"`
	State           string `json:"state"`
	APIKey          string `json:"-"`
	// HasServerAPIKey is populated by the dashboard handler so the UI can
	// distinguish a per-server key from the global one without exposing the
	// secret value.
	HasServerAPIKey bool   `json:"has_server_api_key,omitempty"`
	APIKeyMode      string `json:"api_key_mode,omitempty"`
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
	managedContainerLabel = "vllm-manager.managed"
	containerKindLabel    = "vllm-manager.kind"
	containerModelLabel   = "vllm-manager.model"
	containerKindServer   = "server"
	containerKindDownload = "download"
	managerVersion        = "0.1.0"
)

const DefaultVLLMImage = "vllm/vllm-openai:v0.23.0-cu129"

var VLLMImage = DefaultVLLMImage

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

func sanitizeDockerNamePart(value string) string {
	cleaned := invalidDockerNameChars.ReplaceAllString(value, "-")
	cleaned = strings.Trim(cleaned, "-_.")
	if cleaned == "" {
		return "model"
	}
	return cleaned
}

// SanitizeDockerNamePart is the exported form of sanitizeDockerNamePart.
// It is needed by callers outside the api package (e.g. the dashboard
// handler persisting per-server API keys) that must reproduce the same
// container name StartModel used.
func SanitizeDockerNamePart(value string) string {
	return sanitizeDockerNamePart(value)
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
