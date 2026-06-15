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
	ContainerID   string `json:"container_id"`
	ContainerName string `json:"container_name"`
	Model         string `json:"model"`
	Port          string `json:"port"`
	GPUs          string `json:"gpus"`
	GPUIDs        string `json:"gpu_ids"`
	MemUtil       string `json:"mem_util"`
	MaxLen        string `json:"max_len"`
	MaxNumSeqs    string `json:"max_num_seqs"`
	KVCacheDType  string `json:"kv_cache_dtype"`
	ToolParser    string `json:"tool_parser"`
	APIUrl        string `json:"api_url"`
	IsReady       bool   `json:"is_ready"`
	Arch          string `json:"arch"`
	State         string `json:"state"`
	APIKey        string `json:"-"`
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
	vllmImage             = "vllm/vllm-openai:latest"
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

func sanitizeDockerNamePart(value string) string {
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

	units := []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
	value := float64(bytes)
	unitIndex := 0
	for value >= 1024 && unitIndex < len(units)-1 {
		value /= 1024
		unitIndex++
	}
	return fmt.Sprintf("%.2f %s", value, units[unitIndex])
}
