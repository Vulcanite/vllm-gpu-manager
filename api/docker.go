package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ActiveModel struct {
	ContainerID string `json:"container_id"`
	Model       string `json:"model"`
	Port        string `json:"port"`
	GPUs        string `json:"gpus"`
	GPUIDs      string `json:"gpu_ids"`
	MemUtil     string `json:"mem_util"`
	MaxLen      string `json:"max_len"`
	ToolParser  string `json:"tool_parser"`
	APIUrl      string `json:"api_url"`
	IsReady     bool   `json:"is_ready"`
	Arch        string `json:"arch"`
	State       string `json:"state"`
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

const (
	execTimeoutShort = 10 * time.Second // ps, inspect, rm, image inspect
	execTimeoutLong  = 30 * time.Second // stop
)

func execOutput(timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Output()
}

func execCombinedOutput(timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

func execRun(timeout time.Duration, name string, args ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).Run()
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

func redactDockerArgs(args []string) string {
	redacted := append([]string(nil), args...)
	for i := range redacted {
		if strings.HasPrefix(redacted[i], "HF_TOKEN=") {
			redacted[i] = "HF_TOKEN=<redacted>"
		}
		if redacted[i] == "-e" && i+1 < len(redacted) && strings.HasPrefix(redacted[i+1], "HF_TOKEN=") {
			redacted[i+1] = "HF_TOKEN=<redacted>"
		}
	}
	return strings.Join(redacted, " ")
}

func validateGPUSelection(gpus int, gpuIds string) (string, []GPUInfo, error) {
	gpuList, err := GetGPUInfo()
	if err == nil && len(gpuList) > 0 && gpus > len(gpuList) {
		return "", nil, validationError("requested %d GPUs, but only %d are available on this server", gpus, len(gpuList))
	}

	gpuIds = strings.TrimSpace(gpuIds)
	if gpuIds == "" || gpuIds == "all" {
		if err != nil || len(gpuList) == 0 {
			// No nvidia-smi data — fall back to "all" and skip VRAM preflight
			return "all", nil, nil
		}

		sort.Slice(gpuList, func(i, j int) bool {
			return gpuList[i].ID < gpuList[j].ID
		})
		selected := gpuList[:gpus]

		// Fix GPT-P2a: never pass --gpus all to Docker — it gives the container
		// visibility into every GPU on the host, not just the TP-assigned ones.
		// Convert to an explicit device list so the container is scoped correctly
		// and our VRAM/overlap preflight matches what the container actually sees.
		ids := make([]string, len(selected))
		for i, g := range selected {
			ids[i] = strconv.Itoa(g.ID)
		}
		return fmt.Sprintf("device=%s", strings.Join(ids, ",")), selected, nil
	}

	if err != nil {
		return "", nil, fmt.Errorf("cannot validate gpu_ids because nvidia-smi failed: %v", err)
	}

	available := make(map[int]GPUInfo, len(gpuList))
	for _, gpu := range gpuList {
		available[gpu.ID] = gpu
	}

	parts := strings.Split(gpuIds, ",")
	seen := make(map[int]bool, len(parts))
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return "", nil, validationError("gpu_ids must be 'all' or comma-separated GPU IDs")
		}

		id, err := strconv.Atoi(part)
		if err != nil || id < 0 {
			return "", nil, validationError("invalid GPU ID: %s", part)
		}
		if _, ok := available[id]; !ok {
			return "", nil, validationError("GPU ID %d is not available on this server", id)
		}
		if seen[id] {
			return "", nil, validationError("duplicate GPU ID: %d", id)
		}
		seen[id] = true
		ids = append(ids, id)
	}

	if len(ids) != gpus {
		return "", nil, validationError("gpu_ids contains %d GPUs, but tensor parallel size is %d", len(ids), gpus)
	}

	sort.Ints(ids)
	values := make([]string, len(ids))
	selected := make([]GPUInfo, len(ids))
	for i, id := range ids {
		values[i] = strconv.Itoa(id)
		selected[i] = available[id]
	}
	return fmt.Sprintf("device=%s", strings.Join(values, ",")), selected, nil
}

func ensureManagedContainer(containerID string) error {
	containerID = strings.TrimSpace(containerID)
	if containerID == "" {
		return validationError("container id is required")
	}

	out, err := execOutput(execTimeoutShort, "docker", "inspect", containerID)
	if err != nil {
		return fmt.Errorf("failed to inspect container %s: %v", containerID, err)
	}

	var containers []struct {
		Name   string `json:"Name"`
		Config struct {
			Labels map[string]string `json:"Labels"`
		} `json:"Config"`
	}
	if err := json.Unmarshal(out, &containers); err != nil {
		return fmt.Errorf("failed to parse container metadata: %v", err)
	}
	if len(containers) == 0 {
		return fmt.Errorf("container %s not found", containerID)
	}

	name := strings.TrimPrefix(containers[0].Name, "/")
	labels := containers[0].Config.Labels
	if labels["app"] == "vllm-manager" || labels[managedContainerLabel] == "true" || strings.HasPrefix(name, "vllm-") {
		return nil
	}

	return validationError("container %s is not managed by vllm-manager", containerID)
}

func GetActiveModels() ([]ActiveModel, error) {
	out, err := execOutput(execTimeoutShort, "docker", "ps", "-a", "--format", "{{.Names}}||{{.Status}}||{{.ID}}||{{.State}}")
	if err != nil {
		return nil, fmt.Errorf("docker ps failed: %v", err)
	}

	var models []ActiveModel
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	for _, line := range lines {
		if line == "" || !strings.HasPrefix(line, "vllm-") {
			continue
		}
		// Download containers (vllm-download-*) hold no GPU allocation and no
		// vLLM server port — exclude them from the active model list so they
		// don't poison GPU overlap accounting or the UI model table.
		if strings.HasPrefix(line, "vllm-download-") {
			continue
		}
		parts := strings.Split(line, "||")
		if len(parts) < 4 {
			continue
		}
		name := parts[0]
		containerID := parts[2]
		state := parts[3]

		// Inspect the container to get args + GPU device assignments
		inspectOut, err := execOutput(execTimeoutShort, "docker", "inspect", containerID)
		if err != nil {
			continue
		}

		var inspectData []struct {
			Args   []string `json:"Args"`
			Config struct {
				Labels map[string]string `json:"Labels"`
			} `json:"Config"`
			HostConfig struct {
				DeviceRequests []struct {
					DeviceIDs    []string   `json:"DeviceIDs"`
					Capabilities [][]string `json:"Capabilities"`
					Driver       string     `json:"Driver"`
				} `json:"DeviceRequests"`
			} `json:"HostConfig"`
		}
		if err := json.Unmarshal(inspectOut, &inspectData); err != nil || len(inspectData) == 0 {
			continue
		}

		args := inspectData[0].Args
		labels := inspectData[0].Config.Labels
		deviceRequests := inspectData[0].HostConfig.DeviceRequests

		model := ActiveModel{
			ContainerID: containerID,
			Model:       name, // fallback
			Port:        "unknown",
			GPUs:        "unknown",
			GPUIDs:      "all",
			MemUtil:     "unknown",
			MaxLen:      "default",
			ToolParser:  "disabled",
			APIUrl:      "unknown",
			IsReady:     false,
			State:       state,
		}

		if labels[containerModelLabel] != "" {
			model.Model = labels[containerModelLabel]
		}

		for i := 0; i < len(args); i++ {
			if args[i] == "--model" && i+1 < len(args) && labels[containerModelLabel] == "" {
				model.Model = args[i+1]
			}
			if args[i] == "--port" && i+1 < len(args) {
				model.Port = args[i+1]
			}
			if args[i] == "--tensor-parallel-size" && i+1 < len(args) {
				model.GPUs = args[i+1]
			}
			if args[i] == "--gpu-memory-utilization" && i+1 < len(args) {
				model.MemUtil = args[i+1]
			}
			if args[i] == "--max-model-len" && i+1 < len(args) {
				model.MaxLen = args[i+1]
			}
			if args[i] == "--tool-call-parser" && i+1 < len(args) {
				model.ToolParser = args[i+1]
			}
		}

		// Parse actual GPU IDs from DeviceRequests (--gpus device=0,1 → DeviceIDs: ["0","1"])
		// If DeviceIDs is empty, the container was launched with --gpus all
		for _, dr := range deviceRequests {
			if len(dr.DeviceIDs) > 0 {
				model.GPUIDs = strings.Join(dr.DeviceIDs, ",")
			}
			// else: GPUIDs stays "all"
		}

		if model.Port != "unknown" {
			model.APIUrl = fmt.Sprintf("http://localhost:%s/v1", model.Port)

			// Vision Heuristics
			lower := strings.ToLower(model.Model)
			if strings.Contains(lower, "vl") || strings.Contains(lower, "vision") || strings.Contains(lower, "llava") || strings.Contains(lower, "pixtral") {
				model.Arch = "Vision + Text"
			} else {
				model.Arch = "Text-Only"
			}
		}

		models = append(models, model)
	}

	// Health checks: run in parallel to avoid sequential 1s timeouts stacking up
	var wg sync.WaitGroup
	for i := range models {
		if models[i].Port == "unknown" || models[i].State != "running" {
			continue
		}
		wg.Add(1)
		go func(m *ActiveModel) {
			defer wg.Done()
			client := http.Client{Timeout: 1 * time.Second}
			resp, err := client.Get(fmt.Sprintf("http://localhost:%s/v1/models", m.Port))
			if err != nil {
				return
			}
			defer resp.Body.Close()
			m.IsReady = (resp.StatusCode == http.StatusOK)
			if m.IsReady {
				var modelResp struct {
					Data []struct {
						MaxModelLen int `json:"max_model_len"`
					} `json:"data"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&modelResp); err == nil && len(modelResp.Data) > 0 {
					if m.MaxLen == "default" {
						m.MaxLen = strconv.Itoa(modelResp.Data[0].MaxModelLen)
					}
				}
			}
		}(&models[i])
	}
	wg.Wait()

	return models, nil
}

func GetVLLMContainerNames() ([]string, error) {
	out, err := execOutput(execTimeoutShort, "docker", "ps", "-a", "--format", "{{.Names}}")
	if err != nil {
		return nil, fmt.Errorf("docker ps failed: %v", err)
	}

	var names []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		name := strings.TrimSpace(line)
		if name == "" || !strings.HasPrefix(name, "vllm-") {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

type modelSizeInfo struct {
	Display  string
	Bytes    uint64
	CachedAt time.Time
}

var (
	modelSizeCache   = make(map[string]modelSizeInfo)
	modelSizeCacheMu sync.RWMutex
)

type modelKVCacheEstimate struct {
	Layers             int
	KeyValueHeads      int
	HeadDim            int
	DTypeBytes         int
	DefaultMaxModelLen int
}

const (
	defaultGPUMemoryUtilization = 0.9
	modelVRAMSafetyMargin       = 1.08
	modelSizeCacheTTL           = 2 * time.Second
	vllmImage                   = "vllm/vllm-openai:latest"
)

func formatBytes(bytes uint64) string {
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

func modelCacheDir(hfCache string, model string) string {
	return filepath.Join(HFHubPath(hfCache), modelCacheDirName(model))
}

func ceilDiv(a int, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func (estimate modelKVCacheEstimate) isComplete() bool {
	return estimate.Layers > 0 && estimate.KeyValueHeads > 0 && estimate.HeadDim > 0 && estimate.DTypeBytes > 0
}

func (estimate modelKVCacheEstimate) fullBytesPerToken() uint64 {
	if !estimate.isComplete() {
		return 0
	}
	return uint64(2 * estimate.Layers * estimate.KeyValueHeads * estimate.HeadDim * estimate.DTypeBytes)
}

func (estimate modelKVCacheEstimate) bytesPerTokenPerGPU(tensorParallelSize int) uint64 {
	if !estimate.isComplete() || tensorParallelSize <= 0 {
		return 0
	}

	kvHeadsPerGPU := ceilDiv(estimate.KeyValueHeads, tensorParallelSize)
	if kvHeadsPerGPU < 1 {
		kvHeadsPerGPU = 1
	}
	return uint64(2 * estimate.Layers * kvHeadsPerGPU * estimate.HeadDim * estimate.DTypeBytes)
}

func findModelConfigPath(hfCache string, model string) (string, error) {
	root := modelCacheDir(hfCache, model)

	// Primary path: follow refs/main → snapshots/<revision>/config.json
	if data, err := os.ReadFile(filepath.Join(root, "refs", "main")); err == nil {
		revision := strings.TrimSpace(string(data))
		if revision != "" && !filepath.IsAbs(revision) && !strings.Contains(revision, "..") {
			candidate := filepath.Join(root, "snapshots", revision, "config.json")
			if stat, err := os.Stat(candidate); err == nil && !stat.IsDir() {
				return candidate, nil
			}
		}
	}

	// Fallback: walk the cache tree. Sort snapshot configs by the ModTime of their
	// parent snapshot directory (newest first) so that if multiple revisions exist
	// we use the most recently downloaded one, not an arbitrary lexicographic hash.
	type configCandidate struct {
		path    string
		modTime time.Time
	}
	var snapshotCandidates []configCandidate
	var otherCandidates []configCandidate

	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || entry.Name() != "config.json" {
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}

		// Determine ModTime: use the snapshot directory's mtime, not the file's,
		// because HF hub sometimes preserves file mtimes from the remote.
		var modTime time.Time
		if strings.HasPrefix(filepath.ToSlash(rel), "snapshots/") {
			// path = root/snapshots/<hash>/config.json — stat the <hash> dir
			snapshotDir := filepath.Dir(path)
			if info, err := os.Stat(snapshotDir); err == nil {
				modTime = info.ModTime()
			}
			snapshotCandidates = append(snapshotCandidates, configCandidate{path, modTime})
		} else {
			if info, err := entry.Info(); err == nil {
				modTime = info.ModTime()
			}
			otherCandidates = append(otherCandidates, configCandidate{path, modTime})
		}
		return nil
	}); err != nil {
		return "", err
	}

	// Sort newest first
	sortNewest := func(candidates []configCandidate) {
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].modTime.After(candidates[j].modTime)
		})
	}
	sortNewest(snapshotCandidates)
	sortNewest(otherCandidates)

	if len(snapshotCandidates) > 0 {
		return snapshotCandidates[0].path, nil
	}
	if len(otherCandidates) > 0 {
		return otherCandidates[0].path, nil
	}
	return "", os.ErrNotExist
}

func jsonInt(value interface{}) (int, bool) {
	switch v := value.(type) {
	case float64:
		if v <= 0 {
			return 0, false
		}
		return int(v), true
	case int:
		return v, v > 0
	case int64:
		if v <= 0 {
			return 0, false
		}
		return int(v), true
	case string:
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || parsed <= 0 {
			return 0, false
		}
		return parsed, true
	default:
		return 0, false
	}
}

func jsonString(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		v = strings.TrimSpace(v)
		return v, v != ""
	default:
		return "", false
	}
}

func getConfigObject(config map[string]interface{}, key string) (map[string]interface{}, bool) {
	raw, ok := config[key]
	if !ok {
		return nil, false
	}
	nested, ok := raw.(map[string]interface{})
	return nested, ok
}

func getConfigInt(primary map[string]interface{}, fallback map[string]interface{}, names ...string) (int, bool) {
	configs := []map[string]interface{}{primary}
	if fallback != nil {
		configs = append(configs, fallback)
	}

	for _, config := range configs {
		if config == nil {
			continue
		}
		for _, name := range names {
			if value, ok := jsonInt(config[name]); ok {
				return value, true
			}
		}
	}
	return 0, false
}

func getConfigString(primary map[string]interface{}, fallback map[string]interface{}, names ...string) (string, bool) {
	configs := []map[string]interface{}{primary}
	if fallback != nil {
		configs = append(configs, fallback)
	}

	for _, config := range configs {
		if config == nil {
			continue
		}
		for _, name := range names {
			if value, ok := jsonString(config[name]); ok {
				return value, true
			}
		}
	}
	return "", false
}

func kvElementBytes(dtype string) int {
	dtype = strings.ToLower(strings.TrimSpace(dtype))
	switch {
	case strings.Contains(dtype, "float8") || strings.Contains(dtype, "fp8") || strings.Contains(dtype, "e4m3") || strings.Contains(dtype, "e5m2") || strings.Contains(dtype, "int8"):
		return 1
	case strings.Contains(dtype, "float32") || strings.Contains(dtype, "fp32") || dtype == "float":
		return 4
	default:
		return 2
	}
}

func GetModelDirectorySize(model, hfCache string) (uint64, error) {
	hubPath := HFHubPath(hfCache)
	if hubPath == "" {
		return 0, fmt.Errorf("HF cache path not configured")
	}

	modelPath := filepath.Join(hubPath, modelCacheDirName(model))

	var totalSize int64
	err := filepath.Walk(modelPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if !info.IsDir() {
			totalSize += info.Size()
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return uint64(totalSize), nil
}

func GetModelKVCacheEstimate(hfCache string, model string) (modelKVCacheEstimate, error) {
	configPath, err := findModelConfigPath(hfCache, model)
	if err != nil {
		return modelKVCacheEstimate{}, err
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return modelKVCacheEstimate{}, err
	}

	var root map[string]interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return modelKVCacheEstimate{}, err
	}

	config := root
	if textConfig, ok := getConfigObject(root, "text_config"); ok {
		config = textConfig
	}

	layers, ok := getConfigInt(config, root, "num_hidden_layers", "n_layer", "num_layers", "num_layers_total")
	if !ok {
		return modelKVCacheEstimate{}, validationError("model config does not include num_hidden_layers")
	}

	attentionHeads, ok := getConfigInt(config, root, "num_attention_heads", "n_head", "num_heads")
	if !ok {
		return modelKVCacheEstimate{}, validationError("model config does not include num_attention_heads")
	}

	hiddenSize, ok := getConfigInt(config, root, "hidden_size", "n_embd", "d_model")
	if !ok {
		return modelKVCacheEstimate{}, validationError("model config does not include hidden_size")
	}

	keyValueHeads, ok := getConfigInt(config, root, "num_key_value_heads", "n_kv_head", "num_kv_heads")
	if !ok {
		keyValueHeads = attentionHeads
	}

	headDim, ok := getConfigInt(config, root, "head_dim", "kv_channels")
	if !ok && attentionHeads > 0 {
		headDim = hiddenSize / attentionHeads
	}
	if headDim <= 0 {
		return modelKVCacheEstimate{}, validationError("model config does not include a usable attention head dimension")
	}

	dtype := ""
	if value, ok := getConfigString(config, root, "torch_dtype", "dtype"); ok {
		dtype = value
	}

	maxModelLen, _ := getConfigInt(
		config,
		root,
		"max_position_embeddings",
		"max_sequence_length",
		"max_seq_len",
		"model_max_length",
		"seq_length",
		"n_positions",
		"sliding_window",
	)

	return modelKVCacheEstimate{
		Layers:             layers,
		KeyValueHeads:      keyValueHeads,
		HeadDim:            headDim,
		DTypeBytes:         kvElementBytes(dtype),
		DefaultMaxModelLen: maxModelLen,
	}, nil
}

func calculateDirectorySize(path string) (uint64, error) {
	var size uint64
	err := filepath.WalkDir(path, func(_ string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Size() > 0 {
			size += uint64(info.Size())
		}
		return nil
	})
	return size, err
}

func GetDownloadedModelSize(hfCache string, model string) (uint64, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, "", validationError("model is required")
	}

	path := modelCacheDir(hfCache, model)
	cacheKey := filepath.Clean(path)

	modelSizeCacheMu.RLock()
	cached, ok := modelSizeCache[cacheKey]
	modelSizeCacheMu.RUnlock()
	if ok && time.Since(cached.CachedAt) < modelSizeCacheTTL {
		return cached.Bytes, cached.Display, nil
	}

	stat, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", validationError("model %s is not downloaded in the configured Hugging Face home", model)
		}
		return 0, "", fmt.Errorf("failed to inspect model cache: %v", err)
	}
	if !stat.IsDir() {
		return 0, "", validationError("model cache path is not a directory: %s", path)
	}

	bytes, err := calculateDirectorySize(path)
	if err != nil {
		return 0, "", fmt.Errorf("failed to calculate model size: %v", err)
	}

	display := formatBytes(bytes)
	modelSizeCacheMu.Lock()
	modelSizeCache[cacheKey] = modelSizeInfo{
		Display:  display,
		Bytes:    bytes,
		CachedAt: time.Now(),
	}
	modelSizeCacheMu.Unlock()
	return bytes, display, nil
}

func clearDownloadedModelSizeCache(path string) {
	cacheKey := filepath.Clean(path)
	modelSizeCacheMu.Lock()
	delete(modelSizeCache, cacheKey)
	modelSizeCacheMu.Unlock()
}

func getDownloadContainerModel(name string) (string, error) {
	out, err := execOutput(execTimeoutShort, "docker", "inspect", name)
	if err != nil {
		return "", fmt.Errorf("failed to inspect download container %s: %v", name, err)
	}

	var inspectData []struct {
		Args   []string `json:"Args"`
		Config struct {
			Cmd    []string          `json:"Cmd"`
			Labels map[string]string `json:"Labels"`
		} `json:"Config"`
	}
	if err := json.Unmarshal(out, &inspectData); err != nil || len(inspectData) == 0 {
		return "", fmt.Errorf("failed to parse download container metadata for %s", name)
	}

	if model := inspectData[0].Config.Labels[containerModelLabel]; model != "" {
		return model, nil
	}

	args := inspectData[0].Args
	if len(args) == 0 {
		args = inspectData[0].Config.Cmd
	}
	for i := 0; i < len(args)-1; i++ {
		if args[i] == "download" {
			return args[i+1], nil
		}
	}

	return "", nil
}

func findDownloadContainerForModel(model string) (string, error) {
	names, err := GetVLLMContainerNames()
	if err != nil {
		return "", err
	}

	for _, name := range names {
		if !strings.HasPrefix(name, "vllm-download-") {
			continue
		}

		downloadModel, err := getDownloadContainerModel(name)
		if err != nil {
			return "", err
		}
		if downloadModel == model {
			return name, nil
		}
	}

	return "", nil
}

func ensureModelCacheCanBeDeleted(model string) error {
	active, err := GetActiveModels()
	if err != nil {
		return fmt.Errorf("failed to verify active model containers before deleting cache: %v", err)
	}

	for _, activeModel := range active {
		if activeModel.Model == model {
			return validationError(
				"cannot delete %s because container %s still references it; remove the container first",
				model,
				activeModel.ContainerID,
			)
		}
	}

	containerName, err := findDownloadContainerForModel(model)
	if err != nil {
		return fmt.Errorf("failed to verify download containers before deleting cache: %v", err)
	}
	if containerName != "" {
		return validationError("cannot delete %s because it is currently being downloaded by %s", model, containerName)
	}

	return nil
}

func DeleteDownloadedModel(model string, hfCache string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return validationError("model is required")
	}
	if strings.ContainsRune(model, 0) {
		return validationError("model contains an invalid null byte")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	if err := ensureModelCacheCanBeDeleted(model); err != nil {
		return err
	}

	hubPath := filepath.Clean(HFHubPath(hfCache))
	hubStat, err := os.Stat(hubPath)
	if err != nil {
		if os.IsNotExist(err) {
			return validationError("Hugging Face hub directory does not exist: %s", hubPath)
		}
		return fmt.Errorf("failed to inspect Hugging Face hub directory: %v", err)
	}
	if !hubStat.IsDir() {
		return validationError("Hugging Face hub path is not a directory: %s", hubPath)
	}

	modelDirName := modelCacheDirName(model)
	modelPath := filepath.Clean(filepath.Join(hubPath, modelDirName))
	if filepath.Dir(modelPath) != hubPath {
		return validationError("resolved model cache path escapes Hugging Face hub directory")
	}

	modelStat, err := os.Stat(modelPath)
	if err != nil {
		if os.IsNotExist(err) {
			return validationError("model %s is not downloaded in the configured Hugging Face cache", model)
		}
		return fmt.Errorf("failed to inspect model cache directory: %v", err)
	}
	if !modelStat.IsDir() {
		return validationError("model cache path is not a directory: %s", modelPath)
	}

	args := []string{
		"run", "--rm",
		"-v", fmt.Sprintf("%s:/target", hubPath),
		"alpine",
		"rm", "-rf", "/target/" + modelDirName,
	}

	log.Printf("[DOCKER] Executing: docker %s\n", strings.Join(args, " "))

	out, err := exec.Command("docker", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete downloaded model cache: %v, output: %s", err, string(out))
	}

	clearDownloadedModelSizeCache(modelPath)
	return nil
}

func parseGPUMemoryUtilization(memUtil string) (float64, error) {
	memUtil = strings.TrimSpace(memUtil)
	if memUtil == "" {
		return defaultGPUMemoryUtilization, nil
	}

	utilization, err := strconv.ParseFloat(memUtil, 64)
	if err != nil || utilization <= 0 || utilization > 1.0 {
		return 0, validationError("memory utilization must be a float between 0.01 and 1.0")
	}
	return utilization, nil
}

func parseMaxModelLen(maxLen string) (int, error) {
	maxLen = strings.TrimSpace(maxLen)
	if maxLen == "" {
		return 0, nil
	}

	maxLenNum, err := strconv.Atoi(maxLen)
	if err != nil || maxLenNum <= 0 {
		return 0, validationError("max context length must be a positive integer")
	}
	return maxLenNum, nil
}

// checkVLLMImageExists verifies the configured vLLM image is present locally.
// It warns but does not hard-fail — docker run will attempt a pull automatically
// if the image is absent and network is available.
func checkVLLMImageExists() {
	err := execRun(execTimeoutShort, "docker", "image", "inspect", vllmImage)
	if err != nil {
		log.Printf("[DOCKER] WARNING: image %s not found locally — docker will attempt to pull it, which may take several minutes", vllmImage)
	}
}

// minKVCacheBytesPerGPU is the floor we require after weights are loaded.
// vLLM crashes at startup with "No available memory for cache blocks" below ~1 GiB.
const minKVCacheBytesPerGPU = 1 * 1024 * 1024 * 1024 // 1 GiB

// checkGPUOverlap checks whether the requested GPUs are already in use and
// whether adding the new container's mem_util would exceed 1.0 on any shared GPU.
//
// vLLM pre-allocates gpu_memory_utilization × total_vram at startup, so two
// containers can legitimately share a GPU only if their combined mem_util ≤ 1.0.
// A hard overlap block regardless of mem_util is too aggressive — it prevents
// valid configurations like a 0.5+0.4 split across the same GPU.
//
// requestedIDs is empty when gpu_ids="all"; totalGPUs is used to expand it.
// requestedMemUtil is the new container's parsed mem_util (0.0–1.0).
func checkGPUOverlap(requestedIDs []int, totalGPUs int, requestedMemUtil float64) error {
	active, err := GetActiveModels()
	if err != nil {
		// Can't verify — log and allow rather than hard-block
		log.Printf("[GPU] could not fetch active models for overlap check: %v", err)
		return nil
	}

	// Expand "all" to explicit IDs
	requested := make(map[int]bool, len(requestedIDs))
	for _, id := range requestedIDs {
		requested[id] = true
	}
	if len(requestedIDs) == 0 {
		for i := 0; i < totalGPUs; i++ {
			requested[i] = true
		}
	}

	// Accumulate mem_util already committed per GPU across running model containers.
	// Download containers have no GPU allocation and are excluded by the caller via
	// the isDownloadContainer check in GetActiveModels.
	committedUtil := make(map[int]float64, totalGPUs)
	for _, m := range active {
		if strings.ToLower(m.State) != "running" {
			continue
		}

		existingUtil, err := parseGPUMemoryUtilization(m.MemUtil)
		if err != nil {
			// Unknown mem_util — treat as full allocation to be conservative
			existingUtil = 1.0
		}

		var existingIDs []int
		if m.GPUIDs == "all" || m.GPUIDs == "" {
			tpSize, _ := strconv.Atoi(m.GPUs)
			if tpSize <= 0 {
				tpSize = totalGPUs
			}
			for i := 0; i < tpSize; i++ {
				existingIDs = append(existingIDs, i)
			}
		} else {
			for _, p := range strings.Split(m.GPUIDs, ",") {
				if id, err := strconv.Atoi(strings.TrimSpace(p)); err == nil {
					existingIDs = append(existingIDs, id)
				}
			}
		}

		for _, id := range existingIDs {
			committedUtil[id] += existingUtil
		}
	}

	// Check every requested GPU: committed + new must not exceed 1.0
	var overloaded []string
	for id := range requested {
		combined := committedUtil[id] + requestedMemUtil
		if combined > 1.0 {
			overloaded = append(overloaded,
				fmt.Sprintf("GPU %d (already %.0f%% committed, adding %.0f%% would reach %.0f%%)",
					id,
					committedUtil[id]*100,
					requestedMemUtil*100,
					combined*100,
				),
			)
		}
	}
	if len(overloaded) > 0 {
		sort.Strings(overloaded)
		return validationError(
			"combined GPU memory utilization would exceed 100%% on: %s — stop existing containers or lower mem_util",
			strings.Join(overloaded, "; "),
		)
	}
	return nil
}

func CheckModelFitsInVRAM(model string, hfCache string, selectedGPUs []GPUInfo, gpus int, memUtil string, maxLen string) error {
	if len(selectedGPUs) == 0 {
		return nil
	}

	modelBytes, _, err := GetDownloadedModelSize(hfCache, model)
	if err != nil {
		return err
	}
	if modelBytes == 0 {
		return nil
	}

	utilization, err := parseGPUMemoryUtilization(memUtil)
	if err != nil {
		return err
	}

	requestedMaxLen, err := parseMaxModelLen(maxLen)
	if err != nil {
		return err
	}

	modelBytesPerGPU := uint64((float64(modelBytes) / float64(gpus)) * modelVRAMSafetyMargin)

	estimate, kvErr := GetModelKVCacheEstimate(hfCache, model)
	kvEstimateOK := kvErr == nil && estimate.isComplete()

	// Fix #5: reject max-model-len that exceeds the model's rope/positional limit
	if kvEstimateOK && requestedMaxLen > 0 && estimate.DefaultMaxModelLen > 0 && requestedMaxLen > estimate.DefaultMaxModelLen {
		return validationError(
			"requested max_model_len %d exceeds model's maximum positional encoding length %d; vLLM will refuse to start",
			requestedMaxLen,
			estimate.DefaultMaxModelLen,
		)
	}

	// Fix #6: KV heads must be divisible by tensor parallel size
	if kvEstimateOK && estimate.KeyValueHeads > 0 && estimate.KeyValueHeads%gpus != 0 {
		return validationError(
			"num_key_value_heads (%d) is not divisible by tensor_parallel_size (%d); vLLM will refuse to start — use a divisor of %d GPUs instead",
			estimate.KeyValueHeads,
			gpus,
			estimate.KeyValueHeads,
		)
	}

	var kvCacheBytesPerGPU uint64
	var effectiveMaxLen int
	var kvBytesPerTokenPerGPU uint64
	if kvEstimateOK {
		effectiveMaxLen = requestedMaxLen
		if effectiveMaxLen == 0 {
			effectiveMaxLen = estimate.DefaultMaxModelLen
		}
		if effectiveMaxLen > 0 {
			kvBytesPerTokenPerGPU = estimate.bytesPerTokenPerGPU(gpus)
			kvCacheBytesPerGPU = kvBytesPerTokenPerGPU * uint64(effectiveMaxLen)
		}
	} else if kvErr != nil {
		log.Printf("[VRAM] KV cache estimate unavailable for %s: %v", model, kvErr)
	}

	for _, gpu := range selectedGPUs {
		// Fix #1: budget is total×util, not MemoryFree.
		// MemoryFree is volatile and doesn't represent what vLLM will actually get.
		utilizationBudget := uint64(float64(gpu.MemoryTotal) * utilization)

		// Weights must physically fit in currently free memory (can't page them in)
		if modelBytesPerGPU > gpu.MemoryFree {
			return validationError(
				"model weights need ~%s per GPU but GPU %d only has %s currently free — stop other workloads first",
				formatBytes(modelBytesPerGPU),
				gpu.ID,
				formatBytes(gpu.MemoryFree),
			)
		}

		// Total (weights + KV) must fit within the utilization budget
		requiredPerGPU := modelBytesPerGPU + kvCacheBytesPerGPU
		if requiredPerGPU > utilizationBudget {
			if kvCacheBytesPerGPU > 0 {
				return validationError(
					"model weights ~%s plus KV cache ~%s for max_model_len=%d needs ~%s per GPU, but the %.2f utilization budget on GPU %d is only %s — reduce max_model_len or increase mem_util",
					formatBytes(modelBytesPerGPU),
					formatBytes(kvCacheBytesPerGPU),
					effectiveMaxLen,
					formatBytes(requiredPerGPU),
					utilization,
					gpu.ID,
					formatBytes(utilizationBudget),
				)
			}
			return validationError(
				"model weights ~%s per GPU (with %.0f%% safety margin) exceeds the %.2f utilization budget of %s on GPU %d",
				formatBytes(modelBytesPerGPU),
				(modelVRAMSafetyMargin-1)*100,
				utilization,
				formatBytes(utilizationBudget),
				gpu.ID,
			)
		}

		// Fix #4: ensure minimum viable KV cache remains after weights
		remainingForKV := utilizationBudget - modelBytesPerGPU
		if remainingForKV < minKVCacheBytesPerGPU {
			return validationError(
				"only ~%s remains for KV cache on GPU %d after loading model weights (budget: %s, weights: ~%s) — vLLM requires at least %s; increase mem_util or use fewer GPUs",
				formatBytes(remainingForKV),
				gpu.ID,
				formatBytes(utilizationBudget),
				formatBytes(modelBytesPerGPU),
				formatBytes(minKVCacheBytesPerGPU),
			)
		}
	}

	if kvCacheBytesPerGPU > 0 {
		log.Printf("[VRAM] %s: ~%s/GPU total (%s weights + %s KV cache at max_model_len=%d, %s/token/GPU)",
			model,
			formatBytes(modelBytesPerGPU+kvCacheBytesPerGPU),
			formatBytes(modelBytesPerGPU),
			formatBytes(kvCacheBytesPerGPU),
			effectiveMaxLen,
			formatBytes(kvBytesPerTokenPerGPU),
		)
	}
	return nil
}

func GetDownloadedModels(hfCache string) ([]DownloadedModel, error) {
	hubPath := HFHubPath(hfCache)
	entries, err := os.ReadDir(hubPath)
	if err != nil {
		return nil, nil // Return empty if dir doesn't exist
	}

	var models []DownloadedModel
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelName := strings.ReplaceAll(strings.TrimPrefix(entry.Name(), "models--"), "--", "/")
			size := "unknown"
			var sizeBytes uint64
			if bytes, display, err := GetDownloadedModelSize(hfCache, modelName); err == nil {
				size = display
				sizeBytes = bytes
			}
			kvEstimate, _ := GetModelKVCacheEstimate(hfCache, modelName)

			models = append(models, DownloadedModel{
				Name:                 modelName,
				Size:                 size,
				SizeBytes:            sizeBytes,
				KVCacheBytesPerToken: kvEstimate.fullBytesPerToken(),
				KVCacheLayers:        kvEstimate.Layers,
				KVCacheKeyValueHeads: kvEstimate.KeyValueHeads,
				KVCacheHeadDim:       kvEstimate.HeadDim,
				KVCacheDTypeBytes:    kvEstimate.DTypeBytes,
				DefaultMaxModelLen:   kvEstimate.DefaultMaxModelLen,
			})
		}
	}
	return models, nil
}

func StartModel(model string, port int, gpus int, gpuIds string, memUtil string, maxLen string, toolParser string, hfCache string, hfToken string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return validationError("model is required")
	}
	if port < 1 || port > 65535 {
		return validationError("port must be between 1 and 65535")
	}
	if gpus < 1 {
		return validationError("must specify at least 1 GPU")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	if containerName, err := findDownloadContainerForModel(model); err != nil {
		return fmt.Errorf("failed to verify download containers before starting model: %v", err)
	} else if containerName != "" {
		return validationError("cannot start %s because it is currently being downloaded by %s", model, containerName)
	}

	utilization, err := parseGPUMemoryUtilization(memUtil)
	if err != nil {
		return err
	}
	memUtil = strconv.FormatFloat(utilization, 'f', -1, 64)

	if _, err := parseMaxModelLen(maxLen); err != nil {
		return err
	}

	if toolParser != "" {
		matched, _ := regexp.MatchString(`^[a-zA-Z0-9\-]+$`, toolParser)
		if !matched {
			return validationError("invalid tool parser format")
		}
	}

	// Check if port is already in use
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return validationError("port %d is already in use", port)
	}
	l.Close()

	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	shortName = sanitizeDockerNamePart(shortName)
	containerName := fmt.Sprintf("vllm-%s-%d", shortName, port)

	gpuFlag, selectedGPUs, err := validateGPUSelection(gpus, gpuIds)
	if err != nil {
		return err
	}

	// Fix #2 / GPT-P1a: reject if combined mem_util on any requested GPU would exceed 1.0
	totalGPUCount := len(selectedGPUs)
	if totalGPUCount == 0 {
		if gpuList, err := GetGPUInfo(); err == nil {
			totalGPUCount = len(gpuList)
		}
	}
	var requestedGPUIDs []int
	if gpuIds != "" && gpuIds != "all" {
		for _, part := range strings.Split(gpuIds, ",") {
			if id, err := strconv.Atoi(strings.TrimSpace(part)); err == nil {
				requestedGPUIDs = append(requestedGPUIDs, id)
			}
		}
	}
	// selectedGPUs is now always an explicit list (GPT-P2a fix), so derive IDs from it
	// when gpuIds was "all" — requestedGPUIDs will be empty in that case.
	if len(requestedGPUIDs) == 0 && len(selectedGPUs) > 0 {
		for _, g := range selectedGPUs {
			requestedGPUIDs = append(requestedGPUIDs, g.ID)
		}
	}
	if err := checkGPUOverlap(requestedGPUIDs, totalGPUCount, utilization); err != nil {
		return err
	}

	if err := CheckModelFitsInVRAM(model, hfCache, selectedGPUs, gpus, memUtil, maxLen); err != nil {
		return err
	}

	// Deploy should create a fresh container only. Existing containers keep their
	// original GPU devices, env, image, and volume mounts, so restarting them here
	// can silently run a different configuration than the user requested.
	out, _ := execOutput(execTimeoutShort, "docker", "ps", "-a", "--format", "{{.Names}}")
	if strings.Contains(string(out), containerName+"\n") {
		return validationError("container %q already exists; remove it before deploying a new model with this name and port", containerName)
	}

	args := []string{
		"run", "-d",
		"--ipc=host",
		"--runtime", "nvidia",
		"--gpus", gpuFlag,
		"--network", "host",
		"--label", "app=vllm-manager",
		"--label", managedContainerLabel + "=true",
		"--label", containerKindLabel + "=" + containerKindServer,
		"--label", containerModelLabel + "=" + model,
		"--name", containerName,
		"-v", fmt.Sprintf("%s:/hf_cache", hfCache),
		"-e", "HF_HOME=/hf_cache",
	}

	if hfToken != "" {
		args = append(args, "-e", fmt.Sprintf("HF_TOKEN=%s", hfToken))
	}

	// Warn if the configured image is not present locally.
	checkVLLMImageExists()

	args = append(args,
		vllmImage,
		"--model", model,
		"--tensor-parallel-size", strconv.Itoa(gpus),
		"--port", strconv.Itoa(port),
		"--gpu-memory-utilization", memUtil,
	)

	if maxLen != "" {
		args = append(args, "--max-model-len", maxLen)
	}
	if toolParser != "" {
		args = append(args, "--tool-call-parser", toolParser, "--enable-auto-tool-choice")
	}

	log.Printf("[DOCKER] Executing: docker %s\n", redactDockerArgs(args))

	runOut, err := exec.Command("docker", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start vLLM container: %v, output: %s", err, string(runOut))
	}
	return nil
}

func StopModel(port int) error {
	// Find container name by port
	out, _ := execOutput(execTimeoutShort, "docker", "ps", "-a", "--format", "{{.Names}}")
	lines := strings.Split(string(out), "\n")
	suffix := fmt.Sprintf("-%d", port)

	for _, line := range lines {
		if strings.HasPrefix(line, "vllm-") && strings.HasSuffix(line, suffix) {
			return execRun(execTimeoutLong, "docker", "stop", line)
		}
	}
	return validationError("no vLLM server found on port %d", port)
}

func RemoveModel(containerID string) error {
	if err := ensureManagedContainer(containerID); err != nil {
		return err
	}
	out, err := execCombinedOutput(execTimeoutShort, "docker", "rm", "-f", containerID)
	if err != nil {
		return fmt.Errorf("failed to remove container %s: %v, output: %s", containerID, err, string(out))
	}
	return nil
}

func RestartModel(containerID string) error {
	if err := ensureManagedContainer(containerID); err != nil {
		return err
	}
	out, err := execCombinedOutput(execTimeoutShort, "docker", "start", containerID)
	if err != nil {
		return fmt.Errorf("failed to restart container %s: %v, output: %s", containerID, err, string(out))
	}
	return nil
}

func DownloadModel(model string, hfCache string, hfToken string) (string, error) {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return "", validationError("hugging face cache directory is required")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return "", validationError("model is required")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	if containerName, err := findDownloadContainerForModel(model); err != nil {
		return "", fmt.Errorf("failed to verify download containers before downloading model: %v", err)
	} else if containerName != "" {
		return "", validationError("%s is already being downloaded by %s", model, containerName)
	}

	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	containerName := fmt.Sprintf("vllm-download-%s-%d", sanitizeDockerNamePart(shortName), time.Now().Unix())

	args := []string{
		"run", "-d", "--rm",
		"--network", "host",
		"--label", "app=vllm-manager",
		"--label", managedContainerLabel + "=true",
		"--label", containerKindLabel + "=" + containerKindDownload,
		"--label", containerModelLabel + "=" + model,
		"--name", containerName,
		"-v", fmt.Sprintf("%s:/hf_cache", hfCache),
		"-e", "HF_HOME=/hf_cache",
	}

	if hfToken != "" {
		args = append(args, "-e", fmt.Sprintf("HF_TOKEN=%s", hfToken))
	}

	args = append(args,
		"--entrypoint", "hf",
		vllmImage,
		"download", model,
	)

	log.Printf("[DOCKER] Executing: docker %s\n", redactDockerArgs(args))

	out, err := exec.Command("docker", args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to start download container: %v, output: %s", err, string(out))
	}
	return strings.TrimSpace(string(out)), nil
}
