package api

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
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
	defaultMaxModelLen          = 32768
	modelVRAMSafetyMargin       = 1.08
)

// kvCacheDTypeOptions is the single source of truth for accepted KV cache
// dtypes. It is consumed by normalizeKVCacheDType and surfaced to the frontend
// via LaunchDefaults so the UI and backend cannot drift.
var kvCacheDTypeOptions = []string{"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}

// LaunchConfig exposes the tuning constants the dashboard needs to reproduce the
// backend's VRAM preflight preview. Serving these from one place keeps the
// frontend estimate in sync with CheckModelFitsInVRAM.
type LaunchConfig struct {
	ModelVRAMSafetyMargin       float64  `json:"model_vram_safety_margin"`
	DefaultGPUMemoryUtilization float64  `json:"default_gpu_memory_utilization"`
	DefaultMaxModelLen          int      `json:"default_max_model_len"`
	HFCache                     string   `json:"hf_cache,omitempty"`
	VLLMImage                   string   `json:"vllm_image"`
	KVCacheDTypeOptions         []string `json:"kv_cache_dtype_options"`
	ToolParserOptions           []string `json:"tool_parser_options"`
	TokenizerModeOptions        []string `json:"tokenizer_mode_options"`
	ReasoningParserOptions      []string `json:"reasoning_parser_options"`
	BlockSizeOptions            []int    `json:"block_size_options"`
	MoeBackendOptions           []string `json:"moe_backend_options"`
}

// LaunchDefaults returns the launch tuning constants shared with the frontend.
func LaunchDefaults() LaunchConfig {
	return LaunchConfig{
		ModelVRAMSafetyMargin:       modelVRAMSafetyMargin,
		DefaultGPUMemoryUtilization: defaultGPUMemoryUtilization,
		DefaultMaxModelLen:          defaultMaxModelLen,
		VLLMImage:                   VLLMImage,
		KVCacheDTypeOptions:         append([]string(nil), kvCacheDTypeOptions...),
		ToolParserOptions:           allowedToolParserNames(),
		TokenizerModeOptions:        append([]string(nil), tokenizerModeOptions...),
		ReasoningParserOptions:      allowedReasoningParserNames(),
		BlockSizeOptions:            append([]int(nil), blockSizeOptions...),
		MoeBackendOptions:           append([]string(nil), moeBackendOptions...),
	}
}

// kvCacheEstimateCache memoizes parsed config.json estimates keyed by config
// path. Entries are invalidated when the file's mtime changes (e.g. the model is
// re-downloaded), so a poll of /api/models/downloaded no longer re-reads and
// re-decodes every model config on every request. Uses LRU-style eviction when
// the cache exceeds maxEntries to prevent unbounded growth.
type kvCacheEstimateEntry struct {
	estimate   modelKVCacheEstimate
	modTime    time.Time
	accessTime time.Time
}

const kvCacheEstimateMaxEntries = 500

var (
	kvCacheEstimateCache   = make(map[string]kvCacheEstimateEntry)
	kvCacheEstimateCacheMu sync.RWMutex
)

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
	value, ok := checkedMulUint64(
		2,
		uint64(estimate.Layers),
		uint64(estimate.KeyValueHeads),
		uint64(estimate.HeadDim),
		uint64(estimate.DTypeBytes),
	)
	if !ok {
		return 0
	}
	return value
}

func (estimate modelKVCacheEstimate) bytesPerTokenPerGPU(tensorParallelSize int) (uint64, error) {
	if !estimate.isComplete() || tensorParallelSize <= 0 {
		return 0, nil
	}

	kvHeadsPerGPU := ceilDiv(estimate.KeyValueHeads, tensorParallelSize)
	if kvHeadsPerGPU < 1 {
		kvHeadsPerGPU = 1
	}
	value, ok := checkedMulUint64(
		2,
		uint64(estimate.Layers),
		uint64(kvHeadsPerGPU),
		uint64(estimate.HeadDim),
		uint64(estimate.DTypeBytes),
	)
	if !ok {
		return 0, validationError("KV cache size estimate overflows uint64 before applying max_model_len; increase tensor_parallel_size or use a smaller model")
	}
	return value, nil
}

func checkedMulUint64(values ...uint64) (uint64, bool) {
	result := uint64(1)
	maxUint64 := ^uint64(0)
	for _, value := range values {
		if value != 0 && result > maxUint64/value {
			return 0, false
		}
		result *= value
	}
	return result, true
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

	// Fallback: walk the cache tree, but only descend into snapshots/ and
	// other known subdirectories.  Large model caches (100 GB+) can take
	// seconds to walk fully; limiting depth to the snapshot level keeps this
	// under ~100 ms even for huge models.
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
			// Only walk one level deep for non-snapshot configs to avoid
			// scanning deep nested directories in large model caches.
			depth := strings.Count(rel, string(filepath.Separator))
			if depth <= 2 {
				if info, err := entry.Info(); err == nil {
					modTime = info.ModTime()
				}
				otherCandidates = append(otherCandidates, configCandidate{path, modTime})
			}
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

func GetModelKVCacheEstimate(hfCache string, model string) (modelKVCacheEstimate, error) {
	configPath, err := findModelConfigPath(hfCache, model)
	if err != nil {
		return modelKVCacheEstimate{}, err
	}

	var modTime time.Time
	if info, statErr := os.Stat(configPath); statErr == nil {
		modTime = info.ModTime()
		kvCacheEstimateCacheMu.RLock()
		cached, ok := kvCacheEstimateCache[configPath]
		kvCacheEstimateCacheMu.RUnlock()
		if ok && cached.modTime.Equal(modTime) {
			kvCacheEstimateCacheMu.Lock()
			if entry, found := kvCacheEstimateCache[configPath]; found {
				entry.accessTime = time.Now()
				kvCacheEstimateCache[configPath] = entry
			}
			kvCacheEstimateCacheMu.Unlock()
			return cached.estimate, nil
		}
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
	estimate := modelKVCacheEstimate{DefaultMaxModelLen: maxModelLen}

	layers, ok := getConfigInt(config, root, "num_hidden_layers", "n_layer", "num_layers", "num_layers_total")
	if !ok {
		return estimate, validationError("model config does not include num_hidden_layers")
	}

	attentionHeads, ok := getConfigInt(config, root, "num_attention_heads", "n_head", "num_heads")
	if !ok {
		return estimate, validationError("model config does not include num_attention_heads")
	}

	hiddenSize, ok := getConfigInt(config, root, "hidden_size", "n_embd", "d_model")
	if !ok {
		return estimate, validationError("model config does not include hidden_size")
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
		return estimate, validationError("model config does not include a usable attention head dimension")
	}

	dtype := ""
	if value, ok := getConfigString(config, root, "torch_dtype", "dtype"); ok {
		dtype = value
	}

	estimate.Layers = layers
	estimate.KeyValueHeads = keyValueHeads
	estimate.HeadDim = headDim
	estimate.DTypeBytes = kvElementBytes(dtype)

	if !modTime.IsZero() {
		kvCacheEstimateCacheMu.Lock()
		evictKVCacheIfNeeded()
		kvCacheEstimateCache[configPath] = kvCacheEstimateEntry{estimate: estimate, modTime: modTime, accessTime: time.Now()}
		kvCacheEstimateCacheMu.Unlock()
	}

	return estimate, nil
}

func evictKVCacheIfNeeded() {
	for len(kvCacheEstimateCache) >= kvCacheEstimateMaxEntries {
		var oldest string
		oldestTime := time.Now()
		for path, entry := range kvCacheEstimateCache {
			if entry.accessTime.Before(oldestTime) {
				oldestTime = entry.accessTime
				oldest = path
			}
		}
		if oldest == "" {
			return
		}
		delete(kvCacheEstimateCache, oldest)
	}
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

func parseMaxNumSeqs(maxNumSeqs string) (int, error) {
	maxNumSeqs = strings.TrimSpace(maxNumSeqs)
	if maxNumSeqs == "" {
		return 0, nil
	}

	maxNumSeqsNum, err := strconv.Atoi(maxNumSeqs)
	if err != nil || maxNumSeqsNum <= 0 {
		return 0, validationError("max num seqs must be a positive integer")
	}
	return maxNumSeqsNum, nil
}

func normalizeKVCacheDType(kvCacheDType string) (string, error) {
	kvCacheDType = strings.ToLower(strings.TrimSpace(kvCacheDType))
	if kvCacheDType == "" {
		return "auto", nil
	}

	for _, option := range kvCacheDTypeOptions {
		if kvCacheDType == option {
			return kvCacheDType, nil
		}
	}
	return "", validationError("kv_cache_dtype must be one of %s", strings.Join(kvCacheDTypeOptions, ", "))
}

func kvCacheDTypeElementBytes(kvCacheDType string, modelDTypeBytes int) int {
	if kvCacheDType == "auto" {
		return modelDTypeBytes
	}
	return 1
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
// active is the caller's snapshot of running models; passing it in avoids a
// redundant docker enumeration (and the readiness probes) on the launch path.
func checkGPUOverlap(active []ActiveModel, requestedIDs []int, totalGPUs int, requestedMemUtil float64) error {
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

func CheckModelFitsInVRAM(model string, hfCache string, selectedGPUs []GPUInfo, gpus int, memUtil string, maxLen string, kvCacheDType string) error {
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

	kvCacheDType, err = normalizeKVCacheDType(kvCacheDType)
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
	if kvEstimateOK {
		estimate.DTypeBytes = kvCacheDTypeElementBytes(kvCacheDType, estimate.DTypeBytes)
	}

	// Reject max-model-len that exceeds the model's rope/positional limit.
	// This only needs DefaultMaxModelLen, so enforce it even when other KV
	// metadata is incomplete and KV memory estimation has to be skipped.
	if requestedMaxLen > 0 && estimate.DefaultMaxModelLen > 0 && requestedMaxLen > estimate.DefaultMaxModelLen {
		return validationError(
			"requested max_model_len %d exceeds the selected model's default maximum context length %d; reduce max_model_len before launching",
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
			kvBytesPerTokenPerGPU, err = estimate.bytesPerTokenPerGPU(gpus)
			if err != nil {
				return err
			}
			var ok bool
			kvCacheBytesPerGPU, ok = checkedMulUint64(kvBytesPerTokenPerGPU, uint64(effectiveMaxLen))
			if !ok {
				return validationError("KV cache size estimate overflows uint64 for max_model_len=%d; reduce max_model_len, increase tensor_parallel_size, or use a smaller model", effectiveMaxLen)
			}
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
				FormatBytes(modelBytesPerGPU),
				gpu.ID,
				FormatBytes(gpu.MemoryFree),
			)
		}

		// Total (weights + KV) must fit within the utilization budget
		requiredPerGPU := modelBytesPerGPU + kvCacheBytesPerGPU
		if requiredPerGPU < modelBytesPerGPU {
			return validationError("combined model weights plus KV cache estimate overflows uint64; reduce max_model_len, increase tensor_parallel_size, or use a smaller model")
		}
		if requiredPerGPU > gpu.MemoryFree {
			return validationError(
				"model weights ~%s plus KV cache ~%s for max_model_len=%d needs ~%s per GPU, but GPU %d only has %s currently free — stop other workloads first",
				FormatBytes(modelBytesPerGPU),
				FormatBytes(kvCacheBytesPerGPU),
				effectiveMaxLen,
				FormatBytes(requiredPerGPU),
				gpu.ID,
				FormatBytes(gpu.MemoryFree),
			)
		}
		if requiredPerGPU > utilizationBudget {
			if kvCacheBytesPerGPU > 0 {
				return validationError(
					"model weights ~%s plus KV cache ~%s for max_model_len=%d needs ~%s per GPU, but the %.2f utilization budget on GPU %d is only %s — reduce max_model_len or increase mem_util",
					FormatBytes(modelBytesPerGPU),
					FormatBytes(kvCacheBytesPerGPU),
					effectiveMaxLen,
					FormatBytes(requiredPerGPU),
					utilization,
					gpu.ID,
					FormatBytes(utilizationBudget),
				)
			}
			return validationError(
				"model weights ~%s per GPU (with %.0f%% safety margin) exceeds the %.2f utilization budget of %s on GPU %d",
				FormatBytes(modelBytesPerGPU),
				(modelVRAMSafetyMargin-1)*100,
				utilization,
				FormatBytes(utilizationBudget),
				gpu.ID,
			)
		}

		// Fix #4: ensure minimum viable KV cache remains after weights
		remainingForKV := utilizationBudget - modelBytesPerGPU
		if remainingForKV < minKVCacheBytesPerGPU {
			return validationError(
				"only ~%s remains for KV cache on GPU %d after loading model weights (budget: %s, weights: ~%s) — vLLM requires at least %s; increase mem_util or use fewer GPUs",
				FormatBytes(remainingForKV),
				gpu.ID,
				FormatBytes(utilizationBudget),
				FormatBytes(modelBytesPerGPU),
				FormatBytes(minKVCacheBytesPerGPU),
			)
		}
	}

	if kvCacheBytesPerGPU > 0 {
		log.Printf("[VRAM] %s: ~%s/GPU total (%s weights + %s KV cache at max_model_len=%d, kv_cache_dtype=%s, %s/token/GPU)",
			model,
			FormatBytes(modelBytesPerGPU+kvCacheBytesPerGPU),
			FormatBytes(modelBytesPerGPU),
			FormatBytes(kvCacheBytesPerGPU),
			effectiveMaxLen,
			kvCacheDType,
			FormatBytes(kvBytesPerTokenPerGPU),
		)
	}
	return nil
}
