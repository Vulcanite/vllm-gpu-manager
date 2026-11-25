package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"
)

type HFModelInfo struct {
	Model                 string   `json:"model"`
	SizeBytes             int64    `json:"size_bytes"`
	Downloads             int      `json:"downloads"`
	Likes                 int      `json:"likes"`
	Tags                  []string `json:"tags,omitempty"`
	Quantization          string   `json:"quantization,omitempty"`
	CompatibilityWarning  string   `json:"compatibility_warning,omitempty"`
	DownloadWarning       string   `json:"download_warning,omitempty"`
	DownloadBlockedReason string   `json:"download_blocked_reason,omitempty"`
	// DownloadExcludes are the --exclude patterns the download will use;
	// SizeBytes already leaves those files out.
	DownloadExcludes []string `json:"-"`
}

// Hugging Face repo ids: optional "org/" then a name, ASCII letters, digits,
// "_", "-" and ".", with no "--" or ".." (the Hub rejects both).
var modelIDPattern = regexp.MustCompile(`^(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z0-9][A-Za-z0-9_.-]*$`)

// ValidateModelID rejects anything that is not a Hugging Face repo id before it
// reaches a URL, a cache path or the download container.
func ValidateModelID(model string) error {
	if len(model) > 200 || !modelIDPattern.MatchString(model) || strings.Contains(model, "--") || strings.Contains(model, "..") {
		return validationError("invalid model id %q: expected a Hugging Face repo id like org/model-name", model)
	}
	return nil
}

// unusedWeightPatterns are weight formats vLLM never loads when a repo also
// ships safetensors. They go to `hf download --exclude` (fnmatch rules) and to
// excluded below, so every pattern must be "*.ext" or "dir/*".
var unusedWeightPatterns = []string{
	"*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.ot", "*.gguf",
	"*.onnx", "*.onnx_data", "*.tflite", "*.mlmodel", "onnx/*", "coreml/*", "openvino/*",
}

func downloadExcludes(files []string) []string {
	for _, file := range files {
		if strings.HasSuffix(file, ".safetensors") {
			return unusedWeightPatterns
		}
	}
	return nil
}

// excluded matches the two pattern shapes the way fnmatch does: "*.ext" is a
// suffix anywhere in the path ("*" crosses "/"), "dir/*" is a path prefix.
func excluded(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if ext, ok := strings.CutPrefix(pattern, "*"); ok && strings.HasSuffix(name, ext) {
			return true
		}
		if dir, ok := strings.CutSuffix(pattern, "*"); ok && strings.HasPrefix(name, dir) {
			return true
		}
	}
	return false
}

func FetchHFModelInfo(model string, hfToken string) (HFModelInfo, error) {
	model = strings.TrimSpace(model)
	if err := ValidateModelID(model); err != nil {
		return HFModelInfo{}, err
	}

	client := http.Client{Timeout: 5 * time.Second}
	// blobs=true adds per-file sizes; usedStorage counts old file versions too
	// and overstates the download several times over.
	reqHF, err := http.NewRequest(http.MethodGet, "https://huggingface.co/api/models/"+model+"?blobs=true", nil)
	if err != nil {
		return HFModelInfo{}, validationError("invalid model id %q", model)
	}
	if strings.TrimSpace(hfToken) != "" {
		reqHF.Header.Set("Authorization", "Bearer "+strings.TrimSpace(hfToken))
	}

	resp, err := client.Do(reqHF)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return HFModelInfo{}, fmt.Errorf("model not found or Hugging Face API error")
	}
	defer resp.Body.Close()

	var hfResp struct {
		Downloads int      `json:"downloads"`
		Likes     int      `json:"likes"`
		Tags      []string `json:"tags"`
		Siblings  []struct {
			RFilename string `json:"rfilename"`
			Size      int64  `json:"size"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&hfResp); err != nil {
		return HFModelInfo{}, fmt.Errorf("failed to parse Hugging Face model metadata: %v", err)
	}

	files := make([]string, 0, len(hfResp.Siblings))
	for _, sibling := range hfResp.Siblings {
		files = append(files, sibling.RFilename)
	}
	excludes := downloadExcludes(files)
	var size int64
	for _, sibling := range hfResp.Siblings {
		if !excluded(sibling.RFilename, excludes) {
			size += sibling.Size
		}
	}

	return HFModelInfo{
		Model:            model,
		SizeBytes:        size,
		Downloads:        hfResp.Downloads,
		Likes:            hfResp.Likes,
		Tags:             hfResp.Tags,
		DownloadExcludes: excludes,
	}, nil
}

// HFInfoSource returns Hugging Face metadata for one model. Callers share one
// source across checks so a request makes at most one Hub call.
type HFInfoSource func() (HFModelInfo, error)

// LazyHFInfo fetches on first use and reuses the result.
func LazyHFInfo(model string, hfToken string) HFInfoSource {
	return sync.OnceValues(func() (HFModelInfo, error) {
		return FetchHFModelInfo(model, hfToken)
	})
}

// KnownHFInfo wraps a result the caller already fetched.
func KnownHFInfo(info HFModelInfo, err error) HFInfoSource {
	return func() (HFModelInfo, error) { return info, err }
}

func CheckDownloadPreflight(model string, hfCache string, hfInfo HFInfoSource) (string, error) {
	if isGGUFModelHint(model) {
		return "", validationError("GGUF models are not supported by vllm-manager downloads; vLLM GGUF support is experimental and requires a separate plugin")
	}
	info, err := hfInfo()
	if err != nil || info.SizeBytes <= 0 {
		return "Model size could not be confirmed before download; disk space was not preflighted.", nil
	}
	if isGGUFModelInfo(info) {
		return "", validationError("GGUF models are not supported by vllm-manager downloads; vLLM GGUF support is experimental and requires a separate plugin")
	}
	free, err := FreeBytesForPath(NormalizeHFHome(hfCache))
	if err != nil {
		return "", fmt.Errorf("failed to check free space for Hugging Face cache: %v", err)
	}
	_, cachedBytes, _ := IsModelDownloaded(model, hfCache)
	required := uint64(info.SizeBytes)
	if cachedBytes >= required {
		return "", nil
	}
	required -= cachedBytes
	required = uint64(float64(required) * 1.10)
	if required > free {
		return "", validationError("not enough free disk space to download %s: need about %s, free %s", model, FormatBytes(required), FormatBytes(free))
	}
	return "", nil
}

func FreeBytesForPath(path string) (uint64, error) {
	if strings.TrimSpace(path) == "" {
		return 0, validationError("path is required")
	}
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, err
	}
	return stat.Bavail * uint64(stat.Bsize), nil
}

func CheckModelQuantizationCompatibility(model string, hfCache string, hfInfo HFInfoSource, selectedGPUs []GPUInfo) (string, error) {
	quant, warning := DetectModelQuantization(model, hfCache, hfInfo)
	if quant == "" {
		return warning, nil
	}
	if strings.Contains(strings.ToLower(quant), "fp8") {
		for _, gpu := range selectedGPUs {
			if gpu.ComputeCapabilityAtLeast(8, 9) {
				return warning, nil
			}
		}
		return warning, validationError("model %s appears to use FP8 quantization, which requires an NVIDIA GPU with compute capability 8.9 or newer", model)
	}
	return warning, nil
}

func DetectModelQuantization(model string, hfCache string, hfInfo HFInfoSource) (string, string) {
	if quant := detectLocalQuantization(model, hfCache); quant != "" {
		return quant, ""
	}
	if info, err := hfInfo(); err == nil {
		if quant := quantizationFromStrings(info.Tags...); quant != "" {
			return quant, ""
		}
	} else {
		return quantizationFromStrings(model), "Model quantization metadata could not be confirmed from Hugging Face."
	}
	return quantizationFromStrings(model), ""
}

func detectLocalQuantization(model string, hfCache string) string {
	configPath, err := findModelConfigPath(hfCache, model)
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	var root map[string]interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return ""
	}
	for _, key := range []string{"quantization", "quant_method", "quantization_method", "load_format"} {
		if value, ok := getConfigString(root, root, key); ok {
			if quant := quantizationFromStrings(value); quant != "" {
				return quant
			}
		}
	}
	if config, ok := getConfigObject(root, "quantization_config"); ok {
		for _, key := range []string{"quant_method", "bits", "fmt"} {
			if value, ok := getConfigString(config, root, key); ok {
				if quant := quantizationFromStrings(value); quant != "" {
					return quant
				}
			}
		}
	}
	return ""
}

func quantizationFromStrings(values ...string) string {
	for _, value := range values {
		lower := strings.ToLower(value)
		switch {
		case strings.Contains(lower, "gguf"):
			return "GGUF"
		case strings.Contains(lower, "fp8") || strings.Contains(lower, "float8"):
			return "FP8"
		case strings.Contains(lower, "awq"):
			return "AWQ"
		case strings.Contains(lower, "gptq"):
			return "GPTQ"
		case strings.Contains(lower, "bitsandbytes") || strings.Contains(lower, "bnb"):
			return "BITSANDBYTES"
		}
	}
	return ""
}

func isGGUFModelInfo(info HFModelInfo) bool {
	values := append([]string{info.Model}, info.Tags...)
	return isGGUFModelHint(values...)
}

func isGGUFModelHint(values ...string) bool {
	return quantizationFromStrings(values...) == "GGUF"
}
