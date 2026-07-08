package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
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
}

var fetchHFModelInfoForTest func(model string, hfToken string) (HFModelInfo, error)

func FetchHFModelInfo(model string, hfToken string) (HFModelInfo, error) {
	if fetchHFModelInfoForTest != nil {
		return fetchHFModelInfoForTest(model, hfToken)
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return HFModelInfo{}, validationError("model required")
	}

	client := http.Client{Timeout: 5 * time.Second}
	reqHF, _ := http.NewRequest("GET", "https://huggingface.co/api/models/"+model, nil)
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
		UsedStorage int64    `json:"usedStorage"`
		Downloads   int      `json:"downloads"`
		Likes       int      `json:"likes"`
		Tags        []string `json:"tags"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&hfResp); err != nil {
		return HFModelInfo{}, fmt.Errorf("failed to parse Hugging Face model metadata: %v", err)
	}

	return HFModelInfo{
		Model:     model,
		SizeBytes: hfResp.UsedStorage,
		Downloads: hfResp.Downloads,
		Likes:     hfResp.Likes,
		Tags:      hfResp.Tags,
	}, nil
}

func CheckDownloadPreflight(model string, hfCache string, hfToken string) (string, error) {
	if isGGUFModelHint(model) {
		return "", validationError("GGUF models are not supported by vllm-manager downloads; vLLM GGUF support is experimental and requires a separate plugin")
	}
	info, err := FetchHFModelInfo(model, hfToken)
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

func CheckModelQuantizationCompatibility(model string, hfCache string, hfToken string, selectedGPUs []GPUInfo) (string, error) {
	quant, warning := DetectModelQuantization(model, hfCache, hfToken)
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

func DetectModelQuantization(model string, hfCache string, hfToken string) (string, string) {
	if quant := detectLocalQuantization(model, hfCache); quant != "" {
		return quant, ""
	}
	if info, err := FetchHFModelInfo(model, hfToken); err == nil {
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
