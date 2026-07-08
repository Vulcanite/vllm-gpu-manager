package api

import (
	"embed"
	"encoding/json"
	"math"
	"sort"
	"strconv"
	"strings"
)

//go:embed data/recommend_models.json
var recommendCatalogFS embed.FS

const (
	recommendDefaultMemUtil = 0.9
	recommendOverheadGB     = 1.0
)

type RecommendationOptions struct {
	UseCase string
	FitOnly bool
	Limit   int
}

type RecommendedModel struct {
	Name               string   `json:"name"`
	ParameterCount     string   `json:"parameter_count"`
	Quantization       string   `json:"quantization"`
	UseCases           []string `json:"use_cases"`
	ContextLength      int      `json:"context_length"`
	RecommendedGPUs    int      `json:"recommended_gpus"`
	RecommendedGPUIDs  string   `json:"recommended_gpu_ids"`
	RecommendedMemUtil string   `json:"recommended_mem_util"`
	RecommendedMaxLen  int      `json:"recommended_max_len"`
	RecommendedKVDType string   `json:"recommended_kv_cache_dtype"`
	RequiredGBPerGPU   float64  `json:"required_gb_per_gpu"`
	AvailableGBPerGPU  float64  `json:"available_gb_per_gpu"`
	FitLevel           string   `json:"fit_level"`
	Score              float64  `json:"score"`
	ModelQuality       float64  `json:"model_quality"`
	EstimatedTokensSec float64  `json:"estimated_tokens_sec"`
	Reason             string   `json:"reason"`
}

type RecommendationResponse struct {
	UseCase string             `json:"use_case"`
	GPUs    []GPUInfo          `json:"gpus"`
	Models  []RecommendedModel `json:"models"`
}

type recommendCatalogModel struct {
	Name             string   `json:"name"`
	ParameterB       float64  `json:"parameter_b"`
	ActiveParameterB float64  `json:"active_parameter_b"`
	Quantization     string   `json:"quantization"`
	ContextLength    int      `json:"context_length"`
	UseCases         []string `json:"use_cases"`
	Quality          float64  `json:"quality"`
}

type recommendGPUGroup struct {
	Name        string
	GPUs        []GPUInfo
	TotalVRAMGB float64
	MinTotalGB  float64
	GPUIDs      []int
}

func RecommendModels(opts RecommendationOptions) (RecommendationResponse, error) {
	useCase := strings.ToLower(strings.TrimSpace(opts.UseCase))
	if useCase == "" {
		useCase = "general"
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	if limit > 200 {
		limit = 200
	}

	gpus, err := getGPUInfo()
	if err != nil {
		return RecommendationResponse{}, err
	}
	catalog, err := loadRecommendationCatalog()
	if err != nil {
		return RecommendationResponse{}, err
	}

	groups := recommendationGroups(gpus)
	var recommendations []RecommendedModel
	for _, model := range catalog {
		rec := recommendModelForHardware(model, groups, useCase)
		if rec.Name == "" {
			continue
		}
		if opts.FitOnly && rec.FitLevel == "too_tight" {
			continue
		}
		recommendations = append(recommendations, rec)
	}

	sort.SliceStable(recommendations, func(i, j int) bool {
		if recommendations[i].Score == recommendations[j].Score {
			return recommendations[i].RequiredGBPerGPU < recommendations[j].RequiredGBPerGPU
		}
		return recommendations[i].Score > recommendations[j].Score
	})
	if len(recommendations) > limit {
		recommendations = recommendations[:limit]
	}

	return RecommendationResponse{UseCase: useCase, GPUs: gpus, Models: recommendations}, nil
}

func loadRecommendationCatalog() ([]recommendCatalogModel, error) {
	data, err := recommendCatalogFS.ReadFile("data/recommend_models.json")
	if err != nil {
		return nil, err
	}
	var models []recommendCatalogModel
	if err := json.Unmarshal(data, &models); err != nil {
		return nil, err
	}
	return models, nil
}

func recommendationGroups(gpus []GPUInfo) []recommendGPUGroup {
	if len(gpus) == 0 {
		return nil
	}
	sorted := append([]GPUInfo(nil), gpus...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Name == sorted[j].Name {
			return sorted[i].ID < sorted[j].ID
		}
		return sorted[i].Name < sorted[j].Name
	})

	byKey := make(map[string]*recommendGPUGroup)
	var keys []string
	for _, gpu := range sorted {
		key := gpu.Name + "\x00" + FormatBytes(gpu.MemoryTotal)
		group := byKey[key]
		if group == nil {
			// Recommendation budgets use total VRAM (modulated by mem_util)
			// rather than currently-free VRAM so that foreign processes
			// holding memory on the host do not push every model into
			// "too_tight". The actual launch still gates against free
			// memory in CheckModelFitsInVRAM; this only affects scoring.
			group = &recommendGPUGroup{Name: gpu.Name, MinTotalGB: bytesToGB(gpu.MemoryTotal)}
			byKey[key] = group
			keys = append(keys, key)
		}
		group.GPUs = append(group.GPUs, gpu)
		group.TotalVRAMGB += bytesToGB(gpu.MemoryTotal)
		group.GPUIDs = append(group.GPUIDs, gpu.ID)
		group.MinTotalGB = math.Min(group.MinTotalGB, bytesToGB(gpu.MemoryTotal))
	}

	groups := make([]recommendGPUGroup, 0, len(keys))
	for _, key := range keys {
		group := byKey[key]
		sort.Ints(group.GPUIDs)
		groups = append(groups, *group)
	}
	sort.SliceStable(groups, func(i, j int) bool {
		return groups[i].TotalVRAMGB > groups[j].TotalVRAMGB
	})
	return groups
}

func recommendModelForHardware(model recommendCatalogModel, groups []recommendGPUGroup, useCase string) RecommendedModel {
	if len(groups) == 0 {
		return tooTightRecommendation(model, useCase, "No NVIDIA GPUs detected.")
	}

	best := RecommendedModel{}
	for _, group := range groups {
		for count := 1; count <= len(group.GPUs); count++ {
			rec := scoreRecommendation(model, group, count, useCase)
			if rec.Score > best.Score || best.Name == "" {
				best = rec
			}
		}
	}
	return best
}

func scoreRecommendation(model recommendCatalogModel, group recommendGPUGroup, gpuCount int, useCase string) RecommendedModel {
	ctx := model.ContextLength
	if ctx <= 0 {
		ctx = 4096
	}
	maxLen := ctx
	if maxLen > 32768 {
		maxLen = 32768
	}
	activeB := model.ActiveParameterB
	if activeB <= 0 {
		activeB = model.ParameterB
	}
	weightGB := model.ParameterB * quantBytesPerParam(model.Quantization) * modelVRAMSafetyMargin
	kvGB := activeB * 0.000008 * float64(maxLen)
	requiredPerGPU := (weightGB+kvGB)/float64(gpuCount) + recommendOverheadGB
	budgetPerGPU := group.MinTotalGB * recommendDefaultMemUtil
	fitLevel, fitScore := recommendationFit(requiredPerGPU, budgetPerGPU)
	useScore := recommendationUseCaseScore(model, useCase)
	quality := clamp(model.Quality, 0, 100)
	speed := estimateRecommendationSpeed(model, gpuCount, group)
	speedScore := clamp((speed/40.0)*100, 0, 100)
	score := quality*0.45 + fitScore*0.30 + speedScore*0.15 + useScore*0.10
	if fitLevel == "too_tight" {
		score *= 0.25
	}

	ids := group.GPUIDs
	if gpuCount < len(ids) {
		ids = ids[:gpuCount]
	}
	return RecommendedModel{
		Name:               model.Name,
		ParameterCount:     formatParams(model.ParameterB),
		Quantization:       model.Quantization,
		UseCases:           append([]string(nil), model.UseCases...),
		ContextLength:      ctx,
		RecommendedGPUs:    gpuCount,
		RecommendedGPUIDs:  joinInts(ids),
		RecommendedMemUtil: "0.9",
		RecommendedMaxLen:  maxLen,
		RecommendedKVDType: recommendKVDType(model.Quantization),
		RequiredGBPerGPU:   round1(requiredPerGPU),
		AvailableGBPerGPU:  round1(budgetPerGPU),
		FitLevel:           fitLevel,
		Score:              round1(score),
		ModelQuality:       quality,
		EstimatedTokensSec: round1(speed),
		Reason:             recommendationReason(fitLevel, gpuCount, ids, requiredPerGPU, budgetPerGPU),
	}
}

func tooTightRecommendation(model recommendCatalogModel, useCase string, reason string) RecommendedModel {
	return RecommendedModel{
		Name:               model.Name,
		ParameterCount:     formatParams(model.ParameterB),
		Quantization:       model.Quantization,
		UseCases:           append([]string(nil), model.UseCases...),
		ContextLength:      model.ContextLength,
		RecommendedGPUs:    0,
		RecommendedMemUtil: "0.9",
		RecommendedKVDType: recommendKVDType(model.Quantization),
		FitLevel:           "too_tight",
		Score:              round1(recommendationUseCaseScore(model, useCase) * 0.05),
		ModelQuality:       clamp(model.Quality, 0, 100),
		Reason:             reason,
	}
}

func recommendationFit(required float64, available float64) (string, float64) {
	if available <= 0 || required > available {
		return "too_tight", 0
	}
	ratio := required / available
	if ratio <= 0.65 {
		return "perfect", 100
	}
	if ratio <= 0.80 {
		return "good", 85
	}
	return "marginal", 60
}

func recommendationUseCaseScore(model recommendCatalogModel, useCase string) float64 {
	if useCase == "general" || useCase == "" {
		return 80
	}
	for _, candidate := range model.UseCases {
		if strings.EqualFold(candidate, useCase) {
			return 100
		}
	}
	return 45
}

func quantBytesPerParam(quant string) float64 {
	switch strings.ToUpper(strings.TrimSpace(quant)) {
	case "BF16", "F16", "FP16":
		return 2.0
	case "FP8", "INT8", "AWQ-8BIT", "GPTQ-INT8":
		return 1.0
	case "FP4", "NVFP4", "MXFP4", "NF4", "INT4", "AWQ-4BIT", "GPTQ-INT4":
		return 0.5
	default:
		return 2.0
	}
}

func recommendKVDType(quant string) string {
	if strings.Contains(strings.ToLower(quant), "fp8") {
		return "fp8"
	}
	return "auto"
}

func estimateRecommendationSpeed(model recommendCatalogModel, gpuCount int, group recommendGPUGroup) float64 {
	params := model.ActiveParameterB
	if params <= 0 {
		params = model.ParameterB
	}
	if params <= 0 {
		return 0
	}
	bandwidth := recommendationBandwidth(group.Name)
	if bandwidth <= 0 {
		bandwidth = 500
	}
	return (bandwidth * float64(gpuCount) / (params * quantBytesPerParam(model.Quantization))) * 0.45
}

func recommendationBandwidth(name string) float64 {
	lower := strings.ToLower(name)
	known := []struct {
		key       string
		bandwidth float64
	}{
		{"h200", 4800}, {"h100", 2039}, {"a100", 1555}, {"l40s", 864}, {"l4", 300},
		{"5090", 1792}, {"4090", 1008}, {"4080", 717}, {"3090", 936}, {"3080", 760},
		{"4070", 504}, {"4060", 272}, {"3060", 360}, {"t4", 320},
	}
	for _, item := range known {
		if strings.Contains(lower, item.key) {
			return item.bandwidth
		}
	}
	return 0
}

func recommendationReason(fitLevel string, gpuCount int, gpuIDs []int, required float64, available float64) string {
	if fitLevel == "too_tight" {
		return "Needs " + formatGB(required) + "/GPU, but only " + formatGB(available) + "/GPU is available."
	}
	return "Use " + formatGPUCount(gpuCount) + " on GPU IDs " + joinInts(gpuIDs) + "; needs about " + formatGB(required) + "/GPU."
}

func formatParams(value float64) string {
	return formatFloat(value) + "B"
}

func formatGPUCount(value int) string {
	if value == 1 {
		return "1 GPU"
	}
	return strconv.Itoa(value) + " GPUs"
}

func formatGB(value float64) string {
	return formatFloat(round1(value)) + " GiB"
}

func bytesToGB(value uint64) float64 {
	return float64(value) / (1024 * 1024 * 1024)
}

func round1(value float64) float64 {
	return math.Round(value*10) / 10
}

func clamp(value float64, min float64, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func joinInts(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}

func formatFloat(value float64) string {
	formatted := strconv.FormatFloat(value, 'f', -1, 64)
	if !strings.Contains(formatted, ".") {
		return formatted
	}
	return strings.TrimRight(strings.TrimRight(formatted, "0"), ".")
}
