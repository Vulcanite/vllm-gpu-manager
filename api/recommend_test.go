package api

import "testing"

func TestRecommendModelsRanksFittingModels(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		// MemoryFree is intentionally set to 0 to simulate a host with
		// other workloads holding VRAM. Recommendations should still
		// surface because the budget is derived from total VRAM, not
		// free VRAM. The launch path still gates against MemoryFree in
		// CheckModelFitsInVRAM.
		return []GPUInfo{{
			ID:          0,
			Name:        "NVIDIA RTX 4090",
			MemoryTotal: 24 * 1024 * 1024 * 1024,
			MemoryFree:  0,
		}}, nil
	})

	result, err := RecommendModels(RecommendationOptions{UseCase: "coding", FitOnly: true, Limit: 5})
	if err != nil {
		t.Fatalf("RecommendModels() error=%v", err)
	}
	if result.UseCase != "coding" {
		t.Fatalf("UseCase=%q, want coding", result.UseCase)
	}
	if len(result.Models) == 0 {
		t.Fatal("expected at least one fitting recommendation")
	}
	for _, model := range result.Models {
		if model.FitLevel == "too_tight" {
			t.Fatalf("fit-only result included too_tight model: %+v", model)
		}
		if model.RecommendedGPUs != 1 || model.RecommendedGPUIDs != "0" {
			t.Fatalf("recommendation GPU target=(%d,%q), want 1 GPU 0", model.RecommendedGPUs, model.RecommendedGPUIDs)
		}
	}
}

func TestRecommendationCatalogIsValid(t *testing.T) {
	catalog, err := loadRecommendationCatalog()
	if err != nil {
		t.Fatalf("loadRecommendationCatalog() error=%v", err)
	}
	if len(catalog) < 50 {
		t.Fatalf("catalog has %d models, want at least 50", len(catalog))
	}

	seen := make(map[string]bool, len(catalog))
	validUseCases := map[string]bool{
		"general":   true,
		"coding":    true,
		"reasoning": true,
		"chat":      true,
		"vision":    true,
	}
	for _, model := range catalog {
		if model.Name == "" {
			t.Fatal("catalog includes a model with an empty name")
		}
		if seen[model.Name] {
			t.Fatalf("catalog includes duplicate model %q", model.Name)
		}
		seen[model.Name] = true

		if model.ParameterB <= 0 {
			t.Fatalf("%s parameter_b=%v, want positive", model.Name, model.ParameterB)
		}
		if model.ActiveParameterB < 0 || model.ActiveParameterB > model.ParameterB {
			t.Fatalf("%s active_parameter_b=%v must be between 0 and parameter_b=%v", model.Name, model.ActiveParameterB, model.ParameterB)
		}
		if model.ContextLength <= 0 {
			t.Fatalf("%s context_length=%d, want positive", model.Name, model.ContextLength)
		}
		if model.Quality < 0 || model.Quality > 100 {
			t.Fatalf("%s quality=%v, want 0-100", model.Name, model.Quality)
		}
		if len(model.UseCases) == 0 {
			t.Fatalf("%s has no use_cases", model.Name)
		}
		for _, useCase := range model.UseCases {
			if !validUseCases[useCase] {
				t.Fatalf("%s has unknown use case %q", model.Name, useCase)
			}
		}
	}
}

func TestRecommendationFormattingPreservesZeroAndTrailingZeroes(t *testing.T) {
	if got := joinInts([]int{0, 10}); got != "0,10" {
		t.Fatalf("joinInts([0,10])=%q, want 0,10", got)
	}
	if got := formatParams(30); got != "30B" {
		t.Fatalf("formatParams(30)=%q, want 30B", got)
	}
}
