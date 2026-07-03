package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type HFModelInfo struct {
	Model     string `json:"model"`
	SizeBytes int64  `json:"size_bytes"`
	Downloads int    `json:"downloads"`
	Likes     int    `json:"likes"`
}

// fetchHFModelInfoForTest is an optional indirection that lets unit tests
// stub out the real network call to huggingface.co. Production callers
// always go through fetchHFModelInfo, which dispatches via this variable
// when set.
var fetchHFModelInfoForTest func(model string) (HFModelInfo, error)

func fetchHFModelInfo(model string) (HFModelInfo, error) {
	if fetchHFModelInfoForTest != nil {
		return fetchHFModelInfoForTest(model)
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return HFModelInfo{}, newAppError(http.StatusBadRequest, "model required")
	}

	client := http.Client{Timeout: 5 * time.Second}
	reqHF, _ := http.NewRequest("GET", "https://huggingface.co/api/models/"+model, nil)
	settings := getSettings()
	if settings.HFToken != "" {
		reqHF.Header.Set("Authorization", "Bearer "+settings.HFToken)
	}

	resp, err := client.Do(reqHF)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return HFModelInfo{}, newAppError(http.StatusNotFound, "model not found or Hugging Face API error")
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
	}, nil
}
