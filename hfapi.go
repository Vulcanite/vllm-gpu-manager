package main

import (
	"net/http"
	"strings"

	"vllm-manager/api"
)

type HFModelInfo = api.HFModelInfo

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
	info, err := api.FetchHFModelInfo(model, getSettings().HFToken)
	if err != nil {
		return HFModelInfo{}, newAppError(http.StatusNotFound, "model not found or Hugging Face API error")
	}
	return info, nil
}
