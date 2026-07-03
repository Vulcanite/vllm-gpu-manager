package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"vllm-manager/api"
)

const dashboardShutdownTimeout = 10 * time.Second

func newDashboardMux(proxyEnabled bool) (*http.ServeMux, dashboardAuth) {
	mux := http.NewServeMux()
	auth := newDashboardAuth()
	apiHandler := func(next http.HandlerFunc) http.HandlerFunc {
		return corsMiddleware(auth.require(next))
	}
	authHandler := func(next http.HandlerFunc) http.HandlerFunc {
		return corsMiddleware(next)
	}

	mux.HandleFunc("/api/auth/status", authHandler(auth.statusHandler))
	mux.HandleFunc("/api/auth/login", authHandler(auth.loginHandler))
	mux.HandleFunc("/api/auth/logout", authHandler(auth.logoutHandler))
	registerAPIRoutes(mux, apiHandler, proxyEnabled)
	registerStaticRoutes(mux)

	// Proxy routes
	if proxyEnabled {
		mux.HandleFunc("/v1/", inferenceProxyHandler())
	}

	return mux, auth
}

func validateRoutingTableForActiveModels(table map[string]int, models []api.ActiveModel) (map[string]int, error) {
	liveByPort := make(map[int]string, len(models))
	for _, model := range models {
		port, ok := parseActiveModelPort(model)
		if !ok {
			continue
		}
		modelName := strings.TrimSpace(model.Model)
		if modelName == "" {
			continue
		}
		if existing := liveByPort[port]; existing != "" {
			return nil, newAppError(http.StatusBadRequest, "multiple running managed models report port %d: %q and %q", port, existing, modelName)
		}
		liveByPort[port] = modelName
	}

	validated := make(map[string]int, len(table))
	seenModels := make(map[string]string, len(table))
	seenPorts := make(map[int]string, len(table))
	for rawModel, port := range table {
		modelName := strings.TrimSpace(rawModel)
		if api.NormalizeModelKey(modelName) == "" {
			return nil, newAppError(http.StatusBadRequest, "invalid model name %q", rawModel)
		}
		if !validRoutingPort(port) {
			return nil, newAppError(http.StatusBadRequest, "port for %q must be between 1 and 65535", modelName)
		}
		modelKey := api.NormalizeModelKeyForMatch(modelName)
		if existing := seenModels[modelKey]; existing != "" {
			return nil, newAppError(http.StatusBadRequest, "duplicate route for model %q and %q", existing, modelName)
		}
		if existing := seenPorts[port]; existing != "" {
			return nil, newAppError(http.StatusBadRequest, "duplicate route port %d for %q and %q", port, existing, modelName)
		}
		liveModel, ok := liveByPort[port]
		if !ok {
			return nil, newAppError(http.StatusBadRequest, "port %d is not a running managed vLLM container", port)
		}
		if api.NormalizeModelKeyForMatch(liveModel) != modelKey {
			return nil, newAppError(http.StatusBadRequest, "route %q on port %d does not match running managed model %q", modelName, port, liveModel)
		}
		seenModels[modelKey] = modelName
		seenPorts[port] = modelName
		validated[modelName] = port
	}
	return validated, nil
}

func registerAPIRoutes(mux *http.ServeMux, apiHandler func(http.HandlerFunc) http.HandlerFunc, proxyEnabled bool) {
	mux.HandleFunc("/api/settings", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeJSON(w, getSettingsStatusWithProxy(proxyEnabled))
			return
		}
		if r.Method == http.MethodPost {
			var req struct {
				HFCache                 string  `json:"hf_cache"`
				HFToken                 string  `json:"hf_token"`
				ClearHFToken            bool    `json:"clear_hf_token"`
				VLLMAPIKey              string  `json:"vllm_api_key"`
				ClearVLLMAPIKey         bool    `json:"clear_vllm_api_key"`
				InferenceAPIKey         string  `json:"inference_api_key"`
				ClearInferenceAPIKey    bool    `json:"clear_inference_api_key"`
				EnableCUDACompatibility *bool   `json:"enable_cuda_compatibility"`
				HFHubOffline            *bool   `json:"hf_hub_offline"`
				VLLMImage               *string `json:"vllm_image"`
			}
			if !decodeJSONBody(w, r, &req) {
				return
			}

			if _, err := updateSettings(SettingsUpdate{
				HFCache:                 req.HFCache,
				HFToken:                 req.HFToken,
				ClearHFToken:            req.ClearHFToken,
				VLLMAPIKey:              req.VLLMAPIKey,
				ClearVLLMAPIKey:         req.ClearVLLMAPIKey,
				InferenceAPIKey:         req.InferenceAPIKey,
				ClearInferenceAPIKey:    req.ClearInferenceAPIKey,
				EnableCUDACompatibility: req.EnableCUDACompatibility,
				HFHubOffline:            req.HFHubOffline,
				VLLMImage:               req.VLLMImage,
			}); err != nil {
				writeAPIError(w, err)
				return
			}
			writeJSON(w, map[string]string{"status": "success"})
			return
		}
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
	}))

	mux.HandleFunc("/api/settings/vllm-api-key", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		writeJSON(w, map[string]string{"vllm_api_key": getSettings().VLLMAPIKey})
	}))

	mux.HandleFunc("/api/proxy-users", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeJSON(w, listProxyUsers())
			return
		}
		if r.Method == http.MethodPost {
			var req struct {
				Username string `json:"username"`
			}
			if !decodeJSONBody(w, r, &req) {
				return
			}
			user, err := createProxyUser(req.Username)
			if err != nil {
				writeAPIError(w, err)
				return
			}
			writeJSON(w, map[string]interface{}{
				"username":      user.Username,
				"api_key":       user.APIKey,
				"request_count": user.RequestCount,
			})
			return
		}
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
	}))

	mux.HandleFunc("/api/proxy-users/key", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		key, err := revealProxyUserKey(r.URL.Query().Get("username"))
		if err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"api_key": key})
	}))

	mux.HandleFunc("/api/proxy-users/revoke", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			Username string `json:"username"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if err := revokeProxyUser(req.Username); err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "revoked"})
	}))

	mux.HandleFunc("/api/models/server-api-key", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		if proxyEnabled {
			writeJSONError(w, http.StatusNotFound, "per-server API keys are not exposed in proxy mode")
			return
		}
		containerName := strings.TrimSpace(r.URL.Query().Get("container_name"))
		if containerName == "" {
			writeJSONError(w, http.StatusBadRequest, "container_name is required")
			return
		}
		key := getSettings().ServerAPIKeys[containerName]
		if key == "" {
			writeJSONError(w, http.StatusNotFound, "no per-server API key configured for this container")
			return
		}
		writeJSON(w, map[string]string{"server_api_key": key})
	}))

	if proxyEnabled {
		mux.HandleFunc("/api/routing", apiHandler(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodGet {
				table := getSettings().RoutingTable
				if table == nil {
					table = make(map[string]int)
				}
				writeJSON(w, table)
				return
			}
			if r.Method == http.MethodPost {
				var table map[string]int
				if !decodeJSONBody(w, r, &table) {
					return
				}
				settings := getSettings()
				models, err := api.GetActiveModels(settings.VLLMAPIKey)
				if err != nil {
					writeAPIError(w, err)
					return
				}
				validated, err := validateRoutingTableForActiveModels(table, models)
				if err != nil {
					writeAPIError(w, err)
					return
				}
				if err := replaceRoutingTable(validated); err != nil {
					writeAPIError(w, err)
					return
				}
				writeJSON(w, map[string]string{"status": "success"})
				return
			}
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		}))
	} else {
		mux.HandleFunc("/api/routing", apiHandler(func(w http.ResponseWriter, r *http.Request) {
			writeJSONError(w, http.StatusNotFound, "proxy gateway is disabled")
		}))
	}

	mux.HandleFunc("/api/config", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, api.LaunchDefaults())
	}))

	mux.HandleFunc("/api/recommendations", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		limit := 50
		if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed < 1 {
				writeJSONError(w, http.StatusBadRequest, "limit must be a positive integer")
				return
			}
			limit = parsed
		}
		fitOnly := strings.EqualFold(r.URL.Query().Get("fit_only"), "true") || r.URL.Query().Get("fit_only") == "1"
		recommendations, err := api.RecommendModels(api.RecommendationOptions{
			UseCase: r.URL.Query().Get("use_case"),
			FitOnly: fitOnly,
			Limit:   limit,
		})
		if err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, recommendations)
	}))

	mux.HandleFunc("/api/gpus", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		gpus, err := api.GetGPUInfo()
		if err != nil {
			writeJSONError(w, http.StatusServiceUnavailable, fmt.Sprintf("GPU detection failed: %v", err))
			return
		}
		writeJSON(w, gpus)
	}))

	mux.HandleFunc("/api/models/active", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		settings := getSettings()
		models, err := api.GetActiveModels(settings.VLLMAPIKey)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if models == nil {
			models = []api.ActiveModel{}
		}
		// Enrich the response with per-server key status without exposing the
		// secret itself. A missing entry means the global key is in use.
		for i := range models {
			_, hasKey := settings.ServerAPIKeys[models[i].ContainerName]
			models[i].HasServerAPIKey = hasKey
			if hasKey {
				models[i].APIKeyMode = "custom"
			} else if strings.TrimSpace(models[i].APIKey) != "" {
				models[i].APIKeyMode = "global"
			} else {
				models[i].APIKeyMode = "none"
			}
		}
		writeJSON(w, models)
	}))

	mux.HandleFunc("/api/models/downloaded", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		settings := getSettings()
		models, _ := api.GetDownloadedModels(settings.HFCache)
		if models == nil {
			models = []api.DownloadedModel{}
		}
		writeJSON(w, models)
	}))

	mux.HandleFunc("/api/models/start", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			Model        string `json:"model"`
			Port         int    `json:"port"`
			GPUs         int    `json:"gpus"`
			GPUIDs       string `json:"gpu_ids"`
			MemUtil      string `json:"mem_util"`
			MaxLen       string `json:"max_len"`
			MaxNumSeqs   string `json:"max_num_seqs"`
			KVCacheDType string `json:"kv_cache_dtype"`
			ToolParser   string `json:"tool_parser"`
			UseAPIKey    bool   `json:"use_api_key"`
			ServerAPIKey string `json:"server_api_key"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}

		log.Printf("[API] /api/models/start requested! Model=%s Port=%d GPUs=%d GPUIDs=%s MemUtil=%s MaxLen=%s MaxNumSeqs=%s KVCacheDType=%s Parser=%s\n",
			req.Model, req.Port, req.GPUs, req.GPUIDs, req.MemUtil, req.MaxLen, req.MaxNumSeqs, req.KVCacheDType, req.ToolParser)

		_, err := api.RequireDetectedGPUs()
		if err != nil {
			status := http.StatusServiceUnavailable
			if api.IsValidationError(err) {
				status = http.StatusBadRequest
			}
			writeJSONError(w, status, err.Error())
			return
		}

		settings := getSettings()
		hostBindIP := "0.0.0.0"
		if proxyEnabled {
			hostBindIP = "127.0.0.1"
		}
		serverAPIKey := ""
		vllmAPIKey := ""
		if proxyEnabled {
			vllmAPIKey = settings.VLLMAPIKey
		} else if req.UseAPIKey {
			serverAPIKey = strings.TrimSpace(req.ServerAPIKey)
			vllmAPIKey = settings.VLLMAPIKey
		}
		if serverAPIKey != "" {
			vllmAPIKey = ""
		}
		if err := api.StartModel(req.Model, req.Port, req.GPUs, req.GPUIDs, req.MemUtil, req.MaxLen, req.MaxNumSeqs, req.KVCacheDType, req.ToolParser, settings.HFCache, settings.HFToken, vllmAPIKey, settings.EnableCUDACompatibility, settings.HFHubOffline, hostBindIP, serverAPIKey); err != nil {
			writeAPIError(w, err)
			return
		}
		// Persist the per-server API key against the container name we just
		// created. The proxy and the per-server-key map both use this name
		// as a stable identifier, so the routing table alone is not enough.
		if serverAPIKey != "" {
			shortName := req.Model
			if idx := strings.LastIndex(req.Model, "/"); idx != -1 {
				shortName = req.Model[idx+1:]
			}
			shortName = api.SanitizeDockerNamePart(shortName)
			containerName := fmt.Sprintf("vllm-%s-%d", shortName, req.Port)
			upsertServerAPIKey(containerName, serverAPIKey)
		}
		upsertRoute(req.Model, req.Port)
		writeJSON(w, map[string]string{"status": "success"})
	}))

	mux.HandleFunc("/api/models/stop", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			Port int `json:"port"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if req.Port < 1 || req.Port > 65535 {
			writeJSONError(w, http.StatusBadRequest, "port must be between 1 and 65535")
			return
		}

		if err := api.StopModel(req.Port); err != nil {
			writeAPIError(w, err)
			return
		}
		cleanupModelStateForPort(req.Port)
		writeJSON(w, map[string]string{"status": "stopped"})
	}))

	mux.HandleFunc("/api/models/remove", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			ContainerID string `json:"container_id"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if strings.TrimSpace(req.ContainerID) == "" {
			writeJSONError(w, http.StatusBadRequest, "container_id is required")
			return
		}

		modelState, cleanupState, cleanupErr := api.GetManagedModelByContainerID(req.ContainerID)
		if cleanupErr != nil {
			log.Printf("[API] warning: failed to resolve managed container %s for state cleanup: %v", req.ContainerID, cleanupErr)
		}
		if err := api.RemoveModel(req.ContainerID); err != nil {
			writeAPIError(w, err)
			return
		}
		if cleanupState {
			cleanupModelStateForActiveModel(modelState)
		}
		writeJSON(w, map[string]string{"status": "removed"})
	}))

	mux.HandleFunc("/api/models/downloaded/delete", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if strings.TrimSpace(req.Model) == "" {
			writeJSONError(w, http.StatusBadRequest, "model is required")
			return
		}

		if err := api.DeleteDownloadedModel(req.Model, getSettings().HFCache); err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "deleted"})
	}))

	mux.HandleFunc("/api/models/restart", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			ContainerID string `json:"container_id"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if strings.TrimSpace(req.ContainerID) == "" {
			writeJSONError(w, http.StatusBadRequest, "container_id is required")
			return
		}

		if err := api.RestartModel(req.ContainerID); err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "restarted"})
	}))

	mux.HandleFunc("/api/models/info", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		model := r.URL.Query().Get("model")
		info, err := fetchHFModelInfo(model)
		if err != nil {
			writeAPIError(w, err)
			return
		}
		settings := getSettings()
		downloaded, cachedBytes, cacheErr := api.IsModelDownloaded(model, settings.HFCache)
		cacheWarning := ""
		if cacheErr != nil {
			cacheWarning = cacheErr.Error()
			downloaded = false
			cachedBytes = 0
		}
		writeJSON(w, map[string]interface{}{
			"size_bytes":    info.SizeBytes,
			"downloads":     info.Downloads,
			"likes":         info.Likes,
			"downloaded":    downloaded,
			"cached_bytes":  cachedBytes,
			"cache_warning": cacheWarning,
		})
	}))

	mux.HandleFunc("/api/models/download", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		var req struct {
			Model string `json:"model"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}
		if strings.TrimSpace(req.Model) == "" {
			writeJSONError(w, http.StatusBadRequest, "model is required")
			return
		}

		settings := getSettings()
		// Refuse the download when the model is already on disk. Returning
		// 409 (Conflict) with a typed error code lets the UI render a
		// specific message ("already downloaded, click here to manage
		// cache") instead of a generic 400. A misconfigured cache path
		// silently falls through so the real error from DownloadModel
		// surfaces.
		if downloaded, size, err := api.IsModelDownloaded(req.Model, settings.HFCache); err == nil && downloaded {
			// Headers must be set before the body. Best-effort hints
			// for clients that want to render state without re-parsing
			// the detail string.
			w.Header().Set("X-Cache-State", "downloaded")
			w.Header().Set("X-Cached-Bytes", strconv.FormatUint(size, 10))
			writeJSONError(w, http.StatusConflict, "model is already downloaded")
			return
		}

		containerID, err := api.DownloadModel(req.Model, settings.HFCache, settings.HFToken)
		if err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "success", "container_id": containerID})
	}))

	mux.Handle("/api/models/download/progress", apiHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		model := r.URL.Query().Get("model")
		if model == "" {
			writeJSONError(w, http.StatusBadRequest, "model required")
			return
		}
		size, err := api.GetModelDirectorySize(model, getSettings().HFCache)
		if err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]interface{}{"downloaded_bytes": size})
	})))

	mux.HandleFunc("/api/models/logs", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		containerID := r.URL.Query().Get("container_id")
		if containerID == "" {
			writeJSONError(w, http.StatusBadRequest, "container_id is required")
			return
		}
		logs, err := api.GetContainerLogs(containerID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to fetch logs: %v", err))
			return
		}
		writeJSON(w, map[string]string{"logs": logs})
	}))

	mux.HandleFunc("/api/models/logs/stream", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		containerID := r.URL.Query().Get("container_id")
		if containerID == "" {
			writeJSONError(w, http.StatusBadRequest, "container_id is required")
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeJSONError(w, http.StatusInternalServerError, "streaming unsupported")
			return
		}

		ctx := r.Context()
		stream := api.StreamContainerLogs(ctx, containerID)

		for {
			select {
			case <-ctx.Done():
				return
			case chunk, ok := <-stream:
				if !ok {
					// Channel closed — stream ended.
					fmt.Fprint(w, "event: close\ndata: \n\n")
					flusher.Flush()
					return
				}
				if chunk.Err != nil {
					escaped := strings.ReplaceAll(chunk.Err.Error(), `\`, `\\`)
					escaped = strings.ReplaceAll(escaped, "\r", `\r`)
					fmt.Fprintf(w, "event: error\ndata:%s\n\n", escaped)
					flusher.Flush()
					return
				}
				event := "log"
				if chunk.Stdout {
					event = "stdout"
				} else {
					event = "stderr"
				}
				// Send the exact chunk content (which may contain internal \n).
				// Use multiple "data:" lines for multi-line chunks (standard SSE).
				// The client joins them back with \n, preserving original newlines.
				fmt.Fprintf(w, "event: %s\n", event)
				lines := strings.Split(chunk.Data, "\n")
				for _, line := range lines {
					escaped := strings.ReplaceAll(line, `\`, `\\`)
					escaped = strings.ReplaceAll(escaped, "\r", `\r`)
					fmt.Fprintf(w, "data:%s\n", escaped)
				}
				fmt.Fprint(w, "\n")
				flusher.Flush()
				if ctx.Err() != nil {
					return
				}
			}
		}
	}))
}

func registerStaticRoutes(mux *http.ServeMux) {
	distFS, err := fs.Sub(frontendDist, "frontend/dist")
	if err != nil {
		log.Fatalf("Failed to sub embedded dist fs: %v", err)
	}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if path == "/" {
			path = "/index.html"
		}

		f, err := distFS.Open(strings.TrimPrefix(path, "/"))
		if err == nil {
			f.Close()
			http.FileServer(http.FS(distFS)).ServeHTTP(w, r)
			return
		}

		file, err := distFS.Open("index.html")
		if err != nil {
			http.Error(w, "index.html not found", http.StatusInternalServerError)
			return
		}
		defer file.Close()
		stat, _ := file.Stat()
		http.ServeContent(w, r, "index.html", stat.ModTime(), file.(io.ReadSeeker))
	})
}

func runDashboard(port string, proxyEnabled bool) {
	mux, auth := newDashboardMux(proxyEnabled)
	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		// ReadTimeout/WriteTimeout intentionally left at Go defaults (0 = no timeout).
		// Long-lived SSE streams (/api/models/logs/stream) must not be terminated
		// by global deadlines; client disconnects are handled via r.Context().
	}

	if auth.required() {
		fmt.Println("Dashboard authentication enabled via VLLM_MANAGER_PASSWORD")
	}

	if err := ensureCerts("server.crt", "server.key"); err != nil {
		log.Fatalf("failed to generate TLS certificates: %v", err)
	}

	fmt.Printf("Starting vllm-manager server on :%s (HTTPS)\n", port)

	reconcileRoutingTable()

	heartbeatStop := startHeartbeatRefresher()
	defer func() {
		heartbeatStop()
		clearHeartbeat()
	}()

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServeTLS("server.crt", "server.key")
	}()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	case sig := <-signals:
		log.Printf("Received %s; shutting down dashboard", sig)
		ctx, cancel := context.WithTimeout(context.Background(), dashboardShutdownTimeout)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Fatalf("failed to shut down dashboard cleanly: %v", err)
		}
	}
}

const heartbeatRefreshInterval = 5 * time.Second

// startHeartbeatRefresher writes the dashboard PID and a fresh LastSeen
// timestamp into settings.json on a regular interval so the CLI can detect
// a running dashboard. It returns a stop function the caller should defer.
func startHeartbeatRefresher() func() {
	pid := os.Getpid()
	stop := make(chan struct{})
	if err := writeHeartbeat(pid, time.Now()); err != nil {
		log.Printf("[HEARTBEAT] failed to write initial heartbeat: %v", err)
	}
	go func() {
		ticker := time.NewTicker(heartbeatRefreshInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := writeHeartbeat(pid, time.Now()); err != nil {
					log.Printf("[HEARTBEAT] failed to refresh heartbeat: %v", err)
				}
			case <-stop:
				return
			}
		}
	}()
	return func() { close(stop) }
}
