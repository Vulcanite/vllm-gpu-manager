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
	"strings"
	"syscall"
	"time"

	"vllm-manager/api"
)

const dashboardShutdownTimeout = 10 * time.Second

func newDashboardMux() (*http.ServeMux, dashboardAuth) {
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
	registerAPIRoutes(mux, apiHandler)
	registerStaticRoutes(mux)
	return mux, auth
}

func registerAPIRoutes(mux *http.ServeMux, apiHandler func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("/api/settings", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeJSON(w, getSettingsStatus())
			return
		}
		if r.Method == http.MethodPost {
			var req struct {
				HFCache         string `json:"hf_cache"`
				HFToken         string `json:"hf_token"`
				ClearHFToken    bool   `json:"clear_hf_token"`
				VLLMAPIKey      string `json:"vllm_api_key"`
				ClearVLLMAPIKey bool   `json:"clear_vllm_api_key"`
			}
			if !decodeJSONBody(w, r, &req) {
				return
			}

			if _, err := updateSettings(SettingsUpdate{
				HFCache:         req.HFCache,
				HFToken:         req.HFToken,
				ClearHFToken:    req.ClearHFToken,
				VLLMAPIKey:      req.VLLMAPIKey,
				ClearVLLMAPIKey: req.ClearVLLMAPIKey,
			}); err != nil {
				writeAPIError(w, err)
				return
			}
			writeJSON(w, map[string]string{"status": "success"})
			return
		}
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
	}))

	mux.HandleFunc("/api/config", apiHandler(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, api.LaunchDefaults())
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
		models, err := api.GetActiveModels()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, err.Error())
			return
		}
		if models == nil {
			models = []api.ActiveModel{}
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
			KVCacheDType string `json:"kv_cache_dtype"`
			ToolParser   string `json:"tool_parser"`
		}
		if !decodeJSONBody(w, r, &req) {
			return
		}

		log.Printf("[API] /api/models/start requested! Model=%s Port=%d GPUs=%d GPUIDs=%s MemUtil=%s MaxLen=%s KVCacheDType=%s Parser=%s\n",
			req.Model, req.Port, req.GPUs, req.GPUIDs, req.MemUtil, req.MaxLen, req.KVCacheDType, req.ToolParser)

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
		if err := api.StartModel(req.Model, req.Port, req.GPUs, req.GPUIDs, req.MemUtil, req.MaxLen, req.KVCacheDType, req.ToolParser, settings.HFCache, settings.HFToken, settings.VLLMAPIKey); err != nil {
			writeAPIError(w, err)
			return
		}
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

		if err := api.RemoveModel(req.ContainerID); err != nil {
			writeAPIError(w, err)
			return
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

		writeJSON(w, map[string]interface{}{
			"size_bytes": info.SizeBytes,
			"downloads":  info.Downloads,
			"likes":      info.Likes,
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

func runDashboard(port string) {
	mux, auth := newDashboardMux()
	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      60 * time.Second,
	}

	if auth.required() {
		fmt.Println("Dashboard authentication enabled via VLLM_MANAGER_PASSWORD")
	}
	fmt.Printf("Starting vllm-manager server on :%s\n", port)

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
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
