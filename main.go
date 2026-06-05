package main

import (
	"bufio"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"vllm-manager/api"
)

//go:embed all:frontend/dist
var frontendDist embed.FS

const settingsFile = "settings.json"

type Settings struct {
	HFCache string `json:"hf_cache"`
	HFToken string `json:"hf_token"`
}

var (
	settingsCacheMu     sync.RWMutex
	settingsCache       Settings
	settingsCacheLoaded bool
)

func defaultSettings() Settings {
	home, err := os.UserHomeDir()
	defaultCache := "/root/.cache/huggingface"
	if err == nil {
		defaultCache = home + "/.cache/huggingface"
	}
	return Settings{HFCache: defaultCache}
}

func normalizeSettings(s Settings) Settings {
	defaults := defaultSettings()
	if strings.TrimSpace(s.HFCache) == "" {
		s.HFCache = defaults.HFCache
	}
	s.HFCache = api.NormalizeHFHome(s.HFCache)
	return s
}

func loadSettingsFromDisk() Settings {
	s := defaultSettings()
	data, err := os.ReadFile(settingsFile)
	if err == nil {
		if err := json.Unmarshal(data, &s); err != nil {
			log.Printf("[SETTINGS] warning: failed to parse %s: %v", settingsFile, err)
		}
	}
	return normalizeSettings(s)
}

func getSettings() Settings {
	settingsCacheMu.RLock()
	if settingsCacheLoaded {
		s := settingsCache
		settingsCacheMu.RUnlock()
		return s
	}
	settingsCacheMu.RUnlock()

	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	if !settingsCacheLoaded {
		settingsCache = loadSettingsFromDisk()
		settingsCacheLoaded = true
	}
	return settingsCache
}

func saveSettings(s Settings) error {
	s = normalizeSettings(s)
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}

	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	if err := os.WriteFile(settingsFile, data, 0600); err != nil {
		return err
	}
	if err := os.Chmod(settingsFile, 0600); err != nil {
		return err
	}
	settingsCache = s
	settingsCacheLoaded = true
	return nil
}

func getSystemRAM() (total uint64, used uint64, free uint64) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, 0
	}
	defer file.Close()

	var memTotal, memFree, memAvailable, buffers, cached uint64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		val, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		val *= 1024 // KB to Bytes

		switch fields[0] {
		case "MemTotal:":
			memTotal = val
		case "MemFree:":
			memFree = val
		case "MemAvailable:":
			memAvailable = val
		case "Buffers:":
			buffers = val
		case "Cached:":
			cached = val
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[SYSTEM] warning: failed to scan /proc/meminfo: %v", err)
		return 0, 0, 0
	}

	total = memTotal
	if memAvailable > 0 {
		free = memAvailable
	} else {
		free = memFree + buffers + cached
	}
	used = total - free
	return
}

func writeJSON(w http.ResponseWriter, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(value)
}

func writeJSONError(w http.ResponseWriter, status int, detail string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"detail": detail})
}

func writeAPIError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	if api.IsValidationError(err) {
		status = http.StatusBadRequest
	}
	writeJSONError(w, status, err.Error())
}

func decodeJSONBody(w http.ResponseWriter, r *http.Request, value interface{}) bool {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return false
	}
	return true
}

func isAllowedOrigin(origin string, requestHost string) bool {
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Host == "" {
		return false
	}

	if parsed.Host == requestHost {
		return true
	}

	host := parsed.Hostname()
	if host == "localhost" || host == "127.0.0.1" || host == "::1" {
		return true
	}

	for _, allowed := range strings.Split(os.Getenv("VLLM_MANAGER_ALLOWED_ORIGINS"), ",") {
		if strings.TrimSpace(allowed) == origin {
			return true
		}
	}
	return false
}

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[API] %s %s", r.Method, r.URL.Path)
		origin := r.Header.Get("Origin")
		if origin != "" {
			if !isAllowedOrigin(origin, r.Host) {
				writeJSONError(w, http.StatusForbidden, "origin not allowed")
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
		}
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func main() {
	port := flag.String("port", "9090", "Port to run the vllm-manager on")
	flag.Parse()

	mux := http.NewServeMux()

	// API Routes
	mux.HandleFunc("/api/settings", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			s := getSettings()
			var total, used, free uint64
			var stat syscall.Statfs_t
			if err := syscall.Statfs(s.HFCache, &stat); err == nil {
				total = stat.Blocks * uint64(stat.Bsize)
				free = stat.Bavail * uint64(stat.Bsize)
				used = total - free
			}
			ramTotal, ramUsed, ramFree := getSystemRAM()
			writeJSON(w, map[string]interface{}{
				"hf_cache":     s.HFCache,
				"hf_token_set": s.HFToken != "",
				"disk_total":   total,
				"disk_used":    used,
				"disk_free":    free,
				"ram_total":    ramTotal,
				"ram_used":     ramUsed,
				"ram_free":     ramFree,
			})
			return
		}
		if r.Method == http.MethodPost {
			current := getSettings()
			var req struct {
				HFCache      string `json:"hf_cache"`
				HFToken      string `json:"hf_token"`
				ClearHFToken bool   `json:"clear_hf_token"`
			}
			if !decodeJSONBody(w, r, &req) {
				return
			}

			hfCache := api.NormalizeHFHome(req.HFCache)
			if hfCache == "" {
				writeJSONError(w, http.StatusBadRequest, "Cache directory is required")
				return
			}

			// Validate that the HF Cache directory exists
			stat, err := os.Stat(hfCache)
			if err != nil || !stat.IsDir() {
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("Cache directory does not exist or is not a folder: %s", hfCache))
				return
			}

			if hfCache != current.HFCache {
				containerNames, err := api.GetVLLMContainerNames()
				if err != nil {
					writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to inspect vLLM containers: %v", err))
					return
				}
				if len(containerNames) > 0 {
					writeJSONError(w, http.StatusBadRequest, "Failed to change HF Cache Path: Delete all running model containers before changing the path.")
					return
				}
			}

			s := Settings{
				HFCache: hfCache,
				HFToken: current.HFToken,
			}
			if req.ClearHFToken {
				s.HFToken = ""
			} else if req.HFToken != "" {
				s.HFToken = req.HFToken
			}

			if err := saveSettings(s); err != nil {
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to save settings: %v", err))
				return
			}
			writeJSON(w, map[string]string{"status": "success"})
			return
		}
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
	}))

	mux.HandleFunc("/api/gpus", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		gpus, err := api.GetGPUInfo()
		if err != nil {
			writeJSONError(w, http.StatusServiceUnavailable, fmt.Sprintf("GPU detection failed: %v", err))
			return
		}
		writeJSON(w, gpus)
	}))

	mux.HandleFunc("/api/models/active", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

	mux.HandleFunc("/api/models/downloaded", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		settings := getSettings()
		models, _ := api.GetDownloadedModels(settings.HFCache)
		if models == nil {
			models = []api.DownloadedModel{}
		}
		writeJSON(w, models)
	}))

	mux.HandleFunc("/api/models/start", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

		gpuList, err := api.RequireDetectedGPUs()
		if err != nil {
			status := http.StatusServiceUnavailable
			if api.IsValidationError(err) {
				status = http.StatusBadRequest
			}
			writeJSONError(w, status, err.Error())
			return
		}
		if req.GPUs > len(gpuList) {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("Requested %d GPUs, but only %d are available on this server", req.GPUs, len(gpuList)))
			return
		}

		active, _ := api.GetActiveModels()
		for _, m := range active {
			if m.Port == strconv.Itoa(req.Port) {
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("Requested port %d is already in use by %s", req.Port, m.Model))
				return
			}
		}

		settings := getSettings()
		if err := api.StartModel(req.Model, req.Port, req.GPUs, req.GPUIDs, req.MemUtil, req.MaxLen, req.KVCacheDType, req.ToolParser, settings.HFCache, settings.HFToken); err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "success"})
	}))

	mux.HandleFunc("/api/models/stop", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

	mux.HandleFunc("/api/models/remove", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

	mux.HandleFunc("/api/models/downloaded/delete", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

	mux.HandleFunc("/api/models/restart", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

	mux.HandleFunc("/api/models/info", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
		model := r.URL.Query().Get("model")
		if model == "" {
			writeJSONError(w, http.StatusBadRequest, "model required")
			return
		}

		client := http.Client{Timeout: 5 * time.Second}
		reqHF, _ := http.NewRequest("GET", "https://huggingface.co/api/models/"+model, nil)
		settings := getSettings()
		if settings.HFToken != "" {
			reqHF.Header.Set("Authorization", "Bearer "+settings.HFToken)
		}

		resp, err := client.Do(reqHF)
		if err != nil || resp.StatusCode != http.StatusOK {
			writeJSONError(w, http.StatusNotFound, "model not found or Hugging Face API error")
			if resp != nil {
				resp.Body.Close()
			}
			return
		}
		defer resp.Body.Close()

		var hfResp struct {
			UsedStorage int64    `json:"usedStorage"`
			Downloads   int      `json:"downloads"`
			Likes       int      `json:"likes"`
			Tags        []string `json:"tags"`
		}
		json.NewDecoder(resp.Body).Decode(&hfResp)

		writeJSON(w, map[string]interface{}{
			"size_bytes": hfResp.UsedStorage,
			"downloads":  hfResp.Downloads,
			"likes":      hfResp.Likes,
		})
	}))

	mux.HandleFunc("/api/models/download", corsMiddleware(func(w http.ResponseWriter, r *http.Request) {
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

		containerID, err := api.DownloadModel(req.Model, getSettings().HFCache, getSettings().HFToken)
		if err != nil {
			writeAPIError(w, err)
			return
		}
		writeJSON(w, map[string]string{"status": "success", "container_id": containerID})
	}))

	mux.Handle("/api/models/download/progress", corsMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	// Static SPA Router
	distFS, err := fs.Sub(frontendDist, "frontend/dist")
	if err != nil {
		log.Fatalf("Failed to sub embedded dist fs: %v", err)
	}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if path == "/" {
			path = "/index.html"
		}

		// Try to serve the static file
		f, err := distFS.Open(strings.TrimPrefix(path, "/"))
		if err == nil {
			f.Close()
			http.FileServer(http.FS(distFS)).ServeHTTP(w, r)
			return
		}

		// Fallback to index.html for SPA routing
		file, err := distFS.Open("index.html")
		if err != nil {
			http.Error(w, "index.html not found", http.StatusInternalServerError)
			return
		}
		defer file.Close()
		stat, _ := file.Stat()
		http.ServeContent(w, r, "index.html", stat.ModTime(), file.(io.ReadSeeker))
	})

	fmt.Printf("Starting vllm-manager server on :%s\n", *port)
	log.Fatal(http.ListenAndServe(":"+*port, mux))
}
