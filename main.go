package main

import (
	"bufio"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
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
	"unicode"

	"vllm-manager/api"
)

//go:embed all:frontend/dist
var frontendDist embed.FS

const settingsFile = "settings.json"
const defaultDashboardPort = "9090"
const authCookieName = "vllm_manager_session"

type dashboardAuth struct {
	password     string
	sessionToken string
}

type Settings struct {
	HFCache    string `json:"hf_cache"`
	HFToken    string `json:"hf_token"`
	VLLMAPIKey string `json:"vllm_api_key"`
}

type SettingsStatus struct {
	HFCache       string `json:"hf_cache"`
	HFTokenSet    bool   `json:"hf_token_set"`
	VLLMAPIKeySet bool   `json:"vllm_api_key_set"`
	DiskTotal     uint64 `json:"disk_total"`
	DiskUsed      uint64 `json:"disk_used"`
	DiskFree      uint64 `json:"disk_free"`
	RAMTotal      uint64 `json:"ram_total"`
	RAMUsed       uint64 `json:"ram_used"`
	RAMFree       uint64 `json:"ram_free"`
}

type SettingsUpdate struct {
	HFCache         string
	HFToken         string
	ClearHFToken    bool
	VLLMAPIKey      string
	ClearVLLMAPIKey bool
}

type HFModelInfo struct {
	Model     string `json:"model"`
	SizeBytes int64  `json:"size_bytes"`
	Downloads int    `json:"downloads"`
	Likes     int    `json:"likes"`
}

type appError struct {
	Status int
	Detail string
}

func (e appError) Error() string {
	return e.Detail
}

func newAppError(status int, format string, args ...interface{}) error {
	return appError{Status: status, Detail: fmt.Sprintf(format, args...)}
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

func getSettingsStatus() SettingsStatus {
	s := getSettings()
	var total, used, free uint64
	var stat syscall.Statfs_t
	if err := syscall.Statfs(s.HFCache, &stat); err == nil {
		total = stat.Blocks * uint64(stat.Bsize)
		free = stat.Bavail * uint64(stat.Bsize)
		used = total - free
	}
	ramTotal, ramUsed, ramFree := getSystemRAM()
	return SettingsStatus{
		HFCache:       s.HFCache,
		HFTokenSet:    s.HFToken != "",
		VLLMAPIKeySet: s.VLLMAPIKey != "",
		DiskTotal:     total,
		DiskUsed:      used,
		DiskFree:      free,
		RAMTotal:      ramTotal,
		RAMUsed:       ramUsed,
		RAMFree:       ramFree,
	}
}

func updateSettings(req SettingsUpdate) (Settings, error) {
	current := getSettings()
	hfCache := api.NormalizeHFHome(req.HFCache)
	if hfCache == "" {
		return Settings{}, newAppError(http.StatusBadRequest, "Cache directory is required")
	}
	vllmAPIKey, err := normalizeVLLMAPIKey(req.VLLMAPIKey)
	if err != nil {
		return Settings{}, err
	}
	if req.ClearVLLMAPIKey && vllmAPIKey != "" {
		return Settings{}, newAppError(http.StatusBadRequest, "Cannot set and clear the vLLM API key in the same request")
	}

	stat, err := os.Stat(hfCache)
	if err != nil || !stat.IsDir() {
		return Settings{}, newAppError(http.StatusBadRequest, "Cache directory does not exist or is not a folder: %s", hfCache)
	}

	if hfCache != current.HFCache {
		containerNames, err := api.GetVLLMContainerNames()
		if err != nil {
			return Settings{}, newAppError(http.StatusInternalServerError, "failed to inspect vLLM containers: %v", err)
		}
		if len(containerNames) > 0 {
			return Settings{}, newAppError(http.StatusBadRequest, "Failed to change HF Cache Path: Delete all running model containers before changing the path.")
		}
	}

	s := Settings{
		HFCache:    hfCache,
		HFToken:    current.HFToken,
		VLLMAPIKey: current.VLLMAPIKey,
	}
	if req.ClearHFToken {
		s.HFToken = ""
	} else if req.HFToken != "" {
		s.HFToken = req.HFToken
	}
	if req.ClearVLLMAPIKey {
		s.VLLMAPIKey = ""
	} else if vllmAPIKey != "" {
		s.VLLMAPIKey = vllmAPIKey
	}

	if err := saveSettings(s); err != nil {
		return Settings{}, newAppError(http.StatusInternalServerError, "failed to save settings: %v", err)
	}
	return s, nil
}

func normalizeVLLMAPIKey(value string) (string, error) {
	key := strings.TrimSpace(value)
	if strings.IndexFunc(key, unicode.IsSpace) >= 0 {
		return "", newAppError(http.StatusBadRequest, "vLLM API key cannot contain whitespace")
	}
	return key, nil
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
	var appErr appError
	if errors.As(err, &appErr) {
		status = appErr.Status
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
			w.Header().Set("Access-Control-Allow-Credentials", "true")
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

func newDashboardAuth() dashboardAuth {
	password := strings.TrimSpace(os.Getenv("VLLM_MANAGER_PASSWORD"))
	if password == "" {
		return dashboardAuth{}
	}
	return dashboardAuth{
		password:     password,
		sessionToken: randomSessionToken(),
	}
}

func randomSessionToken() string {
	var data [32]byte
	if _, err := rand.Read(data[:]); err != nil {
		log.Fatalf("failed to initialize dashboard auth session: %v", err)
	}
	return hex.EncodeToString(data[:])
}

func (a dashboardAuth) required() bool {
	return a.password != ""
}

func (a dashboardAuth) authenticated(r *http.Request) bool {
	if !a.required() {
		return true
	}
	cookie, err := r.Cookie(authCookieName)
	if err != nil {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(cookie.Value), []byte(a.sessionToken)) == 1
}

func (a dashboardAuth) require(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !a.authenticated(r) {
			writeJSONError(w, http.StatusUnauthorized, "authentication required")
			return
		}
		next(w, r)
	}
}

func (a dashboardAuth) statusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, map[string]bool{
		"required":      a.required(),
		"authenticated": a.authenticated(r),
	})
}

func (a dashboardAuth) loginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !a.required() {
		writeJSON(w, map[string]bool{"required": false, "authenticated": true})
		return
	}

	var req struct {
		Password string `json:"password"`
	}
	if !decodeJSONBody(w, r, &req) {
		return
	}
	if subtle.ConstantTimeCompare([]byte(req.Password), []byte(a.password)) != 1 {
		writeJSONError(w, http.StatusUnauthorized, "invalid password")
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     authCookieName,
		Value:    a.sessionToken,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
		MaxAge:   int((24 * time.Hour).Seconds()),
	})
	writeJSON(w, map[string]bool{"required": true, "authenticated": true})
}

func (a dashboardAuth) logoutHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     authCookieName,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   r.TLS != nil,
		MaxAge:   -1,
	})
	writeJSON(w, map[string]bool{"authenticated": false})
}

func main() {
	if len(os.Args) == 1 {
		printRootHelp(os.Stdout)
		return
	}

	if os.Args[1] == "-h" || os.Args[1] == "--help" {
		printRootHelp(os.Stdout)
		return
	}

	if hasDashboardFlag(os.Args[1:]) || hasPortFlag(os.Args[1:]) {
		fs := flag.NewFlagSet("vllm-manager", flag.ExitOnError)
		fs.Bool("dashboard", false, "Run the web dashboard")
		port := fs.String("port", defaultDashboardPort, "Port to run the dashboard on")
		fs.Usage = func() { printRootHelp(os.Stderr) }
		if err := fs.Parse(os.Args[1:]); err != nil {
			os.Exit(2)
		}
		if fs.NArg() != 0 {
			fmt.Fprintf(os.Stderr, "error: dashboard mode does not accept CLI commands: %s\n", strings.Join(fs.Args(), " "))
			os.Exit(1)
		}
		runDashboard(*port)
		return
	}

	if err := runCLI(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func hasDashboardFlag(args []string) bool {
	for _, arg := range args {
		if arg == "--dashboard" || arg == "-dashboard" || strings.HasPrefix(arg, "--dashboard=") || strings.HasPrefix(arg, "-dashboard=") {
			return true
		}
	}
	return false
}

func hasPortFlag(args []string) bool {
	for _, arg := range args {
		if arg == "--port" || arg == "-port" || strings.HasPrefix(arg, "--port=") || strings.HasPrefix(arg, "-port=") {
			return true
		}
	}
	return false
}

func fetchHFModelInfo(model string) (HFModelInfo, error) {
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

func runDashboard(port string) {

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

	// API Routes
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

		containerID, err := api.DownloadModel(req.Model, getSettings().HFCache, getSettings().HFToken)
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

	if auth.required() {
		fmt.Println("Dashboard authentication enabled via VLLM_MANAGER_PASSWORD")
	}
	fmt.Printf("Starting vllm-manager server on :%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
