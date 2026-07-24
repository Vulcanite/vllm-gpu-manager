package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"

	"vllm-manager/api"

	"github.com/distribution/reference"
)

type Heartbeat struct {
	PID      int       `json:"pid"`
	LastSeen time.Time `json:"last_seen"`
}

type Settings struct {
	HFCache                 string            `json:"hf_cache"`
	HFToken                 string            `json:"hf_token"`
	VLLMAPIKey              string            `json:"vllm_api_key"`
	InferenceAPIKey         string            `json:"inference_api_key"`
	ProxyUsers              []ProxyUser       `json:"proxy_users,omitempty"`
	EnableCUDACompatibility bool              `json:"enable_cuda_compatibility"`
	HFHubOffline            bool              `json:"hf_hub_offline"`
	VLLMImage               string            `json:"vllm_image"`
	ServerBindIP            string            `json:"server_bind_ip"`
	RoutingTable            map[string]int    `json:"routing_table"`
	ServerAPIKeys           map[string]string `json:"server_api_keys,omitempty"`
	Heartbeat               *Heartbeat        `json:"heartbeat,omitempty"`
}

type ProxyUser struct {
	Username     string `json:"username"`
	APIKey       string `json:"api_key"`
	RequestCount int64  `json:"request_count"`
}

type ProxyUserStatus struct {
	Username      string `json:"username"`
	APIKeyPreview string `json:"api_key_preview"`
	RequestCount  int64  `json:"request_count"`
}

type SettingsStatus struct {
	HFCache                  string   `json:"hf_cache"`
	HFTokenSet               bool     `json:"hf_token_set"`
	HFTokenSuffix            string   `json:"hf_token_suffix,omitempty"`
	VLLMAPIKeySet            bool     `json:"vllm_api_key_set"`
	VLLMAPIKeySuffix         string   `json:"vllm_api_key_suffix,omitempty"`
	VLLMAPIKeyContainerCount int      `json:"vllm_api_key_container_count"`
	InferenceAPIKeySet       bool     `json:"inference_api_key_set"`
	ProxyEnabled             bool     `json:"proxy_enabled"`
	EnableCUDACompatibility  bool     `json:"enable_cuda_compatibility"`
	HFHubOffline             bool     `json:"hf_hub_offline"`
	VLLMImage                string   `json:"vllm_image"`
	ServerBindIP             string   `json:"server_bind_ip"`
	DiskTotal                uint64   `json:"disk_total"`
	DiskUsed                 uint64   `json:"disk_used"`
	DiskFree                 uint64   `json:"disk_free"`
	RAMTotal                 uint64   `json:"ram_total"`
	RAMUsed                  uint64   `json:"ram_used"`
	RAMFree                  uint64   `json:"ram_free"`
	ServerAPIKeyCount        int      `json:"server_api_key_count"`
	ServerAPIKeyNames        []string `json:"server_api_key_names,omitempty"`
	ProxyUserCount           int      `json:"proxy_user_count"`
}

// SettingsUpdate carries optional tri-state fields. A *bool of nil means the
// caller did not include the field; a non-nil value is the explicit choice.
// This matches the ClearHFToken/ClearVLLMAPIKey discipline already used for
// secrets so the dashboard cannot silently toggle a boolean off.
type SettingsUpdate struct {
	HFCache                 string
	HFToken                 string
	ClearHFToken            bool
	VLLMAPIKey              string
	ClearVLLMAPIKey         bool
	InferenceAPIKey         string
	ClearInferenceAPIKey    bool
	EnableCUDACompatibility *bool
	HFHubOffline            *bool
	VLLMImage               *string
	ServerBindIP            *string
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
	return Settings{HFCache: defaultCache, ServerBindIP: "0.0.0.0", RoutingTable: make(map[string]int)}
}

func cloneRoutingTable(table map[string]int) map[string]int {
	if table == nil {
		return make(map[string]int)
	}
	clone := make(map[string]int, len(table))
	for model, port := range table {
		clone[model] = port
	}
	return clone
}

func cloneProxyUsers(users []ProxyUser) []ProxyUser {
	if len(users) == 0 {
		return nil
	}
	clone := make([]ProxyUser, len(users))
	copy(clone, users)
	return clone
}

// cloneSettings is a deep copy that callers can mutate freely. The cache holds
// the canonical copy and is replaced only via copy-on-write helpers below.
func cloneSettings(s Settings) Settings {
	s.RoutingTable = cloneRoutingTable(s.RoutingTable)
	s.ProxyUsers = cloneProxyUsers(s.ProxyUsers)
	if s.ServerAPIKeys != nil {
		keys := make(map[string]string, len(s.ServerAPIKeys))
		for k, v := range s.ServerAPIKeys {
			keys[k] = v
		}
		s.ServerAPIKeys = keys
	}
	if s.Heartbeat != nil {
		hb := *s.Heartbeat
		s.Heartbeat = &hb
	}
	return s
}

func normalizeSettings(s Settings) Settings {
	defaults := defaultSettings()
	if strings.TrimSpace(s.HFCache) == "" {
		s.HFCache = defaults.HFCache
	}
	s.HFCache = api.NormalizeHFHome(s.HFCache)
	s.RoutingTable = cloneRoutingTable(s.RoutingTable)
	if s.ServerAPIKeys == nil {
		s.ServerAPIKeys = map[string]string{}
	} else {
		cleaned := make(map[string]string, len(s.ServerAPIKeys))
		for k, v := range s.ServerAPIKeys {
			cleaned[strings.TrimSpace(k)] = v
		}
		s.ServerAPIKeys = cleaned
	}
	s.ProxyUsers = normalizeProxyUsersForStorage(s.ProxyUsers)
	if strings.TrimSpace(s.VLLMImage) == "" {
		s.VLLMImage = api.DefaultVLLMImage
	}
	if strings.TrimSpace(s.ServerBindIP) == "" {
		s.ServerBindIP = "0.0.0.0"
	}
	api.VLLMImage = s.VLLMImage
	return s
}

func loadSettingsFromDisk() Settings {
	s := defaultSettings()
	path := settingsFilePath()
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &s); err != nil {
			log.Printf("[SETTINGS] warning: failed to parse %s: %v", path, err)
		}
	}
	return normalizeSettings(s)
}

func loadAndCacheSettingsLocked() {
	settingsCache = loadSettingsFromDisk()
	settingsCacheLoaded = true
}

func refreshSettingsCacheFromDiskIfPresentLocked() {
	if _, err := os.Stat(settingsFilePath()); err == nil {
		loadAndCacheSettingsLocked()
		return
	}
	if !settingsCacheLoaded {
		loadAndCacheSettingsLocked()
	}
}

// getSettings returns a deep copy of the current settings under a read lock so
// concurrent readers (e.g. the proxy hot path) don't allocate against the
// shared cache map.
func getSettings() Settings {
	settingsCacheMu.RLock()
	defer settingsCacheMu.RUnlock()
	if !settingsCacheLoaded {
		settingsCacheMu.RUnlock()
		settingsCacheMu.Lock()
		if !settingsCacheLoaded {
			loadAndCacheSettingsLocked()
		}
		settingsCacheMu.Unlock()
		settingsCacheMu.RLock()
	}
	return cloneSettings(settingsCache)
}

func saveSettings(s Settings) error {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	return saveSettingsLocked(s)
}

// saveSettingsLocked writes settings to disk and updates the cache. The caller
// must hold settingsCacheMu.
func saveSettingsLocked(s Settings) error {
	s = normalizeSettings(s)
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	path := settingsFilePath()
	if err := ensureRuntimeDir(path, 0o750); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		return err
	}
	settingsCache = s
	settingsCacheLoaded = true
	return nil
}

func secretSuffix(value string) string {
	if value == "" {
		return ""
	}
	if len(value) <= 4 {
		return value
	}
	return value[len(value)-4:]
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
	ram := api.GetSystemRAM()
	vllmKeyContainerCount := 0
	if s.VLLMAPIKey != "" {
		if models, err := api.GetServerContainersUsingAPIKey(s.VLLMAPIKey); err == nil {
			vllmKeyContainerCount = len(models)
		} else {
			log.Printf("[SETTINGS] warning: failed to inspect vLLM API key container usage: %v", err)
		}
	}
	return SettingsStatus{
		HFCache:                  s.HFCache,
		HFTokenSet:               s.HFToken != "",
		HFTokenSuffix:            secretSuffix(s.HFToken),
		VLLMAPIKeySet:            s.VLLMAPIKey != "",
		VLLMAPIKeySuffix:         secretSuffix(s.VLLMAPIKey),
		VLLMAPIKeyContainerCount: vllmKeyContainerCount,
		InferenceAPIKeySet:       s.InferenceAPIKey != "",
		EnableCUDACompatibility:  s.EnableCUDACompatibility,
		HFHubOffline:             s.HFHubOffline,
		VLLMImage:                s.VLLMImage,
		ServerBindIP:             s.ServerBindIP,
		DiskTotal:                total,
		DiskUsed:                 used,
		DiskFree:                 free,
		RAMTotal:                 ram.Total,
		RAMUsed:                  ram.Used,
		RAMFree:                  ram.Free,
		ServerAPIKeyCount:        len(s.ServerAPIKeys),
		ServerAPIKeyNames:        sortedServerAPIKeyNames(s.ServerAPIKeys),
		ProxyUserCount:           len(s.ProxyUsers),
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

	if req.VLLMImage != nil && *req.VLLMImage != "" {
		named, err := reference.ParseNormalizedNamed(*req.VLLMImage)
		if err != nil {
			return Settings{}, newAppError(http.StatusBadRequest, "Invalid vLLM Docker image reference format: %v", err)
		}
		if named.Name() == "docker.io/vllm/vllm-openai" {
			if tagged, ok := named.(reference.Tagged); ok {
				tag := tagged.Tag()

				url := fmt.Sprintf("https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/%s/", tag)
				client := &http.Client{Timeout: 5 * time.Second}
				resp, err := client.Get(url)
				if err == nil {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusNotFound {
						return Settings{}, newAppError(http.StatusBadRequest, "Docker image tag %q does not exist on Docker Hub for vllm/vllm-openai", tag)
					}
				}
			}
		}
	}
	if req.ServerBindIP != nil {
		bindIP := strings.TrimSpace(*req.ServerBindIP)
		if bindIP == "" {
			bindIP = "0.0.0.0"
		}
		addr, err := api.ValidateLocalBindIP(bindIP)
		if err != nil {
			return Settings{}, newAppError(http.StatusBadRequest, "%s", err.Error())
		}
		normalized := addr.String()
		req.ServerBindIP = &normalized
	}

	inferenceAPIKey, err := normalizeVLLMAPIKey(req.InferenceAPIKey)
	if err != nil {
		return Settings{}, err
	}
	if req.ClearInferenceAPIKey && inferenceAPIKey != "" {
		return Settings{}, newAppError(http.StatusBadRequest, "Cannot set and clear the Inference API key in the same request")
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
		HFCache:                 hfCache,
		HFToken:                 current.HFToken,
		VLLMAPIKey:              current.VLLMAPIKey,
		InferenceAPIKey:         current.InferenceAPIKey,
		EnableCUDACompatibility: current.EnableCUDACompatibility,
		HFHubOffline:            current.HFHubOffline,
		RoutingTable:            current.RoutingTable,
		ServerAPIKeys:           current.ServerAPIKeys,
		ProxyUsers:              current.ProxyUsers,
		Heartbeat:               current.Heartbeat,
		VLLMImage:               current.VLLMImage,
		ServerBindIP:            current.ServerBindIP,
	}
	if req.VLLMImage != nil {
		s.VLLMImage = *req.VLLMImage
	}
	if req.ServerBindIP != nil {
		s.ServerBindIP = *req.ServerBindIP
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
	if req.ClearInferenceAPIKey {
		s.InferenceAPIKey = ""
	} else if inferenceAPIKey != "" {
		s.InferenceAPIKey = inferenceAPIKey
	}
	if req.EnableCUDACompatibility != nil {
		s.EnableCUDACompatibility = *req.EnableCUDACompatibility
	}
	if req.HFHubOffline != nil {
		s.HFHubOffline = *req.HFHubOffline
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

const (
	proxyAPIKeyPrefix = "vllm_"
	proxyAPIKeyChars  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	maxProxyUsername  = 64
)

func normalizeProxyUsersForStorage(users []ProxyUser) []ProxyUser {
	if len(users) == 0 {
		return nil
	}
	normalized := make([]ProxyUser, 0, len(users))
	seen := make(map[string]struct{}, len(users))
	for _, user := range users {
		username := strings.TrimSpace(user.Username)
		apiKey := strings.TrimSpace(user.APIKey)
		if username == "" || apiKey == "" {
			continue
		}
		key := strings.ToLower(username)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		if user.RequestCount < 0 {
			user.RequestCount = 0
		}
		user.Username = username
		user.APIKey = apiKey
		normalized = append(normalized, user)
	}
	sort.Slice(normalized, func(i, j int) bool {
		return strings.ToLower(normalized[i].Username) < strings.ToLower(normalized[j].Username)
	})
	return normalized
}

func validateProxyUsername(username string) (string, error) {
	username = strings.TrimSpace(username)
	if username == "" {
		return "", newAppError(http.StatusBadRequest, "username is required")
	}
	if len(username) > maxProxyUsername {
		return "", newAppError(http.StatusBadRequest, "username must be %d characters or fewer", maxProxyUsername)
	}
	for i, r := range username {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			continue
		}
		if i > 0 && (r == '_' || r == '-' || r == '.' || r == '@') {
			continue
		}
		return "", newAppError(http.StatusBadRequest, "username may contain only letters, numbers, underscore, dash, dot, or @ and must start with a letter or number")
	}
	return username, nil
}

func generateProxyAPIKey() (string, error) {
	var bytes [40]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return "", err
	}
	suffix := make([]byte, len(bytes))
	for i, b := range bytes {
		suffix[i] = proxyAPIKeyChars[int(b)%len(proxyAPIKeyChars)]
	}
	return proxyAPIKeyPrefix + string(suffix), nil
}

func proxyAPIKeyPreview(apiKey string) string {
	if len(apiKey) <= len(proxyAPIKeyPrefix)+8 {
		return apiKey
	}
	return apiKey[:len(proxyAPIKeyPrefix)+4] + "..." + apiKey[len(apiKey)-4:]
}

func listProxyUsers() []ProxyUserStatus {
	s := getSettings()
	users := make([]ProxyUserStatus, 0, len(s.ProxyUsers))
	for _, user := range s.ProxyUsers {
		users = append(users, ProxyUserStatus{
			Username:      user.Username,
			APIKeyPreview: proxyAPIKeyPreview(user.APIKey),
			RequestCount:  user.RequestCount,
		})
	}
	return users
}

func createProxyUser(username string) (ProxyUser, error) {
	username, err := validateProxyUsername(username)
	if err != nil {
		return ProxyUser{}, err
	}
	apiKey, err := generateProxyAPIKey()
	if err != nil {
		return ProxyUser{}, newAppError(http.StatusInternalServerError, "failed to generate API key: %v", err)
	}

	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	for _, user := range settingsCache.ProxyUsers {
		if strings.EqualFold(user.Username, username) {
			return ProxyUser{}, newAppError(http.StatusConflict, "proxy user %q already exists", username)
		}
	}
	user := ProxyUser{Username: username, APIKey: apiKey}
	settingsCache.ProxyUsers = append(settingsCache.ProxyUsers, user)
	if err := saveSettingsLocked(settingsCache); err != nil {
		return ProxyUser{}, newAppError(http.StatusInternalServerError, "failed to save proxy user: %v", err)
	}
	return user, nil
}

func revealProxyUserKey(username string) (string, error) {
	username = strings.TrimSpace(username)
	if username == "" {
		return "", newAppError(http.StatusBadRequest, "username is required")
	}
	s := getSettings()
	for _, user := range s.ProxyUsers {
		if strings.EqualFold(user.Username, username) {
			return user.APIKey, nil
		}
	}
	return "", newAppError(http.StatusNotFound, "proxy user %q not found", username)
}

func revokeProxyUser(username string) error {
	username = strings.TrimSpace(username)
	if username == "" {
		return newAppError(http.StatusBadRequest, "username is required")
	}
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	next := settingsCache.ProxyUsers[:0]
	removed := false
	for _, user := range settingsCache.ProxyUsers {
		if strings.EqualFold(user.Username, username) {
			removed = true
			continue
		}
		next = append(next, user)
	}
	if !removed {
		return newAppError(http.StatusNotFound, "proxy user %q not found", username)
	}
	settingsCache.ProxyUsers = next
	if err := saveSettingsLocked(settingsCache); err != nil {
		return newAppError(http.StatusInternalServerError, "failed to revoke proxy user: %v", err)
	}
	return nil
}

func authenticateProxyUser(authHeader string, users []ProxyUser) (string, bool) {
	if len(users) == 0 {
		return "", false
	}
	const bearerPrefix = "Bearer "
	if !strings.HasPrefix(authHeader, bearerPrefix) {
		return "", false
	}
	token := strings.TrimSpace(strings.TrimPrefix(authHeader, bearerPrefix))
	if token == "" {
		return "", false
	}
	for _, user := range users {
		if subtle.ConstantTimeCompare([]byte(token), []byte(user.APIKey)) == 1 {
			return user.Username, true
		}
	}
	return "", false
}

func incrementProxyUserRequestCount(username string) {
	if username == "" {
		return
	}
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	for i := range settingsCache.ProxyUsers {
		if strings.EqualFold(settingsCache.ProxyUsers[i].Username, username) {
			settingsCache.ProxyUsers[i].RequestCount++
			_ = saveSettingsLocked(settingsCache)
			return
		}
	}
}

// sortedServerAPIKeyNames returns the keys in deterministic order for
// display. We only return container names, not the key values, so the
// dashboard can show a count/list without leaking secrets.
func sortedServerAPIKeyNames(keys map[string]string) []string {
	if len(keys) == 0 {
		return nil
	}
	names := make([]string, 0, len(keys))
	for name := range keys {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// upsertRoute adds or updates model→port in the persisted routing table.
func upsertRoute(model string, port int) {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	settingsCache.RoutingTable[model] = port
	_ = saveSettingsLocked(settingsCache)
}

// removeRouteByPort removes any routing entry whose value equals port.
func removeRouteByPort(port int) {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	for model, p := range settingsCache.RoutingTable {
		if p == port {
			delete(settingsCache.RoutingTable, model)
		}
	}
	_ = saveSettingsLocked(settingsCache)
}

// upsertServerAPIKey records the API key for a specific container (named
// vllm-<short>-<port>). Empty values are rejected by the caller; the helper
// only mutates state. Existing entries for the same container are overwritten.
func upsertServerAPIKey(containerName string, apiKey string) {
	if apiKey == "" {
		return
	}
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	if settingsCache.ServerAPIKeys == nil {
		settingsCache.ServerAPIKeys = map[string]string{}
	}
	settingsCache.ServerAPIKeys[containerName] = apiKey
	_ = saveSettingsLocked(settingsCache)
}

// removeServerAPIKey drops any per-server API key whose container name matches.
// The map can stay with empty entries, but we keep it tidy so settings.json
// doesn't accumulate stale keys.
func removeServerAPIKey(containerName string) bool {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	if settingsCache.ServerAPIKeys == nil {
		return false
	}
	_, existed := settingsCache.ServerAPIKeys[containerName]
	delete(settingsCache.ServerAPIKeys, containerName)
	_ = saveSettingsLocked(settingsCache)
	return existed
}

// removeServerAPIKeyByPort walks the per-server key map and removes the entry
// whose container name ends with the given port suffix. This mirrors
// removeRouteByPort so callers don't have to know the container name.
func removeServerAPIKeyByPort(port int) []string {
	suffix := fmt.Sprintf("-%d", port)
	removed := []string{}
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	if settingsCache.ServerAPIKeys == nil {
		return removed
	}
	for name := range settingsCache.ServerAPIKeys {
		if strings.HasSuffix(name, suffix) {
			delete(settingsCache.ServerAPIKeys, name)
			removed = append(removed, name)
		}
	}
	_ = saveSettingsLocked(settingsCache)
	sort.Strings(removed)
	return removed
}

func cleanupModelStateForPort(port int) []string {
	removeRouteByPort(port)
	return removeServerAPIKeyByPort(port)
}

func cleanupModelStateForActiveModel(model api.ActiveModel) []string {
	removed := []string{}
	if port, err := strconv.Atoi(model.Port); err == nil {
		removed = append(removed, cleanupModelStateForPort(port)...)
	}
	if strings.TrimSpace(model.ContainerName) != "" {
		if removeServerAPIKey(model.ContainerName) {
			removed = append(removed, model.ContainerName)
		}
	}
	sort.Strings(removed)
	return removed
}

func cleanupModelStateForContainerID(containerID string) {
	model, ok, err := api.GetManagedModelByContainerID(containerID)
	if err != nil {
		log.Printf("[SETTINGS] warning: failed to resolve managed container %s for state cleanup: %v", containerID, err)
		return
	}
	if !ok {
		return
	}
	cleanupModelStateForActiveModel(model)
}

func parseActiveModelPort(model api.ActiveModel) (int, bool) {
	if model.State != "running" || model.Port == "unknown" {
		return 0, false
	}
	port, err := strconv.Atoi(model.Port)
	if err != nil || port < 1 || port > 65535 {
		return 0, false
	}
	return port, true
}

func routingModelName(model, servedModelName string) string {
	if servedModelName = strings.TrimSpace(servedModelName); servedModelName != "" {
		return servedModelName
	}
	return strings.TrimSpace(model)
}

func buildRoutingTableFromActiveModels(models []api.ActiveModel) map[string]int {
	type route struct {
		model string
		key   string
		port  int
	}
	routes := make([]route, 0, len(models))
	for _, model := range models {
		port, ok := parseActiveModelPort(model)
		if !ok {
			continue
		}
		modelName := routingModelName(model.Model, model.ServedModelName)
		key := api.NormalizeModelKeyForMatch(modelName)
		if key == "" {
			continue
		}
		routes = append(routes, route{model: modelName, key: key, port: port})
	}
	sort.Slice(routes, func(i, j int) bool {
		if routes[i].key == routes[j].key {
			return routes[i].port < routes[j].port
		}
		return routes[i].key < routes[j].key
	})

	table := make(map[string]int, len(routes))
	seenModels := make(map[string]bool, len(routes))
	seenPorts := make(map[int]string, len(routes))
	for _, route := range routes {
		if seenModels[route.key] {
			log.Printf("[SETTINGS] skipping duplicate live route for model %q on port %d", route.model, route.port)
			continue
		}
		if existing := seenPorts[route.port]; existing != "" {
			log.Printf("[SETTINGS] skipping duplicate live route for port %d: %q conflicts with %q", route.port, route.model, existing)
			continue
		}
		seenModels[route.key] = true
		seenPorts[route.port] = route.model
		table[route.model] = route.port
	}
	return table
}

func replaceRoutingTable(table map[string]int) error {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	settingsCache.RoutingTable = cloneRoutingTable(table)
	return saveSettingsLocked(settingsCache)
}

// reconcileRoutingTable queries Docker for running vLLM containers and rebuilds
// the routing table from that live set. This picks up already-running models
// after a restart and prunes stale entries for stopped or removed containers.
func reconcileRoutingTable(exposure string) {
	settings := getSettings()
	models, err := api.GetActiveModels(settings.VLLMAPIKey, settings.HFCache)
	if err != nil {
		log.Printf("[SETTINGS] failed to reconcile routing table: %v", err)
		return
	}
	if strings.TrimSpace(exposure) != "" {
		filtered := make([]api.ActiveModel, 0, len(models))
		for _, model := range models {
			if model.Exposure == exposure && model.LaunchSpec != nil {
				filtered = append(filtered, model)
			}
		}
		models = filtered
	}
	if err := replaceRoutingTable(buildRoutingTableFromActiveModels(models)); err != nil {
		log.Printf("[SETTINGS] failed to persist reconciled routing table: %v", err)
	}
}

// writeHeartbeat updates only the heartbeat field on the persisted settings
// without disturbing the rest of the configuration. It is safe to call from a
// background goroutine on a regular interval.
func writeHeartbeat(pid int, seen time.Time) error {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	refreshSettingsCacheFromDiskIfPresentLocked()
	if settingsCache.Heartbeat == nil {
		settingsCache.Heartbeat = &Heartbeat{}
	}
	settingsCache.Heartbeat.PID = pid
	settingsCache.Heartbeat.LastSeen = seen
	data, err := json.Marshal(settingsCache)
	if err != nil {
		return err
	}
	path := settingsFilePath()
	if err := ensureRuntimeDir(path, 0o750); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		return err
	}
	return nil
}

// clearHeartbeat removes the heartbeat field from the persisted settings. Used
// on graceful dashboard shutdown so the CLI does not see a stale entry.
func clearHeartbeat() {
	settingsCacheMu.Lock()
	defer settingsCacheMu.Unlock()
	if !settingsCacheLoaded {
		return
	}
	if settingsCache.Heartbeat == nil {
		return
	}
	settingsCache.Heartbeat = nil
	data, err := json.Marshal(settingsCache)
	if err != nil {
		log.Printf("[SETTINGS] failed to marshal settings while clearing heartbeat: %v", err)
		return
	}
	path := settingsFilePath()
	if err := ensureRuntimeDir(path, 0o750); err != nil {
		log.Printf("[SETTINGS] failed to prepare settings directory while clearing heartbeat: %v", err)
		return
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		log.Printf("[SETTINGS] failed to write settings while clearing heartbeat: %v", err)
		return
	}
}

const heartbeatFreshnessWindow = 15 * time.Second

// dashboardIsLive reports whether the dashboard appears to be running based on
// a recent heartbeat with a live PID.
func dashboardIsLive() bool {
	s := getSettings()
	if s.Heartbeat == nil {
		return false
	}
	if time.Since(s.Heartbeat.LastSeen) > heartbeatFreshnessWindow {
		return false
	}
	if s.Heartbeat.PID <= 0 {
		return false
	}
	return pidAlive(s.Heartbeat.PID)
}

func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	// Signal 0 does not deliver a signal but returns an error if the process
	// does not exist. On Unix, this is a reliable liveness probe for the
	// current PID; zombie reuse is unlikely inside the freshness window.
	if err := proc.Signal(syscall.Signal(0)); err != nil {
		return false
	}
	return true
}
