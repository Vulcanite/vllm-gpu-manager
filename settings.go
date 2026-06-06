package main

import (
	"bufio"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"unicode"

	"vllm-manager/api"
)

const settingsFile = "settings.json"

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

var (
	settingsCacheMu     sync.Mutex
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
