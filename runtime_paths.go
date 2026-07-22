package main

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	configDirEnv = "VLLM_MANAGER_CONFIG_DIR"
	stateDirEnv  = "VLLM_MANAGER_STATE_DIR"
)

// These defaults are overridden with -ldflags for packaged builds. Portable
// and source builds intentionally keep all runtime files relative to the
// current working directory.
var (
	defaultConfigDir = "."
	defaultStateDir  = "."
)

func runtimeConfigDir() string {
	return runtimeDir(configDirEnv, defaultConfigDir)
}

func runtimeStateDir() string {
	return runtimeDir(stateDirEnv, defaultStateDir)
}

func runtimeDir(envName string, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(envName)); value != "" {
		return filepath.Clean(value)
	}
	if strings.TrimSpace(fallback) == "" {
		return "."
	}
	return filepath.Clean(fallback)
}

func settingsFilePath() string {
	return filepath.Join(runtimeStateDir(), "settings.json")
}

func auditLogFilePath() string {
	return filepath.Join(runtimeStateDir(), "audit.log")
}

func dashboardCertPaths() (string, string) {
	certDir := filepath.Join(runtimeStateDir(), "certs")
	return filepath.Join(certDir, "server.crt"), filepath.Join(certDir, "server.key")
}

func serviceEnvironmentFilePath() string {
	return filepath.Join(runtimeConfigDir(), "service.env")
}

func defaultMonitoringTargetsPath() string {
	return filepath.Join(runtimeStateDir(), "monitoring", "vllm-targets.yml")
}

func ensureRuntimeDir(path string, mode os.FileMode) error {
	return os.MkdirAll(filepath.Dir(path), mode)
}
