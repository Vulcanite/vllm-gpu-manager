package main

import (
	"path/filepath"
	"testing"
)

func TestRuntimePathsDefaultToWorkingDirectory(t *testing.T) {
	t.Setenv(configDirEnv, "")
	t.Setenv(stateDirEnv, "")

	if got := runtimeConfigDir(); got != "." {
		t.Fatalf("runtimeConfigDir()=%q, want .", got)
	}
	if got := settingsFilePath(); got != "settings.json" {
		t.Fatalf("settingsFilePath()=%q, want settings.json", got)
	}
	if got := defaultMonitoringTargetsPath(); got != filepath.Join("monitoring", "vllm-targets.yml") {
		t.Fatalf("defaultMonitoringTargetsPath()=%q", got)
	}
}

func TestRuntimePathsHonorEnvironmentOverrides(t *testing.T) {
	configDir := filepath.Join(t.TempDir(), "config")
	stateDir := filepath.Join(t.TempDir(), "state")
	t.Setenv(configDirEnv, configDir)
	t.Setenv(stateDirEnv, stateDir)

	if got := serviceEnvironmentFilePath(); got != filepath.Join(configDir, "service.env") {
		t.Fatalf("serviceEnvironmentFilePath()=%q", got)
	}
	if got := settingsFilePath(); got != filepath.Join(stateDir, "settings.json") {
		t.Fatalf("settingsFilePath()=%q", got)
	}
	if got := auditLogFilePath(); got != filepath.Join(stateDir, "audit.log") {
		t.Fatalf("auditLogFilePath()=%q", got)
	}
	certFile, keyFile := dashboardCertPaths()
	if certFile != filepath.Join(stateDir, "certs", "server.crt") || keyFile != filepath.Join(stateDir, "certs", "server.key") {
		t.Fatalf("dashboardCertPaths()=(%q,%q)", certFile, keyFile)
	}
}

func TestDisplayVersionDefaultsAndTrims(t *testing.T) {
	previous := version
	t.Cleanup(func() { version = previous })

	version = "  "
	if got := displayVersion(); got != "dev" {
		t.Fatalf("displayVersion()=%q, want dev", got)
	}
	version = " v1.2.3 "
	if got := displayVersion(); got != "v1.2.3" {
		t.Fatalf("displayVersion()=%q, want v1.2.3", got)
	}
}
