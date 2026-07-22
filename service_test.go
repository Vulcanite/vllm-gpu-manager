package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestDashboardDefaultsFromEnv(t *testing.T) {
	t.Setenv(serviceHostEnv, "0.0.0.0")
	t.Setenv(servicePortEnv, "9443")
	t.Setenv(serviceHTTPSEnv, "yes")
	t.Setenv(serviceNoProxyEnv, "true")
	t.Setenv(serviceMonitoringEnv, "1")
	t.Setenv(serviceMonitoringTargetsEnv, "/tmp/targets.yml")

	got := dashboardDefaultsFromEnv()
	want := dashboardEnvDefaults{
		Host: "0.0.0.0", Port: "9443", HTTPS: true, NoProxy: true,
		Monitoring: true, MonitoringTargets: "/tmp/targets.yml",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("dashboardDefaultsFromEnv()=%+v, want %+v", got, want)
	}
}

func TestServiceConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(configDirEnv, dir)
	t.Setenv(stateDirEnv, filepath.Join(dir, "state"))
	path := filepath.Join(dir, "service.env")
	want := serviceConfig{
		Host: "127.0.0.1", Port: "9090", HTTPS: true, Proxy: false,
		Monitoring: true, MonitoringTargets: "/var/lib/vllm-manager/monitoring/vllm-targets.yml",
		Password: "test password",
	}
	if err := writeServiceConfig(path, want); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o640 {
		t.Fatalf("service env mode=%o, want 640", got)
	}
	got, err := readServiceConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("readServiceConfig()=%+v, want %+v", got, want)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, configDirEnv+"="+`"`+dir+`"`) || !strings.Contains(text, stateDirEnv+"=") {
		t.Fatalf("service env does not include runtime directories:\n%s", text)
	}
}

func TestValidateServiceConfig(t *testing.T) {
	valid := serviceConfig{Host: "127.0.0.1", Port: "9090", Proxy: true}
	if err := validateServiceConfig(valid); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*serviceConfig)
		match  string
	}{
		{name: "bad host", mutate: func(c *serviceConfig) { c.Host = "hostname" }, match: "invalid dashboard host"},
		{name: "bad port", mutate: func(c *serviceConfig) { c.Port = "70000" }, match: "between 1 and 65535"},
		{name: "missing targets", mutate: func(c *serviceConfig) { c.Monitoring = true }, match: "targets file is required"},
		{name: "multiline password", mutate: func(c *serviceConfig) { c.Password = "one\ntwo" }, match: "control characters"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := valid
			tt.mutate(&config)
			if err := validateServiceConfig(config); err == nil || !strings.Contains(err.Error(), tt.match) {
				t.Fatalf("validateServiceConfig() error=%v, want containing %q", err, tt.match)
			}
		})
	}
}

func TestReadPasswordFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "password")
	if err := os.WriteFile(path, []byte("secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got, err := readPasswordFile(path); err != nil || got != "secret" {
		t.Fatalf("readPasswordFile()=%q, %v", got, err)
	}
}

func TestConfigureServiceRestartsOnlyActiveService(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(configDirEnv, dir)
	t.Setenv(stateDirEnv, filepath.Join(dir, "state"))

	previous := serviceSystemctl
	t.Cleanup(func() { serviceSystemctl = previous })
	var calls [][]string
	serviceSystemctl = func(args ...string) error {
		calls = append(calls, append([]string(nil), args...))
		return nil
	}
	cmd := &cobra.Command{}
	cmd.SetOut(&bytes.Buffer{})
	cmd.Flags().String("host", "", "")
	cmd.Flags().String("port", "", "")
	cmd.Flags().Bool("https", false, "")
	cmd.Flags().Bool("proxy", true, "")
	cmd.Flags().Bool("monitoring", false, "")
	cmd.Flags().String("monitoring-targets-file", "", "")
	cmd.Flags().String("password-file", "", "")
	cmd.Flags().Bool("clear-password", false, "")
	if err := cmd.Flags().Set("port", "9443"); err != nil {
		t.Fatal(err)
	}
	if err := configureService(cmd, serviceConfigureOptions{Port: "9443"}); err != nil {
		t.Fatal(err)
	}
	want := [][]string{{"is-active", "--quiet", "vllm-manager.service"}, {"restart", "vllm-manager.service"}}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("systemctl calls=%v, want %v", calls, want)
	}

	calls = nil
	serviceSystemctl = func(args ...string) error {
		calls = append(calls, append([]string(nil), args...))
		return errors.New("inactive")
	}
	if err := configureService(cmd, serviceConfigureOptions{Port: "9443"}); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0][0] != "is-active" {
		t.Fatalf("inactive service calls=%v", calls)
	}
}

func TestServiceConfigureRequiresRoot(t *testing.T) {
	previous := currentEUID
	t.Cleanup(func() { currentEUID = previous })
	currentEUID = func() int { return 1000 }
	cmd := newServiceConfigureCommand()
	cmd.SetArgs([]string{"--port", "9090"})
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "requires root") {
		t.Fatalf("Execute() error=%v", err)
	}
}
