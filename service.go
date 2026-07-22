package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net/netip"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"unicode"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"vllm-manager/api"
)

const (
	serviceHostEnv              = "VLLM_MANAGER_HOST"
	servicePortEnv              = "VLLM_MANAGER_PORT"
	serviceHTTPSEnv             = "VLLM_MANAGER_HTTPS"
	serviceNoProxyEnv           = "VLLM_MANAGER_NO_PROXY"
	serviceMonitoringEnv        = "VLLM_MANAGER_MONITORING"
	serviceMonitoringTargetsEnv = "VLLM_MANAGER_MONITORING_TARGETS_FILE"
	servicePasswordEnv          = "VLLM_MANAGER_PASSWORD"
)

type serviceConfig struct {
	Host              string
	Port              string
	HTTPS             bool
	Proxy             bool
	Monitoring        bool
	MonitoringTargets string
	Password          string
}

type serviceConfigureOptions struct {
	Host              string
	Port              string
	HTTPS             bool
	Proxy             bool
	Monitoring        bool
	MonitoringTargets string
	PasswordFile      string
	ClearPassword     bool
	NoRestart         bool
}

type dashboardEnvDefaults struct {
	Host              string
	Port              string
	HTTPS             bool
	NoProxy           bool
	Monitoring        bool
	MonitoringTargets string
}

var serviceSystemctl = func(args ...string) error {
	cmd := exec.Command("systemctl", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

var currentEUID = os.Geteuid

func dashboardDefaultsFromEnv() dashboardEnvDefaults {
	host := strings.TrimSpace(os.Getenv(serviceHostEnv))
	if host == "" {
		host = "127.0.0.1"
	}
	port := strings.TrimSpace(os.Getenv(servicePortEnv))
	if port == "" {
		port = defaultDashboardPort
	}
	targets := strings.TrimSpace(os.Getenv(serviceMonitoringTargetsEnv))
	if targets == "" {
		targets = defaultMonitoringTargetsPath()
	}
	return dashboardEnvDefaults{
		Host:              host,
		Port:              port,
		HTTPS:             environmentBool(serviceHTTPSEnv),
		NoProxy:           environmentBool(serviceNoProxyEnv),
		Monitoring:        environmentBool(serviceMonitoringEnv),
		MonitoringTargets: targets,
	}
}

func environmentBool(name string) bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func newServiceCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "service",
		Short: "Configure the packaged systemd service",
		RunE:  showCommandHelp,
	}
	cmd.AddCommand(newServiceConfigureCommand())
	return cmd
}

func newServiceConfigureCommand() *cobra.Command {
	opts := serviceConfigureOptions{}
	cmd := &cobra.Command{
		Use:   "configure",
		Short: "Update systemd dashboard parameters",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if currentEUID() != 0 {
				return errors.New("service configuration requires root; rerun with sudo")
			}
			return configureService(cmd, opts)
		},
	}
	cmd.Flags().StringVar(&opts.Host, "host", "", "dashboard bind IP")
	cmd.Flags().StringVar(&opts.Port, "port", "", "dashboard port")
	cmd.Flags().BoolVar(&opts.HTTPS, "https", false, "enable or disable HTTPS")
	cmd.Flags().BoolVar(&opts.Proxy, "proxy", true, "enable or disable the /v1 proxy")
	cmd.Flags().BoolVar(&opts.Monitoring, "monitoring", false, "enable or disable Prometheus target sync")
	cmd.Flags().StringVar(&opts.MonitoringTargets, "monitoring-targets-file", "", "Prometheus file-SD target path")
	cmd.Flags().StringVar(&opts.PasswordFile, "password-file", "", "read dashboard password from a file")
	cmd.Flags().BoolVar(&opts.ClearPassword, "clear-password", false, "disable dashboard password authentication")
	cmd.Flags().BoolVar(&opts.NoRestart, "no-restart", false, "do not restart an active service")
	return cmd
}

func configureService(cmd *cobra.Command, opts serviceConfigureOptions) error {
	path := serviceEnvironmentFilePath()
	config, err := readServiceConfig(path)
	if err != nil {
		return err
	}
	flags := cmd.Flags()
	anyChanged := flags.Changed("host") || flags.Changed("port") || flags.Changed("https") || flags.Changed("proxy") ||
		flags.Changed("monitoring") || flags.Changed("monitoring-targets-file") || flags.Changed("password-file") || flags.Changed("clear-password")
	if !anyChanged {
		if !term.IsTerminal(int(os.Stdin.Fd())) {
			return errors.New("no service settings provided; pass flags or run interactively")
		}
		config, err = promptServiceConfig(cmd.InOrStdin(), cmd.OutOrStdout(), config)
		if err != nil {
			return err
		}
	} else {
		if flags.Changed("host") {
			config.Host = opts.Host
		}
		if flags.Changed("port") {
			config.Port = opts.Port
		}
		if flags.Changed("https") {
			config.HTTPS = opts.HTTPS
		}
		if flags.Changed("proxy") {
			config.Proxy = opts.Proxy
		}
		if flags.Changed("monitoring") {
			config.Monitoring = opts.Monitoring
		}
		if flags.Changed("monitoring-targets-file") {
			config.MonitoringTargets = opts.MonitoringTargets
		}
		if opts.ClearPassword && strings.TrimSpace(opts.PasswordFile) != "" {
			return errors.New("--clear-password and --password-file cannot be used together")
		}
		if opts.ClearPassword {
			config.Password = ""
		}
		if strings.TrimSpace(opts.PasswordFile) != "" {
			password, err := readPasswordFile(opts.PasswordFile)
			if err != nil {
				return err
			}
			config.Password = password
		}
	}
	if err := validateServiceConfig(config); err != nil {
		return err
	}
	if err := writeServiceConfig(path, config); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(cmd.OutOrStdout(), "Service configuration written to %s\n", path); err != nil {
		return err
	}
	if opts.NoRestart {
		return nil
	}
	if err := serviceSystemctl("is-active", "--quiet", "vllm-manager.service"); err != nil {
		_, err = fmt.Fprintln(cmd.OutOrStdout(), "Service is not active; start it with `sudo systemctl enable --now vllm-manager`.")
		return err
	}
	if err := serviceSystemctl("restart", "vllm-manager.service"); err != nil {
		return fmt.Errorf("configuration saved but service restart failed: %v", err)
	}
	_, err = fmt.Fprintln(cmd.OutOrStdout(), "Active vllm-manager service restarted.")
	return err
}

func validateServiceConfig(config serviceConfig) error {
	host := strings.TrimSpace(config.Host)
	if host == "" {
		return errors.New("dashboard host is required")
	}
	addr, err := netip.ParseAddr(host)
	if err != nil {
		return fmt.Errorf("invalid dashboard host: %v", err)
	}
	if !addr.IsUnspecified() {
		if _, err := api.ValidateLocalBindIP(host); err != nil {
			return fmt.Errorf("invalid dashboard host: %v", err)
		}
	}
	port, err := strconv.Atoi(strings.TrimSpace(config.Port))
	if err != nil || port < 1 || port > 65535 {
		return errors.New("dashboard port must be between 1 and 65535")
	}
	if config.Monitoring && strings.TrimSpace(config.MonitoringTargets) == "" {
		return errors.New("monitoring targets file is required when monitoring is enabled")
	}
	if strings.IndexFunc(config.Password, unicode.IsControl) >= 0 {
		return errors.New("dashboard password cannot contain control characters")
	}
	return nil
}

func readServiceConfig(path string) (serviceConfig, error) {
	defaults := dashboardDefaultsFromEnv()
	config := serviceConfig{
		Host:              defaults.Host,
		Port:              defaults.Port,
		HTTPS:             defaults.HTTPS,
		Proxy:             !defaults.NoProxy,
		Monitoring:        defaults.Monitoring,
		MonitoringTargets: defaults.MonitoringTargets,
		Password:          os.Getenv(servicePasswordEnv),
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return config, nil
		}
		return serviceConfig{}, err
	}
	values := map[string]string{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if unquoted, err := strconv.Unquote(value); err == nil {
			value = unquoted
		}
		values[strings.TrimSpace(key)] = value
	}
	if value := values[serviceHostEnv]; value != "" {
		config.Host = value
	}
	if value := values[servicePortEnv]; value != "" {
		config.Port = value
	}
	config.HTTPS = configBool(values, serviceHTTPSEnv, config.HTTPS)
	config.Proxy = !configBool(values, serviceNoProxyEnv, !config.Proxy)
	config.Monitoring = configBool(values, serviceMonitoringEnv, config.Monitoring)
	if value := values[serviceMonitoringTargetsEnv]; value != "" {
		config.MonitoringTargets = value
	}
	if value, ok := values[servicePasswordEnv]; ok {
		config.Password = value
	}
	return config, nil
}

func configBool(values map[string]string, key string, fallback bool) bool {
	value, ok := values[key]
	if !ok {
		return fallback
	}
	parsed, err := strconv.ParseBool(strings.TrimSpace(value))
	if err != nil {
		return fallback
	}
	return parsed
}

func writeServiceConfig(path string, config serviceConfig) error {
	if err := ensureRuntimeDir(path, 0o750); err != nil {
		return err
	}
	contents := strings.Join([]string{
		"# Managed by `vllm-manager service configure`.",
		serviceHostEnv + "=" + strconv.Quote(config.Host),
		servicePortEnv + "=" + strconv.Quote(config.Port),
		serviceHTTPSEnv + "=" + strconv.FormatBool(config.HTTPS),
		serviceNoProxyEnv + "=" + strconv.FormatBool(!config.Proxy),
		serviceMonitoringEnv + "=" + strconv.FormatBool(config.Monitoring),
		serviceMonitoringTargetsEnv + "=" + strconv.Quote(config.MonitoringTargets),
		servicePasswordEnv + "=" + strconv.Quote(config.Password),
		configDirEnv + "=" + strconv.Quote(runtimeConfigDir()),
		stateDirEnv + "=" + strconv.Quote(runtimeStateDir()),
		"",
	}, "\n")
	tmp, err := os.CreateTemp(filepath.Dir(path), ".service.env-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o640); err != nil {
		tmp.Close()
		return err
	}
	if info, err := os.Stat(path); err == nil {
		if stat, ok := info.Sys().(*syscall.Stat_t); ok {
			if err := tmp.Chown(int(stat.Uid), int(stat.Gid)); err != nil {
				tmp.Close()
				return err
			}
		}
	}
	if _, err := io.WriteString(tmp, contents); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func readPasswordFile(path string) (string, error) {
	data, err := os.ReadFile(strings.TrimSpace(path))
	if err != nil {
		return "", fmt.Errorf("failed to read password file: %v", err)
	}
	password := strings.TrimSuffix(strings.TrimSuffix(string(data), "\n"), "\r")
	if password == "" {
		return "", errors.New("password file is empty")
	}
	if strings.ContainsAny(password, "\r\n\x00") {
		return "", errors.New("password file must contain exactly one line")
	}
	return password, nil
}

func promptServiceConfig(in io.Reader, out io.Writer, config serviceConfig) (serviceConfig, error) {
	reader := bufio.NewReader(in)
	var err error
	if config.Host, err = promptValue(reader, out, "Dashboard host", config.Host); err != nil {
		return config, err
	}
	if config.Port, err = promptValue(reader, out, "Dashboard port", config.Port); err != nil {
		return config, err
	}
	if config.HTTPS, err = promptBool(reader, out, "Enable HTTPS", config.HTTPS); err != nil {
		return config, err
	}
	if config.Proxy, err = promptBool(reader, out, "Enable proxy gateway", config.Proxy); err != nil {
		return config, err
	}
	if config.Monitoring, err = promptBool(reader, out, "Enable monitoring target sync", config.Monitoring); err != nil {
		return config, err
	}
	if config.Monitoring {
		if config.MonitoringTargets, err = promptValue(reader, out, "Monitoring targets file", config.MonitoringTargets); err != nil {
			return config, err
		}
	}
	if file, ok := in.(*os.File); ok && term.IsTerminal(int(file.Fd())) {
		if _, err := fmt.Fprint(out, "Dashboard password (blank keeps current): "); err != nil {
			return config, err
		}
		password, err := term.ReadPassword(int(file.Fd()))
		fmt.Fprintln(out)
		if err != nil {
			return config, err
		}
		if len(password) > 0 {
			config.Password = string(password)
		}
	}
	return config, nil
}

func promptValue(reader *bufio.Reader, out io.Writer, label string, current string) (string, error) {
	if _, err := fmt.Fprintf(out, "%s [%s]: ", label, current); err != nil {
		return "", err
	}
	value, err := reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return "", err
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return current, nil
	}
	return value, nil
}

func promptBool(reader *bufio.Reader, out io.Writer, label string, current bool) (bool, error) {
	defaultLabel := "y/N"
	if current {
		defaultLabel = "Y/n"
	}
	if _, err := fmt.Fprintf(out, "%s [%s]: ", label, defaultLabel); err != nil {
		return false, err
	}
	value, err := reader.ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return current, nil
	}
	if value == "y" || value == "yes" {
		return true, nil
	}
	if value == "n" || value == "no" {
		return false, nil
	}
	return false, fmt.Errorf("expected yes or no for %s", label)
}
