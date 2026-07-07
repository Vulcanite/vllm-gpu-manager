package main

import (
	"embed"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

// The frontend/dist directory is cleaned by build.sh before npm run build,
// so this embed directive only picks up the production build artifacts.
// Do not manually place files in frontend/dist — they will be embedded
// and may include stale or unwanted content (e.g. node_modules/.cache).
//
//go:embed all:frontend/dist
var frontendDist embed.FS

const defaultDashboardPort = "9090"

func main() {
	if len(os.Args) == 1 {
		printRootHelp(os.Stdout)
		return
	}

	if os.Args[1] == "-h" || os.Args[1] == "--help" {
		printRootHelp(os.Stdout)
		return
	}

	if shouldRunDashboard(os.Args[1:]) {
		port, proxyEnabled, err := parseDashboardFlags(os.Args[1:], os.Stderr)
		if err != nil {
			os.Exit(2)
		}
		if len(port.extraArgs) != 0 {
			fmt.Fprintf(os.Stderr, "error: dashboard mode does not accept CLI commands: %s\n", strings.Join(port.extraArgs, " "))
			os.Exit(1)
		}
		runDashboard(dashboardOptions{
			Host:         port.host,
			Port:         port.value,
			ProxyEnabled: proxyEnabled,
			Monitoring: monitoringOptions{
				Enabled:     port.monitoringEnabled,
				TargetsFile: port.monitoringTargetsFile,
			},
		})
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

func shouldRunDashboard(args []string) bool {
	return hasDashboardFlag(args)
}

type dashboardFlagValues struct {
	value                 string
	host                  string
	monitoringEnabled     bool
	monitoringTargetsFile string
	extraArgs             []string
}

func parseDashboardFlags(args []string, errOut io.Writer) (dashboardFlagValues, bool, error) {
	fs := flag.NewFlagSet("vllm-manager", flag.ContinueOnError)
	fs.SetOutput(errOut)
	fs.Bool("dashboard", false, "Run the web dashboard")
	port := fs.String("port", defaultDashboardPort, "Port to run the dashboard on")
	host := fs.String("host", "0.0.0.0", "Host IP address to bind the dashboard on")
	noProxy := fs.Bool("no-proxy", false, "disable the OpenAI-compatible proxy gateway on /v1/")
	monitoring := fs.Bool("monitoring", false, "sync running vLLM servers into the Prometheus file-SD targets file")
	monitoringTargetsFile := fs.String("monitoring-targets-file", defaultMonitoringTargetsFile, "Prometheus file-SD targets file for vLLM metrics")
	fs.Usage = func() { printRootHelp(errOut) }
	if err := fs.Parse(args); err != nil {
		return dashboardFlagValues{}, false, err
	}
	return dashboardFlagValues{
		value:                 *port,
		host:                  *host,
		monitoringEnabled:     *monitoring,
		monitoringTargetsFile: *monitoringTargetsFile,
		extraArgs:             fs.Args(),
	}, !*noProxy, nil
}
