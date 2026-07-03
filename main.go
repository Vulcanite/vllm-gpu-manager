package main

import (
	"embed"
	"flag"
	"fmt"
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
		fs := flag.NewFlagSet("vllm-manager", flag.ExitOnError)
		fs.Bool("dashboard", false, "Run the web dashboard")
		port := fs.String("port", defaultDashboardPort, "Port to run the dashboard on")
		fs.Bool("proxy", true, "Enable the OpenAI-compatible proxy gateway on /v1/ (always enabled in dashboard mode)")
		fs.Usage = func() { printRootHelp(os.Stderr) }
		if err := fs.Parse(os.Args[1:]); err != nil {
			os.Exit(2)
		}
		if fs.NArg() != 0 {
			fmt.Fprintf(os.Stderr, "error: dashboard mode does not accept CLI commands: %s\n", strings.Join(fs.Args(), " "))
			os.Exit(1)
		}
		runDashboard(*port, true)
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
