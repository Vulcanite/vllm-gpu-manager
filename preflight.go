package main

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

var (
	operationalPreflight = strictOperationalPreflight
	runDoctorReport      = runDoctor
)

func strictOperationalPreflight() error {
	report := runDoctorReport()
	if report.OK {
		return nil
	}
	var failures []string
	for _, check := range report.Checks {
		if check.Require && check.Status == doctorFail {
			failures = append(failures, fmt.Sprintf("%s: %s", check.Name, check.Detail))
		}
	}
	return fmt.Errorf("prerequisite check failed: %s; run `vllm-manager doctor` for details", strings.Join(failures, "; "))
}

func commandRequiresOperationalPreflight(cmd *cobra.Command) bool {
	if cmd == nil || cmd.HasSubCommands() {
		return false
	}
	parts := strings.Fields(cmd.CommandPath())
	if len(parts) < 2 {
		return false
	}
	switch parts[1] {
	case "gpu", "model", "cache":
		return true
	default:
		return false
	}
}
