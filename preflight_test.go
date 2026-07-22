package main

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestCommandRequiresOperationalPreflight(t *testing.T) {
	tests := []struct {
		path        []string
		hasChildren bool
		want        bool
	}{
		{path: []string{"vllm-manager", "gpu", "list"}, want: true},
		{path: []string{"vllm-manager", "model", "active"}, want: true},
		{path: []string{"vllm-manager", "cache", "delete"}, want: true},
		{path: []string{"vllm-manager", "settings", "get"}, want: false},
		{path: []string{"vllm-manager", "doctor"}, want: false},
		{path: []string{"vllm-manager", "version"}, want: false},
		{path: []string{"vllm-manager", "model"}, hasChildren: true, want: false},
	}
	for _, tt := range tests {
		root := &cobra.Command{Use: tt.path[0]}
		parent := root
		for _, part := range tt.path[1:] {
			child := &cobra.Command{Use: part}
			parent.AddCommand(child)
			parent = child
		}
		if tt.hasChildren {
			parent.AddCommand(&cobra.Command{Use: "child"})
		}
		if got := commandRequiresOperationalPreflight(parent); got != tt.want {
			t.Fatalf("commandRequiresOperationalPreflight(%q)=%v, want %v", parent.CommandPath(), got, tt.want)
		}
	}
}

func TestStrictOperationalPreflightReportsRequiredFailures(t *testing.T) {
	previous := runDoctorReport
	t.Cleanup(func() { runDoctorReport = previous })
	runDoctorReport = func() doctorReport {
		return doctorReport{
			OK: false,
			Checks: []doctorCheck{
				{Name: "Docker API", Status: doctorFail, Detail: "unavailable", Require: true},
				{Name: "CUDA toolkit", Status: doctorWarn, Detail: "optional", Require: false},
			},
		}
	}

	err := strictOperationalPreflight()
	if err == nil || !strings.Contains(err.Error(), "Docker API: unavailable") || strings.Contains(err.Error(), "CUDA toolkit") {
		t.Fatalf("strictOperationalPreflight() error=%v", err)
	}
}

func TestStrictOperationalPreflightPassesHealthyReport(t *testing.T) {
	previous := runDoctorReport
	t.Cleanup(func() { runDoctorReport = previous })
	runDoctorReport = func() doctorReport { return doctorReport{OK: true} }
	if err := strictOperationalPreflight(); err != nil {
		t.Fatalf("strictOperationalPreflight() error=%v", err)
	}
}
