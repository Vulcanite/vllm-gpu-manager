package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"

	"vllm-manager/api"

	"github.com/spf13/cobra"
)

type doctorStatus string

const (
	doctorPass doctorStatus = "PASS"
	doctorWarn doctorStatus = "WARN"
	doctorFail doctorStatus = "FAIL"
)

type doctorCheck struct {
	Name    string       `json:"name"`
	Status  doctorStatus `json:"status"`
	Detail  string       `json:"detail"`
	Require bool         `json:"required"`
}

type doctorReport struct {
	Checks []doctorCheck `json:"checks"`
	OK     bool          `json:"ok"`
}

func newDoctorCommand(opts *cliOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Check whether this host can run vllm-manager",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			report := runDoctor()
			if opts.JSON {
				if err := printJSON(report); err != nil {
					return err
				}
			} else {
				rows := make([][]string, 0, len(report.Checks))
				for _, check := range report.Checks {
					rows = append(rows, []string{string(check.Status), check.Name, check.Detail})
				}
				printMarkdownTable(os.Stdout, []string{"Status", "Check", "Detail"}, rows)
			}
			if !report.OK {
				return fmt.Errorf("doctor found required failures")
			}
			return nil
		},
	}
}

func runDoctor() doctorReport {
	var checks []doctorCheck
	add := func(name string, status doctorStatus, detail string, required bool) {
		checks = append(checks, doctorCheck{Name: name, Status: status, Detail: detail, Require: required})
	}

	if runtime.GOOS != "linux" {
		add("Operating system", doctorFail, "vllm-manager serving support requires Linux", true)
	} else {
		add("Operating system", doctorPass, "linux", true)
	}

	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		add("nvidia-smi", doctorFail, "nvidia-smi not found in PATH", true)
	} else {
		add("nvidia-smi", doctorPass, "nvidia-smi found", true)
	}

	gpus, err := api.GetGPUInfo()
	if err != nil {
		add("NVIDIA GPUs", doctorFail, err.Error(), true)
	} else if len(gpus) == 0 {
		add("NVIDIA GPUs", doctorFail, "no NVIDIA GPUs detected", true)
	} else {
		detail := fmt.Sprintf("%d GPU(s) detected", len(gpus))
		if gpus[0].ComputeCapability != "" {
			detail += fmt.Sprintf("; first compute capability %s", gpus[0].ComputeCapability)
		}
		add("NVIDIA GPUs", doctorPass, detail, true)
	}

	if _, err := exec.LookPath("docker"); err != nil {
		add("Docker CLI", doctorWarn, "docker CLI not found in PATH; Docker API may still work via DOCKER_HOST", false)
	} else {
		add("Docker CLI", doctorPass, "docker CLI found", false)
	}
	if err := dockerPingForDoctor(); err != nil {
		add("Docker API", doctorFail, err.Error(), true)
	} else {
		add("Docker API", doctorPass, "Docker API ping succeeded", true)
	}
	if detail, ok := dockerNVIDIASignal(); ok {
		add("NVIDIA Container Toolkit", doctorPass, detail, true)
	} else {
		add("NVIDIA Container Toolkit", doctorWarn, "could not confirm Docker NVIDIA runtime/toolkit from docker info", false)
	}

	if _, err := exec.LookPath("nvcc"); err != nil {
		add("CUDA toolkit", doctorWarn, "nvcc not found; host CUDA toolkit is optional for Docker-based serving", false)
	} else {
		add("CUDA toolkit", doctorPass, "nvcc found", false)
	}

	settings := getSettings()
	if stat, err := os.Stat(settings.HFCache); err != nil || !stat.IsDir() {
		add("HF cache path", doctorFail, fmt.Sprintf("%s is not a readable directory", settings.HFCache), true)
	} else {
		add("HF cache path", doctorPass, settings.HFCache, true)
		if err := checkWritable(settings.HFCache); err != nil {
			add("HF cache writable", doctorFail, err.Error(), true)
		} else {
			add("HF cache writable", doctorPass, "cache directory is writable", true)
		}
		if free, err := freeBytes(settings.HFCache); err != nil {
			add("HF cache free space", doctorWarn, err.Error(), false)
		} else {
			add("HF cache free space", doctorPass, api.FormatBytes(free)+" available", false)
		}
	}
	add("vLLM image", doctorPass, settings.VLLMImage, true)
	checkSettingsPermissions(&checks)
	checkDashboardCerts(&checks)

	ok := true
	for _, check := range checks {
		if check.Require && check.Status == doctorFail {
			ok = false
			break
		}
	}
	return doctorReport{Checks: checks, OK: ok}
}

func dockerPingForDoctor() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "docker", "info")
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("docker info failed: %s", strings.TrimSpace(string(out)))
	}
	return nil
}

func dockerNVIDIASignal() (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "docker", "info", "--format", "{{json .Runtimes}}").CombinedOutput()
	if err != nil {
		return "", false
	}
	text := strings.ToLower(string(out))
	if strings.Contains(text, "nvidia") || strings.Contains(text, "containerd") {
		return "docker info exposes NVIDIA/containerd runtime signals", true
	}
	return "", false
}

func checkWritable(dir string) error {
	f, err := os.CreateTemp(dir, ".vllm-manager-doctor-*")
	if err != nil {
		return err
	}
	name := f.Name()
	_ = f.Close()
	return os.Remove(name)
}

func freeBytes(path string) (uint64, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, err
	}
	return stat.Bavail * uint64(stat.Bsize), nil
}

func checkSettingsPermissions(checks *[]doctorCheck) {
	info, err := os.Stat(settingsFile)
	if err != nil {
		if os.IsNotExist(err) {
			*checks = append(*checks, doctorCheck{Name: "settings.json", Status: doctorWarn, Detail: "settings.json has not been created yet", Require: false})
			return
		}
		*checks = append(*checks, doctorCheck{Name: "settings.json", Status: doctorWarn, Detail: err.Error(), Require: false})
		return
	}
	mode := info.Mode().Perm()
	if mode&0o077 != 0 {
		*checks = append(*checks, doctorCheck{Name: "settings.json permissions", Status: doctorWarn, Detail: fmt.Sprintf("mode %s is readable by group/other", mode), Require: false})
		return
	}
	*checks = append(*checks, doctorCheck{Name: "settings.json permissions", Status: doctorPass, Detail: fmt.Sprintf("mode %s", mode), Require: false})
}

func checkDashboardCerts(checks *[]doctorCheck) {
	if _, err := os.Stat("server.crt"); err != nil {
		*checks = append(*checks, doctorCheck{Name: "dashboard certificate", Status: doctorWarn, Detail: "server.crt missing; dashboard will generate a self-signed cert", Require: false})
		return
	}
	if _, err := os.Stat("server.key"); err != nil {
		*checks = append(*checks, doctorCheck{Name: "dashboard private key", Status: doctorWarn, Detail: "server.key missing; dashboard will generate a self-signed key", Require: false})
		return
	}
	*checks = append(*checks, doctorCheck{Name: "dashboard TLS files", Status: doctorPass, Detail: "server.crt and server.key exist", Require: false})
}
