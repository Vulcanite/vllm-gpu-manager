package main

import (
	"errors"
	"path/filepath"
	"testing"

	"vllm-manager/api"
)

func TestExtractNvidiaSMIField(t *testing.T) {
	banner := `| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |`

	if got := extractNvidiaSMIField(banner, "Driver Version:"); got != "560.35.03" {
		t.Fatalf("driver=%q, want 560.35.03", got)
	}
	if got := extractNvidiaSMIField(banner, "CUDA Version:"); got != "12.6" {
		t.Fatalf("cuda=%q, want 12.6", got)
	}
	if got := extractNvidiaSMIField(banner, "Missing:"); got != "" {
		t.Fatalf("missing field=%q, want empty", got)
	}
}

func stubDoctorDependencies(t *testing.T, cache string) {
	t.Helper()
	previousGPU := doctorGetGPUInfo
	previousAPI := doctorCheckDockerAPI
	previousDockerGPU := doctorCheckDockerGPU
	previousImage := doctorCheckDockerImage
	previousWritable := doctorCheckWritablePath
	previousLookPath := doctorLookPath
	previousNvidiaInfo := doctorNvidiaInfo
	previousRuntime := doctorRuntimeSignal
	t.Cleanup(func() {
		doctorGetGPUInfo = previousGPU
		doctorCheckDockerAPI = previousAPI
		doctorCheckDockerGPU = previousDockerGPU
		doctorCheckDockerImage = previousImage
		doctorCheckWritablePath = previousWritable
		doctorLookPath = previousLookPath
		doctorNvidiaInfo = previousNvidiaInfo
		doctorRuntimeSignal = previousRuntime
	})

	resetSettingsCache(t, Settings{HFCache: cache, VLLMImage: api.DefaultVLLMImage, ServerBindIP: "0.0.0.0"}, true)
	doctorGetGPUInfo = func() ([]api.GPUInfo, error) {
		return []api.GPUInfo{{ID: 0, ComputeCapability: "8.9"}}, nil
	}
	doctorCheckDockerAPI = func() error { return nil }
	doctorCheckDockerGPU = func() error { return nil }
	doctorCheckDockerImage = func(string) error { return nil }
	doctorCheckWritablePath = func(string) error { return nil }
	doctorLookPath = func(file string) (string, error) { return filepath.Join("/usr/bin", file), nil }
	doctorNvidiaInfo = func() (string, error) { return "driver 560; CUDA 12.6", nil }
	doctorRuntimeSignal = func() (string, bool) { return "nvidia runtime", true }
}

func TestRunDoctorHealthyHost(t *testing.T) {
	cache := t.TempDir()
	stubDoctorDependencies(t, cache)
	report := runDoctor()
	if !report.OK {
		t.Fatalf("runDoctor() report=%+v", report)
	}
}

func TestRunDoctorRequiresDockerGPUAccess(t *testing.T) {
	cache := t.TempDir()
	stubDoctorDependencies(t, cache)
	doctorCheckDockerGPU = func() error { return errors.New("GPU request rejected") }
	report := runDoctor()
	if report.OK {
		t.Fatalf("runDoctor() unexpectedly passed: %+v", report)
	}
	found := false
	for _, check := range report.Checks {
		if check.Name == "Docker GPU access" && check.Require && check.Status == doctorFail {
			found = true
		}
	}
	if !found {
		t.Fatalf("runDoctor() missing required Docker GPU failure: %+v", report)
	}
}

func TestRunDoctorTreatsMissingVLLMImageAsWarning(t *testing.T) {
	cache := t.TempDir()
	stubDoctorDependencies(t, cache)
	doctorCheckDockerImage = func(string) error { return errors.New("not found") }
	report := runDoctor()
	if !report.OK {
		t.Fatalf("missing local image should not fail doctor: %+v", report)
	}
	for _, check := range report.Checks {
		if check.Name == "vLLM image" && check.Status == doctorWarn && !check.Require {
			return
		}
	}
	t.Fatalf("runDoctor() missing optional image warning: %+v", report)
}
