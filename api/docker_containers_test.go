package api

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func stubGPUInfo(t *testing.T, fn func() ([]GPUInfo, error)) {
	t.Helper()
	original := getGPUInfo
	getGPUInfo = fn
	t.Cleanup(func() {
		getGPUInfo = original
	})
}

func TestValidateGPUSelectionRejectsDiscoveryFailure(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return nil, errors.New("nvidia-smi unavailable")
	})

	_, _, err := validateGPUSelection(1, "all")
	if err == nil {
		t.Fatal("validateGPUSelection returned nil error, want discovery failure")
	}
	if !strings.Contains(err.Error(), "GPU detection failed") {
		t.Fatalf("error=%q, want GPU detection failure", err.Error())
	}
}

func TestValidateGPUSelectionRejectsNoDetectedGPUs(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return []GPUInfo{}, nil
	})

	_, _, err := validateGPUSelection(1, "all")
	if err == nil {
		t.Fatal("validateGPUSelection returned nil error, want no-GPU validation error")
	}
	if !IsValidationError(err) {
		t.Fatalf("error type=%T, want ValidationError", err)
	}
	if !strings.Contains(err.Error(), "no GPUs detected") {
		t.Fatalf("error=%q, want no GPUs detected", err.Error())
	}
}

func TestValidateGPUSelectionAllUsesDetectedDevices(t *testing.T) {
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return []GPUInfo{{ID: 1}, {ID: 0}}, nil
	})

	gpuFlag, selected, err := validateGPUSelection(2, "all")
	if err != nil {
		t.Fatalf("validateGPUSelection returned error: %v", err)
	}
	if gpuFlag != "device=0,1" {
		t.Fatalf("gpuFlag=%q, want device=0,1", gpuFlag)
	}
	if !reflect.DeepEqual(selected, []GPUInfo{{ID: 0}, {ID: 1}}) {
		t.Fatalf("selected=%v, want GPU IDs 0 and 1", selected)
	}
}

func TestIsManagedContainerRequiresLabels(t *testing.T) {
	tests := []struct {
		name   string
		labels map[string]string
		want   bool
	}{
		{
			name:   "managed label",
			labels: map[string]string{managedContainerLabel: "true"},
			want:   true,
		},
		{
			name:   "legacy app label",
			labels: map[string]string{"app": "vllm-manager"},
			want:   true,
		},
		{
			name:   "vllm name without labels is not enough",
			labels: map[string]string{},
			want:   false,
		},
		{
			name:   "nil labels",
			labels: nil,
			want:   false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isManagedContainer(test.labels); got != test.want {
				t.Fatalf("isManagedContainer()=%v, want %v", got, test.want)
			}
		})
	}
}

func TestIsManagedServerContainerRejectsDownloads(t *testing.T) {
	if !isManagedServerContainer(map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindServer}) {
		t.Fatal("managed server container was rejected")
	}
	if !isManagedServerContainer(map[string]string{"app": "vllm-manager"}) {
		t.Fatal("legacy managed server container without kind was rejected")
	}
	if isManagedServerContainer(map[string]string{managedContainerLabel: "true", containerKindLabel: containerKindDownload}) {
		t.Fatal("managed download container was accepted as server container")
	}
}
