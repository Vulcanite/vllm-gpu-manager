package api

import (
	"strings"
	"testing"

	"github.com/moby/moby/api/types/container"
)

func restartTestContainer(t *testing.T, id, model string, port int, gpuIDs, memUtil, state, hfCache string) managedVLLMContainer {
	t.Helper()
	spec := LaunchSpec{
		Model:           model,
		Port:            port,
		GPUs:            len(strings.Split(gpuIDs, ",")),
		GPUIDs:          gpuIDs,
		MemUtil:         memUtil,
		MaxLen:          "128",
		ParallelismMode: ParallelismTensor,
		KVCacheDType:    "auto",
		TokenizerMode:   "auto",
		MoeBackend:      "auto",
		BindIP:          "127.0.0.1",
	}
	labels, err := serverContainerLabels(model, ContainerExposureProxy, spec)
	if err != nil {
		t.Fatalf("serverContainerLabels: %v", err)
	}
	hostConfig := &container.HostConfig{
		Resources: container.Resources{
			DeviceRequests: []container.DeviceRequest{{Driver: "nvidia", DeviceIDs: strings.Split(gpuIDs, ",")}},
		},
	}
	if hfCache != "" {
		hostConfig.Binds = []string{hfCache + ":/hf_cache"}
	}
	return managedVLLMContainer{
		id:     id,
		name:   "vllm-" + id,
		state:  state,
		labels: labels,
		metadata: container.InspectResponse{
			Config:     &container.Config{Labels: labels},
			HostConfig: hostConfig,
		},
	}
}

// stubRestart replaces the docker snapshot and restart seams and returns a
// pointer to the list of container IDs passed to docker restart.
func stubRestart(t *testing.T, snapshot []managedVLLMContainer) *[]string {
	t.Helper()
	originalSnapshot := snapshotManagedContainersFn
	originalRestart := restartContainerFn
	t.Cleanup(func() {
		snapshotManagedContainersFn = originalSnapshot
		restartContainerFn = originalRestart
	})
	snapshotManagedContainersFn = func() ([]managedVLLMContainer, error) {
		return snapshot, nil
	}
	var restarted []string
	restartContainerFn = func(id string) error {
		restarted = append(restarted, id)
		return nil
	}
	return &restarted
}

func stubTwoGPUs(t *testing.T, free uint64) {
	t.Helper()
	stubGPUInfo(t, func() ([]GPUInfo, error) {
		return []GPUInfo{
			{ID: 0, MemoryTotal: 16 * 1024 * 1024 * 1024, MemoryFree: free, ComputeCapability: "9.0"},
			{ID: 1, MemoryTotal: 16 * 1024 * 1024 * 1024, MemoryFree: free, ComputeCapability: "9.0"},
		}, nil
	})
}

func TestRestartModelBlocksStoppedContainerWhenGPUTaken(t *testing.T) {
	stubTwoGPUs(t, 16*1024*1024*1024)
	restarted := stubRestart(t, []managedVLLMContainer{
		restartTestContainer(t, "aaaa", "test/alpha", 8001, "0", "0.9", "exited", ""),
		restartTestContainer(t, "bbbb", "test/beta", 8002, "0", "0.9", "running", ""),
	})

	err := RestartModel("aaaa")
	if err == nil {
		t.Fatal("RestartModel succeeded, want GPU overlap rejection")
	}
	if !IsValidationError(err) || !strings.Contains(err.Error(), "combined GPU memory utilization") {
		t.Fatalf("RestartModel error=%v, want GPU overlap validation error", err)
	}
	if len(*restarted) != 0 {
		t.Fatalf("docker restart called for %v, want no restart", *restarted)
	}
}

func TestRestartModelAllowsStoppedContainerWhenGPUsFree(t *testing.T) {
	hfCache := t.TempDir()
	writeVRAMTestModel(t, hfCache, "test/alpha", tinyVRAMConfig(1024), 1024)
	stubTwoGPUs(t, 16*1024*1024*1024)
	restarted := stubRestart(t, []managedVLLMContainer{
		restartTestContainer(t, "aaaa", "test/alpha", 8001, "0", "0.9", "exited", hfCache),
		restartTestContainer(t, "bbbb", "test/beta", 8002, "1", "0.9", "running", ""),
	})

	if err := RestartModel("aaaa"); err != nil {
		t.Fatalf("RestartModel returned error: %v", err)
	}
	if len(*restarted) != 1 || (*restarted)[0] != "aaaa" {
		t.Fatalf("restarted=%v, want [aaaa]", *restarted)
	}
}

func TestRestartModelRunsVRAMFitCheckForStoppedContainer(t *testing.T) {
	hfCache := t.TempDir()
	writeVRAMTestModel(t, hfCache, "test/alpha", tinyVRAMConfig(1024), 1024*1024)
	// GPUs report almost no free memory (a non-managed workload holds it).
	stubTwoGPUs(t, 1024)
	restarted := stubRestart(t, []managedVLLMContainer{
		restartTestContainer(t, "aaaa", "test/alpha", 8001, "0", "0.5", "exited", hfCache),
	})

	err := RestartModel("aaaa")
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "currently free") {
		t.Fatalf("RestartModel error=%v, want VRAM fit validation error", err)
	}
	if len(*restarted) != 0 {
		t.Fatalf("docker restart called for %v, want no restart", *restarted)
	}
}

func TestRestartModelRunningContainerNotBlockedBySelf(t *testing.T) {
	stubTwoGPUs(t, 0) // a running container's own memory is in use
	restarted := stubRestart(t, []managedVLLMContainer{
		restartTestContainer(t, "aaaa", "test/alpha", 8001, "0", "0.9", "running", t.TempDir()),
		restartTestContainer(t, "bbbb", "test/beta", 8002, "1", "0.9", "running", ""),
	})

	if err := RestartModel("vllm-aaaa"); err != nil {
		t.Fatalf("RestartModel returned error: %v", err)
	}
	if len(*restarted) != 1 || (*restarted)[0] != "aaaa" {
		t.Fatalf("restarted=%v, want [aaaa]", *restarted)
	}
}

func TestRestartModelRejectsUnmanagedContainer(t *testing.T) {
	restarted := stubRestart(t, nil)
	err := RestartModel("unknown")
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "not managed") {
		t.Fatalf("RestartModel error=%v, want not-managed validation error", err)
	}
	if len(*restarted) != 0 {
		t.Fatalf("docker restart called for %v, want no restart", *restarted)
	}
}
