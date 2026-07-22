package api

import "testing"

func TestGPUProbeCreateOptionsRequestsAllNVIDIAGPUs(t *testing.T) {
	opts := gpuProbeCreateOptions("probe")
	if opts.Name != "probe" || opts.Config == nil || opts.Config.Image != gpuProbeImage {
		t.Fatalf("gpuProbeCreateOptions()=%+v", opts)
	}
	if len(opts.Config.Cmd) != 1 || opts.Config.Cmd[0] != "true" {
		t.Fatalf("probe command=%v", opts.Config.Cmd)
	}
	if opts.HostConfig == nil || len(opts.HostConfig.Resources.DeviceRequests) != 1 {
		t.Fatalf("device requests=%+v", opts.HostConfig)
	}
	request := opts.HostConfig.Resources.DeviceRequests[0]
	if request.Driver != "nvidia" || request.Count != -1 {
		t.Fatalf("device request=%+v", request)
	}
}
