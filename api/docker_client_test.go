package api

import (
	"reflect"
	"testing"
)

func TestDockerDeviceRequestExplicitDevices(t *testing.T) {
	request := dockerDeviceRequest("device=0, 1")

	if request.Count != 0 {
		t.Fatalf("explicit device request set Count=%d, want 0", request.Count)
	}
	if !reflect.DeepEqual(request.DeviceIDs, []string{"0", "1"}) {
		t.Fatalf("DeviceIDs=%v, want [0 1]", request.DeviceIDs)
	}
}

func TestDockerDeviceRequestAll(t *testing.T) {
	request := dockerDeviceRequest("all")

	if request.Count != -1 {
		t.Fatalf("all device request set Count=%d, want -1", request.Count)
	}
	if len(request.DeviceIDs) != 0 {
		t.Fatalf("DeviceIDs=%v, want empty", request.DeviceIDs)
	}
}

func TestDockerDeviceRequestEmptyDeviceListFallsBackToAll(t *testing.T) {
	request := dockerDeviceRequest("device=")

	if request.Count != -1 {
		t.Fatalf("empty device request set Count=%d, want -1", request.Count)
	}
	if len(request.DeviceIDs) != 0 {
		t.Fatalf("DeviceIDs=%v, want empty", request.DeviceIDs)
	}
}
