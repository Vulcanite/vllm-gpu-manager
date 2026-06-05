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

func TestRedactDockerArgsRedactsHFToken(t *testing.T) {
	got := redactDockerArgs([]string{"-e", "HF_TOKEN=secret", "image=vllm"})
	want := "-e HF_TOKEN=<redacted> image=vllm"

	if got != want {
		t.Fatalf("redactDockerArgs()=%q, want %q", got, want)
	}
}

func TestRedactDockerArgsRedactsVLLMAPIKeyInCommand(t *testing.T) {
	got := redactDockerArgs([]string{"cmd=--model Qwen/Qwen2 --api-key token-secret --port 8000"})
	want := "cmd=--model Qwen/Qwen2 --api-key <redacted> --port 8000"

	if got != want {
		t.Fatalf("redactDockerArgs()=%q, want %q", got, want)
	}
}

func TestRedactDockerArgsRedactsVLLMAPIKeyEqualsForm(t *testing.T) {
	got := redactDockerArgs([]string{"cmd=--model Qwen/Qwen2 --api-key=token-secret --port 8000"})
	want := "cmd=--model Qwen/Qwen2 --api-key=<redacted> --port 8000"

	if got != want {
		t.Fatalf("redactDockerArgs()=%q, want %q", got, want)
	}
}

func TestRedactDockerArgsRedactsVLLMAPIKeyWhenFirstCommandArg(t *testing.T) {
	got := redactDockerArgs([]string{"cmd=--api-key token-secret --model Qwen/Qwen2"})
	want := "cmd=--api-key <redacted> --model Qwen/Qwen2"

	if got != want {
		t.Fatalf("redactDockerArgs()=%q, want %q", got, want)
	}
}

func TestRedactDockerArgsRedactsVLLMAPIKeySeparateArgs(t *testing.T) {
	got := redactDockerArgs([]string{"container create", "--api-key", "token-secret"})
	want := "container create --api-key <redacted>"

	if got != want {
		t.Fatalf("redactDockerArgs()=%q, want %q", got, want)
	}
}
