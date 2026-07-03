package api

import (
	"bytes"
	"context"
	"encoding/binary"
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

// makeDockerLogFrame builds a valid 8-byte header + payload using the
// real moby multiplex format (size at bytes 4-7).
func makeDockerLogFrame(stream byte, payload string) []byte {
	data := []byte(payload)
	h := make([]byte, 8)
	h[0] = stream
	// bytes 1-3 stay 0 (reserved)
	binary.BigEndian.PutUint32(h[4:8], uint32(len(data)))
	return append(h, data...)
}

func TestStreamDockerLogsParsesStdoutStderrFrames(t *testing.T) {
	// Simulate a small sequence of frames that Docker would send.
	frames := append([]byte{}, makeDockerLogFrame(1, "hello stdout\n")...)
	frames = append(frames, makeDockerLogFrame(2, "some stderr\n")...)
	frames = append(frames, makeDockerLogFrame(1, "more output")...) // no trailing nl

	r := bytes.NewReader(frames)
	ch := make(chan LogChunk, 8)
	done := make(chan struct{})

	go func() {
		// run the parser (ctx never canceled)
		_ = streamDockerLogs(context.Background(), r, ch)
		close(done)
	}()

	var got []LogChunk
	for {
		select {
		case c := <-ch:
			got = append(got, c)
		case <-done:
			// drain any last
			for len(ch) > 0 {
				got = append(got, <-ch)
			}
			goto collected
		}
	}
collected:

	if len(got) != 3 {
		t.Fatalf("got %d chunks, want 3; %+v", len(got), got)
	}
	if !got[0].Stdout || got[0].Data != "hello stdout\n" {
		t.Errorf("chunk0: stdout=%v data=%q", got[0].Stdout, got[0].Data)
	}
	if got[1].Stdout || got[1].Data != "some stderr\n" {
		t.Errorf("chunk1: stdout=%v data=%q", got[1].Stdout, got[1].Data)
	}
	if !got[2].Stdout || got[2].Data != "more output" {
		t.Errorf("chunk2: stdout=%v data=%q", got[2].Stdout, got[2].Data)
	}
}
