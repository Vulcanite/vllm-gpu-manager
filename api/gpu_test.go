package api

import "testing"

func TestParseNvidiaSMICSV(t *testing.T) {
	gpus := parseNvidiaSMICSV([]byte("0, GPU-abc-123, NVIDIA H100 NVL, 95830, 70465, 25365, 98\n1, GPU-def-456, NVIDIA H100 NVL, 95830, 67023, 28807, 0\n"))

	if len(gpus) != 2 {
		t.Fatalf("len(gpus)=%d, want 2", len(gpus))
	}
	if gpus[0].ID != 0 || gpus[0].Name != "NVIDIA H100 NVL" {
		t.Fatalf("gpu[0]=%+v, want parsed ID/name", gpus[0])
	}
	if gpus[0].UUID != "GPU-abc-123" {
		t.Fatalf("gpu[0].UUID=%q, want GPU-abc-123", gpus[0].UUID)
	}
	if gpus[0].MemoryTotal != 95830*1024*1024 {
		t.Fatalf("MemoryTotal=%d, want MiB converted to bytes", gpus[0].MemoryTotal)
	}
	if gpus[0].GPUUtilization != 98 {
		t.Fatalf("GPUUtilization=%d, want 98", gpus[0].GPUUtilization)
	}
	if gpus[1].MemoryPercent <= 0 {
		t.Fatalf("MemoryPercent=%f, want positive value", gpus[1].MemoryPercent)
	}
}

func TestParseNvidiaSMICSVSkipsMalformedLines(t *testing.T) {
	gpus := parseNvidiaSMICSV([]byte("bad line\n2, GPU-xyz, GPU, 100, 50, 50, 12\n"))

	if len(gpus) != 1 {
		t.Fatalf("len(gpus)=%d, want 1", len(gpus))
	}
	if gpus[0].ID != 2 {
		t.Fatalf("ID=%d, want 2", gpus[0].ID)
	}
	if gpus[0].UUID != "GPU-xyz" {
		t.Fatalf("UUID=%q, want GPU-xyz", gpus[0].UUID)
	}
}

func TestParseNvidiaSMIProcessCSV(t *testing.T) {
	csv := []byte("GPU-abc-123, 1234, python, 512\nGPU-abc-123, 5678, /usr/bin/cuda_app, 1024\nGPU-def-456, 9012, java, 2048\n")
	parsed := parseNvidiaSMIProcessCSV(csv)

	if got := len(parsed["GPU-abc-123"]); got != 2 {
		t.Fatalf("len(parsed[GPU-abc-123])=%d, want 2", got)
	}
	if parsed["GPU-abc-123"][0].PID != 1234 || parsed["GPU-abc-123"][0].Name != "python" {
		t.Fatalf("first process=%+v, want PID 1234 name python", parsed["GPU-abc-123"][0])
	}
	if parsed["GPU-abc-123"][0].UsedMemory != 512*1024*1024 {
		t.Fatalf("UsedMemory=%d, want 512 MiB in bytes", parsed["GPU-abc-123"][0].UsedMemory)
	}
	if got := len(parsed["GPU-def-456"]); got != 1 {
		t.Fatalf("len(parsed[GPU-def-456])=%d, want 1", got)
	}
}

func TestParseNvidiaSMIProcessCSVEmptyAndMalformed(t *testing.T) {
	if got := parseNvidiaSMIProcessCSV([]byte("")); len(got) != 0 {
		t.Fatalf("empty CSV yielded %d entries, want 0", len(got))
	}
	parsed := parseNvidiaSMIProcessCSV([]byte("GPU-x, not-a-pid, name, 100\nGPU-y, 42, name\n"))
	if len(parsed) != 0 {
		t.Fatalf("malformed CSV yielded %d entries, want 0 (no valid PIDs)", len(parsed))
	}
}
