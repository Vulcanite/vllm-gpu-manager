package api

import "testing"

func TestParseNvidiaSMICSV(t *testing.T) {
	gpus := parseNvidiaSMICSV([]byte("0, NVIDIA H100 NVL, 95830, 70465, 25365, 98\n1, NVIDIA H100 NVL, 95830, 67023, 28807, 0\n"))

	if len(gpus) != 2 {
		t.Fatalf("len(gpus)=%d, want 2", len(gpus))
	}
	if gpus[0].ID != 0 || gpus[0].Name != "NVIDIA H100 NVL" {
		t.Fatalf("gpu[0]=%+v, want parsed ID/name", gpus[0])
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
	gpus := parseNvidiaSMICSV([]byte("bad line\n2, GPU, 100, 50, 50, 12\n"))

	if len(gpus) != 1 {
		t.Fatalf("len(gpus)=%d, want 1", len(gpus))
	}
	if gpus[0].ID != 2 {
		t.Fatalf("ID=%d, want 2", gpus[0].ID)
	}
}
