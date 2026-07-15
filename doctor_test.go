package main

import "testing"

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
