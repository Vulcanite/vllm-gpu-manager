package api

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type GPUInfo struct {
	ID             int     `json:"id"`
	Name           string  `json:"name"`
	MemoryTotal    uint64  `json:"memory_total"`
	MemoryUsed     uint64  `json:"memory_used"`
	MemoryFree     uint64  `json:"memory_free"`
	MemoryPercent  float64 `json:"memory_percent"`
	GPUUtilization uint32  `json:"gpu_utilization"`
}

func GetGPUInfo() ([]GPUInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi failed: %v", err)
	}

	var gpus []GPUInfo
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}

		parts := strings.Split(line, ", ")
		if len(parts) < 6 {
			continue
		}

		id, _ := strconv.Atoi(parts[0])
		name := parts[1]
		memTotal, _ := strconv.ParseUint(parts[2], 10, 64)
		memUsed, _ := strconv.ParseUint(parts[3], 10, 64)
		memFree, _ := strconv.ParseUint(parts[4], 10, 64)
		gpuUtil, _ := strconv.ParseUint(parts[5], 10, 32)

		// Convert MiB to bytes.
		memTotalBytes := memTotal * 1024 * 1024
		memUsedBytes := memUsed * 1024 * 1024
		memFreeBytes := memFree * 1024 * 1024

		memPercent := 0.0
		if memTotalBytes > 0 {
			memPercent = (float64(memUsedBytes) / float64(memTotalBytes)) * 100.0
		}

		gpus = append(gpus, GPUInfo{
			ID:             id,
			Name:           name,
			MemoryTotal:    memTotalBytes,
			MemoryUsed:     memUsedBytes,
			MemoryFree:     memFreeBytes,
			MemoryPercent:  memPercent,
			GPUUtilization: uint32(gpuUtil),
		})
	}

	return gpus, nil
}
