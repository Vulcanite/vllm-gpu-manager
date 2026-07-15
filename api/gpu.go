package api

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type GPUProcess struct {
	PID         int    `json:"pid"`
	Name        string `json:"name"`
	UsedMemory  uint64 `json:"used_memory"`
	UsedMemoryH string `json:"used_memory_human"`
}

type GPUInfo struct {
	ID                int          `json:"id"`
	UUID              string       `json:"uuid"`
	Name              string       `json:"name"`
	MemoryTotal       uint64       `json:"memory_total"`
	MemoryUsed        uint64       `json:"memory_used"`
	MemoryFree        uint64       `json:"memory_free"`
	MemoryPercent     float64      `json:"memory_percent"`
	GPUUtilization    uint32       `json:"gpu_utilization"`
	ComputeCapability string       `json:"compute_capability,omitempty"`
	Processes         []GPUProcess `json:"processes,omitempty"`
}

type GPUTopologyLink struct {
	GPUA int    `json:"gpu_a"`
	GPUB int    `json:"gpu_b"`
	Link string `json:"link"`
}

var getGPUInfo = GetGPUInfo

func RequireDetectedGPUs() ([]GPUInfo, error) {
	gpus, err := getGPUInfo()
	if err != nil {
		return nil, fmt.Errorf("GPU detection failed: %v", err)
	}
	if len(gpus) == 0 {
		return nil, validationError("no GPUs detected; deployment requires at least one NVIDIA GPU")
	}
	return gpus, nil
}

func GetGPUInfo() ([]GPUInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	gpus, err := queryNvidiaSMI(ctx)
	if err != nil {
		return nil, err
	}
	// Process list is best-effort: a failure to enumerate PIDs (e.g. nvidia-smi
	// without permissions, or a container that can't see the driver) must
	// never block the rest of the GPU telemetry from being surfaced.
	processesByUUID := queryNvidiaSMIProcesses(ctx)
	for i := range gpus {
		if procs, ok := processesByUUID[gpus[i].UUID]; ok {
			gpus[i].Processes = procs
		}
	}
	return gpus, nil
}

func queryNvidiaSMI(ctx context.Context) ([]GPUInfo, error) {
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_cap",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi failed: %v", err)
	}
	return parseNvidiaSMICSV(out), nil
}

func queryNvidiaSMIProcesses(ctx context.Context) map[string][]GPUProcess {
	procsByUUID := make(map[string][]GPUProcess)
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return procsByUUID
	}
	for uuid, list := range parseNvidiaSMIProcessCSV(out) {
		procsByUUID[uuid] = list
	}
	return procsByUUID
}

func GetGPUTopologyLinks() ([]GPUTopologyLink, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "topo", "-m").Output()
	if err != nil {
		return nil, fmt.Sprintf("nvidia-smi topo -m failed: %v", err)
	}
	links := parseNvidiaSMITopology(out)
	if len(links) == 0 {
		return nil, "nvidia-smi topo -m returned no GPU links"
	}
	return links, ""
}

func parseNvidiaSMITopology(out []byte) []GPUTopologyLink {
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) < 2 {
		return nil
	}
	header := strings.Fields(lines[0])
	gpuColumns := make(map[int]int)
	for column, value := range header {
		id, ok := parseTopologyGPUHeader(value)
		if ok {
			gpuColumns[column] = id
		}
	}
	var links []GPUTopologyLink
	for _, line := range lines[1:] {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		rowGPU, ok := parseTopologyGPUHeader(fields[0])
		if !ok {
			continue
		}
		for column, otherGPU := range gpuColumns {
			valueIndex := column + 1
			if valueIndex >= len(fields) || rowGPU >= otherGPU {
				continue
			}
			link := strings.TrimSpace(fields[valueIndex])
			if link == "" || link == "X" {
				continue
			}
			links = append(links, GPUTopologyLink{GPUA: rowGPU, GPUB: otherGPU, Link: link})
		}
	}
	return links
}

func parseTopologyGPUHeader(value string) (int, bool) {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "GPU") {
		return 0, false
	}
	id, err := strconv.Atoi(strings.TrimPrefix(value, "GPU"))
	if err != nil {
		return 0, false
	}
	return id, true
}

func parseNvidiaSMICSV(out []byte) []GPUInfo {
	var gpus []GPUInfo
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}

		// Field 1 (UUID) may contain hyphens and is space-delimited in the
		// middle of the CSV. Splitting on ", " works for the index/name/...
		// tail but eats trailing fields when UUIDs include leading spaces,
		// so we re-join defensively.
		parts := strings.Split(line, ", ")
		if len(parts) < 7 {
			continue
		}

		id, _ := strconv.Atoi(parts[0])
		uuid := strings.TrimSpace(parts[1])
		name := parts[2]
		memTotal, _ := strconv.ParseUint(parts[3], 10, 64)
		memUsed, _ := strconv.ParseUint(parts[4], 10, 64)
		memFree, _ := strconv.ParseUint(parts[5], 10, 64)
		gpuUtil, _ := strconv.ParseUint(parts[6], 10, 32)
		computeCapability := ""
		if len(parts) > 7 {
			computeCapability = strings.TrimSpace(parts[7])
		}

		// Convert MiB to bytes.
		memTotalBytes := memTotal * 1024 * 1024
		memUsedBytes := memUsed * 1024 * 1024
		memFreeBytes := memFree * 1024 * 1024

		memPercent := 0.0
		if memTotalBytes > 0 {
			memPercent = (float64(memUsedBytes) / float64(memTotalBytes)) * 100.0
		}

		gpus = append(gpus, GPUInfo{
			ID:                id,
			UUID:              uuid,
			Name:              name,
			MemoryTotal:       memTotalBytes,
			MemoryUsed:        memUsedBytes,
			MemoryFree:        memFreeBytes,
			MemoryPercent:     memPercent,
			GPUUtilization:    uint32(gpuUtil),
			ComputeCapability: computeCapability,
		})
	}

	return gpus
}

func (g GPUInfo) ComputeCapabilityAtLeast(major int, minor int) bool {
	parts := strings.Split(strings.TrimSpace(g.ComputeCapability), ".")
	if len(parts) < 1 || strings.TrimSpace(parts[0]) == "" {
		return false
	}
	gotMajor, err := strconv.Atoi(parts[0])
	if err != nil {
		return false
	}
	gotMinor := 0
	if len(parts) > 1 {
		gotMinor, _ = strconv.Atoi(parts[1])
	}
	if gotMajor != major {
		return gotMajor > major
	}
	return gotMinor >= minor
}

func parseNvidiaSMIProcessCSV(out []byte) map[string][]GPUProcess {
	procsByUUID := make(map[string][]GPUProcess)
	text := strings.TrimSpace(string(out))
	if text == "" {
		return procsByUUID
	}
	for _, line := range strings.Split(text, "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, ", ")
		if len(parts) < 4 {
			continue
		}
		uuid := strings.TrimSpace(parts[0])
		pid, err := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			continue
		}
		name := parts[2]
		usedMiB, _ := strconv.ParseUint(strings.TrimSpace(parts[3]), 10, 64)
		procsByUUID[uuid] = append(procsByUUID[uuid], GPUProcess{
			PID:         pid,
			Name:        name,
			UsedMemory:  usedMiB * 1024 * 1024,
			UsedMemoryH: FormatBytes(usedMiB * 1024 * 1024),
		})
	}
	return procsByUUID
}
