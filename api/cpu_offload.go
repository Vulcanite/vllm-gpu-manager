package api

import (
	"strconv"
	"strings"

	"github.com/moby/moby/api/types/container"
)

const (
	bytesPerGiB                  = uint64(1024 * 1024 * 1024)
	cpuOffloadRAMSafetyNumerator = uint64(5)
	cpuOffloadRAMSafetyDenom     = uint64(4)
	CPUOffloadRAMSafetyFactor    = 1.25
)

func normalizeCPUOffloadGB(value string) (int, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, nil
	}
	cpuOffloadGB, err := strconv.Atoi(value)
	if err != nil || cpuOffloadGB < 0 {
		return 0, validationError("CPU offload must be a non-negative whole number of GiB per GPU")
	}
	return cpuOffloadGB, nil
}

func requiredCPUOffloadRAMBytes(cpuOffloadGB int, gpuCount int) uint64 {
	if cpuOffloadGB <= 0 || gpuCount <= 0 {
		return 0
	}
	base := uint64(cpuOffloadGB) * uint64(gpuCount) * bytesPerGiB
	return (base * cpuOffloadRAMSafetyNumerator) / cpuOffloadRAMSafetyDenom
}

func validateCPUOffloadRAM(cpuOffloadGB int, gpuCount int, ram SystemRAM) error {
	if cpuOffloadGB <= 0 {
		return nil
	}
	required := requiredCPUOffloadRAMBytes(cpuOffloadGB, gpuCount)
	if ram.Free == 0 {
		return validationError("could not read available system RAM to validate CPU offload")
	}
	if ram.Free < required {
		requested := uint64(cpuOffloadGB) * uint64(gpuCount) * bytesPerGiB
		return validationError("CPU offload requests %s (%d GiB/GPU x %d GPUs) and requires %s with safety margin, but only %s is available", FormatBytes(requested), cpuOffloadGB, gpuCount, FormatBytes(required), FormatBytes(ram.Free))
	}
	return nil
}

func cpuOffloadUlimits(cpuOffloadGB int) []*container.Ulimit {
	if cpuOffloadGB <= 0 {
		return nil
	}
	return []*container.Ulimit{
		{Name: "memlock", Soft: -1, Hard: -1},
	}
}
