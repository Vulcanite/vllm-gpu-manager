package api

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type SystemRAM struct {
	Total uint64
	Used  uint64
	Free  uint64
}

func GetSystemRAM() SystemRAM {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return SystemRAM{}
	}
	defer file.Close()

	var memTotal, memFree, memAvailable, buffers, cached uint64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		val, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		val *= 1024

		switch fields[0] {
		case "MemTotal:":
			memTotal = val
		case "MemFree:":
			memFree = val
		case "MemAvailable:":
			memAvailable = val
		case "Buffers:":
			buffers = val
		case "Cached:":
			cached = val
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[SYSTEM] warning: failed to scan /proc/meminfo: %v", err)
		return SystemRAM{}
	}

	free := memAvailable
	if free == 0 {
		free = memFree + buffers + cached
	}
	used := uint64(0)
	if memTotal >= free {
		used = memTotal - free
	}
	return SystemRAM{Total: memTotal, Used: used, Free: free}
}
