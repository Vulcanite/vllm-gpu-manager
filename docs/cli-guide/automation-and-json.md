---
title: Automation & JSON Scripting
sidebar_position: 5
---

# Automation & JSON Scripting

Use the `--json` flag to integrate **vLLM GPU Manager** into automated deployment scripts, CI/CD pipelines, and cluster health monitoring.

---

## JSON Output Structure

All inspection commands produce schema-consistent JSON when `--json` is supplied:

### GPU Telemetry JSON
```bash
vllm-manager gpu list --json | jq .
```

Output:
```json
[
  {
    "id": 0,
    "uuid": "GPU-8a12f94c-34e8-4a11-872e-d018c1e40001",
    "name": "NVIDIA H100 80GB HBM3",
    "memory_total": 85899345920,
    "memory_used": 36700160000,
    "memory_free": 49199185920,
    "memory_percent": 42.7,
    "gpu_utilization": 68,
    "compute_capability": "9.0",
    "processes": [
      {
        "pid": 24819,
        "user": "vllm-manager",
        "name": "python3 (vLLM::Engine)",
        "used_memory": 36700160000,
        "used_memory_human": "34.18 GiB"
      }
    ]
  }
]
```

---

## Automation Examples

### 1. Wait for GPU Memory Availability in Bash
```bash
#!/bin/bash
REQUIRED_GIB=40

FREE_GIB=$(vllm-manager gpu list --json | jq '.[0].memory_free / 1073741824')

if (( $(echo "$FREE_GIB < $REQUIRED_GIB" | bc -l) )); then
  echo "Insufficient VRAM: only ${FREE_GIB} GiB free."
  exit 1
fi

echo "VRAM OK: ${FREE_GIB} GiB available. Launching container..."
```

### 2. Auto-Discover Active Model Endpoints
```bash
ACTIVE_PORT=$(vllm-manager model active --json | jq -r '.[] | select(.model=="Qwen/Qwen2.5-Coder-32B-Instruct") | .port')
echo "Model running on internal port: ${ACTIVE_PORT}"
```
