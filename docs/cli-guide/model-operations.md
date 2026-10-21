---
title: Model Lifecycle Operations
sidebar_position: 3
---

# Model Lifecycle Operations

Deploy, stop, restart, list, and download vLLM model containers from the command line.

---

## Starting a Model Server (`model start`)

Launch a vLLM container with automated VRAM preflight checks:

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct \
  --port 8080 \
  --gpus 2 \
  --kv-cache-dtype fp8
```

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>AUTOMATED VRAM PREFLIGHT &amp; CONTAINER LAUNCH</span>
    <span>vllm-manager model start</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/cli/03-cli-model-start.png" alt="vllm-manager model start output" />
</div>

### Key Flags
- `--port <int>`: Host port to publish. CLI launches are direct (non-proxy) containers bound to `server_bind_ip` (default `0.0.0.0`) unless `--bind-ip` is given.
- `--gpus <int>`: Number of GPUs to allocate.
- `--gpu-ids <string>`: Explicit GPU devices to pin (e.g. `0,1`).
- `--kv-cache-dtype <string>`: KV cache quantization (`auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`).
- `--max-len <int>`: Max context length passed as `--max-model-len`.
- `--mem-util <float>`: Memory fraction passed as `--gpu-memory-utilization` (default `0.9`).
- `--parallelism <tensor|pipeline>`: Multi-GPU mode (default `tensor`).

`--served-model-name` is not a CLI flag; set it from the web console's Deploy Server dialog.

---

## Managing Active Containers

### Listing Active Models
```bash
vllm-manager model active
```

### Stopping a Model
```bash
vllm-manager model stop --port 8080
```

### Restarting a Model
```bash
# Container ID from `vllm-manager model active`
vllm-manager model restart --container-id <container-id>
```

### Downloading a Model in the Background
```bash
vllm-manager model download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```
