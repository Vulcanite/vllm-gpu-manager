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
- `--port <int>`: Container published port on `127.0.0.1`.
- `--gpus <int>`: Number of GPUs to allocate.
- `--gpu-ids <string>`: Explicit GPU devices to pin (e.g. `0,1`).
- `--kv-cache-dtype <string>`: KV cache quantization (`auto`, `fp8`).
- `--max-model-len <int>`: Max context token ceiling.
- `--gpu-memory-utilization <float>`: Memory fraction budget (default `0.90`).
- `--served-model-name <string>`: Friendly model alias for OpenAI API clients.

---

## Managing Active Containers

### Listing Active Models
```bash
vllm-manager model active
```

### Stopping a Model
```bash
vllm-manager model stop vllm-Qwen2.5-Coder-32B-Instruct-8080
```

### Restarting a Model
```bash
vllm-manager model restart vllm-Qwen2.5-Coder-32B-Instruct-8080
```

### Downloading a Model in the Background
```bash
vllm-manager model download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```
