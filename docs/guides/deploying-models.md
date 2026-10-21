---
title: Deploying Models
sidebar_position: 1
---

# Deploying Models

Learn how to configure, optimize, and deploy large language models using vLLM GPU Manager.

---

## Launch Options & Parameters

When deploying a model via the web console (**Deploy Server** card) or CLI (`vllm-manager model start`), you can configure the following launch parameters:

```bash
vllm-manager model start <model-id> [flags]
```

### Essential Parameters

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | Integer | Required | Port for the serving container. |
| `--gpus` | Integer | Required | Number of GPUs to allocate (tensor parallel size by default). |
| `--gpu-ids` | String | First *N* GPUs | Specific GPU IDs to pin (e.g. `0,1`). |
| `--mem-util` | Float | `0.9` | Fraction of GPU memory allocated to vLLM (`0.1` to `1.0`). |
| `--max-len` | Integer | Auto | Maximum context length in tokens (capped by model limits). |
| `--max-num-seqs` | Integer | vLLM default | Maximum number of concurrent sequences in a batch. |
| `--dtype` | String | `auto` | Model weight data type (`auto`, `bfloat16`, `float16`, `float32`). |
| Served model name | String | None | Web console and API only (no CLI flag). Custom alias returned by `/v1/models` and used for proxy routing. |

---

## Advanced Performance & Optimization Options

### 1. KV Cache Quantization (`--kv-cache-dtype`)
Reduce VRAM used by the key-value cache by up to 50% with FP8 precision:

- `auto`: Uses the model's native data type (typically 16-bit, 2 bytes/token/head).
- `fp8` / `fp8_e4m3`: Uses 8-bit floating point representation (1 byte/token/head), doubling maximum sequence capacity on NVIDIA Ada Lovelace, Hopper, and Blackwell GPUs.
- `fp8_e5m2`: Extended dynamic range FP8 for deep attention architectures.

```bash
vllm-manager model start meta-llama/Llama-3.3-70B-Instruct \
  --port 8080 --gpus 4 --kv-cache-dtype fp8
```

### 2. Prefix Caching (`--enable-prefix-caching`)
Automatically caches prompt prefixes across requests (e.g. system prompts, few-shot examples, multi-turn chat history). Enabled by default (`--enable-prefix-caching=true`).

### 3. CPU Offloading (`--cpu-offload-gb`)
Offload model weights to system host RAM when GPU VRAM is tight:

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct \
  --port 8080 --gpus 1 --cpu-offload-gb 8
```

> [!NOTE]
> The backend validates host RAM availability using a safety factor of $1.25\times$ to avoid triggering Linux host OOM killers.

### 4. Tool Call & Reasoning Parsers
Enable native structured function calling or reasoning token separation:

- `--tool-call-parser`: `hermes`, `mistral`, `llama3_json`, `granite-20b-fc`, `internlm2`.
- `--reasoning-parser`: `deepseek_r1`, `deepseek_v4`, `glm45`, `qwen3`.

```bash
vllm-manager model start deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --port 8080 --gpus 2 --reasoning-parser deepseek_r1
```

### 5. Speculative Decoding (`--speculative-config`)
Accelerate token generation using draft models or speculative verification:

```bash
vllm-manager model start meta-llama/Llama-3.1-70B-Instruct \
  --port 8080 --gpus 4 \
  --speculative-config '{"method":"eagle","num_speculative_tokens":5}'
```

---

## Container Restarts & Lifecycle

- **Restarting**: If a container exits or needs a reboot, restart it using the web UI or CLI:
  ```bash
  vllm-manager model restart <container-id>
  ```
  Containers restart with their original command, port, GPU pinning, and volume mounts.
- **Removing**: Stop the container first, then remove it:
  ```bash
  vllm-manager model stop --port 8080
  vllm-manager model remove <container-id>
  ```
