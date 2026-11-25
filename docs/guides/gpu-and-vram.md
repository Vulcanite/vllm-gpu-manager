---
title: GPU placement & VRAM
sidebar_position: 3
---

# GPU placement & VRAM

vLLM reserves its GPU memory when it starts. If the weights and KV cache do not fit, the container fails during start-up, often after minutes of loading. The manager estimates the memory first and refuses launches that will not fit.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/vram-preflight-engine.png" alt="How the VRAM check works" />
  <div className="diagram-caption">Weights and KV cache are estimated per GPU and checked against free memory and the memory fraction.</div>
</div>

## How memory is estimated

Per GPU:

```text
memory_per_gpu = (weights / num_gpus × 1.08) − cpu_offload + kv_cache
```

- **Weights**: the size of the model's folder in the Hugging Face cache. It counts every cached file, so it slightly overestimates when tokenizer or extra files are present. A model that is not cached cannot be launched.
- **1.08**: an 8% margin for the CUDA context and activation buffers.
- **CPU offload**: `--cpu-offload-gb` is subtracted from the GPU share.
- **KV cache**: bytes per token × context length (`--max-len`), where

```text
kv_bytes_per_token = 2 × layers × ceil(kv_heads / tp_size) × head_dim × element_bytes
```

| Value | From `config.json` |
| :--- | :--- |
| `layers` | `num_hidden_layers` |
| `kv_heads` | `num_key_value_heads` (handles GQA) |
| `head_dim` | `head_dim`, or `hidden_size / num_attention_heads` |
| `element_bytes` | 2 for 16-bit, 1 for an FP8 `--kv-cache-dtype` |

## What is checked

On every selected GPU:

1. The weights fit in the memory that is free right now.
2. Weights plus KV cache fit in both the free memory and `total × --mem-util`.
3. At least 1 GiB of that budget is left for KV cache after the weights.
4. The `--mem-util` values of all containers on that GPU add up to no more than `1.0`.

Launches and restarts take a process-wide lock while they check and start, so two launches cannot both claim the same free memory.

## Using several GPUs

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/multi-gpu-placement.png" alt="Tensor and pipeline parallelism" />
  <div className="diagram-caption">Tensor parallelism splits each layer; pipeline parallelism splits the stack of layers.</div>
</div>

- **Tensor parallelism** (`--parallelism tensor`, default) splits every layer across the GPUs. It is fastest with NVLink or a fast PCIe link. The number of KV heads must divide evenly by the GPU count (MLA models such as DeepSeek are exempt).
- **Pipeline parallelism** (`--parallelism pipeline`) gives each GPU a block of layers. Use it without NVLink, or when the head count does not divide by the GPU count.

The manager reads `nvidia-smi topo -m` and shows how the GPUs are connected in the dashboard's deploy form.

### Pin exact GPUs

By default the first free GPUs are used. To choose:

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct --port 8080 --gpus 2 --gpu-ids 0,1
```

This becomes Docker's `--gpus '"device=0,1"'`.
