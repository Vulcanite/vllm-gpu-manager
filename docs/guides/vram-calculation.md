---
title: VRAM Preflight & Memory Math
sidebar_position: 1
---

# VRAM Preflight & Memory Math

vLLM allocates its KV cache block pool eagerly during startup. Without precise memory estimation, container launches frequently trigger unrecoverable CUDA Out-of-Memory (OOM) failures.

**vLLM GPU Manager** executes proactive preflight mathematics before launching containers.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/vram-preflight-engine.png" alt="VRAM Preflight & Memory Budgeting Engine" />
  <div className="diagram-caption">Figure 2: VRAM preflight engine, weight sizing, and KV cache allocation math.</div>
</div>

---

## The VRAM Calculation Engine

The total memory required per GPU is governed by:

```text
Memory_per_GPU = (Model_Weights / Num_GPUs * 1.08) - CPU_Offload + KV_Cache_Total
```

### 1. Model Weights Sizing
The engine measures the size of the model's local Hugging Face cache directory (`<hf_home>/hub/models--<org>--<name>/`). This counts every cached file, not only `.safetensors` or `.bin` weights, so it slightly overestimates when tokenizer or extra files are present. A model that is not in the cache cannot be launched.

- **Multi-GPU Division**: Slices weights evenly across Tensor Parallel (TP) or Pipeline Parallel (PP) ranks.
- **Safety Overhead (1.08x)**: Allocates an additional 8% buffer for non-parameter CUDA context overhead, PyTorch activation buffers, and temporary tensor buffers.
- **CPU Offloading**: If `--cpu-offload-gb` is configured, offloaded gigabytes are deducted directly from the required GPU footprint.

---

## 2. Token KV Cache Sizing Formula

The memory required to store the Key-Value (KV) attention cache for a single context token on a single GPU is computed as:

```text
KV_Bytes_per_Token = 2 * Num_Layers * (Num_KV_Heads / Tensor_Parallel_Size) * Head_Dimension * Element_Size
```

| Parameter | Source in `config.json` | Description |
| :--- | :--- | :--- |
| `Num_Layers` | `num_hidden_layers` | Total transformer block depth. |
| `Num_KV_Heads` | `num_key_value_heads` | Number of KV attention heads (handles GQA). |
| `Head_Dimension` | `head_dim` or `hidden_size / num_attention_heads` | Dimensionality of each individual attention head. |
| `Element_Size` | `kv_cache_dtype` | 2 bytes for `float16`/`bfloat16`, 1 byte for `fp8`. |

---

## 3. Preflight Safety Floor & Overlap Checks

Before initiating container creation, the manager enforces two critical safety invariants:

1. **Fit Checks**: On every selected GPU, the weights must fit in currently free memory, weights plus KV cache must fit in both currently free memory and the `total × gpu_memory_utilization` budget, and at least 1 GiB of that budget must remain for KV cache after the weights.
2. **Multi-Container Overlap Guard**: The sum of `gpu_memory_utilization` across all active containers mapped to the same physical GPU device cannot exceed `1.0`.
