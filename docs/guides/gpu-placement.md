---
title: Multi-GPU Placement & Parallelism
sidebar_position: 2
---

# Multi-GPU Placement & Parallelism

Deploying large language models (such as 70B, 120B, or 405B parameter checkpoints) requires distributing computation across multiple GPUs.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/multi-gpu-placement.png" alt="Multi-GPU Placement & Parallelism Modes" />
  <div className="diagram-caption">Figure 3: Tensor Parallelism vs. Pipeline Parallelism and physical device pinning.</div>
</div>

---

## Parallelism Strategies

### 1. Tensor Parallelism (TP)
- **Mechanism**: Splits individual matrix multiplications (linear layers, MLP projections, attention projections) across GPU ranks simultaneously.
- **Interconnect**: Requires high-bandwidth GPU interconnects (NVLink or PCIe Gen 4/5).
- **Head Divisibility**: Total KV attention heads must be evenly divisible by the TP size (unless utilizing DeepSeek Multi-head Latent Attention / MLA architectures).

### 2. Pipeline Parallelism (PP)
- **Mechanism**: Partitions sequential transformer layers across GPU ranks in a conveyor-belt fashion.
- **Use Cases**: Useful when NVLink is unavailable or when the number of attention heads is not divisible by the GPU count.

---

## Explicit Device Pinning

By default, vLLM GPU Manager allocates the first $N$ available GPUs. To pin containers to specific physical devices:

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct \
  --port 8080 \
  --gpu-ids "0,1"
```

This passes the exact device requests down to Docker:
```text
--gpus '"device=0,1"'
```
