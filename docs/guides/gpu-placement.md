---
title: GPU Placement & Multi-GPU
sidebar_position: 2
---

# GPU Placement & Multi-GPU

vLLM GPU Manager provides fine-grained controls for deploying models across single or multiple NVIDIA GPUs.

---

## Parallelism Modes

### 1. Tensor Parallelism (`--parallelism-mode tensor`)

**Tensor Parallelism (TP)** shards individual linear layers across GPUs. This is the default and highest-performance multi-GPU serving mode for models that fit within a single machine.

```bash
vllm-manager model start meta-llama/Llama-3.3-70B-Instruct \
  --port 8080 \
  --gpus 4 \
  --parallelism-mode tensor
```

This passes `--tensor-parallel-size 4` to vLLM and leaves pipeline parallelism at `1`.

> [!IMPORTANT]
> For standard Grouped Query Attention (GQA) models, the number of key-value attention heads (`num_key_value_heads`) must be divisible by the tensor parallel size. The preflight check automatically verifies this constraint before starting the container.

---

### 2. Pipeline Parallelism (`--parallelism-mode pipeline`)

**Pipeline Parallelism (PP)** divides sequential model layers across different GPUs. Useful when serving massive models across GPUs where NVLink high-speed interconnect is unavailable or tensor parallel head divisibility is not possible.

```bash
vllm-manager model start mistralai/Mistral-Large-Instruct-2407 \
  --port 8080 \
  --gpus 4 \
  --parallelism-mode pipeline
```

This passes `--pipeline-parallel-size 4` to vLLM and sets tensor parallelism to `1`.

---

## Explicit Device Pinning (`--gpu-ids`)

By default (`--gpu-ids all`), vLLM GPU Manager selects the first N available GPUs sorted by ID. To target specific physical GPUs:

```bash
# Run model A on GPUs 0 and 1
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct --port 8080 --gpus 2 --gpu-ids 0,1

# Run model B on GPUs 2 and 3
vllm-manager model start deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --port 8081 --gpus 2 --gpu-ids 2,3
```

### Docker Device Request Scoping

Unlike passing `--gpus all` to Docker (which makes all host GPUs visible inside the container), vLLM GPU Manager translates GPU selection into explicit device lists:

```text
--gpus device=0,1
```

This ensures the containerized process can only access its allocated GPUs, preventing CUDA device index confusion and protecting other workloads.

---

## GPU Memory Overlap & Multi-Tenancy

vLLM pre-allocates memory based on `--gpu-memory-utilization` (`mem_util`). Two containers can safely share the same physical GPU if their combined utilization does not exceed 100%:

```text
Container A (0.50 mem_util) + Container B (0.40 mem_util) = 0.90 (90%) -> ALLOWED
Container A (0.60 mem_util) + Container B (0.50 mem_util) = 1.10 (110%) -> REJECTED
```

The preflight engine inspects all running model containers and calculates committed memory per GPU before permitting a launch.

---

## GPU Topology & Interconnect Discovery

The web dashboard and `/api/config` endpoint query `nvidia-smi topo -m` to surface inter-GPU interconnects (NVLink, NVSwitch, PCIe, NUMA nodes):

```text
       GPU0   GPU1   GPU2   GPU3
GPU0    X     NV12   SYS    SYS
GPU1   NV12    X     SYS    SYS
GPU2   SYS    SYS     X     NV12
GPU3   SYS    SYS    NV12    X
```

Pairing tensor-parallel workloads across high-bandwidth NVLink links significantly reduces inter-token communication latency.
