---
title: Hardware & GPU Compatibility
sidebar_position: 4
---

# Hardware & GPU Compatibility

Architecture requirements, compute capability matrix, and quantization guidelines for NVIDIA GPUs.

---

## NVIDIA GPU Compute Capabilities

| Architecture | Example GPUs | Compute Capability | Native FP8 | Tensor Parallel |
| :--- | :--- | :--- | :--- | :--- |
| **Blackwell** | B200, B100, GB200 | `10.0` | Yes | Yes (NVLink / PCIe) |
| **Hopper** | H100, H200, H800 | `9.0` | Yes | Yes (NVLink) |
| **Ada Lovelace** | RTX 4090, L40, L40S, 6000 Ada | `8.9` | Yes | Yes (PCIe) |
| **Ampere** | A100, A10, A30, A40, RTX 3090 | `8.0`, `8.6` | No | Yes (NVLink / PCIe) |
| **Turing** | T4, Quadro RTX 8000, RTX 2080 | `7.5` | No | Yes (PCIe) |
| **Volta** | V100, Titan V | `7.0` | No | Yes (NVLink / PCIe) |

---

## Quantization Support Matrix

### 1. FP8 (W8A8 / E4M3 / E5M2)
- **Requirements**: NVIDIA Compute Capability $\ge 8.9$ (Ada Lovelace, Hopper, Blackwell).
- **Behavior**: Attempting to launch native FP8 checkpoints on Ampere ($8.0$) or older GPUs is blocked during launch preflight to prevent CUDA kernel aborts.

### 2. AWQ & GPTQ (4-Bit Weight Quantization)
- **Requirements**: Supported on Compute Capability $\ge 7.0$ (Volta and newer).
- **Behavior**: Drastically reduces VRAM footprint with minimal perplexity degradation.

### 3. BF16 vs FP16
- `bfloat16` is recommended for Ampere, Ada, and Hopper GPUs to match the training precision of modern open-weights models (Llama 3, Qwen 2.5, DeepSeek).
- `float16` is supported on Turing and older architectures.

---

## Docker & Container Toolkit Requirements

1. **NVIDIA Driver**: Version $\ge 535.86.10$ for CUDA 12.x workloads.
2. **NVIDIA Container Toolkit**: Must be installed and configured in `/etc/docker/daemon.json`:
   ```json
   {
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```
