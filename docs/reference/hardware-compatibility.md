---
title: Hardware compatibility
sidebar_position: 4
---

# Hardware compatibility

Which NVIDIA GPUs support which precisions. vLLM itself sets the real limits; this page covers what the manager checks.

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
- **What the manager does**: an FP8 model on a GPU below 8.9 is refused before launch and before download, instead of failing inside vLLM.

### 2. AWQ & GPTQ (4-Bit Weight Quantization)
- **Requirements**: Supported on Compute Capability $\ge 7.0$ (Volta and newer).
- **Effect**: roughly a quarter of the 16-bit weight memory.

### 3. BF16 vs FP16
- `bfloat16` is recommended for Ampere, Ada, and Hopper GPUs to match the training precision of modern open-weights models (Llama 3, Qwen 2.5, DeepSeek).
- `float16` is supported on Turing and older architectures.

---

## Driver and Docker

Docker must be able to start GPU containers (`docker run --gpus all ...`). See [Install](../install.md#set-up-docker-gpu-access) for the NVIDIA Container Toolkit setup, and run `vllm-manager doctor` to confirm.
