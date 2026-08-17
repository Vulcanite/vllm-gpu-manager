---
title: Prerequisites & System Requirements
sidebar_position: 2
---

# Prerequisites & System Requirements

Before installing and operating **vLLM GPU Manager**, verify that your host server meets the following hardware, driver, and container runtime requirements.

---

## 1. Hardware Matrix

| Hardware | Requirement | Recommendation |
| :--- | :--- | :--- |
| **NVIDIA GPUs** | Pascal or newer (Compute Capability &ge; 6.0) | Hopper (H100/H200), Ada Lovelace (RTX 4090/L40S), or Ampere (A100) |
| **Native FP8 Support** | Compute Capability &ge; 8.9 | Required for FP8 weights and FP8 KV cache quantization |
| **Host System RAM** | Minimum 32 GB RAM | &ge; 2x the size of weights if utilizing CPU offloading (`--cpu-offload-gb`) |
| **Disk Storage** | NVMe SSD with &ge; 200 GB free | High-speed storage for Hugging Face model weight caching |
| **Interconnect** | PCIe Gen 4/5 or NVLink | NVLink is strongly recommended for high-bandwidth Tensor Parallelism (TP) |

---

## 2. Host Operating System

vLLM GPU Manager runs on modern 64-bit Linux distributions:

- **Debian / Ubuntu**: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12 (Bookworm)
- **Red Hat Enterprise Linux / Rocky / Alma / Fedora**: RHEL 8/9, Rocky Linux 9, Fedora 39+
- **Architectures**: `x86_64` (AMD64) and `aarch64` (ARM64)

---

## 3. NVIDIA Driver & CUDA

- **NVIDIA Linux Driver**: Version &ge; 535.86.10 (CUDA 12.2+) or &ge; 550.54.14 (CUDA 12.4+).
- **Driver Verification**: Verify `nvidia-smi` is installed and operational:

```bash
nvidia-smi
```

---

## 4. Docker Engine & NVIDIA Container Toolkit

vLLM GPU Manager manages official `vllm/vllm-openai` containers through the Docker Engine API (`/var/run/docker.sock`).

### 1. Install Docker Engine
Install Docker Community Edition (version &ge; 20.10):

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
```

### 2. Install the NVIDIA Container Toolkit
Install the `nvidia-container-toolkit` package so Docker can expose GPU devices to containers:

```bash
# On Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Test GPU Container Execution
Verify that Docker can successfully execute a GPU container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 5. Host Preflight Verification (`doctor`)

After installing the `vllm-manager` package or binary, run the automated diagnostic tool to verify all prerequisites in one step:

```bash
vllm-manager doctor
```

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>DIAGNOSTIC PREFLIGHT VERIFICATION</span>
    <span>vllm-manager doctor</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/cli/02-cli-doctor.png" alt="vllm-manager doctor output" />
</div>
