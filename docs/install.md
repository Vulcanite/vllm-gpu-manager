---
title: Install
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install

## Requirements

| What | Needed |
| :--- | :--- |
| OS | 64-bit Linux (`x86_64` or `aarch64`). Packages are built for Debian/Ubuntu (`.deb`) and RHEL/Rocky/Fedora (`.rpm`). |
| GPU | NVIDIA, with a working driver and `nvidia-smi` in `PATH`. See [Hardware compatibility](./reference/hardware-compatibility.md) for FP8 and quantization limits. |
| Docker | Docker Engine with the NVIDIA Container Toolkit, so `docker run --gpus` works. |
| Disk | Room for the models you plan to cache. The download preflight checks free space per model. |

### Set up Docker GPU access

If `docker run --gpus all ...` does not work yet, install the NVIDIA Container Toolkit. On Ubuntu/Debian:

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Check it works
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Install vLLM GPU Manager

Download from [GitHub Releases](https://github.com/Vulcanite/vllm-gpu-manager/releases). The commands below use v1.1.1; replace the version for other releases.

<Tabs groupId="install">
<TabItem value="deb" label="Debian / Ubuntu" default>

```bash
sudo apt install ./vllm-manager_1.1.1_amd64.deb   # or _arm64.deb
```

</TabItem>
<TabItem value="rpm" label="RHEL / Rocky / Fedora">

```bash
sudo dnf install ./vllm-manager-1.1.1-1.amd64.rpm   # or .arm64.rpm
```

</TabItem>
<TabItem value="binary" label="Standalone binary">

```bash
curl -fsSL https://github.com/Vulcanite/vllm-gpu-manager/releases/download/v1.1.1/vllm-manager-v1.1.1-linux-amd64 -o vllm-manager
chmod +x vllm-manager
./vllm-manager version
```

A standalone binary keeps `settings.json`, the audit log and TLS certificates in the current directory. Run it as a user in the `docker` group.

</TabItem>
<TabItem value="source" label="From source">

Needs Go (the version in `go.mod`) and Node.js 20+.

```bash
git clone https://github.com/Vulcanite/vllm-gpu-manager.git
cd vllm-gpu-manager
(cd frontend && npm ci && npm run build)
go test ./...
CGO_ENABLED=0 go build -trimpath -o vllm-manager .

# Optional: install to /usr/local/bin with a systemd service
sudo ./scripts/install.sh
```

</TabItem>
</Tabs>

The `.deb` and `.rpm` packages install:

- `/usr/bin/vllm-manager`
- the systemd unit `vllm-manager.service`
- `/etc/vllm-manager/service.env` (service settings)
- `/var/lib/vllm-manager/` (settings, audit log, certificates and the default model cache)
- helper scripts in `/usr/share/vllm-manager/scripts/`

To run the dashboard as a service, see [Run as a service](./guides/service.md).

## Check the host

`doctor` checks every prerequisite in one go:

```bash
vllm-manager doctor
```

Example output (sample data):

| Status | Check | Detail |
| :--- | :--- | :--- |
| PASS | Operating system | linux |
| PASS | nvidia-smi | nvidia-smi found |
| PASS | NVIDIA driver CUDA | Driver 550.90.07; CUDA 12.4 |
| PASS | NVIDIA GPUs | 2 GPU(s) detected; first compute capability 8.9 |
| PASS | Docker CLI | docker CLI found |
| PASS | Docker API | Docker API ping succeeded |
| PASS | Docker NVIDIA runtime | Short-lived GPU test container succeeded |
| PASS | Hugging Face cache | Directory is readable and writable |

The dashboard and the `gpu`, `model` and `cache` commands run the same checks before they start, and stop if a required one fails. `help`, `version`, `doctor`, `settings` and `service` always run, so you can repair a broken setup.
