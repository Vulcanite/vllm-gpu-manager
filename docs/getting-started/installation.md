---
title: Installation
sidebar_position: 1
---

# Installation

**vLLM GPU Manager** is distributed as native Debian/Ubuntu packages (`.deb`), Red Hat/Rocky/Fedora packages (`.rpm`), portable standalone Linux binaries (AMD64 & ARM64), and source code.

---

## 1. Debian / Ubuntu Package (`.deb`)

Download the appropriate `.deb` package from the [GitHub Releases](https://github.com/Vulcanite/vllm-gpu-manager/releases) page:

```bash
# For AMD64 (x86_64)
sudo apt update
sudo apt install ./vllm-manager_1.0.0_amd64.deb

# For ARM64 (aarch64)
sudo apt update
sudo apt install ./vllm-manager_1.0.0_arm64.deb
```

The Debian package installs:
- **Binary**: `/usr/bin/vllm-manager`
- **Systemd Unit**: `/usr/lib/systemd/system/vllm-manager.service`
- **Service Configuration**: `/etc/vllm-manager/service.env`
- **State & Model Cache**: `/var/lib/vllm-manager/`
- **Helper Scripts**: `/usr/share/vllm-manager/scripts/`

Configure and enable the service:

```bash
sudo vllm-manager service configure --port 9090
sudo systemctl enable --now vllm-manager
```

---

## 2. Red Hat / Rocky / Fedora Package (`.rpm`)

Download the `.rpm` package from the [GitHub Releases](https://github.com/Vulcanite/vllm-gpu-manager/releases) page:

```bash
# For AMD64 (x86_64)
sudo dnf install ./vllm-manager-1.0.0-1.amd64.rpm
# or with rpm:
sudo rpm -ivh ./vllm-manager-1.0.0-1.amd64.rpm

# For ARM64 (aarch64)
sudo dnf install ./vllm-manager-1.0.0-1.arm64.rpm
```

Configure and start the service:

```bash
sudo vllm-manager service configure --port 9090
sudo systemctl enable --now vllm-manager
```

---

## 3. Portable Standalone Binary

For portable or non-root environments, download the standalone Linux binary:

```bash
# Download AMD64 binary
curl -fsSL https://github.com/Vulcanite/vllm-gpu-manager/releases/download/v1.0.0/vllm-manager-v1.0.0-linux-amd64 -o vllm-manager
chmod +x vllm-manager

# Verify installation
./vllm-manager version
./vllm-manager doctor
```

Portable binaries store their `settings.json`, audit logs, and self-signed certificates in the current working directory.

---

## 4. Build from Source

### Prerequisites
- **Go**: 1.23 or newer
- **Node.js & npm**: Node.js 20+

### Build Steps

```bash
git clone https://github.com/Vulcanite/vllm-gpu-manager.git
cd vllm-gpu-manager

# Install frontend dependencies and compile web assets
cd frontend
npm ci
npm run build
cd ..

# Run backend unit tests and build standalone binary
go test ./...
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X main.version=1.0.0" -o vllm-manager .
```

### System Installation Helper

To install the built binary to `/usr/local/bin` and create the `vllm-manager` systemd service:

```bash
sudo ./scripts/install.sh
```

---

## Host Validation (`doctor`)

After installation, run the diagnostic check to verify that NVIDIA drivers, Docker, and the container toolkit are operational:

```bash
vllm-manager doctor
```

Example output:

| Status | Check | Detail |
| :--- | :--- | :--- |
| **PASS** | Operating system | linux |
| **PASS** | nvidia-smi | nvidia-smi found |
| **PASS** | NVIDIA driver CUDA | Driver 550.90.07; CUDA 12.4 |
| **PASS** | NVIDIA GPUs | 2 GPU(s) detected; first compute capability 8.9 |
| **PASS** | Docker CLI | docker CLI found |
| **PASS** | Docker API | Docker API ping succeeded |
| **PASS** | Docker NVIDIA runtime | Short-lived GPU test container succeeded |
| **PASS** | Hugging Face cache | Directory is readable and writable |
