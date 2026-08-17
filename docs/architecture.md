---
title: System Architecture
sidebar_position: 1
---

# System Architecture

**vLLM GPU Manager** is engineered as a single, statically-linked Go binary with an embedded React frontend and zero external database dependencies.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/system-architecture.png" alt="vLLM GPU Manager End-to-End System Architecture" />
  <div className="diagram-caption">Figure 5: End-to-end component layers, synchronization mutexes, and Docker engine controller.</div>
</div>

---

## Architectural Layers

### 1. Ingestion & HTTP Gateway
- **ServeMux Router**: Multiplexes Web UI assets, administrative REST APIs (`/api/*`), SSE log streams, and OpenAI proxy routes (`/v1/*`).
- **Session Authentication & Sanitization**: Verifies admin sessions, validates proxy bearer tokens, and redacts sensitive credentials from logs.

### 2. Daemon Core
- **In-Memory Settings Cache**: Protected by `sync.RWMutex`, synchronizing state atomically to `/etc/vllm-manager/settings.json`.
- **VRAM Preflight Engine**: Proactively parses model architectures, computes layer weights and KV cache sizing, and prevents CUDA Out-of-Memory faults.
- **Docker SDK Lifecycle Controller**: Dispatches container creation, health monitoring, and lifecycle commands directly to the Docker API socket.

### 3. Host Infrastructure & NVIDIA Hardware
- **Docker Engine (`/var/run/docker.sock`)**: Runs isolated `vllm/vllm-openai` containers.
- **NVIDIA Driver & Container Toolkit**: Maps physical GPU devices (`--gpus device=0,1`) and executes high-bandwidth tensor parallel communication over NVLink/PCIe.
- **Hugging Face Model Cache**: Shared local host volume (`/var/lib/vllm-manager/.cache/huggingface`) mounted read-only into serving containers.
