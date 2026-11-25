---
title: Architecture
sidebar_position: 1
---

# Architecture

vLLM GPU Manager is one statically linked Go binary. The React dashboard is embedded in it with `go:embed`, and state is a JSON file, so there is no database or other runtime to install.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/system-architecture.png" alt="vLLM GPU Manager architecture" />
  <div className="diagram-caption">Clients, the HTTP layer, the core and the host it drives.</div>
</div>

## HTTP layer (dashboard mode)

- One `net/http` mux serves the dashboard assets, the JSON API under `/api/`, live log streams (Server-Sent Events) and the OpenAI-compatible proxy under `/v1/`.
- `/api/` requires a dashboard session when `VLLM_MANAGER_PASSWORD` is set, and only answers local or allow-listed Host names when it is not. `/v1/` requires a proxy user key. See [Security](./security.md).

## Core

- **Settings**: an in-memory copy of `settings.json` behind a lock, written to the state directory with a temp file and rename so a crash cannot truncate it.
- **Memory check**: reads the model's `config.json` and cache size and refuses launches that will not fit. See [GPU placement & VRAM](./guides/gpu-and-vram.md).
- **Container control**: creates, inspects, restarts and removes containers through the Docker Engine API. Launches and restarts hold one process-wide lock while they check and start.

The CLI calls the same core directly, so it works without the dashboard running.

## Host

- **Docker** (`/var/run/docker.sock`) runs the `vllm/vllm-openai` containers, the background download containers and short-lived `alpine` containers that delete cached models.
- **NVIDIA driver and Container Toolkit** give containers their GPUs (`--gpus device=0,1`); `nvidia-smi` supplies memory, utilization, processes and topology.
- **Hugging Face cache**: one host directory mounted into every container as `/hf_cache`.
