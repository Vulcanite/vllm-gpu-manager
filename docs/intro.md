---
title: Introduction
sidebar_position: 1
slug: /intro
---

# Introduction

**vLLM GPU Manager** is one Go binary with a CLI and an embedded web dashboard for running [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible servers as Docker containers on a single Linux host with NVIDIA GPUs.

It is deliberately small. It is not a cluster scheduler, a Kubernetes operator or a model platform. It is an operations tool for one GPU server where you want a clear view over Docker, `nvidia-smi`, the Hugging Face cache and vLLM's launch flags.

Read [why I built it](https://amishtp.tech/blog/vllm-gpu-manager) for the problem it came from and the decisions behind it.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/system-architecture.png" alt="vLLM GPU Manager architecture" />
  <div className="diagram-caption">How the CLI, dashboard, proxy and Docker fit together.</div>
</div>

## What it does

- **Checks memory before launch.** It estimates model weights plus KV cache for the context length you ask for, and refuses a launch that will not fit in the GPU memory you gave it.
- **Puts every model behind one endpoint.** A `/v1/` proxy on the dashboard port routes OpenAI-style requests to the right container, with per-user API keys.
- **Manages the model cache.** It downloads Hugging Face models in background containers, shows progress and failures, and deletes cached models safely.
- **Works from SSH or a browser.** Every lifecycle operation is a CLI command; the dashboard adds live logs, the proxy and a deploy form with a memory gauge.

## Core concepts

### Managed containers

Every container the manager creates carries labels, and only labelled containers appear in the model list and the proxy:

| Label | Value | Meaning |
| :--- | :--- | :--- |
| `vllm-manager.managed` | `true` | Created by vLLM GPU Manager. |
| `vllm-manager.kind` | `server` or `download` | A serving container or a background download. |
| `vllm-manager.model` | `<org>/<model>` | The Hugging Face model id. |
| `vllm-manager.launch_spec` | JSON | The launch parameters, used for restarts. |

Containers started outside the manager are ignored by the model list, but their GPU memory still shows up through `nvidia-smi`, so the memory checks account for it.

### Proxy mode and direct mode

```mermaid
graph TD
    subgraph Proxy mode, the default
        ClientP[Client] -->|Bearer proxy key| GW[Dashboard :9090 /v1]
        GW -->|127.0.0.1:8080| C1[Model 1]
        GW -->|127.0.0.1:8081| C2[Model 2]
    end

    subgraph Direct mode, --no-proxy
        ClientD1[Client] -->|:8080| C3[Model 1]
        ClientD2[Client] -->|:8081| C4[Model 2]
    end
```

- **Proxy mode** (default): model containers listen only on `127.0.0.1`. Clients call `http://<host>:9090/v1` with a proxy API key, and the manager forwards each request to the container named in its `model` field. See [Proxy & routing](./guides/proxy.md).
- **Direct mode** (`--no-proxy`, and all CLI launches): containers listen on `server_bind_ip` (default `0.0.0.0`) and clients call each port directly, optionally with a vLLM `--api-key`.

### The Hugging Face cache

All containers mount one Hugging Face home directory as `/hf_cache` with `HF_HOME=/hf_cache`, so a model is stored once and shared:

```text
/var/lib/vllm-manager/.cache/huggingface/
└── hub/
    └── models--Qwen--Qwen2.5-Coder-7B-Instruct/
        ├── blobs/
        ├── refs/main
        └── snapshots/<commit>/
            ├── config.json
            ├── tokenizer.json
            └── model.safetensors
```

See [Download & cache models](./guides/downloads.md).

## Where to go next

1. [Install](./install.md) the package or binary and run `vllm-manager doctor`.
2. Follow the [Quickstart](./quickstart.md) to download, start and query a model.
3. Read the [Security model](./security.md) before exposing the dashboard beyond `localhost`.
