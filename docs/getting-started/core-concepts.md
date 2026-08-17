---
title: Core Concepts
sidebar_position: 3
---

# Core Concepts

Understanding the design principles and operational model of **vLLM GPU Manager**.

---

## 1. Managed vs. Unmanaged Containers

vLLM GPU Manager tags every Docker container it creates with metadata labels:

| Label | Value | Description |
| :--- | :--- | :--- |
| `vllm-manager.managed` | `true` | Declares that the container is owned by vLLM GPU Manager. |
| `vllm-manager.kind` | `server` \| `download` | Identifies serving containers vs background download containers. |
| `vllm-manager.model` | `<huggingface-model-id>` | Stores the normalized model repository identifier. |
| `vllm-manager.launch_spec` | `<json-encoded-spec>` | Preserves the exact parameters used to launch the container. |

> [!IMPORTANT]
> Containers created outside vLLM GPU Manager are ignored by the active model list and proxy router. They will, however, be detected during `nvidia-smi` GPU memory inspections to prevent memory oversubscription.

---

## 2. Container Exposure Modes

vLLM GPU Manager supports two networking and exposure architectures:

```mermaid
graph TD
    subgraph Proxy Mode Default
        ClientP[External Client] -->|Bearer sk-proxy| GW[Dashboard Gateway :9090]
        GW -->|127.0.0.1:8080| C1[Model 1 Container]
        GW -->|127.0.0.1:8081| C2[Model 2 Container]
    end

    subgraph Direct Mode --no-proxy
        ClientD1[External Client] -->|Direct :8080| C3[Model 1 Container :8080]
        ClientD2[External Client] -->|Direct :8081| C4[Model 2 Container :8081]
    end
```

### Proxy Mode (Default)
- Model containers bind strictly to `127.0.0.1:<port>`.
- Clients send requests to the central `/v1/` gateway on the dashboard port (e.g. `http://<host>:9090/v1`).
- Incoming requests are authenticated with a Proxy API key.
- The manager inspects the JSON payload (`{"model": "..."}`) and transparently forwards traffic to the matching model's local port.

### Direct Mode (`--no-proxy`)
- Containers bind to the configured `server_bind_ip` (e.g. `0.0.0.0` or a specific LAN IP).
- Clients connect directly to container ports (e.g., `http://<host>:8080/v1`).
- Global or per-container API keys can be passed directly as `--api-key` to vLLM.

---

## 3. VRAM Preflight & Memory Budgeting

vLLM allocates GPU memory eagerly at startup:

```text
Total VRAM Allocated = GPU Memory Utilization * Total GPU VRAM
```

Memory is divided into two primary zones:
1. **Model Weights**: Uncompressed parameter tensors in memory.
2. **KV Cache Blocks**: Pre-allocated memory for attention key-value caches across batch sequences.

The manager runs a multi-step preflight algorithm before launching any container:
- **Weight Calculation**: Inspects disk size of model files in the Hugging Face cache.
- **KV Cache Estimate**: Computes KV bytes per token:
  ```text
  KV Bytes/Token = 2 * Layers * ceil(KV Heads / TP Size) * Head Dimension * Element Bytes
  ```
- **Context Length Cap**: Multiplies KV bytes per token by requested `--max-model-len`.
- **Overlap Verification**: Ensures the sum of `mem_util` for all containers running on any single GPU does not exceed `1.0` (100%).

---

## 4. Hugging Face Hub Cache Hierarchy

vLLM GPU Manager uses the standard Hugging Face hub cache layout:

```text
/var/lib/vllm-manager/.cache/huggingface/
└── hub/
    ├── models--Qwen--Qwen2.5-Coder-7B-Instruct/
    │   ├── refs/
    │   │   └── main -> (commit hash)
    │   └── snapshots/
    │       └── <commit-hash>/
    │           ├── config.json
    │           ├── tokenizer.json
    │           └── model.safetensors
    └── tmp...
```

Containers mount the cache directory as `/hf_cache` with `HF_HOME=/hf_cache`. Model weights are shared across containers without duplicating files on disk.

---

## 5. Security & Isolation Model

- **Restricted File Modes**: `settings.json`, `service.env`, and TLS certificates are stored with `0600` or `0640` permissions, owned by the dedicated `vllm-manager` system account.
- **Automatic Log Redaction**: `HF_TOKEN`, `VLLM_API_KEY`, passwords, and header credentials are automatically redacted from dashboard logs and `audit.log`.
- **Sandboxed Cache Purging**: Model deletion is executed inside an ephemeral `alpine` container with target paths strictly verified to avoid escaping the hub directory.
