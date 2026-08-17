---
title: Hugging Face Cache Ops
sidebar_position: 4
---

# Hugging Face Cache Ops

Manage local model storage, track background downloads, and safely remove cached model directories.

---

## Cache Configuration

vLLM GPU Manager uses the standard Hugging Face home directory. In packaged installations, the default path is:

```text
/var/lib/vllm-manager/.cache/huggingface
```

The hub cache files reside in the `hub/` subdirectory (e.g. `hub/models--<org>--<model>`).

### Changing the Cache Path

You can configure a custom cache directory (such as a mounted NVMe array or high-speed network volume `/mnt/models`) via:

1. **Web Console**: Navigate to **Settings** $\rightarrow$ **Hugging Face Cache Path**.
2. **CLI**:
   ```bash
   vllm-manager settings set --hf-cache /mnt/models
   ```

> [!CAUTION]
> The cache directory can only be changed when **no managed vLLM containers are currently running**. This prevents active containers from pointing to stale storage.

---

## Hugging Face Authentication Token

Private or gated models (such as `meta-llama/Llama-3.3-70B-Instruct` or `google/gemma-2-27b-it`) require an authenticated token:

```bash
vllm-manager settings set --hf-token hf_yourTokenHere
```

The token is stored in `settings.json` with file permissions `0600` and is automatically injected as `HF_TOKEN` into download and serving containers.

---

## Downloading Models

Start a background download container:

```bash
vllm-manager model download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

Downloads run in a detached container with host networking, equivalent to:

```bash
docker run -d --rm \
  --network host \
  -v <hf_cache>:/hf_cache \
  -e HF_HOME=/hf_cache \
  --entrypoint hf \
  vllm/vllm-openai:v0.23.0 \
  download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

### Preflight Space Verification
Before starting a download, the backend queries the Hugging Face API (`/api/models/<id>`) to retrieve the total model weight size in bytes. It inspects free disk space on the cache filesystem and blocks the download if free space is insufficient.

---

## Deleting Cached Models

Delete models cleanly through the web console or CLI:

```bash
vllm-manager cache delete Qwen/Qwen2.5-7B-Instruct
```

### Safety Protections

Deletion is strictly blocked if:
1. An active container is currently serving that model.
2. A background download container is currently fetching that model.
3. The target model directory attempts to escape the root Hugging Face hub path.

Deletion is executed inside an ephemeral `alpine` container mounted strictly at the hub path to ensure correct file permission handling.
