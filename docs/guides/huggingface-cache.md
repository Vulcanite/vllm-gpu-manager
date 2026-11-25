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
docker run -d \
  --network host \
  -v <hf_cache>:/hf_cache \
  -e HF_HOME=/hf_cache \
  --entrypoint hf \
  vllm/vllm-openai:v0.23.0 \
  download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --exclude "*.bin" "*.pt" "*.pth" "*.onnx" "*.gguf" ...
```

The `--exclude` list is added only when the repo ships `.safetensors`; it skips weight formats vLLM does not load. The container is kept after it exits so its result can be read.

Check on a download:

```bash
vllm-manager model progress deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

It reports `running`, `succeeded`, `failed` (with the exit code and the end of the log) or `stopped`, then removes the finished container. A model only counts as downloaded when a snapshot has `config.json` (or `params.json`) and no `*.incomplete` blobs remain, so running `model download` again after an interruption resumes it.

### Model IDs
Model IDs must be Hugging Face repo ids such as `org/model-name` (letters, digits, `_`, `-`, `.`). Anything else is rejected before any network or Docker call.

### Preflight Space Verification
Before starting a download, the backend lists the repo's files from the Hugging Face API (`/api/models/<id>?blobs=true`) and adds up the files that will actually be downloaded. It compares that, plus 10%, with free space on the cache filesystem and blocks the download if it will not fit.

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
