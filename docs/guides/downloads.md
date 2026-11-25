---
title: Download & cache models
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Download & cache models

## Where models are stored

The manager uses one Hugging Face home directory for every container. Packaged installs default to `/var/lib/vllm-manager/.cache/huggingface`; model files live under its `hub/` folder (`hub/models--<org>--<model>`).

To use another disk, change it while no managed containers are running:

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

**Settings** → **Hugging Face Cache Path**.

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager settings set --hf-cache /mnt/models
```

</TabItem>
</Tabs>

### Gated models

Gated or private models (for example `meta-llama/Llama-3.3-70B-Instruct`) need a Hugging Face token. Set it once; it is stored in `settings.json` (mode `0600`) and passed as `HF_TOKEN` to download and serving containers.

```bash
vllm-manager settings set --hf-token <your-token>
```

## Download a model

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

On **Models**, type a model id and click **Download**. The progress bar compares bytes on disk with the size of the files being downloaded. When it ends, the panel shows **Download completed**, or the exit code and the end of the container log if it failed.

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>MODEL CACHE</span>
    <span>Demo build, sample data</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/04-huggingface-cache-browser.png" alt="Downloaded models in the dashboard" />
</div>

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager model download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
vllm-manager model progress deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
vllm-manager model cached
```

</TabItem>
</Tabs>

### What happens during a download

1. **The id is checked.** It must be a Hugging Face repo id such as `org/model-name` (letters, digits, `_`, `-`, `.`). Anything else is rejected before any network or Docker call.
2. **The size is checked against free disk.** The manager lists the repo's files from the Hugging Face API (`/api/models/<id>?blobs=true`), adds up the files it will download, and refuses to start if that plus 10% does not fit. If the API cannot be reached, the download starts with a warning instead.
3. **A background container fetches the files.** It uses the configured vLLM image with host networking, roughly:

   ```bash
   docker run -d --network host \
     -v <hf_cache>:/hf_cache -e HF_HOME=/hf_cache \
     --entrypoint hf vllm/vllm-openai:v0.23.0 \
     download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
     --exclude "*.bin" "*.pt" "*.pth" "*.onnx" "*.gguf" ...
   ```

   The `--exclude` list is added only when the repo ships `.safetensors`. It skips weight formats vLLM does not load, which can be most of the repo.
4. **Progress reports the result.** `model progress` (or the dashboard panel) shows `running`, `succeeded`, `failed` with the exit code and the end of the log, or `stopped`. The finished container is removed once its result has been read.

### Interrupted downloads

A model only counts as downloaded when a snapshot has `config.json` (or `params.json`) and no `*.incomplete` files remain. If a download stops early, start it again: it resumes from what is already on disk.

GGUF repos are refused; vLLM's GGUF support needs a separate plugin.

## Delete a cached model

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

On **Models**, click **Delete** next to a downloaded model.

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager cache delete Qwen/Qwen2.5-7B-Instruct
```

</TabItem>
</Tabs>

Deletion is refused while a container serves the model or a download for it is still running. The path is checked to stay inside the `hub/` folder, and the files are removed by a short-lived `alpine` container so file ownership inside the cache does not matter.
