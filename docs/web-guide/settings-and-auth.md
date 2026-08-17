---
title: Settings & Web Authentication
sidebar_position: 7
---

# Settings & Web Authentication

Configure storage paths, update authentication secrets, and customize daemon parameters.

---

## Settings Interface

The **Settings** view displays current system configuration, host RAM, and cache storage statistics:

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>SETTINGS CONFIGURATION</span>
    <span>vLLM GPU Manager</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/06-settings-configuration.png" alt="Settings Configuration" />
</div>

---

## Configurable Options

- **Hugging Face Cache Path**: Modify the storage directory used for model weight downloads (only changeable when no containers are active).
- **Hugging Face Access Token**: Set or update the `HF_TOKEN` secret used to download gated models (e.g. Llama 3, Gemma).
- **Global vLLM API Key**: Configure the default authentication token passed to downstream containers.
- **CUDA Compatibility Mode**: Toggle `VLLM_ENABLE_CUDA_COMPATIBILITY=1` for forward compatibility on newer CUDA drivers.
- **Server Bind IP**: Default network address for container exposure in direct mode.
