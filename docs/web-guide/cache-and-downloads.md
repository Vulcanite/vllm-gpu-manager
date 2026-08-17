---
title: Cache & Model Downloads
sidebar_position: 5
---

# Cache & Model Downloads

Browse downloaded model repositories in the Hugging Face cache, trigger background downloads, and safely purge unused checkpoints.

---

## Hugging Face Cache Browser

The **Cache** tab displays all downloaded model directories in the Hugging Face storage volume:

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>HUGGING FACE MODEL CACHE</span>
    <span>vLLM GPU Manager</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/04-huggingface-cache-browser.png" alt="Hugging Face Cache Browser" />
</div>

---

## Background Download Containers

1. Enter any Hugging Face model identifier (e.g. `meta-llama/Llama-3.3-70B-Instruct`).
2. Click **Download Model**.
3. The manager launches an isolated background container that streams download progress and tracks byte counts.

---

## Safe Model Deletion

Clicking **Delete** on a cached model triggers a safety evaluation:

1. **Active Server Lock**: Deletion is blocked if the model is currently being served by any container.
2. **Path Traversal Check**: Confirms the path resides strictly inside the root Hugging Face hub folder.
3. **Sandboxed Alpine Purge**: Deletion executes in an ephemeral Alpine container to cleanly resolve file permissions without host root escalation.
