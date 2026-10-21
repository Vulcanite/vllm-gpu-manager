---
title: GPU Management & Diagnostics
sidebar_position: 2
---

# GPU Management & Diagnostics

Query physical NVIDIA GPUs, inspect memory allocations, and diagnose host prerequisites.

---

## Listing GPU Devices (`gpu list`)

Display visible NVIDIA GPUs, memory statistics, compute capabilities, and active processes with process owners:

```bash
vllm-manager gpu list
```

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>GPU TELEMETRY &amp; PROCESS INSPECTION</span>
    <span>vllm-manager gpu list</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/cli/01-cli-gpu-list.png" alt="vllm-manager gpu list output" />
</div>

_Illustrative mock-up rendered from sample data. Real `vllm-manager` output differs in layout and wording._

---

## Host Health Diagnostics (`doctor`)

Run an automated end-to-end prerequisite check:

```bash
vllm-manager doctor
```

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>DIAGNOSTIC HEALTH AUDIT</span>
    <span>vllm-manager doctor</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/cli/02-cli-doctor.png" alt="vllm-manager doctor output" />
</div>

_Illustrative mock-up rendered from sample data. Real `vllm-manager` output differs in layout and wording._

The doctor diagnostic verifies:
1. `nvidia-smi` presence and driver version.
2. GPU compute capabilities.
3. Docker Engine CLI and socket connectivity.
4. NVIDIA Container Toolkit runtime configuration.
5. Hugging Face cache directory permissions and free storage.
6. Service configuration file (`/etc/vllm-manager/service.env`).
