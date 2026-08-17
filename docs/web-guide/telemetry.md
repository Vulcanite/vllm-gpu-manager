---
title: Hardware Telemetry & Monitoring
sidebar_position: 2
---

# Hardware Telemetry & Monitoring

Real-time visibility into physical NVIDIA GPU resources, compute processes, and cluster memory.

---

## Real-Time GPU Gauges

The top of the Web Console displays live telemetry cards for every visible NVIDIA GPU:

- **GPU Utilization**: Real-time core activity percentage queried from `nvidia-smi`.
- **Memory Allocation Bar**: Visual breakdown of used VRAM vs. free VRAM.
- **Compute Capability**: Hardware architecture generation (e.g. `9.0 Hopper`, `8.9 Ada`, `8.0 Ampere`).
- **Device UUID**: Canonical NVIDIA device identifier for hardware troubleshooting.

---

## Active Compute Process Table

Below the GPU cards, the console displays all active GPU compute processes:

| Column | Description |
| :--- | :--- |
| **GPU ID** | Physical device index hosting the process. |
| **PID** | Linux OS process identifier. |
| **User** | Linux OS username owning the process (resolved via `/proc/<pid>`). |
| **Process Name** | Executable command (e.g. `python3 (vLLM::Engine)`). |
| **Mem Used** | Exact physical VRAM consumed by the process. |

> [!TIP]
> Resolving the OS user who owns each GPU process helps system administrators identify rogue training jobs or unauthorized inference instances consuming VRAM on multi-tenant nodes.
