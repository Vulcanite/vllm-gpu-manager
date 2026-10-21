---
title: Web Console Overview
sidebar_position: 1
---

# Web Management Console

The **vLLM GPU Manager Web Console** is an embedded React management dashboard that runs directly inside the single Go binary (`go:embed`).

It provides real-time cluster telemetry, an interactive VRAM preflight deployment calculator, live Server-Sent Events (SSE) log streaming, and an OpenAI proxy management gateway.

---

## Accessing the Web Console

Start the dashboard daemon on your GPU host:

```bash
vllm-manager --dashboard --port 9090 --host 0.0.0.0
```

Open your browser at `http://<your-server-ip>:9090` (or `https://...` if `--https` is enabled).

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>DASHBOARD OVERVIEW</span>
    <span>vLLM GPU Manager Web Console</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/01-dashboard-overview.png" alt="vLLM GPU Manager Dashboard Overview" />
</div>

_Screenshot from a demo build with sample GPU data._

---

## Core Sections & Capabilities

- **[Hardware Telemetry](./telemetry.md)**: Real-time GPU utilization, memory gauges, thermal metrics, and process tables showing PID, User, and VRAM.
- **[Deploying Models](./deploying-models.md)**: Visual deployment modal with live VRAM preflight estimation bar and advanced runtime configuration.
- **[Model Management & Logs](./model-management.md)**: Container lifecycle management (Start, Stop, Restart, Remove) and live SSE log streams.
- **[Cache & Downloads](./cache-and-downloads.md)**: Browse Hugging Face model cache, track download containers, and safely delete models.
- **[Proxy Gateway](./proxy-gateway.md)**: Generate and revoke Proxy API keys, view request counters, and manage `/v1/` routes.
- **[Settings & Authentication](./settings-and-auth.md)**: Configure cache storage path, dashboard passwords, and CUDA compatibility.
