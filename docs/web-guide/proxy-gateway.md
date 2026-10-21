---
title: Proxy Gateway & User Management
sidebar_position: 6
---

# Proxy Gateway & User Management

Manage API access keys, monitor per-user inference request counters, and configure dynamic `/v1/` routes.

---

## Proxy Gateway Dashboard

The **Proxy Gateway** view provides administrative controls for the central OpenAI reverse proxy:

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>PROXY GATEWAY &amp; USER KEYS</span>
    <span>vLLM GPU Manager</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/05-proxy-gateway-users.png" alt="Proxy Gateway and User Keys" />
</div>

_Screenshot from a demo build with sample GPU data._

---

## Creating & Revoking Proxy API Keys

1. Click **Create Proxy User**.
2. Enter a descriptive username (e.g. `cursor-ide`, `developer-alice`).
3. Copy the generated API key (prefixed with `vllm_proxy_`).
4. To revoke access, click **Revoke** next to any active user.

---

## Dynamic Routing Table

The routing table maps model names to internal container ports:

- When a model container starts in proxy mode, one route is registered automatically: its `--served-model-name` if set, otherwise its Hugging Face model ID.
- Edited routes are validated against running containers. Each entry must use the routing name of the managed model listening on that port, so arbitrary aliases (for example `gpt-4o`) are rejected. Set `--served-model-name` at launch to expose a different name.
