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

---

## Creating & Revoking Proxy API Keys

1. Click **Create Proxy User**.
2. Enter a descriptive username (e.g. `cursor-ide`, `developer-alice`).
3. Copy the generated API key (prefixed with `vllm_proxy_`).
4. To revoke access, click **Revoke** next to any active user.

---

## Dynamic Routing Table

The routing table maps model names and aliases to internal container ports:

- When an active model container starts, its model ID and `--served-model-name` alias are registered automatically.
- Custom routing entries can be added to alias third-party model names (e.g. mapping `gpt-4o` to a local `Qwen2.5-Coder-32B` container).
