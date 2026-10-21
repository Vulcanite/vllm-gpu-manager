---
title: Security Model
sidebar_position: 5
---

# Security Model

Security principles, isolation boundaries, authentication mechanisms, and operational best practices.

---

## 1. Threat Model & Isolation Boundaries

vLLM GPU Manager is designed for **internal infrastructure operations** on trusted networks.

```mermaid
graph LR
    subgraph Trusted Network
        Admin[Administrator]
        App[Internal App]
    end

    subgraph Host Server
        Proxy[OpenAI Proxy :9090]
        vLLMDaemon[vLLM Manager Daemon]
        DockerGroup[Docker Group (Root-equivalent)]
        Containers[vLLM Containers]
    end

    Admin -->|Dashboard Auth Session| vLLMDaemon
    App -->|Bearer Proxy Token| Proxy
    Proxy --> Containers
    vLLMDaemon --> DockerGroup
```

> [!WARNING]
> The `vllm-manager` service account belongs to the `docker` group. Docker socket access is effectively root-equivalent on Linux systems. Never expose unauthenticated dashboard ports to public untrusted networks.

---

## 2. Authentication & Credential Handling

### Dashboard Web Authentication
- When `VLLM_MANAGER_PASSWORD` is configured, the web UI and all `/api/*` endpoints require an authenticated session.
- Each login mints a unique 256-bit session token stored server-side with a 24-hour expiry. Logout deletes that token, so a stolen cookie cannot be reused after sign-out.
- Login is rate-limited to five failed attempts per client IP per minute (`429 Too Many Requests`).
- Session verification looks up the cookie against the in-memory session table. Sessions do not survive a dashboard restart.
- The dashboard password is not hashed. In systemd installations it is stored in `/etc/vllm-manager/service.env` (mode `0640`); the `manager.sh` helper stores it in `.vllm-manager.state` (mode `0600`). Treat both files as secrets.

### Proxy Client Tokens
- The `/v1/` OpenAI proxy gateway verifies `Authorization: Bearer <proxy_api_key>`.
- Invalid or revoked tokens return `401 Unauthorized` immediately without forwarding requests to model containers.

---

## 3. Secret Redaction

Redaction is applied at specific points rather than to every log line:

- **API error logs** pass through [`redactSensitive`](https://github.com/Vulcanite/vllm-gpu-manager/blob/master/http_helpers.go#L45), which scrubs `HF_TOKEN=`, `hf_token=`, `VLLM_API_KEY=`, `VLLM_MANAGER_PASSWORD=`, `--api-key` values, and `-H "Authorization: ..."` arguments.
- **Docker API operation logs** redact `HF_TOKEN=` environment values and `--api-key` arguments before logging container commands.
- **Audit log parameters** drop keys that look like secrets (`*api_key*`, `token`, `*_token`, and similar) before the event is written to `audit.log` (mode `0600`).

Free-form error text stored in audit events is not passed through `redactSensitive`.

---

## 4. Path Traversal & Escape Prevention

Hugging Face cache deletion requests verify:
1. The target model directory cannot contain null bytes (`\x00`).
2. The resolved target path must reside strictly inside the root Hugging Face hub directory (`filepath.Dir(modelPath) == hubPath`).
3. Model deletions are executed inside an ephemeral unprivileged `alpine` container with the hub directory as its only volume mount.
