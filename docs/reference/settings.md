---
title: Settings & Config Reference
sidebar_position: 3
---

# Settings & Config Reference

Detailed schema and environment variables for configuring vLLM GPU Manager.

---

## `settings.json` Schema

Settings are persisted in `settings.json` (located at `/var/lib/vllm-manager/settings.json` in packaged installations):

```json
{
  "hf_cache": "/var/lib/vllm-manager/.cache/huggingface",
  "hf_token": "hf_...",
  "vllm_api_key": "token-...",
  "proxy_users": [
    {
      "username": "alice",
      "api_key": "vllm_proxy_5f8a92...",
      "request_count": 42
    }
  ],
  "server_api_keys": {
    "vllm-model-8080": "custom-key-..."
  },
  "enable_cuda_compatibility": false,
  "hf_hub_offline": false,
  "vllm_image": "vllm/vllm-openai:v0.23.0",
  "server_bind_ip": "0.0.0.0",
  "routing_table": {
    "meta-llama/Llama-3.3-70B-Instruct": 8080
  }
}
```

### Field Descriptions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hf_cache` | String | `~/.cache/huggingface` | Hugging Face root home directory. |
| `hf_token` | String | `""` | Hugging Face access token for gated models. |
| `vllm_api_key` | String | `""` | Global upstream API key passed to vLLM containers. |
| `proxy_users` | Array | `[]` | List of authorized proxy client users and request counts. |
| `server_api_keys` | Map | `{}` | Per-container API key overrides indexed by container name. |
| `enable_cuda_compatibility`| Boolean | `false` | Injects `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. |
| `hf_hub_offline` | Boolean | `false` | Injects `HF_HUB_OFFLINE=1` into newly launched servers. |
| `vllm_image` | String | `vllm/vllm-openai:v0.23.0` | Default Docker image used for downloads and serving. |
| `server_bind_ip` | String | `0.0.0.0` | Default direct bind IP for containers outside proxy mode. |
| `routing_table` | Map | `{}` | Model name/alias to container port mapping. |
| `proxy_max_body_bytes` | Integer | `33554432` (32 MiB) | Maximum POST body size the `/v1/` proxy will buffer to sniff the `model` field. Requests over the limit return `413`. |

---

## Environment Variables

| Variable | Scope | Description |
| :--- | :--- | :--- |
| `VLLM_MANAGER_CONFIG_DIR` | System | Directory containing `service.env` (default `/etc/vllm-manager`). |
| `VLLM_MANAGER_STATE_DIR` | System | Directory containing `settings.json`, `audit.log`, and `certs/` (default `/var/lib/vllm-manager`). |
| `VLLM_MANAGER_HOST` | Daemon | Dashboard and proxy bind IP (default `127.0.0.1`). |
| `VLLM_MANAGER_PORT` | Daemon | Dashboard and proxy TCP port (default `9090`). |
| `VLLM_MANAGER_HTTPS` | Daemon | Enables HTTPS with auto-generated TLS certificates (`true`/`false`). |
| `VLLM_MANAGER_NO_PROXY` | Daemon | Disables `/v1/` reverse proxy gateway (`true`/`false`). |
| `VLLM_MANAGER_MONITORING` | Daemon | Enables Prometheus target synchronization (`true`/`false`). |
| `VLLM_MANAGER_PASSWORD` | Daemon | Password required to access dashboard and JSON APIs. |
