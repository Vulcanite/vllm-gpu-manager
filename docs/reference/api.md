---
title: REST API Reference
sidebar_position: 2
---

# REST API Reference

The vLLM GPU Manager web console communicates with the backend via JSON REST APIs under `/api`.

---

## Authentication Endpoints

### `GET /api/auth/status`
Check whether dashboard authentication is required and whether the current session is valid.

**Response**:
```json
{
  "required": true,
  "authenticated": true
}
```

### `POST /api/auth/login`
Create an authenticated browser session cookie.

**Request Body**:
```json
{
  "password": "your-dashboard-password"
}
```

### `POST /api/auth/logout`
Clear the session cookie and log out.

---

## Hardware & Settings Endpoints

### `GET /api/gpus`
Return real-time telemetry from `nvidia-smi`.

**Response**:
```json
[
  {
    "id": 0,
    "uuid": "GPU-12345678-...",
    "name": "NVIDIA A100-SXM4-80GB",
    "memory_total": 87241523200,
    "memory_used": 1288490188,
    "memory_free": 85953033012,
    "memory_percent": 1.48,
    "gpu_utilization": 0,
    "compute_capability": "8.0",
    "processes": []
  }
]
```

### `GET /api/settings`
Read Hugging Face cache configuration, secret statuses, disk usage, and host RAM.

### `POST /api/settings`
Update Hugging Face cache path or token secrets.

---

## Model Lifecycle Endpoints

### `GET /api/models/active`
List managed vLLM serving containers and their health status.

### `GET /api/models/downloaded`
List cached models and their on-disk sizes.

### `POST /api/models/start`
Start a new serving container with VRAM preflight checks.

**Request Body**:
```json
{
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "port": 8080,
  "gpus": 4,
  "gpu_ids": "0,1,2,3",
  "mem_util": "0.90",
  "max_len": "32768",
  "kv_cache_dtype": "fp8",
  "dtype": "bfloat16",
  "parallelism_mode": "tensor"
}
```

### `POST /api/models/stop`
Stop a serving container by port (`{"port": 8080}`).

### `POST /api/models/restart`
Restart an existing container by container ID (`{"container_id": "..."}`).

### `POST /api/models/remove`
Remove a stopped container by container ID (`{"container_id": "..."}`).

### `POST /api/models/download`
Start a background model download (`{"model": "Qwen/Qwen2.5-Coder-7B-Instruct"}`).

### `GET /api/models/download/progress?model=<id>`
Retrieve downloaded byte count for an in-progress download.

### `POST /api/models/downloaded/delete`
Purge a cached model directory (`{"model": "..."}`).

### `GET /api/models/logs/stream?container_id=<id>`
Stream container stdout and stderr logs via Server-Sent Events (SSE).

---

## OpenAI Reverse Proxy Endpoints

### `GET /v1/models`
List routed models available for inference.

### `POST /v1/chat/completions`
Send standard OpenAI chat completion requests (supports streaming).

### `POST /v1/completions`
Send raw text completion requests.
