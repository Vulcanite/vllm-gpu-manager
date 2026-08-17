---
title: CLI Reference
sidebar_position: 1
---

# CLI Reference

Comprehensive reference for all `vllm-manager` command-line commands and flags.

---

## Global Flags

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--json` | Boolean | Output results in JSON format for automation and scripts. |
| `-h`, `--help` | Boolean | Display command help and usage syntax. |
| `--dashboard` | Boolean | Run the web dashboard and REST API daemon. |
| `--port` | String | Port to run the dashboard on (default `9090`). |
| `--host` | String | Host IP to bind dashboard on (default `127.0.0.1`). |
| `--https` | Boolean | Enable HTTPS with local self-signed TLS certificates. |
| `--no-proxy` | Boolean | Disable the OpenAI-compatible proxy gateway on `/v1/`. |
| `--monitoring`| Boolean | Sync active model servers into Prometheus file-SD target file. |

---

## GPU Commands

### `gpu list`
Display visible NVIDIA GPUs, memory utilization, and running compute processes:

```bash
vllm-manager gpu list [--json]
```

---

## Model Commands

### `model active`
List all managed vLLM serving containers:

```bash
vllm-manager model active [--json]
```

### `model cached`
List downloaded models in the Hugging Face cache:

```bash
vllm-manager model cached [--json]
```

### `model download`
Start a background download container:

```bash
vllm-manager model download <model-id>
```

### `model progress`
Check downloaded bytes and download progress for a model:

```bash
vllm-manager model progress <model-id> [--json]
```

### `model recommend`
Rank models against detected GPU VRAM and compute capabilities:

```bash
vllm-manager model recommend [--use-case coding|chat|reasoning] [--fit-only] [--limit 20]
```

### `model start`
Deploy an LLM serving container:

```bash
vllm-manager model start <model-id> \
  --port <port> \
  --gpus <count> \
  [--gpu-ids <ids>] \
  [--mem-util <0.1-1.0>] \
  [--max-len <tokens>] \
  [--max-num-seqs <count>] \
  [--dtype auto|bfloat16|float16] \
  [--parallelism-mode tensor|pipeline] \
  [--kv-cache-dtype auto|fp8|fp8_e4m3|fp8_e5m2] \
  [--cpu-offload-gb <gb>] \
  [--tool-parser <name>] \
  [--reasoning-parser <name>] \
  [--served-model-name <alias>] \
  [--bind-ip <ip>] \
  [--api-key <key>]
```

### `model stop`
Stop a running model server by port:

```bash
vllm-manager model stop --port <port>
```

### `model restart`
Restart a running container or start an existing stopped container:

```bash
vllm-manager model restart <container-id>
```

### `model remove`
Remove a stopped managed container:

```bash
vllm-manager model remove <container-id>
```

---

## Cache Commands

### `cache delete`
Delete a downloaded model directory from the Hugging Face cache:

```bash
vllm-manager cache delete <model-id> [--yes]
```

---

## Proxy & Routing Commands

### `proxy-users list`
List all proxy user API keys and request counts:

```bash
vllm-manager proxy-users list [--json]
```

### `proxy-users create`
Create a new proxy user API key:

```bash
vllm-manager proxy-users create <username>
```

### `proxy-users revoke`
Revoke a proxy user API key:

```bash
vllm-manager proxy-users revoke <username>
```

### `routing list`
Display the current OpenAI reverse proxy routing table:

```bash
vllm-manager routing list [--json]
```

### `routing set`
Add or update a model route:

```bash
vllm-manager routing set <model-name-or-alias> <port>
```

---

## System & Service Commands

### `doctor`
Run diagnostic preflight checks on drivers, GPUs, Docker, and cache:

```bash
vllm-manager doctor [--json]
```

### `service configure`
Atomically update `/etc/vllm-manager/service.env` settings:

```bash
vllm-manager service configure \
  [--host <ip>] \
  [--port <port>] \
  [--https[=true|false]] \
  [--proxy[=true|false]] \
  [--monitoring[=true|false]] \
  [--password-file <path>] \
  [--clear-password] \
  [--no-restart]
```

### `version`
Print version and build information:

```bash
vllm-manager version
```
