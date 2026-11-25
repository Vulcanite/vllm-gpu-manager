---
title: CLI Reference
sidebar_label: CLI
sidebar_position: 1
---

# CLI Reference

Every `vllm-manager` command and flag. Run any command with `--help` for the exact list in your version.

---

## Global Flags

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--json` | Boolean | Output results as JSON (see [Scripting with JSON](#scripting-with-json)). |
| `-h`, `--help` | Boolean | Show help for a command. |
| `--dashboard` | Boolean | Run the web dashboard and API. |
| `--port` | String | Port to run the dashboard on (default `9090`). |
| `--host` | String | Host IP to bind dashboard on (default `127.0.0.1`). |
| `--https` | Boolean | Enable HTTPS with local self-signed TLS certificates. |
| `--no-proxy` | Boolean | Disable the OpenAI-compatible proxy gateway on `/v1/`. |
| `--insecure-no-auth` | Boolean | Allow a non-loopback `--host` without `VLLM_MANAGER_PASSWORD`. Anyone who can reach the port controls Docker. |
| `--monitoring`| Boolean | Sync active model servers into Prometheus file-SD target file. |

---

## Dashboard flags

These start the web dashboard instead of running a command:

```bash
vllm-manager --dashboard [--host <ip>] [--port <port>] [--https] [--no-proxy] [--monitoring] [--insecure-no-auth]
```

Lifecycle commands (`model start/stop/restart/remove`, `cache delete`, `settings set`) refuse to run while the dashboard is running, so two processes do not change the same state. Pass `--force` to run them anyway.

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

### `model info`
Fetch a model's size, downloads and likes from Hugging Face, and whether it is cached:

```bash
vllm-manager model info <model-id> [--json]
```

### `model download`
Start a background download container (see [Download & cache models](../guides/downloads.md)):

```bash
vllm-manager model download <model-id>
```

### `model progress`
Show a download's state (`running`, `succeeded`, `failed` or `stopped`) and bytes on disk. A failed download prints its exit code and the end of its log:

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
  [--dtype auto|bfloat16|float16|float32] \
  [--parallelism tensor|pipeline] \
  [--kv-cache-dtype auto|fp8|fp8_e4m3|fp8_e5m2] \
  [--cpu-offload-gb <gb>] \
  [--tool-parser <name>] \
  [--reasoning-parser <name>] \
  [--tokenizer-mode <mode>] \
  [--block-size <n>] \
  [--trust-remote-code] \
  [--bind-ip <ip>]
```

The CLI uses the global vLLM API key from `settings set --vllm-api-key`; it has no per-launch `--api-key` or `--served-model-name` flag. Served-model aliases and per-server keys are set from the web console or `POST /api/models/start`. Run `vllm-manager model start --help` for the full flag list, including MoE and speculative-decoding options.

### `model stop`
Stop a running model server by port:

```bash
vllm-manager model stop --port <port>
```

### `model restart`
Restart a running container or start an existing stopped container:

```bash
vllm-manager model restart --container-id <container-id>
```

### `model remove`
Remove a stopped managed container:

```bash
vllm-manager model remove --container-id <container-id> [--yes]
```

---

## Cache Commands

### `cache delete`
Delete a downloaded model directory from the Hugging Face cache:

```bash
vllm-manager cache delete <model-id> [--yes]
```

---

## Settings Commands

### `settings get`
Print stored settings with secrets masked:

```bash
vllm-manager settings get [--json] [--field hf_cache]
```

### `settings set`
Update stored settings:

```bash
vllm-manager settings set \
  [--hf-cache <path>] \
  [--hf-token <token> | --clear-hf-token] \
  [--vllm-api-key <key> | --clear-vllm-api-key] \
  [--vllm-image <image:tag>] \
  [--server-bind-ip <ip>] \
  [--vllm-cuda-compatibility] \
  [--hf-hub-offline]
```

---

## Proxy Users & Routing

There are no `proxy-users` or `routing` CLI commands. Manage them on the dashboard's **Proxy Gateway** page or through `/api/proxy-users` and `/api/routing`. See [Proxy & routing](../guides/proxy.md).

---

## System & Service Commands

### `doctor`
Run diagnostic preflight checks on drivers, GPUs, Docker, and cache:

```bash
vllm-manager doctor [--json]
```

### `service configure`
Update `/etc/vllm-manager/service.env` (see [Run as a service](../guides/service.md)):

```bash
vllm-manager service configure \
  [--host <ip>] \
  [--port <port>] \
  [--https[=true|false]] \
  [--proxy[=true|false]] \
  [--monitoring[=true|false]] \
  [--password-file <path>] \
  [--clear-password] \
  [--insecure-no-auth] \
  [--no-restart]
```

### `version`
Print version and build information:

```bash
vllm-manager version
```

---

## Scripting with JSON

Listing and inspection commands accept `--json`. Example output (sample data):

```bash
vllm-manager gpu list --json
```

```json
[
  {
    "id": 0,
    "name": "NVIDIA H100 80GB HBM3",
    "memory_total": 85899345920,
    "memory_used": 36700160000,
    "memory_free": 49199185920,
    "gpu_utilization": 68,
    "compute_capability": "9.0",
    "processes": [
      {"pid": 24819, "user": "vllm-manager", "name": "python3 (vLLM::Engine)", "used_memory": 36700160000}
    ]
  }
]
```

Check free memory on GPU 0 before a launch:

```bash
FREE_GIB=$(vllm-manager gpu list --json | jq '.[0].memory_free / 1073741824')
echo "GPU 0 has ${FREE_GIB} GiB free"
```

Find the port a model is running on:

```bash
vllm-manager model active --json | jq -r '.[] | select(.model=="Qwen/Qwen2.5-Coder-32B-Instruct") | .port'
```
