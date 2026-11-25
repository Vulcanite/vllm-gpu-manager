---
title: Deploy models
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Deploy models

A model must be [downloaded](./downloads.md) before it can start; the memory check reads its files from the cache.

## Start a model

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

Go to **Models** → **Deploy Server**. As you fill in the model, GPUs, memory fraction and context length, the form recalculates how much GPU memory the launch needs and only lets you deploy when it fits.

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>DEPLOY SERVER</span>
    <span>Demo build, sample data</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/02-deploy-server-modal.png" alt="Deploy Server form" />
</div>

Dashboard launches use proxy mode when the proxy is on (the default), so the model is reached through `/v1` on the dashboard port. The form also sets things the CLI cannot: a **served model name** (the alias clients use and the proxy routes on) and a per-server API key in direct mode.

To copy settings from an existing `docker run` or `vllm serve` command, use **Import Command**. See [Command import](./command-import.md).

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct \
  --port 8080 --gpus 2 --kv-cache-dtype fp8
```

CLI launches are always direct: the container listens on `server_bind_ip` (default `0.0.0.0`, or `--bind-ip`) and uses the global vLLM API key from `settings set --vllm-api-key`, if one is set.

</TabItem>
</Tabs>

## Launch options

The CLI flag is shown; the dashboard has the same fields under the main form or **Advanced**. Run `vllm-manager model start --help` for the full list.

| Option | CLI flag | Default | Notes |
| :--- | :--- | :--- | :--- |
| Port | `--port` | required | Host port for the container. |
| GPU count | `--gpus` | required | Tensor or pipeline size. |
| Exact GPUs | `--gpu-ids` | first free | For example `0,1`. See [GPU placement](./gpu-and-vram.md). |
| Parallelism | `--parallelism` | `tensor` | `tensor` or `pipeline`. |
| Memory fraction | `--mem-util` | `0.9` | Passed as `--gpu-memory-utilization`. |
| Context length | `--max-len` | model default | Passed as `--max-model-len`; drives the KV cache estimate. |
| Max sequences | `--max-num-seqs` | vLLM default | Concurrent sequences per batch. |
| Weight dtype | `--dtype` | `auto` | `auto`, `bfloat16`, `float16`, `float32`. |
| KV cache dtype | `--kv-cache-dtype` | `auto` | `fp8`, `fp8_e4m3`, `fp8_e5m2` halve KV memory; FP8 needs compute capability 8.9+. |
| CPU offload | `--cpu-offload-gb` | `0` | GiB of weights per GPU kept in host RAM; host RAM is checked with a 1.25× margin. |
| Tool calling | `--tool-parser` | off | vLLM tool-call parser, for example `hermes`. |
| Reasoning | `--reasoning-parser` | off | For example `deepseek_r1`, `qwen3`. |
| Tokenizer mode | `--tokenizer-mode` | `auto` | `auto`, `hf`, `slow`, `mistral`, `deepseek_v32`, `deepseek_v4`. |
| Speculative decoding | `--speculative-config` | off | JSON, for example `'{"method":"eagle","num_speculative_tokens":5}'`. |
| Remote code | `--trust-remote-code` | off | Lets the model repo run its own Python code. Only for repos you trust. |
| MoE | `--enable-expert-parallel`, `--moe-backend` | off, `auto` | For mixture-of-experts models. |
| Bind IP | `--bind-ip` | `server_bind_ip` | Direct mode only. |

Prefix caching is on by default in the dashboard form (vLLM's own default).

## Manage running models

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

The **Models** page lists managed containers with **Restart**/**Start**, **Stop** and **Remove**. **Logs** streams the container's output live (Server-Sent Events from `/api/models/logs/stream`); closing it stops the stream.

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>ACTIVE MODELS AND LOGS</span>
    <span>Demo build, sample data</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/03-active-models-and-logs.png" alt="Active models and live logs" />
</div>

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager model active                           # list managed containers
vllm-manager model stop --port 8080                 # stop the server on a port
vllm-manager model restart --container-id <id>      # restart, or start a stopped one
vllm-manager model remove --container-id <id>       # remove a stopped container
```

</TabItem>
</Tabs>

A restart reuses the original launch settings and runs the GPU and memory checks again, so it cannot take a GPU another model claimed while this one was stopped.

While the dashboard is running, lifecycle CLI commands refuse to run (to avoid two writers) unless you pass `--force`.
