---
title: Quickstart
sidebar_position: 3
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart

Download a model, start it, and send it a request. This assumes you have [installed](./install.md) the manager and `vllm-manager doctor` passes.

## 1. Check the GPUs

```bash
vllm-manager gpu list
```

## 2. Download a model

```bash
vllm-manager model download Qwen/Qwen2.5-Coder-7B-Instruct
vllm-manager model progress Qwen/Qwen2.5-Coder-7B-Instruct
```

`model progress` shows `running` until the download finishes, then `succeeded` (or `failed` with the reason). See [Download & cache models](./guides/downloads.md).

## 3. Start it and send a request

There are two ways to serve a model. The dashboard puts it behind the shared `/v1` proxy; the CLI starts it on its own port.

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard (proxy)" default>

Start the dashboard. On `127.0.0.1` no password is needed; to reach it from another machine, set `VLLM_MANAGER_PASSWORD` and `--host` (see [Security](./security.md)).

```bash
vllm-manager --dashboard --port 9090
```

1. Open `http://localhost:9090`, go to **Models**, and in **Deploy Server** pick `Qwen/Qwen2.5-Coder-7B-Instruct`, port `8000` and 1 GPU. Deploy once the memory check passes.
2. On **Proxy Gateway**, click **Create User** and copy the key (`vllm_proxy_...`).
3. Call the proxy:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <proxy-key>" \
  -d '{"model": "Qwen/Qwen2.5-Coder-7B-Instruct",
       "messages": [{"role": "user", "content": "Write a Go function that reverses a string."}]}'
```

</TabItem>
<TabItem value="cli" label="CLI (direct port)">

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 --gpus 1 --mem-util 0.85 --max-len 32768
```

Before starting the container it checks that the port is free, that the weights plus the KV cache for 32,768 tokens fit in 85% of GPU 0, and that GPU 0 is not already promised to other containers. CLI launches listen on `server_bind_ip` (default `0.0.0.0`) and are called directly:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Coder-7B-Instruct",
       "messages": [{"role": "user", "content": "Write a Go function that reverses a string."}]}'
```

If you set a global key with `settings set --vllm-api-key`, add `-H "Authorization: Bearer <that-key>"`.

</TabItem>
</Tabs>

vLLM needs a minute or two to load the weights; `vllm-manager model active` shows it as ready when it can take requests.

## 4. Stop it

```bash
vllm-manager model stop --port 8000
```

## Next

- [Deploy models](./guides/deploy.md): every launch option.
- [GPU placement & VRAM](./guides/gpu-and-vram.md): how the memory check works and how to use several GPUs.
- [Run as a service](./guides/service.md): start the dashboard at boot.
