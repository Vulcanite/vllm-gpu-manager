---
title: Quickstart
sidebar_position: 2
---

# Quickstart

This 5-minute guide walks through downloading an LLM, inspecting GPU readiness, launching an OpenAI-compatible serving container, and querying the server through the built-in `/v1/` reverse proxy.

---

## Step 1: Check GPU Status

Verify visible NVIDIA GPUs and current memory utilization:

```bash
vllm-manager gpu list
```

```text
+----+---------------------+------------+-----------+----------+--------+
| ID | NAME                | TOTAL VRAM | USED VRAM | FREE     | UTIL % |
+----+---------------------+------------+-----------+----------+--------+
| 0  | NVIDIA A100-SXM4-80GB| 81.25 GiB  | 1.20 GiB  | 80.05 GiB| 0%     |
| 1  | NVIDIA A100-SXM4-80GB| 81.25 GiB  | 1.20 GiB  | 80.05 GiB| 0%     |
+----+---------------------+------------+-----------+----------+--------+
```

---

## Step 2: Download a Model

Download a model from Hugging Face. Downloads run in a detached background Docker container using the official vLLM image and entrypoint `hf`:

```bash
vllm-manager model download Qwen/Qwen2.5-Coder-7B-Instruct
```

Track the download progress:

```bash
vllm-manager model progress Qwen/Qwen2.5-Coder-7B-Instruct
```

Verify downloaded models in cache:

```bash
vllm-manager model cached
```

---

## Step 3: Start the Model Server

Launch the model as an isolated Docker container with VRAM preflight checks:

```bash
vllm-manager model start Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --gpus 1 \
  --mem-util 0.85 \
  --max-len 32768
```

The preflight engine verifies:
1. Port `8000` is free.
2. GPU 0 has enough available VRAM for the 7B FP16 weights (~14.5 GiB) + KV cache for 32k tokens.
3. Total GPU memory utilization on GPU 0 does not exceed 100%.

View active model containers:

```bash
vllm-manager model active
```

```text
+------------------------------------+------+-------+---------+--------+
| MODEL                              | PORT | GPUS  | STATE   | HEALTH |
+------------------------------------+------+-------+---------+--------+
| Qwen/Qwen2.5-Coder-7B-Instruct     | 8000 | GPU 0 | running | ready  |
+------------------------------------+------+-------+---------+--------+
```

---

## Step 4: Start the Web Dashboard & Proxy Gateway

Start the management dashboard on port `9090`:

```bash
vllm-manager --dashboard --port 9090
```

Open your browser at `http://<your-server-ip>:9090` to view real-time GPU telemetry, stream live container logs, and inspect routes.

---

## Step 5: Query the Model via OpenAI Client

The dashboard includes a built-in reverse proxy gateway on `/v1/`.

1. Generate a Proxy API key via the web console (**Proxy Gateway** page) or CLI:
   ```bash
   vllm-manager proxy-users create developer-alice
   ```
2. Send an OpenAI-compatible chat completion request:

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-proxy-api-key>" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}
    ],
    "temperature": 0.2
  }'
```

```python
# Using the official OpenAI Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="<your-proxy-api-key>",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    messages=[{"role": "user", "content": "Write a quicksort implementation in Go."}],
)

print(response.choices[0].message.content)
```

---

## Step 6: Stop or Remove Containers

Stop the running container cleanly:

```bash
vllm-manager model stop --port 8000
```
