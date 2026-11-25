---
title: Proxy & routing
sidebar_position: 4
---

# Proxy & routing

In proxy mode (the default for dashboard launches), every model is reached through one OpenAI-compatible endpoint on the dashboard port: `http://<host>:9090/v1`. Model containers only listen on `127.0.0.1`.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/proxy-routing-flow.png" alt="How the /v1 proxy routes a request" />
  <div className="diagram-caption">Key check, model lookup, header rewrite, streamed response.</div>
</div>

## How a request is handled

1. **Key check.** `Authorization: Bearer <proxy-key>` must match a proxy user.
2. **Model lookup.** For `POST` requests the proxy reads the `model` field from the JSON body (bodies over `proxy_max_body_bytes`, default 32 MiB, get `413`). `GET /v1/models` is answered from the routing table. A request with no routable model gets `404`.
3. **Routing.** The routing table maps the model name to the container's local port.
4. **Header rewrite.** The proxy key is replaced with the API key that container was started with, or removed if it has none.
5. **Streaming.** Responses are flushed as they arrive, so streamed tokens reach the client immediately.

Disable the proxy with `--no-proxy` (or `VLLM_MANAGER_NO_PROXY=true`); dashboard launches then use direct mode like the CLI.

## Proxy users

Each client gets its own key, and requests are counted per user.

On **Proxy Gateway**, click **Create User**, give it a name (for example `cursor-ide`) and copy the key (`vllm_proxy_...`). **Revoke** removes it.

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>PROXY GATEWAY</span>
    <span>Demo build, sample data</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/05-proxy-gateway-users.png" alt="Proxy users and routing table" />
</div>

There are no CLI commands for proxy users or routes. To script them, call the dashboard API (send the session cookie when a password is set):

```bash
curl http://localhost:9090/api/proxy-users                       # list users and counts
curl -X POST http://localhost:9090/api/proxy-users \
  -H "Content-Type: application/json" -d '{"username": "cursor-ide"}'   # create; response has the key
curl -X POST http://localhost:9090/api/proxy-users/revoke \
  -H "Content-Type: application/json" -d '{"username": "cursor-ide"}'   # revoke
```

## Routing table

A model started in proxy mode adds one route automatically: its served model name if it has one, otherwise its Hugging Face id. The **Proxy Gateway** page shows the table.

Routes are checked against running containers: each name must be the routing name of the managed model on that port, so arbitrary aliases (for example `gpt-4o`) are rejected. To expose a different name, set the served model name when deploying.

```bash
curl http://localhost:9090/api/routing                           # read
curl -X POST http://localhost:9090/api/routing \
  -H "Content-Type: application/json" \
  -d '{"Qwen/Qwen2.5-Coder-7B-Instruct": 8080}'                  # replace the whole table
```

## Calling the proxy

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://gpu-server.lan:9090/v1", api_key="vllm_proxy_...")

stream = client.chat.completions.create(
    model="qwen-coder",
    messages=[{"role": "user", "content": "Write a small Go HTTP server."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### curl

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm_proxy_..." \
  -d '{"model": "qwen-coder", "messages": [{"role": "user", "content": "Hello"}]}'
```
