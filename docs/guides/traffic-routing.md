---
title: OpenAI Reverse Proxy & Routing
sidebar_position: 3
---

# OpenAI Reverse Proxy & Routing

**vLLM GPU Manager** features a built-in reverse proxy gateway on `:9090` that translates standard OpenAI API requests, handles authentication, and dispatches traffic to active vLLM backend containers.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/proxy-routing-flow.png" alt="OpenAI Reverse Proxy & Upstream Token Rewriting" />
  <div className="diagram-caption">Figure 4: Proxy pipeline, user key validation, and upstream request rewriting.</div>
</div>

---

## The Gateway Pipeline

When a client sends an inference request:

1. **Authentication Check**: Validates the `Authorization: Bearer <proxy_key>` token against the `proxy_users` registry.
2. **Model Extraction**: Reads the `model` field from the JSON body of `POST` requests (bodies are capped by `proxy_max_body_bytes`, default 32 MiB). `GET /v1/models` is answered from the routing table; other requests without a routable model return `404`.
3. **Route Resolution**: Looks up the internal container port from the dynamic routing table.
4. **Header Rewriting**: Replaces the client's proxy key with the API key the target container was launched with, or strips the header if it has none.
5. **Streaming Response**: Forwards the response through Go's `httputil.ReverseProxy` with immediate flushing, so SSE token frames stream to the client as they arrive.

---

## Client Integration Examples

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://gpu-server.lan:9090/v1",
    api_key="vllm_proxy_a91b2c3d4e5f...",
)

response = client.chat.completions.create(
    model="qwen-coder",
    messages=[{"role": "user", "content": "Write a high-performance Go HTTP server."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Curl
```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer vllm_proxy_a91b2c3d4e5f..." \
  -d '{
    "model": "qwen-coder",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
