---
title: Traffic Routing & Proxy Gateway
sidebar_position: 5
---

# Traffic Routing & Proxy Gateway

The built-in reverse proxy gateway allows external clients and applications to connect to a single OpenAI-compatible endpoint without tracking underlying container ports.

---

## How the Gateway Works

When the dashboard daemon is running (default port `9090`), the `/v1/` endpoint acts as a smart reverse proxy:

```mermaid
sequenceDiagram
    autonumber
    actor Client as OpenAI API Client
    participant GW as vLLM Proxy Gateway (:9090)
    participant C1 as Llama-3 (Port 8080)
    participant C2 as Qwen-Coder (Port 8081)

    Client->>GW: POST /v1/chat/completions {"model": "Qwen/Qwen2.5-Coder-32B"}
    Note over GW: Authenticate Bearer sk-proxy-user
    Note over GW: Inspect {"model": "..."} -> Port 8081
    Note over GW: Rewrite Auth Header -> Upstream Token
    GW->>C2: POST /v1/chat/completions (Stream)
    C2-->>GW: SSE Stream Chunks
    GW-->>Client: Forwarded Stream Chunks
```

---

## Managing Proxy Users & API Keys

Generate named API keys for developers, applications, or downstream services:

### Creating a User Key
- **Web Console**: Go to **Proxy Gateway** $\rightarrow$ **Create Proxy User**.
- **CLI**:
  ```bash
  vllm-manager proxy-users create analytics-service
  ```

Output:
```text
Created proxy user: analytics-service
API Key: vllm_proxy_5f8a92b3c1d4...
```

### Revoking a User Key
```bash
vllm-manager proxy-users revoke analytics-service
```

---

## The Routing Table

The routing table maps model names and custom aliases to local ports:

```json
{
  "meta-llama/Llama-3.3-70B-Instruct": 8080,
  "llama3": 8080,
  "Qwen/Qwen2.5-Coder-32B-Instruct": 8081,
  "coder": 8081
}
```

When an active container is launched or stopped, the routing table updates automatically. You can also view or edit custom alias routes on the **Traffic Routing** page or via CLI:

```bash
vllm-manager routing list
vllm-manager routing set "gpt-4-turbo" 8080
```

---

## Querying `/v1/models`

Sending a `GET` request to `/v1/models` returns all currently routed models in standard OpenAI format:

```bash
curl http://localhost:9090/v1/models \
  -H "Authorization: Bearer <your-proxy-api-key>"
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.3-70B-Instruct",
      "object": "model",
      "created": 1722384000,
      "owned_by": "vllm-manager"
    },
    {
      "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "object": "model",
      "created": 1722384000,
      "owned_by": "vllm-manager"
    }
  ]
}
```
