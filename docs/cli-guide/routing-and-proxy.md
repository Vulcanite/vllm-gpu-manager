---
title: Routing & Proxy User Management
sidebar_position: 4
---

# Routing & Proxy User Management

The CLI does not manage proxy users or the routing table. There are no `proxy-users` or `routing` subcommands. Both are managed by the running dashboard, from the web console or its JSON API.

---

## Proxy Users

Create, reveal, and revoke proxy API keys on the **Proxy Gateway** page of the web console. See [Proxy Gateway](../web-guide/proxy-gateway.md).

The same operations are available over the dashboard API (send the dashboard session cookie when `VLLM_MANAGER_PASSWORD` is set):

```bash
# List users and request counts
curl http://localhost:9090/api/proxy-users

# Create a user; the response contains the new API key
curl -X POST http://localhost:9090/api/proxy-users \
  -H "Content-Type: application/json" \
  -d '{"username": "cursor-ide"}'

# Revoke a user
curl -X POST http://localhost:9090/api/proxy-users/revoke \
  -H "Content-Type: application/json" \
  -d '{"username": "cursor-ide"}'
```

---

## Routing Table

Routes are added automatically when a model starts in proxy mode. To review or edit them, use the **Traffic Routing** page, or:

```bash
# Read the table
curl http://localhost:9090/api/routing

# Replace the table; each key must be the running model's served name (or model ID if none) on that port
curl -X POST http://localhost:9090/api/routing \
  -H "Content-Type: application/json" \
  -d '{"Qwen/Qwen2.5-Coder-7B-Instruct": 8080}'
```

`POST /api/routing` replaces the whole table, so include every route you want to keep.
