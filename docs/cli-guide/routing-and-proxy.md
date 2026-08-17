---
title: Routing & Proxy User Management
sidebar_position: 4
---

# Routing & Proxy User Management

Manage API access keys and configure reverse proxy routes from the command line.

---

## Listing Proxy Users (`proxy-users list`)

```bash
vllm-manager proxy-users list
```

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>PROXY USERS &amp; REQUEST COUNTERS</span>
    <span>vllm-manager proxy-users list</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/cli/04-cli-proxy-users.png" alt="vllm-manager proxy-users list output" />
</div>

---

## Creating & Revoking Proxy Users

### Create a Proxy Key
```bash
vllm-manager proxy-users create cursor-ide
```

Output:
```text
Created proxy user cursor-ide
API Key: vllm_proxy_a91b2c3d4e5f6g7h8i9j0k1l2m3n4o5p
Keep this key secure; it will not be displayed again.
```

### Revoke a Proxy Key
```bash
vllm-manager proxy-users revoke cursor-ide
```

---

## Custom Route Aliases (`routing`)

### Listing Routes
```bash
vllm-manager routing list
```

### Setting a Model Route Alias
```bash
vllm-manager routing set gpt-4o 8080
```
