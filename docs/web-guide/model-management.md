---
title: Model Lifecycle & Log Streaming
sidebar_position: 4
---

# Model Lifecycle & Log Streaming

Monitor active vLLM containers, inspect health statuses, and stream live stdout/stderr logs over Server-Sent Events (SSE).

---

## Active Containers Table

The **Active Models** view displays all currently running serving containers:

<div className="screenshot-card">
  <div className="screenshot-header">
    <span>ACTIVE MODEL SERVERS</span>
    <span>vLLM GPU Manager</span>
  </div>
  <img src="/vllm-gpu-manager/img/screenshots/web/03-active-models-and-logs.png" alt="Active Model Servers" />
</div>

### Container Lifecycle Actions
- **Restart Container**: Triggers Docker restart preserving the original command, port bindings, and volume mounts.
- **Stop Container**: Sends graceful `SIGTERM` to the container and unpublishes the port.
- **Remove Container**: Deletes stopped containers and frees container state.

---

## Real-Time Log Viewer (SSE)

Clicking the **Logs** button on any container card opens the real-time log drawer:

- Streams live stdout and stderr frames directly from Docker via Server-Sent Events (`/api/models/logs/stream`).
- Includes auto-scroll, search filtering, and ANSI color rendering.
- Disconnecting closes the background stream goroutine immediately, releasing Docker resources.
