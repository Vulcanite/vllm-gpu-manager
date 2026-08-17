---
title: System Overview
sidebar_position: 1
---

# System Overview

**vLLM GPU Manager** is an enterprise-grade operations console and OpenAI reverse proxy engineered specifically for managing **vLLM** instances across single-node NVIDIA GPU servers.

<div className="diagram-container">
  <img src="/vllm-gpu-manager/img/diagrams/system-architecture.png" alt="vLLM GPU Manager System Architecture & Dataflow" />
  <div className="diagram-caption">Figure 1: Full-stack architecture, gateway proxy routing, and Docker engine controller.</div>
</div>

---

## Why vLLM GPU Manager?

Running production LLMs on raw GPU servers often leads to operational friction:

1. **CUDA Out-of-Memory Panics**: vLLM allocates KV cache eagerly on startup. Launching without exact memory budgeting leads to unrecoverable GPU crashes.
2. **Port Proliferation**: Running multiple model containers requires tracking random local ports across internal services.
3. **Multi-Tenant Friction**: Unauthenticated team access and lack of request tracking.
4. **Heavyweight Orchestration Overhead**: Standard Kubernetes/Ray clusters add substantial operational complexity for single-node multi-GPU machines.

**vLLM GPU Manager** solves this by providing:
- **Automated VRAM Preflight Math**: Computes exact model weight sizes and token KV cache allocations before container start.
- **Unified OpenAI Proxy Gateway**: Dynamic model resolution on `/v1/chat/completions` with user API keys and request auditing.
- **Dual Interface**: Operate seamlessly via headless SSH CLI or embedded React Web Console.
- **Zero Heavy Dependencies**: Distributed as a single compiled Go binary or native `.deb` / `.rpm` package.
