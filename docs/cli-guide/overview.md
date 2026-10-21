---
title: CLI Overview & Headless Ops
sidebar_position: 1
---

# CLI Overview & Headless Operations

**vLLM GPU Manager** features a standalone Command Line Interface (CLI) built on Cobra. GPU inspection, model lifecycle, downloads, cache cleanup, settings, diagnostics, and service configuration all run directly from the terminal over SSH without the web daemon. Proxy users, the routing table, and container log streaming are managed from the dashboard.

---

## Command Hierarchy

```mermaid
graph TD
    Root[vllm-manager] --> GPU[gpu]
    Root --> Model[model]
    Root --> Cache[cache]
    Root --> Settings[settings]
    Root --> Service[service]
    Root --> Doctor[doctor]
    Root --> Version[version]

    GPU --> GPUList[gpu list]
    Model --> ModelStart[model start]
    Model --> ModelStop[model stop]
    Model --> ModelRestart[model restart]
    Model --> ModelActive[model active]
    Model --> ModelCached[model cached]
    Model --> ModelDownload[model download]
    Model --> ModelRemove[model remove]
    Cache --> CacheDelete[cache delete]
    Settings --> SettingsGet[settings get]
    Settings --> SettingsSet[settings set]
    Service --> ServiceConfigure[service configure]
```

---

## Global CLI Conventions

- **Structured Output**: All listing and inspection commands accept `--json` for shell scripting, jq piping, and automation.
- **Root-Safe & Non-Root Execution**: Runs under standard user accounts or system service accounts (`vllm-manager`) without requiring root access if the user belongs to the `docker` group.
- **Direct Engine Execution**: CLI commands communicate directly with the Docker Engine socket and `nvidia-smi` without requiring the web daemon to be active.
