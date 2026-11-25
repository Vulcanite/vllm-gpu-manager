---
title: Monitoring
sidebar_position: 6
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Monitoring

## GPUs and processes

<Tabs groupId="interface">
<TabItem value="web" label="Dashboard" default>

The **Dashboard** page shows a card per GPU (memory used, compute utilization, compute capability, and which model has it pinned) and a table of GPU processes with their PID, Linux user, name and memory. The user is read from `/proc/<pid>`, which helps find a process you did not start.

</TabItem>
<TabItem value="cli" label="CLI">

```bash
vllm-manager gpu list            # GPUs, memory and processes
vllm-manager gpu list --json     # the same as JSON
```

</TabItem>
</Tabs>

## Prometheus and Grafana

### Set up the stack

The helper script starts Prometheus, Grafana and the NVIDIA DCGM exporter with Docker Compose:

```bash
./scripts/setup-monitoring.sh                                   # from a checkout
sudo /usr/share/vllm-manager/scripts/setup-monitoring.sh        # packaged install
```

It asks for the Grafana admin login, the bind address and port (default `0.0.0.0:3000`) and TLS. Prometheus scrapes the DCGM exporter on `127.0.0.1:9400` and every vLLM server listed in `vllm-targets.yml`. Grafana comes with the NVIDIA DCGM dashboard (ID 22515) and a vLLM dashboard (ID 25263).

### Keep the targets current

Start the dashboard with `--monitoring` (or `service configure --monitoring`). Whenever a managed model starts, restarts or is removed, the manager rewrites the Prometheus file-SD targets file:

```yaml
- targets: ["127.0.0.1:8080"]
  labels:
    job: "vllm"
    model: "meta-llama/Llama-3.3-70B-Instruct"
    container: "vllm-Llama-3.3-70B-Instruct-8080"
```

Prometheus picks up the change without a restart. Change the file location with `--monitoring-targets-file`.

### Useful vLLM metrics

Metric names change between vLLM versions; check `http://127.0.0.1:<port>/metrics` on a running model for the exact list.

| Metric | Type | Meaning |
| :--- | :--- | :--- |
| `vllm:num_requests_running` | Gauge | Requests being processed. |
| `vllm:num_requests_waiting` | Gauge | Requests queued for KV cache space. |
| `vllm:prompt_tokens_total` | Counter | Prompt tokens processed. |
| `vllm:generation_tokens_total` | Counter | Tokens generated. |
| `vllm:time_to_first_token_seconds` | Histogram | Time to first token. |
| `vllm:time_per_output_token_seconds` | Histogram | Time between generated tokens. |
