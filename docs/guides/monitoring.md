---
title: Prometheus & Grafana Monitoring
sidebar_position: 6
---

# Prometheus & Grafana Monitoring

vLLM GPU Manager includes built-in Prometheus File-SD target synchronization and automated Grafana / NVIDIA DCGM stack setup.

---

## 1. Quick Monitoring Stack Setup

Run the automated setup helper to deploy Prometheus, Grafana, and NVIDIA DCGM exporter via Docker Compose:

```bash
# In source or git checkouts:
./scripts/setup-monitoring.sh

# On packaged systems:
sudo /usr/share/vllm-manager/scripts/setup-monitoring.sh
```

The script will prompt for:
- Grafana admin username and password.
- Bind IP address and port (default `0.0.0.0:3000`).
- HTTPS / TLS configuration.

The helper automatically provisions:
- **NVIDIA DCGM Exporter**: GPU temperature, PCIe throughput, power usage, and tensor core utilization metrics.
- **Prometheus**: Scrapes DCGM exporter on `127.0.0.1:9400` and vLLM servers from `vllm-targets.yml`.
- **Grafana**: Preconfigured with NVIDIA DCGM Dashboard (ID 22515) and vLLM Metrics Dashboard (ID 25263).

---

## 2. Enabling Metrics Target Synchronization

Start the vLLM GPU Manager daemon with `--monitoring`:

```bash
vllm-manager --dashboard --port 9090 --monitoring
```

Or enable it persistently via systemd:

```bash
sudo vllm-manager service configure --monitoring=true
```

### How Target Synchronization Works

Whenever a managed model container is started, restarted, or removed, the manager automatically updates the Prometheus file-SD configuration (`vllm-targets.yml`):

```yaml
- targets:
    - "127.0.0.1:8080"
  labels:
    job: "vllm"
    model: "meta-llama/Llama-3.3-70B-Instruct"
    container: "vllm-Llama-3.3-70B-Instruct-8080"
- targets:
    - "127.0.0.1:8081"
  labels:
    job: "vllm"
    model: "Qwen/Qwen2.5-Coder-32B-Instruct"
    container: "vllm-Qwen2.5-Coder-32B-Instruct-8081"
```

Prometheus reloads this file automatically without needing service restarts.

---

## 3. Key Metrics Exposed by vLLM

| Metric | Type | Description |
| :--- | :--- | :--- |
| `vllm:num_requests_running` | Gauge | Number of requests currently executing. |
| `vllm:num_requests_waiting` | Gauge | Number of queued requests waiting for KV cache allocation. |
| `vllm:gpu_cache_usage_factor` | Gauge | Fraction of GPU KV cache blocks currently in use ($0.0$ to $1.0$). |
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed. |
| `vllm:generation_tokens_total` | Counter | Total generation tokens produced. |
| `vllm:time_to_first_token_seconds` | Histogram | Time to first token (TTFT) latency distribution. |
| `vllm:time_per_output_token_seconds` | Histogram | Inter-token generation latency distribution. |
