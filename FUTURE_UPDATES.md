# Future Updates

This project should stay focused on single-binary deployment for one GPU host. Future features should improve day-to-day operations without turning the tool into a distributed platform.

## Good Fits

- **Model presets:** Save launch presets per model, including GPUs, GPU IDs, port, memory utilization, max model length, KV cache dtype, and tool parser.
- **Container logs view:** Add an API and dashboard panel for recent vLLM container logs, especially startup failures.
- **Readiness and startup progress:** Show states such as created, starting, loading weights, ready, and failed using Docker status, health checks, and logs.
- **Disk usage breakdown:** Show Hugging Face cache usage per downloaded model.
- **One-click recreate:** Recreate a managed model container with the same launch configuration, useful after changing global settings.
- **Proxy request limits:** Add request body limits and simple concurrency controls for the opt-in `/v1/` proxy gateway.
- **Settings export and import:** Export sanitized settings and routing table; import secrets only when explicitly provided.
- **Health diagnostics page:** Check Docker socket, NVIDIA runtime, `nvidia-smi`, vLLM image availability, HF cache path, disk space, and TLS cert status.
- **Configurable vLLM image tag:** Allow `vllm/vllm-openai:<tag>` instead of hardcoding `latest` for reproducible deployments.
- **Systemd unit generator:** Add a CLI command that prints a systemd unit for running the single binary in dashboard mode.
- **Recommended model improvements:** Add sort controls, richer "why this model" explanations, downloaded/not-downloaded filters, family tags, launch-risk indicators, recommendation modes such as Balanced/Max Quality/Fastest/Lowest VRAM, suggested `max_num_seqs`, compare mode, and catalog metadata such as license, source URL, notes, and last-reviewed date.

## Avoid

- Multi-user authorization or RBAC.
- Kubernetes support.
- Distributed scheduling.
- Database-backed history.
- Multi-host orchestration.
- A full model registry abstraction.

These would move the project away from its intended role as a lightweight single-host GPU operations tool.
