# Roadmap

Growth work for after v1.1.0. These items are intentionally not in this release: the current binary is a single-host Docker launcher, and each of these expands that job without changing the existing fail-closed launch path.

## Multi-GPU placement

Auto-pick the least-loaded GPU that still fits the model's weights and KV cache. Operators today have to choose `--gpu-ids` by reading `nvidia-smi` themselves; a placement pass that reuses the existing overlap and VRAM preflight would make the default "just start it" path correct on multi-GPU boxes without turning the tool into a cluster scheduler.

## Prometheus `/metrics` endpoint

Expose GPU memory, per-model request rate, and proxy latency in Prometheus text format next to the existing file-SD target sync. Grafana already ships in `monitoring/`; a first-party `/metrics` scrape target would give the same numbers without requiring DCGM or a sidecar, and it is the natural place to surface the in-memory request counters.

## Per-user token budgets and usage export

Build quotas and a CSV/JSON export on the atomic per-user request counter. Once counts are no longer a disk write on the hot path, a budget check is a compare-and-swap, and usage export is a snapshot of that map — enough for a lab to share one GPU host without one client starving the rest.

## Launch templates / presets for common models

Save named LaunchConfig presets (Qwen 32K, Llama 8K, vision, etc.) so the dashboard and CLI do not re-enter the same 20 flags for every deploy. The LaunchConfig struct is the storage shape; templates are a thin layer over it plus the VRAM defaults already returned by `LaunchDefaults`.

## Crash-restart policy for containers with backoff

Restart managed vLLM containers that exit, with exponential backoff and a cap, instead of leaving a dead container occupying the GPU. A restart policy belongs here rather than in Docker's `--restart` flag so the global launch mutex and VRAM preflight still run before a replacement is created.
