# Future Updates

This project should stay focused on single-binary deployment for one GPU host. Future features should improve day-to-day operations without turning the tool into a distributed platform.

## Next Release Focus

- **AMD GPU support:** Add ROCm device discovery, Docker runtime validation, compatible vLLM image handling, VRAM checks, and UI/CLI terminology that works for both NVIDIA and AMD hosts. NVIDIA remains the only supported GPU platform in the current release.
- **RPM packages:** Extend the existing nFPM packaging and release workflow with RPM artifacts, systemd paths, maintainer scripts, and validation on a supported RPM-based distribution.

## Good Fits

- **Model presets:** Save launch presets per model, including GPUs, GPU IDs, port, memory utilization, max model length, KV cache dtype, and tool parser.
- **Readiness and startup progress:** Show states such as created, starting, loading weights, ready, and failed using Docker status, health checks, and logs.
- **Proxy request limits:** Add request body limits and simple concurrency controls for the opt-in `/v1/` proxy gateway.
- **Settings export and import:** Export sanitized settings and routing table; import secrets only when explicitly provided.
- **DeepGEMM MegaMoE backend support:** Add `deep_gemm_mega_moe` as a guarded advanced MoE backend for DeepSeek-V4-style models. Validate `--enable-expert-parallel`, `model_type=deepseek_v4`, `expert_dtype=fp4`, and SM100-class GPUs before allowing launch.
- **Recommended model improvements:** Add sort controls, richer "why this model" explanations, downloaded/not-downloaded filters, family tags, launch-risk indicators, recommendation modes such as Balanced/Max Quality/Fastest/Lowest VRAM, suggested `max_num_seqs`, compare mode, and catalog metadata such as license, source URL, notes, and last-reviewed date.

## Avoid

- Multi-user authorization or RBAC.
- Kubernetes support.
- Distributed scheduling.
- Database-backed history.
- Multi-host orchestration.
- A full model registry abstraction.

These would move the project away from its intended role as a lightweight single-host GPU operations tool.
