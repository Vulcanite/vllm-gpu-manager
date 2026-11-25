---
title: Changelog
sidebar_position: 7
---

# Changelog

All notable changes to **vLLM GPU Manager** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.2] - 2026-11-25

### Fixed
- Hugging Face downloads pass each `--exclude` pattern separately to `hf download`, so patterns are not mistaken for filenames.
- Download progress reports failure, including logs, when the downloader exits successfully without fetching model files.

---

## [1.1.1] - 2026-11-25

### Changed
- `manager.sh` supports `--insecure-no-auth`, preserves the setting on restart, and warns that an unauthenticated non-loopback dashboard grants control to anyone who can reach it.

### Fixed
- `manager.sh` reports when the dashboard exits during startup and rejects unknown options instead of treating them as a password.

---

## [1.1.0] - 2026-11-25

### Security
- Without `VLLM_MANAGER_PASSWORD`, the dashboard API only answers requests addressed to `localhost`, an IP address, or a hostname in `VLLM_MANAGER_ALLOWED_HOSTS`. This blocks DNS rebinding, where a web page points its own hostname at `127.0.0.1` to drive a local dashboard.
- The dashboard refuses to bind to a non-loopback address without a password unless `--insecure-no-auth` (or `VLLM_MANAGER_INSECURE_NO_AUTH=true`, or `service configure --insecure-no-auth`) is set.
- Model IDs are validated as Hugging Face repo ids before any URL, cache path or container uses them. Input such as `a%zz` no longer crashes the CLI.

### Changed
- **Upgrade note:** a systemd service configured with a non-loopback host and no password no longer starts. Run `service configure --password-file <file>` or `service configure --insecure-no-auth`.
- Download size comes from the repo's file list (`?blobs=true`) instead of `usedStorage`, which counted old file versions and overstated downloads several times over (for example 3.5 GB instead of 1 GB for Qwen2.5-0.5B-Instruct). The disk-space preflight and the progress bar use the corrected size.
- When a repo ships `.safetensors`, downloads skip weight formats vLLM does not load (`*.bin`, `*.pt`, `*.pth`, ONNX, GGUF and similar).
- `GET /api/models/download/progress` and `model progress` report a state (`running`, `succeeded`, `failed`, `stopped`), with the exit code and log tail for failures. Download containers are no longer auto-removed, so a failure stays visible until it is read.
- Model info and download make one Hugging Face API call per request instead of up to four.
- `StartModel` now takes a `LaunchConfig` struct instead of 31 positional arguments.

### Fixed
- A failed download (bad name, gated model, network error) no longer leaves the progress bar stuck with no error.
- An interrupted download is no longer reported as downloaded. A model counts as downloaded only when a snapshot has `config.json` or `params.json` and no `*.incomplete` blobs remain, and downloading again resumes it.
- Expired dashboard sessions and stale failed-login entries are pruned on login.
- Concurrent launches of different models targeting the same GPU can no longer both pass the VRAM preflight; a process-wide launch mutex covers the Docker snapshot, port reservation, overlap check, and container start.
- Restarting a stopped model (dashboard or `model restart`) now takes the launch mutex and runs the GPU overlap and VRAM fit checks against running containers, so it can no longer oversubscribe a GPU that another model claimed while it was stopped.
- Proxy request counts are tracked in memory and flushed periodically instead of rewriting `settings.json` on every inference POST. Settings writes use a temp file plus `rename` so a crash cannot truncate tokens and keys.
- The `/v1/` proxy bounds request-body buffering with `http.MaxBytesReader` (default 32 MiB, configurable via `proxy_max_body_bytes`).
- Dashboard login issues a unique server-side session per login. Logout revokes that session. Failed logins are rate-limited per client IP.

---

## [1.0.0] - 2026-08-18

### Added
- **Single-Binary Engine**: Consolidated CLI operations and embedded React management web console with `go:embed`.
- **VRAM Preflight Guardrails**: Automated weight calculations, KV cache memory estimation per token, context length bounds checking, and multi-container GPU memory overlap validation.
- **Multi-GPU Parallelism**: Support for Tensor Parallelism (`--tensor-parallel-size`), Pipeline Parallelism (`--pipeline-parallel-size`), and explicit GPU device pinning (`--gpu-ids`).
- **OpenAI-Compatible Reverse Proxy Gateway**: Built-in `/v1/` routing gateway with Bearer token authentication, per-user request accounting, and dynamic route mapping.
- **Hugging Face Hub Integration**: Background model downloads, directory size inspection, and sandboxed Alpine cache deletion.
- **Prometheus & DCGM Monitoring**: File-SD target synchronization (`vllm-targets.yml`) and preconfigured Grafana dashboards.
- **Linux Packaging**: Multi-architecture Debian (`.deb`) and Red Hat (`.rpm`) packages for AMD64 and ARM64.
- **Diagnostic Preflight Tool**: `vllm-manager doctor` command for verifying host drivers, Docker access, and GPU readiness.
- **Command Recipe Importer**: Intelligent parser for translating raw `docker run` and `vllm serve` shell commands into validated configuration parameters.
- **Docusaurus Documentation**: Complete documentation suite deployed to GitHub Pages.
