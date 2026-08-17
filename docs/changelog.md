---
title: Changelog
sidebar_position: 7
---

# Changelog

All notable changes to **vLLM GPU Manager** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
