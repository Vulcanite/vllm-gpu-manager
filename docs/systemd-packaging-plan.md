# Systemd Packaging Plan

## Summary

Package `vllm-manager` with a first-pass installer script, a dedicated `vllm-manager` service user, and a single app directory under `/opt/vllm-manager`. Install a CLI wrapper into `/usr/local/bin/vllm-manager` so shell commands and the systemd service always use the same packaged config.

## Key Changes

- Add app-home path support:
  - New env var: `VLLM_MANAGER_HOME`, default `.` for dev mode.
  - Resolve packaged runtime files relative to `VLLM_MANAGER_HOME`: `settings.json`, TLS cert/key, and default monitoring target file.
  - Keep current local-dev behavior unchanged when `VLLM_MANAGER_HOME` is unset.
- Add installer script:
  - New `scripts/install-systemd.sh`.
  - Requires root/sudo.
  - Copies the built binary into `/opt/vllm-manager/bin/vllm-manager`.
  - Creates `/usr/local/bin/vllm-manager` wrapper that exports `VLLM_MANAGER_HOME=/opt/vllm-manager`.
  - Creates system user `vllm-manager` with home `/opt/vllm-manager`.
  - Adds service user to `docker`, plus `video`/`render` when present.
  - Prompts for dashboard host, port, HTTPS, proxy mode, monitoring, monitoring targets path, and Hugging Face cache path.
  - Writes `/opt/vllm-manager/vllm-manager.env`.
  - Creates initial `/opt/vllm-manager/settings.json` only if missing.
- Add systemd unit:
  - Install as `/etc/systemd/system/vllm-manager.service`.
  - Runs as `vllm-manager`.
  - Uses `EnvironmentFile=/opt/vllm-manager/vllm-manager.env`.
  - Uses a small service launcher script to assemble optional flags like `--https`, `--no-proxy`, and `--monitoring`.
  - Logs to journald.
- App directory layout:
  - `/opt/vllm-manager/bin/vllm-manager`
  - `/opt/vllm-manager/bin/vllm-manager-service`
  - `/opt/vllm-manager/settings.json`
  - `/opt/vllm-manager/server.crt`
  - `/opt/vllm-manager/server.key`
  - `/opt/vllm-manager/monitoring/runtime/vllm-targets.yml`
  - Hugging Face cache path chosen during install, default prompt value `/opt/vllm-manager/huggingface`.

## Public Interfaces

- New environment variables:
  - `VLLM_MANAGER_HOME`: base directory for packaged config/runtime files.
  - `VLLM_MANAGER_HOST`: dashboard bind IP for systemd wrapper.
  - `VLLM_MANAGER_PORT`: dashboard port.
  - `VLLM_MANAGER_HTTPS`: enables `--https`.
  - `VLLM_MANAGER_NO_PROXY`: enables `--no-proxy`.
  - `VLLM_MANAGER_MONITORING`: enables `--monitoring`.
  - `VLLM_MANAGER_MONITORING_TARGETS_FILE`: overrides monitoring target file.
  - Existing `VLLM_MANAGER_PASSWORD` remains supported.
- New commands:
  - `sudo scripts/install-systemd.sh`
  - `sudo systemctl enable --now vllm-manager`
  - `sudo systemctl status vllm-manager`
  - `journalctl -u vllm-manager -f`

## Test Plan

- Unit tests:
  - Path resolution uses current directory when `VLLM_MANAGER_HOME` is unset.
  - Path resolution uses `/opt/vllm-manager`-style base when `VLLM_MANAGER_HOME` is set.
  - Default monitoring target path resolves under app home in packaged mode.
- Script checks:
  - `bash -n scripts/install-systemd.sh`
  - `bash -n` for generated service launcher template.
  - Installer dry-run mode validates planned paths without writing system files.
- Manual validation:
  - Fresh install on Linux host.
  - Confirm service starts with systemd.
  - Confirm CLI wrapper reads the same settings as dashboard.
  - Confirm model download writes to configured HF cache.
  - Confirm monitoring writes `/opt/vllm-manager/monitoring/runtime/vllm-targets.yml` when enabled.
  - Confirm Prometheus/DCGM remain loopback-only and only Grafana is exposed on configured host.

## Assumptions

- First packaging target is an install script, not `.deb`/`.rpm`.
- Packaged service runs as dedicated `vllm-manager` user.
- Docker group access is acceptable for the service user, with documentation that Docker group is root-equivalent.
- Single app directory is preferred, with the model cache path prompted during install.
- `manager.sh` remains a dev/manual helper; systemd uses a dedicated service launcher instead.
