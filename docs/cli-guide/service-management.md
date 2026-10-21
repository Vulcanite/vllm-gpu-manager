---
title: Systemd Service Configuration
sidebar_position: 6
---

# Systemd Service Configuration

Configure the background daemon service for automated startup and secure credential isolation.

---

## Interactive Service Configurator

Configure `/etc/vllm-manager/service.env` interactively:

```bash
sudo vllm-manager service configure
```

Run without flags on a terminal, it prompts for:
- Dashboard bind host and port.
- HTTPS on/off (a self-signed certificate is generated on first TLS start).
- Proxy gateway on/off.
- Monitoring target sync on/off, and the targets file path.
- Dashboard password (`VLLM_MANAGER_PASSWORD`; blank keeps the current value).

The Hugging Face cache, Hugging Face token, and vLLM API key are not service settings. Set them with `sudo vllm-manager settings set` or from the web console.

---

## Non-Interactive Configuration via Flags

Passing any flag skips the prompts, which suits cloud-init or Ansible:

```bash
sudo vllm-manager service configure \
  --host 0.0.0.0 \
  --port 9090 \
  --https \
  --proxy=true \
  --password-file /root/vllm-manager-password

sudo vllm-manager settings set \
  --hf-cache /var/lib/vllm-manager/.cache/huggingface \
  --hf-token "hf_..."
```

Other flags: `--monitoring[=false]`, `--monitoring-targets-file <path>`, `--clear-password`, and `--no-restart`. An active service is restarted after a change unless `--no-restart` is given.

---

## Managing the Systemd Daemon

```bash
# Start and enable the service
sudo systemctl enable --now vllm-manager

# Inspect service logs
journalctl -u vllm-manager -f

# Check runtime status
systemctl status vllm-manager
```
