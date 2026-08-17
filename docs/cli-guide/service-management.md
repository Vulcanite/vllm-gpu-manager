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

This prompts for:
- Dashboard port and bind address.
- Hugging Face cache root directory.
- Hugging Face access token (`HF_TOKEN`).
- Global downstream vLLM API key (`VLLM_API_KEY`).
- Dashboard UI authentication password (`DASHBOARD_PASSWORD`).
- HTTPS TLS certificates (if enabled).

---

## Non-Interactive Configuration via Flags

Automate deployment in cloud-init or Ansible:

```bash
sudo vllm-manager service configure --non-interactive \
  --hf-cache /var/lib/vllm-manager/.cache/huggingface \
  --hf-token "hf_xxx..." \
  --vllm-api-key "vllm_internal_secret..." \
  --dashboard-port 9090 \
  --dashboard-password "admin_password"
```

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
