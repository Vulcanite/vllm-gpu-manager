---
title: Systemd Service & Packaging Ops
sidebar_position: 7
---

# Systemd Service & Packaging Ops

How to configure, secure, and operate **vLLM GPU Manager** as a system service.

---

## 1. Systemd Service Unit

The packaged service unit (`/usr/lib/systemd/system/vllm-manager.service`) is configured as follows:

```ini
[Unit]
Description=vLLM GPU Manager Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=vllm-manager
Group=vllm-manager
EnvironmentFile=/etc/vllm-manager/service.env
ExecStart=/usr/bin/vllm-manager --dashboard --port ${VLLM_MANAGER_PORT} --host ${VLLM_MANAGER_HOST}
Restart=always
RestartSec=5s
LimitNOFILE=65536
LimitNPROC=65536

[Install]
WantedBy=multi-user.target
```

---

## 2. Configuring Service Settings

Use the `service configure` subcommand to modify `/etc/vllm-manager/service.env` atomically:

```bash
# Configure host IP and port
sudo vllm-manager service configure --host 0.0.0.0 --port 9090

# Enable HTTPS and monitoring
sudo vllm-manager service configure --https=true --monitoring=true

# Set dashboard authentication password from a secure file
sudo vllm-manager service configure --password-file /root/.vllm_pass
```

### Environment Variables in `service.env`

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VLLM_MANAGER_HOST` | `127.0.0.1` | Bind address for dashboard and proxy gateway. |
| `VLLM_MANAGER_PORT` | `9090` | TCP port for dashboard and proxy gateway. |
| `VLLM_MANAGER_HTTPS` | `false` | Enable HTTPS with local self-signed TLS certificates. |
| `VLLM_MANAGER_NO_PROXY` | `false` | Disable `/v1/` reverse proxy gateway. |
| `VLLM_MANAGER_MONITORING` | `false` | Enable Prometheus target sync to `vllm-targets.yml`. |
| `VLLM_MANAGER_PASSWORD` | `""` | Password required to access dashboard and APIs. |

---

## 3. Service Lifecycle Management

```bash
# Enable service on boot and start immediately
sudo systemctl enable --now vllm-manager

# Check service status
systemctl status vllm-manager

# View live service logs via journald
journalctl -u vllm-manager -f

# Restart service
sudo systemctl restart vllm-manager
```

---

## 4. Uninstallation & Data Retention

To remove the package while retaining cached models and container states:

```bash
# On Debian / Ubuntu:
sudo apt remove vllm-manager

# On Red Hat / Rocky:
sudo dnf remove vllm-manager
```

To run interactive cleanup (optionally removing model containers, cached Hugging Face weights, and monitoring volumes):

```bash
sudo /usr/share/vllm-manager/scripts/uninstall.sh
```
