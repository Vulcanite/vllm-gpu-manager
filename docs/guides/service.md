---
title: Run as a service
sidebar_position: 5
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Run as a service

The `.deb` and `.rpm` packages install a systemd unit that runs the dashboard as the `vllm-manager` user. For a standalone binary, `manager.sh` starts it in the background instead.

<Tabs groupId="runner">
<TabItem value="systemd" label="systemd (packages)" default>

### Configure

`service configure` edits `/etc/vllm-manager/service.env` atomically and restarts the service if it is running. With no flags on a terminal, it asks for each setting:

```bash
sudo vllm-manager service configure
```

With flags it does not prompt, which suits cloud-init or Ansible:

```bash
# Listen on all interfaces; a non-loopback host needs a password
sudo vllm-manager service configure --host 0.0.0.0 --port 9090 --password-file /root/vllm-manager-password

# HTTPS (self-signed) and Prometheus target sync
sudo vllm-manager service configure --https --monitoring
```

Other flags: `--proxy=false`, `--monitoring-targets-file <path>`, `--clear-password`, `--insecure-no-auth` and `--no-restart`.

A non-loopback `--host` without a password is rejected. On a trusted network you can pass `--insecure-no-auth` instead; anyone who can reach the port then controls Docker on the host.

The Hugging Face cache, Hugging Face token and vLLM API key are not service settings; set them with `sudo vllm-manager settings set` or in the dashboard.

### Start and inspect

```bash
sudo systemctl enable --now vllm-manager
systemctl status vllm-manager
journalctl -u vllm-manager -f
```

### The unit

```ini
[Unit]
Description=vLLM Manager dashboard
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=vllm-manager
Group=vllm-manager
SupplementaryGroups=docker
EnvironmentFile=-/etc/vllm-manager/service.env
WorkingDirectory=/var/lib/vllm-manager
ExecStart=/usr/bin/vllm-manager --dashboard
Restart=on-failure
RestartSec=5s
UMask=0027
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

The dashboard reads host, port and the other options from the `VLLM_MANAGER_*` variables in `service.env`; see [Settings & environment](../reference/settings.md#environment-variables).

### Uninstall

```bash
sudo apt remove vllm-manager      # or: sudo dnf remove vllm-manager
```

This keeps cached models and settings. To also remove model containers, cached weights and monitoring volumes, run the interactive cleanup first:

```bash
sudo /usr/share/vllm-manager/scripts/uninstall.sh
```

</TabItem>
<TabItem value="manager" label="manager.sh (standalone binary)">

Run from the folder that holds the `vllm-manager` binary:

```bash
./manager.sh start                                   # 127.0.0.1:9090
./manager.sh start 8080 change-me --host 0.0.0.0     # port, password, host
./manager.sh start 8080 --host 0.0.0.0 --insecure-no-auth
./manager.sh status
./manager.sh restart                                 # reuses the saved options
./manager.sh stop
```

Other options: `--no-proxy`, `--https`, `--monitoring` and `--monitoring-targets-file <path>`. The options and password are saved in `.vllm-manager.state` (mode `0600`) so `restart` can reuse them. If the binary exits during start-up, for example on a non-loopback host without a password, `start` prints the end of `vllm-manager.log` and fails.

</TabItem>
</Tabs>
