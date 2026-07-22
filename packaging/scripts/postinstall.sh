#!/bin/sh
set -eu

if ! getent group vllm-manager >/dev/null 2>&1; then
  addgroup --system vllm-manager >/dev/null
fi
if ! getent passwd vllm-manager >/dev/null 2>&1; then
  adduser --system --ingroup vllm-manager --home /var/lib/vllm-manager \
    --no-create-home --shell /usr/sbin/nologin vllm-manager >/dev/null
fi

for group in docker video render; do
  if getent group "$group" >/dev/null 2>&1; then
    adduser vllm-manager "$group" >/dev/null 2>&1 || true
  fi
done
if ! getent group docker >/dev/null 2>&1; then
  printf '%s\n' 'warning: Docker is not installed or its docker group is missing; install Docker before starting vllm-manager' >&2
fi

install -d -o vllm-manager -g vllm-manager -m 0750 \
  /var/lib/vllm-manager \
  /var/lib/vllm-manager/monitoring \
  /var/lib/vllm-manager/.cache \
  /var/lib/vllm-manager/.cache/huggingface
if [ ! -e /var/lib/vllm-manager/settings.json ]; then
  cat > /var/lib/vllm-manager/settings.json <<'EOF'
{"hf_cache":"/var/lib/vllm-manager/.cache/huggingface","vllm_image":"vllm/vllm-openai:v0.23.0","server_bind_ip":"0.0.0.0","routing_table":{}}
EOF
  chown vllm-manager:vllm-manager /var/lib/vllm-manager/settings.json
  chmod 0600 /var/lib/vllm-manager/settings.json
fi
install -d -o root -g vllm-manager -m 0750 /etc/vllm-manager
if [ ! -e /etc/vllm-manager/service.env ]; then
  cat > /etc/vllm-manager/service.env <<'EOF'
# Dashboard defaults. Update with `sudo vllm-manager service configure`.
VLLM_MANAGER_HOST="127.0.0.1"
VLLM_MANAGER_PORT="9090"
VLLM_MANAGER_HTTPS=false
VLLM_MANAGER_NO_PROXY=false
VLLM_MANAGER_MONITORING=false
VLLM_MANAGER_MONITORING_TARGETS_FILE="/var/lib/vllm-manager/monitoring/vllm-targets.yml"
VLLM_MANAGER_PASSWORD=""
VLLM_MANAGER_CONFIG_DIR="/etc/vllm-manager"
VLLM_MANAGER_STATE_DIR="/var/lib/vllm-manager"
EOF
fi
chown root:vllm-manager /etc/vllm-manager/service.env
chmod 0640 /etc/vllm-manager/service.env

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload || true
fi

printf '%s\n' 'vllm-manager installed but not enabled.'
printf '%s\n' 'Configure it with: sudo vllm-manager service configure'
printf '%s\n' 'Start it with: sudo systemctl enable --now vllm-manager'
