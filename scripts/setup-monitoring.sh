#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITORING_DIR="${VLLM_MANAGER_MONITORING_ASSET_DIR:-$ROOT_DIR/monitoring}"
DEFAULT_RUNTIME_DIR="$MONITORING_DIR/runtime"
if [[ "$ROOT_DIR" == /usr/share/vllm-manager || "$ROOT_DIR" == /usr/local/share/vllm-manager ]]; then
  DEFAULT_RUNTIME_DIR="/var/lib/vllm-manager/monitoring"
fi
RUNTIME_DIR="${VLLM_MANAGER_MONITORING_STATE_DIR:-$DEFAULT_RUNTIME_DIR}"
CERT_DIR="$RUNTIME_DIR/certs"
ENV_FILE="$RUNTIME_DIR/monitoring.env"
PROM_FILE="$RUNTIME_DIR/prometheus.yml"
TARGETS_FILE="$RUNTIME_DIR/vllm-targets.yml"
DASHBOARD_DIR="$RUNTIME_DIR/grafana/dashboards"
DCGM_DASHBOARD_REVISION=1
VLLM_DASHBOARD_REVISION=1

prompt() {
  local label="$1"
  local default_value="$2"
  local value
  if [ -n "$default_value" ]; then
    read -r -p "$label [$default_value]: " value
    printf '%s' "${value:-$default_value}"
  else
    read -r -p "$label: " value
    printf '%s' "$value"
  fi
}

prompt_secret() {
  local label="$1"
  local value
  read -r -s -p "$label: " value
  printf '\n' >&2
  printf '%s' "$value"
}

validate_ip() {
  python3 - "$1" <<'PY'
import ipaddress, sys
try:
    ipaddress.ip_address(sys.argv[1])
except ValueError:
    raise SystemExit(1)
PY
}

download_dashboard() {
  local dashboard_id="$1"
  local revision="$2"
  local output="$3"
  local temporary="${output}.download"
  curl -fsSL "https://grafana.com/api/dashboards/${dashboard_id}/revisions/${revision}/download" -o "$temporary"
  python3 - "$temporary" "$output" <<'PY'
import json, sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    dashboard = json.load(handle)

def normalize(value):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "datasource":
                if isinstance(item, str) and item not in {"-- Grafana --", "grafana"}:
                    value[key] = "Prometheus"
                    continue
                if isinstance(item, dict) and item.get("type") == "prometheus":
                    item["uid"] = "Prometheus"
            normalize(value[key])
    elif isinstance(value, list):
        for item in value:
            normalize(item)

normalize(dashboard)
dashboard.pop("__inputs", None)
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(dashboard, handle, separators=(",", ":"))
PY
  rm -f "$temporary"
}

env_quote() {
  local value="${1//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

write_prometheus_config() {
  local dcgm_port="$1"
  cat > "$PROM_FILE" <<EOF
global:
  scrape_interval: 5s
  evaluation_interval: 15s

scrape_configs:
  - job_name: vllm
    file_sd_configs:
      - files:
          - /etc/prometheus/file_sd/vllm-targets.yml
        refresh_interval: 5s

  - job_name: nvidia-dcgm
    static_configs:
      - targets:
          - 127.0.0.1:${dcgm_port}
        labels:
          exporter: dcgm
EOF
}

for command in docker curl python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to set up monitoring." >&2
    exit 1
  fi
done

mkdir -p "$RUNTIME_DIR" "$CERT_DIR" "$DASHBOARD_DIR"
touch "$TARGETS_FILE"
if [ "$(id -u)" -eq 0 ] && id vllm-manager >/dev/null 2>&1; then
  chown vllm-manager:vllm-manager "$RUNTIME_DIR" "$TARGETS_FILE"
  chmod 0755 "$RUNTIME_DIR"
  chmod 0644 "$TARGETS_FILE"
fi

grafana_user="$(prompt "Grafana admin username" "admin")"
grafana_password="$(prompt_secret "Grafana admin password")"
if [ -z "$grafana_password" ]; then
  echo "Grafana admin password is required." >&2
  exit 1
fi

grafana_bind_ip="$(prompt "Grafana bind IP" "127.0.0.1")"
if ! validate_ip "$grafana_bind_ip"; then
  echo "Invalid Grafana bind IP: $grafana_bind_ip" >&2
  exit 1
fi
grafana_port="$(prompt "Grafana port" "3000")"
prometheus_port="$(prompt "Prometheus loopback port" "9091")"
dcgm_port="$(prompt "DCGM exporter loopback port" "9400")"

enable_https="$(prompt "Enable HTTPS for Grafana? (y/N)" "N")"
grafana_protocol="http"
grafana_cert_file=""
grafana_cert_key=""

case "${enable_https,,}" in
  y|yes)
    if [ "$(id -u)" -ne 0 ]; then
      echo "HTTPS setup needs root to create a Grafana-readable private key; rerun with sudo." >&2
      exit 1
    fi
    grafana_protocol="https"
    cert_mode="$(prompt "Use custom certificate files? (y/N)" "N")"
    if [[ "${cert_mode,,}" =~ ^(y|yes)$ ]]; then
      cert_path="$(prompt "Certificate path" "")"
      key_path="$(prompt "Private key path" "")"
      if [ ! -f "$cert_path" ] || [ ! -f "$key_path" ]; then
        echo "Certificate and key files must exist." >&2
        exit 1
      fi
      cp "$cert_path" "$CERT_DIR/grafana.crt"
      cp "$key_path" "$CERT_DIR/grafana.key"
    else
      if ! command -v openssl >/dev/null 2>&1; then
        echo "openssl is required to generate a self-signed certificate." >&2
        exit 1
      fi
      openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$CERT_DIR/grafana.key" \
        -out "$CERT_DIR/grafana.crt" \
        -days 365 \
        -subj "/CN=$grafana_bind_ip" >/dev/null 2>&1
    fi
    chmod 0644 "$CERT_DIR/grafana.crt"
    chmod 0640 "$CERT_DIR/grafana.key"
    grafana_cert_file="/etc/grafana/certs/grafana.crt"
    grafana_cert_key="/etc/grafana/certs/grafana.key"
    ;;
esac

write_prometheus_config "$dcgm_port"
echo "Downloading pinned Grafana dashboards..."
download_dashboard 22515 "$DCGM_DASHBOARD_REVISION" "$DASHBOARD_DIR/nvidia-dcgm.json"
download_dashboard 25263 "$VLLM_DASHBOARD_REVISION" "$DASHBOARD_DIR/vllm.json"

{
  printf 'GRAFANA_ADMIN_USER=%s\n' "$(env_quote "$grafana_user")"
  printf 'GRAFANA_ADMIN_PASSWORD=%s\n' "$(env_quote "$grafana_password")"
  printf 'GRAFANA_BIND_IP=%s\n' "$(env_quote "$grafana_bind_ip")"
  printf 'GRAFANA_PORT=%s\n' "$(env_quote "$grafana_port")"
  printf 'GRAFANA_PROTOCOL=%s\n' "$(env_quote "$grafana_protocol")"
  printf 'GRAFANA_CERT_FILE=%s\n' "$(env_quote "$grafana_cert_file")"
  printf 'GRAFANA_CERT_KEY=%s\n' "$(env_quote "$grafana_cert_key")"
  printf 'PROMETHEUS_PORT=%s\n' "$(env_quote "$prometheus_port")"
  printf 'DCGM_PORT=%s\n' "$(env_quote "$dcgm_port")"
  printf 'PROMETHEUS_CONFIG_FILE=%s\n' "$(env_quote "$PROM_FILE")"
  printf 'MONITORING_RUNTIME_DIR=%s\n' "$(env_quote "$RUNTIME_DIR")"
  printf 'GRAFANA_DASHBOARDS_DIR=%s\n' "$(env_quote "$DASHBOARD_DIR")"
  printf 'GRAFANA_CERTS_DIR=%s\n' "$(env_quote "$CERT_DIR")"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo
echo "Monitoring files generated:"
echo "  $ENV_FILE"
echo "  $PROM_FILE"
echo "  $TARGETS_FILE"
echo
echo "Run the dashboard with:"
echo "  ./vllm-manager --dashboard --monitoring --monitoring-targets-file $TARGETS_FILE"
echo

start_now="$(prompt "Start monitoring stack now? (Y/n)" "Y")"
if [[ ! "${start_now,,}" =~ ^(n|no)$ ]]; then
  docker compose --env-file "$ENV_FILE" -f "$MONITORING_DIR/docker-compose.yml" up -d
  echo "Grafana: ${grafana_protocol}://${grafana_bind_ip}:${grafana_port}"
fi
