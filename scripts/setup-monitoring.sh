#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITORING_DIR="$ROOT_DIR/monitoring"
RUNTIME_DIR="$MONITORING_DIR/runtime"
CERT_DIR="$RUNTIME_DIR/certs"
ENV_FILE="$MONITORING_DIR/.env"
PROM_FILE="$RUNTIME_DIR/prometheus.yml"
TARGETS_FILE="$RUNTIME_DIR/vllm-targets.yml"

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

mkdir -p "$RUNTIME_DIR" "$CERT_DIR"
touch "$TARGETS_FILE"

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
    chmod 600 "$CERT_DIR/grafana.key"
    grafana_cert_file="/etc/grafana/certs/grafana.crt"
    grafana_cert_key="/etc/grafana/certs/grafana.key"
    ;;
esac

write_prometheus_config "$dcgm_port"

cat > "$ENV_FILE" <<EOF
GRAFANA_ADMIN_USER=$grafana_user
GRAFANA_ADMIN_PASSWORD=$grafana_password
GRAFANA_BIND_IP=$grafana_bind_ip
GRAFANA_PORT=$grafana_port
GRAFANA_PROTOCOL=$grafana_protocol
GRAFANA_CERT_FILE=$grafana_cert_file
GRAFANA_CERT_KEY=$grafana_cert_key
PROMETHEUS_PORT=$prometheus_port
DCGM_PORT=$dcgm_port
EOF
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
  (cd "$MONITORING_DIR" && docker compose up -d)
  echo "Grafana: ${grafana_protocol}://${grafana_bind_ip}:${grafana_port}"
fi
