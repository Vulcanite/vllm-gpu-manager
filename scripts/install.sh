#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-$(git -C "$ROOT_DIR" describe --tags --always --dirty 2>/dev/null || printf 'dev')}"
VERSION="${VERSION#v}"
INSTALL_BIN="/usr/local/bin/vllm-manager"
UNIT_PATH="/etc/systemd/system/vllm-manager.service"
SHARE_DIR="/usr/local/share/vllm-manager"
CONFIG_DIR="/etc/vllm-manager"
STATE_DIR="/var/lib/vllm-manager"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

if [ "$(uname -s)" != "Linux" ]; then
  echo "vllm-manager systemd installation requires Linux." >&2
  exit 1
fi
if ! command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
  echo "sudo or root access is required." >&2
  exit 1
fi
SUDO=()
if [ "$(id -u)" -ne 0 ]; then
  SUDO=(sudo)
fi

for command in go npm; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required to build from source." >&2
    exit 1
  fi
done

echo "Building frontend and vllm-manager ${VERSION}..."
(cd "$ROOT_DIR/frontend" && npm ci && npm run lint && npm run build)
(cd "$ROOT_DIR" && go test ./... && CGO_ENABLED=0 go build -trimpath \
  -ldflags="-s -w -X main.version=${VERSION} -X main.defaultConfigDir=${CONFIG_DIR} -X main.defaultStateDir=${STATE_DIR}" \
  -o "$BUILD_DIR/vllm-manager" .)

if ! getent group docker >/dev/null 2>&1; then
  echo "The docker group does not exist. Install Docker before vllm-manager." >&2
  exit 1
fi

if ! getent group vllm-manager >/dev/null 2>&1; then
  "${SUDO[@]}" groupadd --system vllm-manager
fi
if ! getent passwd vllm-manager >/dev/null 2>&1; then
  "${SUDO[@]}" useradd --system --gid vllm-manager --home-dir "$STATE_DIR" \
    --no-create-home --shell /usr/sbin/nologin vllm-manager
fi
for group in docker video render; do
  if getent group "$group" >/dev/null 2>&1; then
    "${SUDO[@]}" usermod -a -G "$group" vllm-manager
  fi
done

"${SUDO[@]}" install -d -o root -g vllm-manager -m 0750 "$CONFIG_DIR"
"${SUDO[@]}" install -d -o vllm-manager -g vllm-manager -m 0750 \
  "$STATE_DIR" "$STATE_DIR/monitoring" "$STATE_DIR/.cache" "$STATE_DIR/.cache/huggingface"
if [ ! -e "$STATE_DIR/settings.json" ]; then
  printf '%s\n' '{"hf_cache":"/var/lib/vllm-manager/.cache/huggingface","vllm_image":"vllm/vllm-openai:v0.23.0","server_bind_ip":"0.0.0.0","routing_table":{}}' > "$BUILD_DIR/settings.json"
  "${SUDO[@]}" install -o vllm-manager -g vllm-manager -m 0600 "$BUILD_DIR/settings.json" "$STATE_DIR/settings.json"
fi
"${SUDO[@]}" install -o root -g root -m 0755 "$BUILD_DIR/vllm-manager" "$INSTALL_BIN"
"${SUDO[@]}" install -d -o root -g root -m 0755 "$SHARE_DIR/scripts" "$SHARE_DIR/monitoring/grafana"
"${SUDO[@]}" install -o root -g root -m 0755 "$ROOT_DIR/scripts/setup-monitoring.sh" "$SHARE_DIR/scripts/setup-monitoring.sh"
"${SUDO[@]}" install -o root -g root -m 0755 "$ROOT_DIR/scripts/uninstall.sh" "$SHARE_DIR/scripts/uninstall.sh"
"${SUDO[@]}" install -o root -g root -m 0644 "$ROOT_DIR/monitoring/docker-compose.yml" "$SHARE_DIR/monitoring/docker-compose.yml"
"${SUDO[@]}" cp -R "$ROOT_DIR/monitoring/grafana/provisioning" "$SHARE_DIR/monitoring/grafana/"
if [ ! -e "$CONFIG_DIR/service.env" ]; then
  "${SUDO[@]}" install -o root -g vllm-manager -m 0640 "$ROOT_DIR/packaging/service.env" "$CONFIG_DIR/service.env"
fi
sed "s|ExecStart=/usr/bin/vllm-manager|ExecStart=${INSTALL_BIN}|" "$ROOT_DIR/packaging/vllm-manager.service" > "$BUILD_DIR/vllm-manager.service"
"${SUDO[@]}" install -o root -g root -m 0644 "$BUILD_DIR/vllm-manager.service" "$UNIT_PATH"
"${SUDO[@]}" systemctl daemon-reload

echo
echo "Installed $INSTALL_BIN. The service is disabled until explicitly enabled."
echo "Configure: sudo vllm-manager service configure"
echo "Start:     sudo systemctl enable --now vllm-manager"
echo "Monitoring setup: sudo VLLM_MANAGER_MONITORING_STATE_DIR=$STATE_DIR/monitoring $SHARE_DIR/scripts/setup-monitoring.sh"
echo "Uninstall: sudo $SHARE_DIR/scripts/uninstall.sh"
