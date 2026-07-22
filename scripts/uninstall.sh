#!/usr/bin/env bash
set -euo pipefail

PURGE=0
REMOVE_MONITORING_DATA=0
REMOVE_MODEL_CONTAINERS=0
REMOVE_MODEL_CACHE=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    --remove-monitoring-data) REMOVE_MONITORING_DATA=1 ;;
    --remove-model-containers) REMOVE_MODEL_CONTAINERS=1 ;;
    --remove-model-cache) REMOVE_MODEL_CACHE=1 ;;
    --yes) ASSUME_YES=1 ;;
    -h|--help)
      echo "Usage: uninstall.sh [--purge] [--remove-monitoring-data] [--remove-model-containers] [--remove-model-cache] [--yes]"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

SUDO=()
if [ "$(id -u)" -ne 0 ]; then
  SUDO=(sudo)
fi

confirm() {
  local auto_confirm="$1"
  local prompt="$2"
  [ "$auto_confirm" -eq 1 ] && [ "$ASSUME_YES" -eq 1 ] && return 0
  [ -t 0 ] || return 1
  read -r -p "$prompt [y/N]: " answer
  [[ "${answer,,}" =~ ^(y|yes)$ ]]
}

"${SUDO[@]}" systemctl disable --now vllm-manager.service >/dev/null 2>&1 || true

manager_bin="$(command -v vllm-manager || true)"
hf_cache="/var/lib/vllm-manager/.cache/huggingface"
if [ -n "$manager_bin" ]; then
  hf_cache="$("${SUDO[@]}" "$manager_bin" settings get --field hf_cache 2>/dev/null || printf '%s' "$hf_cache")"
fi

declare -A retained_models=()
retained_unknown_model=0

if command -v docker >/dev/null 2>&1; then
  if container_rows="$("${SUDO[@]}" docker ps -a \
    --filter label=vllm-manager.managed=true \
    --format '{{.ID}}\t{{.Names}}\t{{.Label "vllm-manager.model"}}\t{{.Status}}' 2>/dev/null)"; then
    while IFS=$'\t' read -r container_id container_name container_model container_status <&3; do
      [ -n "$container_id" ] || continue
      if confirm "$REMOVE_MODEL_CONTAINERS" "Stop and remove container $container_name ($container_model; $container_status)?"; then
        if ! "${SUDO[@]}" docker rm -f "$container_id" >/dev/null; then
          echo "Warning: failed to remove container $container_name; its model data will be preserved." >&2
          [ -z "$container_model" ] || retained_models["$container_model"]=1
          [ -n "$container_model" ] || retained_unknown_model=1
        fi
      else
        [ -z "$container_model" ] || retained_models["$container_model"]=1
        [ -n "$container_model" ] || retained_unknown_model=1
      fi
    done 3<<<"$container_rows"
  else
    echo "Warning: failed to inspect managed containers; cached models will be preserved." >&2
    retained_unknown_model=1
  fi
else
  echo "Warning: Docker is unavailable; managed containers and cached models will be preserved." >&2
  retained_unknown_model=1
fi

if "${SUDO[@]}" test -d "$hf_cache"; then
  hf_cache="$("${SUDO[@]}" readlink -f -- "$hf_cache")"
  remaining_model=""
  if "${SUDO[@]}" test -d "$hf_cache/hub"; then
    while IFS= read -r -d '' model_dir <&3; do
      model_dir_name="${model_dir##*/}"
      model_name="${model_dir_name#models--}"
      model_name="${model_name//--//}"
      if [ "$retained_unknown_model" -eq 1 ] || [ "${retained_models[$model_name]:-0}" -eq 1 ]; then
        echo "Preserving cached model $model_name because a retained container may reference it."
      elif confirm "$REMOVE_MODEL_CACHE" "Delete cached model $model_name?"; then
        "${SUDO[@]}" rm -rf -- "$model_dir" "$hf_cache/hub/.locks/$model_dir_name"
      fi
    done 3< <("${SUDO[@]}" find "$hf_cache/hub" -mindepth 1 -maxdepth 1 -type d -name 'models--*' -print0)
    remaining_model="$("${SUDO[@]}" find "$hf_cache/hub" -mindepth 1 -maxdepth 1 -type d -name 'models--*' -print -quit)"
  fi
  if [ -z "$remaining_model" ] && \
    confirm "$REMOVE_MODEL_CACHE" "Remove remaining Hugging Face cache metadata under $hf_cache?"; then
    if [ -n "$hf_cache" ] && [ "$hf_cache" != "/" ]; then
      "${SUDO[@]}" rm -rf -- "$hf_cache"
    else
      echo "Warning: refusing to remove unsafe Hugging Face cache path: $hf_cache" >&2
    fi
  fi
fi

if command -v docker >/dev/null 2>&1; then
  monitoring_env="/var/lib/vllm-manager/monitoring/monitoring.env"
  monitoring_compose=""
  for candidate in \
    /usr/share/vllm-manager/monitoring/docker-compose.yml \
    /usr/local/share/vllm-manager/monitoring/docker-compose.yml; do
    if [ -f "$candidate" ]; then
      monitoring_compose="$candidate"
      break
    fi
  done
  if [ -n "$monitoring_compose" ] && [ -f "$monitoring_env" ]; then
    "${SUDO[@]}" docker compose --env-file "$monitoring_env" -f "$monitoring_compose" down >/dev/null 2>&1 || true
  fi
  "${SUDO[@]}" docker rm -f \
    vllm-manager-grafana vllm-manager-prometheus vllm-manager-dcgm >/dev/null 2>&1 || true
fi

if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W -f='${db:Status-Status}' vllm-manager 2>/dev/null | grep -q '^installed$'; then
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get remove -y vllm-manager
else
  "${SUDO[@]}" rm -f /etc/systemd/system/vllm-manager.service /usr/local/bin/vllm-manager
  "${SUDO[@]}" rm -rf /usr/local/share/vllm-manager
  "${SUDO[@]}" systemctl daemon-reload >/dev/null 2>&1 || true
fi

if confirm "$PURGE" "Remove configuration and manager state? This does not remove retained Hugging Face model data"; then
  state_dir="/var/lib/vllm-manager"
  "${SUDO[@]}" rm -rf /etc/vllm-manager
  if "${SUDO[@]}" test -d "$hf_cache" && [[ "$hf_cache" == "$state_dir"/* ]]; then
    cache_top="${hf_cache#"$state_dir"/}"
    cache_top="${cache_top%%/*}"
    "${SUDO[@]}" find "$state_dir" -mindepth 1 -maxdepth 1 ! -name "$cache_top" -exec rm -rf -- {} +
  else
    "${SUDO[@]}" rm -rf "$state_dir"
  fi
  if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W vllm-manager >/dev/null 2>&1; then
    "${SUDO[@]}" dpkg --purge vllm-manager >/dev/null
  fi
  if [ ! -d /var/lib/vllm-manager ] && ! "${SUDO[@]}" test -d "$hf_cache" && getent passwd vllm-manager >/dev/null 2>&1; then
    "${SUDO[@]}" userdel vllm-manager || true
  fi
fi

if confirm "$REMOVE_MONITORING_DATA" "Remove vllm-manager Grafana and Prometheus Docker volumes?"; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Warning: Docker is unavailable; monitoring volumes were not removed." >&2
  else
    volumes=()
    for volume in vllm-manager-grafana-data vllm-manager-prometheus-data; do
      if "${SUDO[@]}" docker volume inspect "$volume" >/dev/null 2>&1; then
        volumes+=("$volume")
      fi
    done
    if [ "${#volumes[@]}" -gt 0 ] && ! "${SUDO[@]}" docker volume rm "${volumes[@]}" >/dev/null; then
      echo "Warning: one or more monitoring volumes could not be removed." >&2
    fi
  fi
fi

echo "vllm-manager removed. Resources not explicitly confirmed for deletion were preserved."
