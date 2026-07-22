#!/bin/sh
set -eu

if [ "${1:-}" = "remove" ]; then
  if command -v deb-systemd-invoke >/dev/null 2>&1; then
    deb-systemd-invoke stop vllm-manager.service >/dev/null 2>&1 || true
  fi
  if command -v deb-systemd-helper >/dev/null 2>&1; then
    deb-systemd-helper disable vllm-manager.service >/dev/null 2>&1 || true
  fi
fi
