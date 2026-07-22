#!/bin/sh
set -eu

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload >/dev/null 2>&1 || true
fi

# State, model cache, monitoring volumes, and the service account are preserved.
# The source uninstall helper offers explicit destructive cleanup options.
