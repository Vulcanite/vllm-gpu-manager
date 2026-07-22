#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

grep -q '^Apache License$' LICENSE
go run github.com/google/go-licenses@v1.6.0 check ./...
node <<'NODE'
const lock = require('./frontend/package-lock.json')
const denied = /^(AGPL|GPL)/i
for (const [name, pkg] of Object.entries(lock.packages || {})) {
  if (!name || pkg.dev === true) continue
  if (!pkg.license) throw new Error(`missing frontend production license: ${name}`)
  if (denied.test(pkg.license)) throw new Error(`unsupported frontend production license ${pkg.license}: ${name}`)
}
console.log('Frontend production dependency licenses: OK')
NODE
