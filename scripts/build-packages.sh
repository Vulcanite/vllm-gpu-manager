#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || printf 'dev')}"
VERSION="${VERSION#v}"
if [[ ! "$VERSION" =~ ^[0-9] ]]; then
  VERSION="0.0.0-$VERSION"
fi
NFPM_VERSION="${NFPM_VERSION:-2.47.0}"
NFPM_BIN="${NFPM_BIN:-$(command -v nfpm || true)}"
if [ -z "$NFPM_BIN" ] && [ -x "$(go env GOPATH)/bin/nfpm" ]; then
  NFPM_BIN="$(go env GOPATH)/bin/nfpm"
fi
ARCHES="${ARCHES:-amd64 arm64}"

if [ -z "$NFPM_BIN" ]; then
  echo "nfpm is required. Install the pinned version with:" >&2
  echo "  go install github.com/goreleaser/nfpm/v2/cmd/nfpm@v${NFPM_VERSION}" >&2
  exit 1
fi

mkdir -p dist
debian_version="${VERSION//-/~}"
changelog_date="$(date -R)"
if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
  changelog_date="$(date -R -d "@${SOURCE_DATE_EPOCH}")"
fi
printf 'vllm-manager (%s) unstable; urgency=medium\n\n  * Release %s.\n\n -- Amish <opensource@vulcanite.dev>  %s\n' \
  "$debian_version" "$VERSION" "$changelog_date" | gzip -9n > dist/changelog.gz

for arch in $ARCHES; do
  echo "Building linux/${arch}..."
  CGO_ENABLED=0 GOOS=linux GOARCH="$arch" go build -trimpath \
    -ldflags="-s -w -X main.version=${VERSION} -X main.defaultConfigDir=/etc/vllm-manager -X main.defaultStateDir=/var/lib/vllm-manager" \
    -o "dist/vllm-manager-linux-${arch}" .
  chmod 0755 "dist/vllm-manager-linux-${arch}"

  echo "Packaging Debian package for ${arch}..."
  ARCH="$arch" VERSION="$VERSION" "$NFPM_BIN" package \
    --config packaging/nfpm.yaml \
    --packager deb \
    --target "dist/vllm-manager_${VERSION}_${arch}.deb"

  echo "Packaging RPM package for ${arch}..."
  ARCH="$arch" VERSION="$VERSION" "$NFPM_BIN" package \
    --config packaging/nfpm.yaml \
    --packager rpm \
    --target "dist/vllm-manager-${VERSION}-1.${arch}.rpm"
done
