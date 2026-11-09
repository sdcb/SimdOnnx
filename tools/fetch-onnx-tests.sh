#!/usr/bin/env bash
# Fetches the ONNX backend node tests (onnx/onnx, Apache-2.0) into artifacts/cache/onnx-node-<tag>/.
# Same pin and hash as tools/fetch-onnx-tests.ps1.
set -euo pipefail
tag=v1.22.0
sha=70bb8b25cf31ea9b1d9f94baacfdc8c4fa27a760f9a10f5d93881bc9eede5fbc
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/artifacts/cache/onnx-node-$tag"
[ -f "$dest/.complete" ] && { echo "already fetched: $dest"; exit 0; }
dl="$root/artifacts/cache/dl"; mkdir -p "$dl"
archive="$dl/onnx-$tag.tar.gz"
[ -f "$archive" ] || curl -sSL --fail -o "$archive" "https://github.com/onnx/onnx/archive/refs/tags/$tag.tar.gz"
got=$( (sha256sum "$archive" 2>/dev/null || shasum -a 256 "$archive") | cut -d' ' -f1)
[ "$got" = "$sha" ] || { rm -f "$archive"; echo "SHA-256 mismatch for $archive: $got" >&2; exit 1; }
rm -rf "$dest"; mkdir -p "$dest"
tar -xzf "$archive" -C "$dest" --strip-components=6 "onnx-${tag#v}/onnx/backend/test/data/node"
touch "$dest/.complete"
echo "fetched $(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l) node tests into $dest"
