# Fetches the ONNX backend node tests (onnx/onnx, Apache-2.0) into artifacts/cache/onnx-node-<tag>/.
# v1.22.0 is the last release that ships the generated node test data in the source tree.
# The archive is checked by SHA-256 and only onnx/backend/test/data/node/** is extracted.
# Usage: pwsh tools/fetch-onnx-tests.ps1
$ErrorActionPreference = 'Stop'
$tag = 'v1.22.0'
$sha = '70BB8B25CF31EA9B1D9F94BAACFDC8C4FA27A760F9A10F5D93881BC9EEDE5FBC'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "artifacts/cache/onnx-node-$tag"
if (Test-Path (Join-Path $dest '.complete')) { Write-Host "already fetched: $dest"; exit 0 }
$dl = Join-Path $root 'artifacts/cache/dl'
New-Item -ItemType Directory -Force $dl | Out-Null
$archive = Join-Path $dl "onnx-$tag.tar.gz"
if (-not (Test-Path $archive)) {
    curl -sSL --fail -o $archive "https://github.com/onnx/onnx/archive/refs/tags/$tag.tar.gz"
    if ($LASTEXITCODE) { throw "download failed" }
}
$got = (Get-FileHash $archive -Algorithm SHA256).Hash
if ($got -ne $sha) { Remove-Item $archive; throw "SHA-256 mismatch for $archive : $got" }
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force $dest | Out-Null
$prefix = "onnx-$($tag.TrimStart('v'))/onnx/backend/test/data/node"
tar -xzf $archive -C $dest --strip-components=6 $prefix
if ($LASTEXITCODE) { throw "extract failed" }
New-Item -ItemType File (Join-Path $dest '.complete') | Out-Null
Write-Host "fetched $((Get-ChildItem $dest -Directory).Count) node tests into $dest"
