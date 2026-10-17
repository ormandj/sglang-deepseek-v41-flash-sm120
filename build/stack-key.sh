#!/usr/bin/env bash
# Write stack.json, the part of stack.lock.json the stack stage is built from, and
# derive its keys. Every input of the stack stage is in the hash: the vendor base,
# the FlashInfer, DeepGEMM and Triton sources and series, their packaging fields,
# and the stack build files. A candidate whose stack inputs are unchanged reuses
# the stack image instead of rebuilding it, and the candidate build checks that
# it builds on the stack its lock describes.
#   key             -> tag of the stack image (stack-<key>)
#   flashinfer_key  -> FlashInfer JIT cache directory: base image, FlashInfer tree
#                      and target architectures, so objects compiled for one
#                      FlashInfer source are never reused for another
# usage: stack-key.sh [dir]   (dir holds Containerfile.stack and stack.lock.json; default:
# the repository root). The build scripts are hashed from this script's own
# directory under their in-image names, so the host and the image derive one key.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dir=${1:-$(cd "$here/.." && pwd)}
lock=$dir/stack.lock.json
if command -v sha256sum >/dev/null; then sha() { sha256sum | cut -d' ' -f1; }; else sha() { shasum -a 256 | cut -d' ' -f1; }; fi
files=$(printf '%s %s\n' Containerfile.stack "$(sha < "$dir/Containerfile.stack")"
  for f in build-base.sh prepare-sources.py; do
    printf '%s %s\n' "scripts/$f" "$(sha < "$here/$f")"; done)
arch=$(sed -n 's/.*FLASHINFER_CUDA_ARCH_LIST=\([^ ]*\).*/\1/p' "$dir/Containerfile.stack")
[[ -n "$arch" && "$arch" != *$'\n'* ]] || { echo "expected one FLASHINFER_CUDA_ARCH_LIST in Containerfile.stack" >&2; exit 1; }
inputs=$(jq -cS --arg files "$files" '{
  base: .base.image,
  sources: (.sources | {flashinfer, deepgemm, triton}),
  flashinfer_patches: (.flashinfer_patches // []),
  triton_patches: (.triton_patches // []),
  packaging: (.packaging | {flashinfer_python, flashinfer_artifacts_sha256, sgl_deep_gemm}),
  files: $files}' "$lock")
key=$(printf '%s' "$inputs" | sha | cut -c1-16)
fi_key=$(jq -cS --arg arch "$arch" '{base: .base.image, tree: .sources.flashinfer.effective_tree, arch: $arch}' "$lock" \
  | sha | cut -c1-16)
jq -S --arg key "$key" --arg fi_key "$fi_key" '. + {key: $key, flashinfer_key: $fi_key}' \
  <<<"$inputs" > "$dir/stack.json"
printf '%s %s\n' "$key" "$fi_key"
