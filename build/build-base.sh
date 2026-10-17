#!/usr/bin/env bash
# Stack image: build FlashInfer (with its cubin bundle), DeepGEMM and Triton from
# the sources in stack.json into the vendor /opt/sglang environment. Candidates
# install SGLang on top (build-stack.sh), so a candidate that changes only the
# SGLang series does not rebuild these.
set -euo pipefail
root=/opt/dsv41-platform
py=/opt/sglang/bin/python
export MAX_JOBS=${MAX_JOBS:-12}
pyrun() { uv run --no-project --python "$py" python "$@"; }
stack() { jq -er "$1" "$root/stack.json"; }

mkdir -p "$root/provenance/stack" "$root/wheels"
cp "$root/stack.json" "$root/provenance/stack/stack.json"
uv pip freeze --python "$py" > "$root/provenance/stack/base-packages.txt"
uv pip install --python "$py" 'setuptools>=77' setuptools-rust setuptools-scm wheel build
pyrun "$root/scripts/prepare-sources.py" "$root" stack.json flashinfer deepgemm triton
src=$root/sources

# Remove the vendor's published FlashInfer/DeepGEMM before installing source builds.
uv pip uninstall --python "$py" flashinfer-python flashinfer-cubin flashinfer-jit-cache sgl-deep-gemm deep-gemm
# Files written after the vendor install survive uninstall; a leftover
# flashinfer_jit_cache directory would import as an empty namespace package
# and fail FlashInfer's AOT-provider version check.
site=$(pyrun -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -rf "$site"/flashinfer "$site"/flashinfer_cubin "$site"/flashinfer_jit_cache "$site"/deep_gemm
# nvidia-cutlass-dsl follows SGLang's pin, which the candidate's SGLang install applies.
(cd "$src/flashinfer" && uv pip install --python "$py" --no-build-isolation '.[cu13]')
# No published wheel matches FlashInfer main: build the cubin package from the
# same commit. Its downloader verifies the pinned manifest and every file hash;
# runtime downloads stay disabled (FLASHINFER_NO_DOWNLOAD). The artifact CDN
# intermittently answers 403; a rerun reuses every verified download.
for attempt in 1 2 3; do
  (cd "$src/flashinfer" && FLASHINFER_CUBIN_DIR="$PWD/flashinfer-cubin/flashinfer_cubin/cubins" \
    pyrun -m build --wheel --no-isolation --outdir "$root/wheels" ./flashinfer-cubin) && break
  (( attempt < 3 )) || exit 1
  sleep 120
done
# Upstream's integration builder emits the sgl-deep-gemm wheel (import deep_gemm).
(cd "$src/deepgemm" && uv run --no-project --python "$py" bash build_sgl_deep_gemm.sh)
cp "$src"/deepgemm/dist/*.whl "$root/wheels/"
uv pip install --python "$py" --no-deps "$root"/wheels/*.whl
sha256sum "$root"/wheels/*.whl > "$root/provenance/stack/wheels.sha256"
# Triton: the pinned release branch plus the triton_patches backports replaces the
# vendor's published wheel of the same version (LLVM is fetched by its setup).
(cd "$src/triton" && TRITON_HOME="$root/triton-home" TRITON_BUILD_WITH_CLANG_LLD=0 \
  TRITON_BUILD_PROTON=OFF TRITON_BUILD_UT=OFF \
  "$py" -m pip wheel --no-deps --no-build-isolation -w "$root/triton-wheel" .)
uv pip install --python "$py" --no-deps --reinstall "$root"/triton-wheel/triton-*.whl
sha256sum "$root"/triton-wheel/*.whl >> "$root/provenance/stack/wheels.sha256"

# FlashInfer compiles its JIT modules at runtime, and ninja rebuilds an object
# whose sources are newer than it. Give the installed sources one fixed time, the
# FlashInfer commit time, so every image built from this stack (and any rebuild of
# it) presents the same sources and reuses the objects already compiled into its
# keyed cache directory (FLASHINFER_WORKSPACE_BASE) instead of recompiling.
fi_time=$(git -C "$src/flashinfer" log -1 --format=%ct HEAD)
find "$site/flashinfer" "$site/flashinfer_cubin" -exec touch -h -d "@$fi_time" {} +

CUDA_VISIBLE_DEVICES='' pyrun - <<'PY'
import importlib, json
from importlib import metadata
stack = json.load(open('/opt/dsv41-platform/stack.json'))
fi = stack['sources']['flashinfer']['commit']
assert importlib.import_module('flashinfer').__git_commit__ == fi
assert importlib.import_module('flashinfer_cubin').__git_version__ == fi
assert metadata.version('flashinfer-python') == stack['packaging']['flashinfer_python']
assert metadata.version('sgl-deep-gemm') == stack['packaging']['sgl_deep_gemm']
url = json.loads(metadata.distribution('triton').read_text('direct_url.json'))['url']
assert url.startswith('file://') and '/triton-wheel/' in url, url
print('stack identity ok:', stack['key'])
PY
uv pip freeze --python "$py" > "$root/provenance/stack/installed-packages.txt"
{ du -sh "$root/wheels" "$src"/*/ "$src/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins" 2>/dev/null || true; } \
  | tee "$root/provenance/stack/removed-build-output.txt"
uv cache clean
rm -rf "$root/wheels" "$root/triton-wheel" "$root/triton-home" "$src/triton/build" \
  "$src/deepgemm/build" "$src/deepgemm/dist" "$src/flashinfer/build" \
  "$src/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins"
