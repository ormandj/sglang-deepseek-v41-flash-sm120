#!/usr/bin/env bash
# Candidate image: install SGLang from its locked source and series on top of the
# stack image (FlashInfer, DeepGEMM and Triton from build-base.sh), then run the
# installed-image gates over the whole stack.
set -euo pipefail
root=/opt/dsv41-platform
py=/opt/sglang/bin/python
export MAX_JOBS=${MAX_JOBS:-12} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-${MAX_JOBS:-12}}
# Expert-parallel MoE all-to-all backends are unused (--moe-a2a-backend none).
export BUILD_NIXL_EP=0 BUILD_NCCL_EP=0
pyrun() { uv run --no-project --python "$py" python "$@"; }
lock() { pyrun -c "import json; print('\n'.join(json.load(open('$root/stack.lock.json'))$1))"; }

# The stack this candidate builds on must be the one its lock describes.
want=$(cd "$root" && bash scripts/stack-key.sh "$root" >/dev/null && jq -r .key stack.json)
have=$(jq -r .key "$root/provenance/stack/stack.json")
[[ "$want" == "$have" ]] || { echo "stack $have does not match the lock ($want)" >&2; exit 1; }

uv pip freeze --python "$py" > "$root/provenance/base-packages.txt"
pyrun "$root/scripts/prepare-sources.py" "$root" stack.lock.json sglang
src=$root/sources
rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' "$src/sglang/rust/rust-toolchain.toml")" --profile minimal

# The editable install owns SGLang's Python source and its Rust extensions.
mapfile -t extras < <(lock "['packaging']['sglang_install_extras']")
uv pip install --python "$py" --no-build-isolation -e "$src/sglang/python" "${extras[@]}"
mapfile -t remove < <(lock "['packaging']['remove']")
uv pip uninstall --python "$py" "${remove[@]}"
mapfile -t no_deps < <(lock "['packaging']['install_no_deps']")
uv pip install --python "$py" --no-deps "${no_deps[@]}"
pyrun "$root/scripts/install-added-dependencies.py"
uv pip freeze --python "$py" > "$root/provenance/installed-packages.txt"

# Run every gate so one build reports all failures, then fail on any.
failed=()
pyrun "$root/scripts/check-dependencies.py" || failed+=(dependencies)
CUDA_VISIBLE_DEVICES='' pyrun "$root/scripts/verify-installed.py" || failed+=(identity)
pyrun "$root/scripts/run-cpu-tests.py" || failed+=(cpu-tests)
if (( ${#failed[@]} )); then echo "failed gates: ${failed[*]}" >&2; exit 1; fi
# Keep one copy of every artifact: the installed packages. Cargo/setuptools build
# output is build-time only (SGLANG_RUST_BUILD_MODE=never at runtime).
{ du -sh "$src/sglang/rust/target" "$src/sglang/python/build" 2>/dev/null || true; } \
  | tee "$root/provenance/removed-build-output.txt"
uv cache clean
rm -rf /root/.cargo/registry /root/.cargo/git "$src/sglang/rust/target" "$src/sglang/python/build"
