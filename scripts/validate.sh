#!/usr/bin/env bash
# Host-side contract check of the image inputs (CI and local); needs bash, jq.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lock=$root/stack.lock.json
jq -e --arg dg "$(jq -r '.sources.deepgemm.commit[:8]' "$lock")" '. as {sources: $root_src} |
  def sha40: test("^[0-9a-f]{40}$");
  def sha64: test("^[0-9a-f]{64}$");
  def pr_url: test("^https://github.com/sgl-project/sglang/pull/[0-9]+$");
  [.patches[] | select(has("hold") | not)] as $active |
  [$active[].id] as $ids |
  .schema == 1 and .lineage == "dsv41-platform" and .platform == "linux/amd64" and
  (.candidate_tag | test("^v[0-9]+\\.[0-9]+\\.[0-9]+-rc\\.[1-9][0-9]*$")) and
  (.base.image | test("@sha256:[0-9a-f]{64}$")) and
  (.sources | keys == ["deepgemm", "flashinfer", "sglang", "triton"]) and
  all(.sources[]; (.commit | sha40) and (.tree | sha40)) and
  (.sources.sglang.effective_tree | sha40) and
  (.sources.deepgemm | has("effective_tree") | not) and
  ((.flashinfer_patches // []) as $fi |
    if ($fi | length) > 0 then (.sources.flashinfer.effective_tree | sha40) and
      all($fi | to_entries[]; .key as $i | .value |
        ((.kind == "branch" and (.branch | startswith("up/")) and
          (.upstream_pr == "TBD" or (.upstream_pr | test("^https://github.com/flashinfer-ai/flashinfer/pull/[0-9]+$")))) or
         (.kind == "pr" and .upstream_pr == ("https://github.com/flashinfer-ai/flashinfer/pull/" + (.pr | tostring)))) and
        (.head | sha40) and (.base | sha40) and (.sha256 | sha64) and
        (.applied == "clean" or .applied == "three-way") and
        (.path == ("patches/fi-" + (($i + 1) | tostring | if length < 2 then "0" + . else . end) + "-" + .id + ".patch")))
    else (.sources.flashinfer | has("effective_tree") | not) end) and
  ((.triton_patches // []) as $tr |
    if ($tr | length) > 0 then (.sources.triton.effective_tree | sha40) and
      all($tr | to_entries[]; .key as $i | .value |
        .kind == "cherry-pick" and (.head | sha40) and (.base == $root_src.triton.commit) and
        (.sha256 | sha64) and .applied == "clean" and
        (.upstream_pr | test("^https://github.com/triton-lang/triton/pull/[0-9]+$")) and
        (.path == ("patches/tr-" + (($i + 1) | tostring | if length < 2 then "0" + . else . end) + "-" + .id + ".patch")))
    else (.sources.triton | has("effective_tree") | not) end) and
  (.packaging.flashinfer_artifacts_sha256 | sha64) and
  (.packaging.sgl_deep_gemm | endswith("+git" + $dg)) and
  ([.patches[].id] | length == (unique | length)) and
  all(.patches[]; (.head | sha40) and ((has("diff_base") | not) or (.diff_base | sha40)) and
    (if .ab_candidate then .kind == "branch" and .upstream_pr == "none - A/B candidate"
     elif .kind == "pr" then .upstream_pr == ("https://github.com/sgl-project/sglang/pull/" + (.pr | tostring))
     elif .kind == "branch" then (.upstream_pr == "TBD" or (.upstream_pr | pr_url) or
         (.upstream_pr == "pending" and (.reason | type == "string" and length > 0))) and
       ((has("pr") | not) or .upstream_pr == ("https://github.com/sgl-project/sglang/pull/" + (.pr | tostring)))
     else false end) and
    (if .kind == "branch" then .branch | test("^(up|rebased)/") else true end)) and
  # A/B candidates follow every upstream patch, so dropping one for a
  # leave-one-out build leaves the upstream part of the series unchanged.
  ([$active[] | .ab_candidate == true] | . == sort) and
  all(.patches[] | select(has("hold")); (.hold | length > 0) and
    ([has("path"), has("base"), has("applied"), has("sha256"), has("cpu_tests")] | any | not)) and
  all($active | to_entries[]; .key as $i | .value |
    (.path == ("patches/" + (($i + 1) | tostring | if length < 2 then "0" + . else . end) + "-" + .id + ".patch")) and
    (.sha256 | sha64) and (.base | sha40) and
    (.applied == "clean" or .applied == "three-way") and
    ((has("resolution") | not) or .applied == "three-way") and
    (.cpu_tests | type == "array" and all(.[]; startswith("test/") and endswith(".py"))) and
    all(([.stacked_on // empty] + (.requires // []))[];
      . as $p | ($ids | index($p)) as $at | $at != null and $at < $i)) and
  (.dependency_gate.allowed_conflicts | length > 0) and
  all(.cpu_test_deselect[]; (.test | startswith("test/")) and (.reason | length > 0)) and
  all(.added_dependencies[]; (.sha256 | sha64) and (.wheel | endswith(".whl"))) and
  .hardware.architecture == "sm120"
' "$lock" >/dev/null || { echo "stack.lock.json violates the contract" >&2; exit 1; }
# The stack image pins the vendor base; the candidate builds FROM the stack image.
grep -Fqx "ARG BASE_IMAGE=$(jq -r .base.image "$lock")" "$root/Containerfile.stack"
grep -Fqx 'FROM ${STACK_IMAGE} AS runtime' "$root/Containerfile"
key_out=$(bash "$root/build/stack-key.sh" "$root")
[[ "$key_out" =~ ^[0-9a-f]{16}\ [0-9a-f]{16}$ ]] || { echo "stack-key.sh: $key_out" >&2; exit 1; }
rm -f "$root/stack.json"
tag=$(jq -r .candidate_tag "$lock")
grep -Fq "org.opencontainers.image.version=\"${tag#v}\"" "$root/Containerfile"
# patches/ holds exactly the active SGLang series and the FlashInfer series, each
# file hash-pinned.
diff <(jq -r '(.patches[] | select(has("hold") | not) | .path), ((.flashinfer_patches // [])[] | .path), ((.triton_patches // [])[] | .path)' "$lock" | sort) \
  <(cd "$root" && ls patches/*.patch | sort)
if command -v sha256sum >/dev/null; then sha=(sha256sum); else sha=(shasum -a 256); fi
while IFS=$'\t' read -r path expected; do
  [[ $("${sha[@]}" "$root/$path" | cut -d' ' -f1) == "$expected" ]] || {
    echo "patch hash mismatch: $path" >&2; exit 1; }
done < <(jq -r '(.patches[] | select(has("hold") | not)), (.flashinfer_patches // [])[], (.triton_patches // [])[] | [.path, .sha256] | @tsv' "$lock")
# The README table lists the active series in order, then the held entries.
diff <(jq -r '.patches[] | select(has("hold") | not) | .path | ltrimstr("patches/")' "$lock") \
  <(grep -oE '^\| [0-9]{2} \| `[0-9]{2}-[^`]+\.patch`' "$root/README.md" | grep -oE '[0-9]{2}-[^`]+\.patch')
diff <(jq -r '.patches[] | select(has("hold")) | .id' "$lock") \
  <(grep -oE '^\| held \| `[^`]+`' "$root/README.md" | grep -oE '`[^`]+`' | tr -d '`')
diff <(jq -r '(.flashinfer_patches // [])[] | .path | ltrimstr("patches/")' "$lock") \
  <(grep -oE '^\| fi-[0-9]{2} \| `fi-[0-9]{2}-[^`]+\.patch`' "$root/README.md" | grep -oE 'fi-[0-9]{2}-[^`]+\.patch')
diff <(jq -r '(.triton_patches // [])[] | .path | ltrimstr("patches/")' "$lock") \
  <(grep -oE '^\| tr-[0-9]{2} \| `tr-[0-9]{2}-[^`]+\.patch`' "$root/README.md" | grep -oE 'tr-[0-9]{2}-[^`]+\.patch')
bash -n "$root"/scripts/*.sh "$root"/build/*.sh
printf 'image contract valid: %s, %s patches, %s held, %s FlashInfer, %s Triton\n' "$tag" \
  "$(jq '[.patches[] | select(has("hold") | not)] | length' "$lock")" \
  "$(jq '[.patches[] | select(has("hold"))] | length' "$lock")" \
  "$(jq '(.flashinfer_patches // []) | length' "$lock")" \
  "$(jq '(.triton_patches // []) | length' "$lock")"
