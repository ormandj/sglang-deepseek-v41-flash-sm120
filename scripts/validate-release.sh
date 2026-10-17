#!/usr/bin/env bash
# Release contract: release.json, stack.lock.json, the Containerfile and the
# patch series describe one candidate. Needs bash and jq.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:-"$repo/release.json"}
lock=${2:-"$repo/stack.lock.json"}

for tool in jq grep sed; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
jq -e . "$manifest" "$lock" >/dev/null

version=$(jq -er '.version' "$manifest")
candidate=$(jq -er '.candidate' "$manifest")
candidate_tag=$(jq -er '.candidate_tag' "$manifest")
stable_tag=$(jq -er '.stable_tag' "$manifest")
cache_schema=$(jq -er '.cache_schema' "$manifest")
product=$(jq -er '.product' "$manifest")

[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]
[[ "$candidate" =~ ^[1-9][0-9]*$ ]]
[[ "$candidate_tag" == "v${version}-rc.${candidate}" ]]
[[ "$stable_tag" == "v${version}" ]]
[[ "$cache_schema" =~ ^v[1-9][0-9]*$ ]]
[[ "$product" == "sglang-deepseek-v41-flash-sm120" ]]
[[ "$(jq -er '.platform' "$manifest")" == "linux/amd64" ]]

# The lock and the image label name the same candidate.
[[ "$(jq -er '.candidate_tag' "$lock")" == "$candidate_tag" ]] || {
  echo "release.json and stack.lock.json disagree on the candidate tag" >&2; exit 1;
}
grep -Fq "org.opencontainers.image.version=\"${candidate_tag#v}\"" "$repo/Containerfile" || {
  echo "Containerfile image version label does not match ${candidate_tag}" >&2; exit 1;
}
[[ "$(jq -er '.hardware.tensor_parallel' "$lock")" == 4 ]]
[[ "$(jq -er '.hardware.architecture' "$lock")" == sm120 ]]

# Lock contract, patch hashes and README patch tables.
"$repo/scripts/validate.sh"

published_tag=$(jq -er '.published.tag' "$manifest")
published_cache_schema=$(jq -er '.published.cache_schema' "$manifest")
[[ "$published_tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || {
  echo "invalid published stable tag" >&2; exit 1;
}
[[ "$published_cache_schema" =~ ^v[1-9][0-9]*$ ]] || {
  echo "invalid published cache schema" >&2; exit 1;
}
[[ "$published_tag" != "$stable_tag" || "$published_cache_schema" == "$cache_schema" ]] || {
  echo "published stable cache schema mismatch" >&2; exit 1;
}

echo "release contract valid: ${candidate_tag} -> ${stable_tag}, cache ${cache_schema}"
