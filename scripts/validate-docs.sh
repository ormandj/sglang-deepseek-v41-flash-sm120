#!/usr/bin/env bash
# Documentation contract: the README, RUN.md and the launcher describe the
# published release and the locked model. Needs bash, jq and grep.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
published_tag=$(jq -er '.published.tag' "$repo/release.json")
published_cache_schema=$(jq -er '.published.cache_schema' "$repo/release.json")
model=$(jq -er '.model.repository' "$repo/stack.lock.json")
revision=$(jq -er '.model.revision' "$repo/stack.lock.json")
image="ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:${published_tag}"
launcher="$repo/examples/serve-deepseek-v41-flash-tp4.sh"

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} missing: $expected" >&2
    exit 1
  }
}

require_text "$repo/README.md" "The current published stable image is \`${published_tag}\`"
require_text "$repo/README.md" "| Image | \`${image}\` |"
require_text "$repo/README.md" "\`${model}\` at revision \`${revision}\`"
require_text "$repo/README.md" "/srv/cache/sglang-deepseek-v41-flash-sm120-${published_cache_schema}-tp4"
require_text "$repo/RUN.md" "\`${image}\`"
require_text "$repo/RUN.md" "export IMAGE=${image}"
require_text "$repo/RUN.md" "/srv/cache/sglang-deepseek-v41-flash-sm120-${published_cache_schema}-tp4"
require_text "$repo/RUN.md" "revision \`${revision}\`"
grep -Fx "IMAGE=\${IMAGE:-${image}}" "$launcher" >/dev/null || {
  echo "launcher default image is not ${image}" >&2; exit 1;
}
for value in '--tp "$TP_SIZE"' '--ep-size 2' '--kv-cache-dtype fp8_e4m3' '--speculative-algorithm DSPARK' \
  '--speculative-dspark-block-size 5' '--speculative-dspark-sps-autotune auto' '--load-format instanttensor' \
  '--enable-decoder-swa-bounded-replay' '--enable-encoder-swa-bounded-replay' '--warmups serving_coverage'; do
  require_text "$launcher" "$value"
done
require_text "$repo/CHANGELOG.md" "## ${published_tag} (stable;"

# Public prose stays free of private infrastructure and em dashes.
if grep -nE 'home\.[a-z0-9.-]+\.(com|net|org)|/Users/|~/git/|homelab|quasar' \
    "$repo"/*.md "$repo"/examples/*.sh "$repo"/stack.lock.json; then
  echo "private references in public files" >&2; exit 1
fi
if grep -n $'\xe2\x80\x94' "$repo"/*.md; then
  echo "em dash in public prose" >&2; exit 1
fi

echo "documentation contract valid: ${published_tag}, cache ${published_cache_schema}, ${model}@${revision:0:12}"
