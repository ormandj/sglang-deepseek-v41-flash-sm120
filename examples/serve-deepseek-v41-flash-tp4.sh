#!/usr/bin/env bash
# TP4 profile for deepseek-ai/DeepSeek-V4.1-Flash on four 96 GB SM120 GPUs over PCIe.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the local deepseek-ai/DeepSeek-V4.1-Flash checkpoint}"
: "${CACHE_DIR:?set CACHE_DIR to a version-specific persistent cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.2}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
MEM_FRACTION=${MEM_FRACTION:-0.85}
# Clients that omit max_tokens are capped here; reasoning counts against it.
MAX_NEW_TOKENS_LIMIT=${MAX_NEW_TOKENS_LIMIT:-393216}

# HiCache keeps evicted prefixes in host RAM (HICACHE_RATIO times the device KV
# pool) and restores them instead of recomputing. Set ENABLE_HICACHE=0 to disable.
ENABLE_HICACHE=${ENABLE_HICACHE:-1}
HICACHE_RATIO=${HICACHE_RATIO:-2}
HICACHE_ARGS=()
if [[ "$ENABLE_HICACHE" == 1 ]]; then
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO")
fi
API_KEY_ARGS=()
if [[ -n "${API_KEY:-}" ]]; then
  API_KEY_ARGS=(--api-key "$API_KEY")
fi
CONTAINER_NAME=${CONTAINER_NAME:-deepseek-v41-flash-sm120-tp4}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR does not contain config.json: $MODEL_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
  exit 2
fi
if [[ "$TP_SIZE" != 4 ]]; then
  echo "This launcher is scoped to TP_SIZE=4" >&2
  exit 2
fi
for value in MAX_RUNNING_REQUESTS CONTEXT_LENGTH MAX_NEW_TOKENS_LIMIT HICACHE_RATIO; do
  current=${!value}
  if ! [[ "$current" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value must be a positive integer" >&2
    exit 2
  fi
done
if (( MAX_RUNNING_REQUESTS > 16 )); then
  echo "This TP4 profile supports at most 16 running requests" >&2
  exit 2
fi
if (( CONTEXT_LENGTH > 1048576 )); then
  echo "CONTEXT_LENGTH exceeds the checkpoint's 1048576-token limit" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)
container_model_path=/models/deepseek-v41-flash

# TP4 with expert parallelism 2 over PCIe. The Engram tables (about 189 GiB) live
# in host RAM, one row shard per rank. SYS permits NCCL peer access across CPU
# roots; the driver and platform must provide working GPU P2P.
exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint sglang \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish "${PORT}:8000" \
  --volume "${model_dir}:${container_model_path}:ro" \
  --volume "${cache_dir}:/root/.cache" \
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P=1.0 \
  --env SGLANG_RAGGED_VERIFY_SUB_WIDTH_TIERS=1 \
  --env SGLANG_DSPARK_BUDGET_SURVIVAL_EMA=0.15 \
  --env SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1 \
  --env SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
  --env SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=41943040 \
  --env SGLANG_SPEC_TP_SYNC=rng \
  --env SGLANG_RAGGED_VERIFY_MODE=compact \
  --env SGLANG_ENABLE_BF16_SKINNY_LM_HEAD=1 \
  --env SGLANG_NUMA_BIND_V2=0 \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env SGLANG_MAX_NEW_TOKENS_LIMIT="$MAX_NEW_TOKENS_LIMIT" \
  --env MAX_JOBS=4 \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  "$IMAGE" \
  serve \
  --model-path "$container_model_path" \
  --served-model-name deepseek-v4.1-flash \
  ${API_KEY_ARGS[@]+"${API_KEY_ARGS[@]}"} \
  --tp-size "$TP_SIZE" \
  --ep-size 2 \
  --moe-a2a-backend none \
  --enable-decoder-swa-bounded-replay \
  --enable-encoder-swa-bounded-replay \
  ${HICACHE_ARGS[@]+"${HICACHE_ARGS[@]}"} \
  --enable-multimodal \
  --warmups serving_coverage \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --min-free-slots-delay 1 \
  --cuda-graph-max-bs-decode 16 \
  --load-format instanttensor \
  --kv-cache-dtype fp8_e4m3 \
  --disable-custom-all-reduce \
  --fp8-gemm-backend flashinfer_cutlass \
  --enable-deepseek-v4-fp4-indexer \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --speculative-dspark-align-verify-tokens-to-graph-tier \
  --speculative-dspark-sps-autotune auto \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --limit-mm-data-per-request '{"image":128}' \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 8000
