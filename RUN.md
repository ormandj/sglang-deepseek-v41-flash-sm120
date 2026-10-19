# Running DeepSeek-V4.1-Flash

Download the checkpoint using [README.md](README.md), then start the TP4 launcher with the published image `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.1`:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export IMAGE=ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.1
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-flash-sm120-v1-tp4
./examples/serve-deepseek-v41-flash-tp4.sh
```

Use `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Use a fresh cache directory for each image version; the directory holds compiled kernels and the measured speculative-decoding cost table.

## Profile

- Tensor parallelism 4 with expert parallelism 2 (`--ep-size 2`) over PCIe, FlashInfer PCIe-IPC all-reduce for decode-sized reductions, NCCL for the rest. `NCCL_P2P_LEVEL=SYS` permits peer access across CPU roots; verify P2P on your platform before keeping that setting.
- FP8 E4M3 KV cache, a 1,048,576-token per-request limit, 16 running requests and 8,192-token prefill chunks. The device KV pool is shared across requests.
- Engram conditional-memory tables in host RAM, one row shard per rank (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`), about 189 GiB in total. The InstantTensor loader (`--load-format instanttensor`) loads the checkpoint.
- Decoder and encoder SWA bounded replay. Together with HiCache they let an evicted prefix be restored from host RAM instead of recomputed. HiCache is on with a host tier twice the device pool (`HICACHE_RATIO=2`); set `ENABLE_HICACHE=0` to disable it.
- DSpark speculative decoding with block size 5. At startup, `--speculative-dspark-sps-autotune auto` measures the verify-cost table on this machine and caches it in `CACHE_DIR`; `SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P=1.0` matches the recommended sampling. Compact ragged verify (`SGLANG_RAGGED_VERIFY_MODE=compact`) with sub-width tiers, and a single request's verify budget sized from a running average of its acceptance (`SGLANG_DSPARK_BUDGET_SURVIVAL_EMA=0.15`).
- Image input, up to 128 images per request.
- `SGLANG_MAX_NEW_TOKENS_LIMIT=393216` caps a request that does not set `max_tokens`.

## Clients

The model is served as `deepseek-v4.1-flash` through the OpenAI chat-completions API, with reasoning and tool-call parsing. The DeepSeek-V4.1-Flash-1017 model card recommends `temperature=1.0` and `top_p=1.0`. Reasoning shares the `max_tokens` budget: leave enough room for it, or a small budget can end before any answer text. Set `API_KEY` before starting the launcher to require `Authorization: Bearer <key>`.

## Build from source

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
read -r stack_key flashinfer_key < <(bash build/stack-key.sh .)
docker build -f Containerfile.stack --target stack \
  --build-arg FLASHINFER_KEY="$flashinfer_key" \
  -t "sglang-deepseek-v41-flash-sm120-stack:stack-$stack_key" .
docker build -f Containerfile --target runtime \
  --build-arg STACK_IMAGE="sglang-deepseek-v41-flash-sm120-stack:stack-$stack_key" \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-deepseek-v41-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-deepseek-v41-flash-sm120:v0.3.0-rc.1 .
rm -f stack.json
```

`build/stack-key.sh` writes `stack.json`, the stack image's inputs, and prints the stack and FlashInfer keys. The stack image (FlashInfer, DeepGEMM, Triton) only changes when those inputs do, so later candidates can reuse it. The build scripts use twelve compiler jobs; on a builder with less memory, prefix `MAX_JOBS=3` to the build-script commands in copies of the two Containerfiles, as the CI workflow does, and keep the tracked `Containerfile.stack` in the build context, because the image hashes it. [BUILDING.md](BUILDING.md) describes the CI workflows and the steps that publish a Release.
