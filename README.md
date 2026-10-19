# SGLang for DeepSeek-V4.1-Flash on SM120

An SGLang image, built from pinned sources, for [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four RTX PRO 6000 Blackwell (SM120) GPUs over PCIe: tensor parallelism 4 with expert parallelism 2, FP8 KV cache, DSpark speculative decoding, a 1,048,576-token context, image input and a host-RAM prefix cache.

The current published stable image is `v0.4.3`.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.4.3` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| GPUs | four 96 GB RTX PRO 6000 Blackwell (SM120) with working PCIe peer-to-peer; no NVLink needed |
| Host RAM | about 250 GiB in use while serving (189 GiB of Engram tables plus the HiCache host tier); 384 GB or more recommended |
| Disk | about 480 GiB for the checkpoint, plus a persistent kernel cache |

## Quick start

Download the checkpoint at the pinned revision:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /srv/models/DeepSeek-V4.1-Flash-1017
```

Start the server with the TP4 launcher:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-flash-sm120-v1-tp4
./examples/serve-deepseek-v41-flash-tp4.sh
```

The first start compiles kernels and measures the speculative-decoding cost table into `CACHE_DIR`; later starts reuse it. Wait for `The server is fired up and ready to roll!`. The server speaks the OpenAI chat-completions API on port 8000 as `deepseek-v4.1-flash`. Set `API_KEY` to require a bearer token. [RUN.md](RUN.md) covers the settings, client parameters and limits.

## Measured

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with the privately built 0.4.2 image (0.4.3 changes only the startup warmup), at the model card's sampling (temperature 1.0, `top_p` 1.0), three repetitions per cell:

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 202 | 322 | 497 | 674 | 1,051 |

- Cold prefill: about 14.1K, 14.9K and 14.7K prompt tok/s at 8K, 32K and 128K tokens.
- Device KV pool: 2,931,712 tokens across requests; one request may use up to 1,048,576.
- The DeepSeek-V4.1-Flash-1017 model writes shorter answers than the earlier V4.1-Flash checkpoint, so answers finish sooner than its output rate alone suggests.

These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance. Details and the benchmark scope are in [CHANGELOG.md](CHANGELOG.md).

## Source and patches

The image is built from four upstream commits, recorded in [`stack.lock.json`](stack.lock.json) with their trees and every patch's hash: SGLang `main`, FlashInfer `main`, sgl-project DeepGEMM `release/v0.2.3` and Triton `release/3.8.x`, on the `lmsysorg/sglang` CUDA 13 nightly, which supplies CUDA, PyTorch and runtime libraries only. SGLang, FlashInfer, DeepGEMM and Triton are rebuilt from source. [`scripts/verify-patches.sh`](scripts/verify-patches.sh) fetches the official commits, applies the series and checks every resulting tree.

SGLang series, applied in order to the pinned SGLang commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| 01 | `01-instanttensor-host-resident.patch` | [sgl-project/sglang#42919](https://github.com/sgl-project/sglang/pull/42919) | Host-resident weights (Engram tables) stay off the InstantTensor GPU stream |
| 02 | `02-engram-positioned-reads.patch` | [sgl-project/sglang#42920](https://github.com/sgl-project/sglang/pull/42920), on 01 | Parallel positioned reads fill each rank's host Engram shard |
| 03 | `03-sm12x-candidate-indexer.patch` | [sgl-project/sglang#42371](https://github.com/sgl-project/sglang/pull/42371) | V4.1 candidate indexer on SM12x through the paged decode path |
| 04 | `04-dsv41-ragged-verify-engram.patch` | [sgl-project/sglang#42718](https://github.com/sgl-project/sglang/pull/42718) | Packed compact-verify tokens mapped to their requests (replaces #40567) |
| 05 | `05-dspark-forced-budget-verify-all.patch` | [sgl-project/sglang#42711](https://github.com/sgl-project/sglang/pull/42711) | Correct SPS cost tables when profiling without a table |
| 06 | `06-flash-mla-sm120-split-buffers.patch` | [sgl-project/sglang#42714](https://github.com/sgl-project/sglang/pull/42714) | SM120 FlashMLA page-split buffers allocated once per size (captured graphs never write freed memory) |
| 07 | `07-dspark-confidence-relay-sync.patch` | [sgl-project/sglang#42717](https://github.com/sgl-project/sglang/pull/42717) | Every TP rank resolves the same relayed DSpark confidence |
| 08 | `08-forced-uniform-capture-token-key.patch` | [sgl-project/sglang#42713](https://github.com/sgl-project/sglang/pull/42713) | Forced-uniform compact captures keyed by token count, with a full-width layout |
| 09 | `09-dspark-sts-accept-labels.patch` | [sgl-project/sglang#42712](https://github.com/sgl-project/sglang/pull/42712) | STS calibration labels from the actual acceptance, unverified positions masked |
| 10 | `10-dsv4-fp4-indexer-exp2-free.patch` | [sgl-project/sglang#42861](https://github.com/sgl-project/sglang/pull/42861) | FP4 index keys decoded without `exp2` (bit-identical) |
| 11 | `11-sm120-mxfp8-skinny.patch` | [sgl-project/sglang#42865](https://github.com/sgl-project/sglang/pull/42865) | Split-K MXFP8 GEMM for tuned decode shapes at up to 16 rows |
| 12 | `12-mhc-mix-vec-reduce.patch` | [sgl-project/sglang#42864](https://github.com/sgl-project/sglang/pull/42864), ported | Vectorized mHC reduce; 160 K slices at SM120 decode |
| 13 | `13-mhc-prefill-stats-overlap.patch` | [sgl-project/sglang#42985](https://github.com/sgl-project/sglang/pull/42985), ported | Wide prefill overlaps the mHC mix stats with the TP all-reduce again (#42245 removed it; measured prefill regression) |
| 14 | `14-pcie-ipc-gather.patch` | [sgl-project/sglang#42863](https://github.com/sgl-project/sglang/pull/42863) | TP4 logits and DSpark vocab gathers on FlashInfer PCIe-IPC |
| 15 | `15-dsv41-request-window-history-check.patch` | [sgl-project/sglang#42921](https://github.com/sgl-project/sglang/pull/42921), rebased | Request-window history checked once per layout with an async device assert (no prefill host sync) |
| 16 | `16-bf16-skinny-lm-head.patch` | [sgl-project/sglang#42867](https://github.com/sgl-project/sglang/pull/42867) | Opt-in weight-streaming BF16 LM head GEMM at <= 16 rows (`SGLANG_ENABLE_BF16_SKINNY_LM_HEAD`) |
| 17 | `17-serving-coverage-warmup.patch` | [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541) | `--warmups serving_coverage`: real requests through every serving path, including decodes at every power-of-two batch size up to the running-request limit |
| 18 | `18-dspark-sps-autotune.patch` | not yet upstream | `--speculative-dspark-sps-autotune off\|auto\|force`: fit the DSpark SPS cost table at startup from real decode steps per batch size and verify tier plus a per-verify-token context term, cached per configuration |
| 19 | `19-ragged-sub-width-tiers.patch` | [sgl-project/sglang#43011](https://github.com/sgl-project/sglang/pull/43011), rebased (closed upstream) | Opt-in ragged-verify tiers below one request's width (`SGLANG_RAGGED_VERIFY_SUB_WIDTH_TIERS`) |
| 20 | `20-dspark-verify-pad-last-row.patch` | [sgl-project/sglang#43374](https://github.com/sgl-project/sglang/pull/43374) | No top-p renormalization on compact-verify padding rows |
| 21 | `21-compact-verify-committed-seq-lens.patch` | [sgl-project/sglang#43225](https://github.com/sgl-project/sglang/pull/43225) | Compact verify passes committed CPU lengths to self-adding backends |
| 22 | `22-dsv41-candidate-scorer.patch` | [sgl-project/sglang#43401](https://github.com/sgl-project/sglang/pull/43401) | Paged decode consumer layers score only their candidate blocks |
| 23 | `23-dsv41-wo-a-fp8-weight.patch` | [sgl-project/sglang#43008](https://github.com/sgl-project/sglang/pull/43008) | SM120 decode/verify WO-A reads an exact FP8 weight copy (`SGLANG_DSV41_WO_A_FP8_COPY`, default on) |
| 24 | `24-encoder-swa-replay-multimodal.patch` | [sgl-project/sglang#43241](https://github.com/sgl-project/sglang/pull/43241) | Under encoder SWA bounded replay, the server warms up the image path on CUDA, where replay carries image inputs |
| 25 | `25-dsv41-indexer-prefill-tile-budget.patch` | [sgl-project/sglang#43376](https://github.com/sgl-project/sglang/pull/43376) | Prefill indexer score tiles sized from free memory, scratch counted, no padded copy (OOM fix) |
| 26 | `26-dspark-smoothed-budget-sizing.patch` | not yet upstream | Host verify-budget sizing from a per-request running average of lagged survival (opt-in) |
| 27 | `27-dspark-read-accept-no-alias.patch` | [sgl-project/sglang#42812](https://github.com/sgl-project/sglang/pull/42812), amended for #43790 | DSpark folded accept results are copied before the next verify replay, so TP ranks commit the same tokens under overlap scheduling with HiCache; the in-graph accept still writes the static buffers |
| 28 | `28-chat-usage-completion-tokens-details.patch` | [sgl-project/sglang#45131](https://github.com/sgl-project/sglang/pull/45131) | Usage also reports reasoning tokens at `usage.completion_tokens_details.reasoning_tokens`, where OpenAI clients and gateways read them |
| 29 | `29-openai-error-envelope.patch` | [sgl-project/sglang#45130](https://github.com/sgl-project/sglang/pull/45130) | OpenAI routes return errors in OpenAI's error format |
| 30 | `30-openai-abort-finish-reason.patch` | [sgl-project/sglang#45132](https://github.com/sgl-project/sglang/pull/45132) | Aborted requests are reported as an error or a stop, never `finish_reason: "abort"` |
| 31 | `31-openai-stream-exception-frame.patch` | [sgl-project/sglang#45133](https://github.com/sgl-project/sglang/pull/45133) | Failed chat and completion streams end with an error frame and `[DONE]` |
| 32 | `32-dsv41-effort-minimal-medium.patch` | [sgl-project/sglang#45134](https://github.com/sgl-project/sglang/pull/45134) | DeepSeek-V4.1 supports `reasoning_effort` `minimal` and `medium` |

FlashInfer series, applied to the pinned FlashInfer commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| fi-01 | `fi-01-sm120-mxfp8-default-tactic.patch` | [flashinfer-ai/flashinfer#6162](https://github.com/flashinfer-ai/flashinfer/pull/6162) | SM120 MXFP8 GEMM calls without a tuned tactic use the swap-AB 128x32 tile |

Triton series, applied to the pinned Triton commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| tr-01 | `tr-01-release-3.8-dot-operand-remat.patch` | [triton-lang/triton#11940](https://github.com/triton-lang/triton/pull/11940), backported to `release/3.8.x` | Restore dot-operand rematerialization reuse: Triton 3.8.0 (issue #11526) sends tf32x3 dot operands through extra shared-memory round trips |

Patches marked "not yet upstream" are under preparation for upstream review. [BUILDING.md](BUILDING.md) describes the build and release workflows.

## License

The repository's own files are licensed under the [Apache License 2.0](LICENSE). The patches modify Apache-2.0 upstream projects and keep their licenses; the DeepSeek-V4.1-Flash weights are distributed by DeepSeek under the MIT License. See [NOTICE.md](NOTICE.md).
