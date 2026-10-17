# SGLang for DeepSeek-V4.1-Flash on SM120

An SGLang image, built from pinned sources, for [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on four RTX PRO 6000 Blackwell (SM120) GPUs over PCIe: tensor parallelism 4 with expert parallelism 2, FP8 KV cache, DSpark speculative decoding, a 1,048,576-token context, image input and a host-RAM prefix cache.

The current published stable image is `v0.2.1`.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-deepseek-v41-flash-sm120:v0.2.1` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` at revision `dba1be0a40aa45a94ad051997016db3960a90277` |
| GPUs | four 96 GB RTX PRO 6000 Blackwell (SM120) with working PCIe peer-to-peer; no NVLink needed |
| Host RAM | about 250 GiB in use while serving (189 GiB of Engram tables plus the HiCache host tier); 384 GB or more recommended |
| Disk | about 480 GiB for the checkpoint, plus a persistent kernel cache |

## Quick start

Download the checkpoint at the pinned revision:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --local-dir /srv/models/DeepSeek-V4.1-Flash
```

Start the server with the TP4 launcher:

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash
export CACHE_DIR=/srv/cache/sglang-deepseek-v41-flash-sm120-v1-tp4
./examples/serve-deepseek-v41-flash-tp4.sh
```

The first start compiles kernels and measures the speculative-decoding cost table into `CACHE_DIR`; later starts reuse it. Wait for `The server is fired up and ready to roll!`. The server speaks the OpenAI chat-completions API on port 8000 as `deepseek-v4.1-flash`. Set `API_KEY` to require a bearer token. [RUN.md](RUN.md) covers the settings, client parameters and limits.

## Measured

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with a separately built image from the same pinned sources and patches, three repetitions per cell:

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 220 | 326 | 515 | 697 | 1,009 |

- Cold prefill: about 10.1K, 12.7K and 12.9K prompt tok/s at 8K, 32K and 128K tokens.
- Device KV pool: 2,078,208 tokens across requests; one request may use up to 1,048,576.
- Ten minutes at C8: 309 requests, no errors or runaway outputs.

These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance. Details and the benchmark scope are in [CHANGELOG.md](CHANGELOG.md).

## Source and patches

The image is built from four upstream commits, recorded in [`stack.lock.json`](stack.lock.json) with their trees and every patch's hash: SGLang `main`, FlashInfer `main`, sgl-project DeepGEMM `dev` and Triton `release/3.8.x`, on the `lmsysorg/sglang` CUDA 13 nightly, which supplies CUDA, PyTorch and runtime libraries only. SGLang, FlashInfer, DeepGEMM and Triton are rebuilt from source. [`scripts/verify-patches.sh`](scripts/verify-patches.sh) fetches the official commits, applies the series and checks every resulting tree.

SGLang series, applied in order to the pinned SGLang commit:

| # | Patch | Origin | Change |
|---|---|---|---|
| 01 | `01-instanttensor-loader.patch` | [sgl-project/sglang#40453](https://github.com/sgl-project/sglang/pull/40453) | InstantTensor loader (`--load-format instanttensor`) |
| 02 | `02-instanttensor-host-resident.patch` | [sgl-project/sglang#42919](https://github.com/sgl-project/sglang/pull/42919) | Host-resident weights (Engram tables) stay off the InstantTensor GPU stream |
| 03 | `03-engram-positioned-reads.patch` | [sgl-project/sglang#42920](https://github.com/sgl-project/sglang/pull/42920), on 02 | Parallel positioned reads fill each rank's host Engram shard |
| 04 | `04-decoder-swa-window.patch` | [sgl-project/sglang#41303](https://github.com/sgl-project/sglang/pull/41303), rebased | Complete SWA window on prefix hits and short final chunks under decoder bounded replay |
| 05 | `05-hicache-swa-request-window-gate.patch` | [sgl-project/sglang#42716](https://github.com/sgl-project/sglang/pull/42716) | HiCache host hits under encoder SWA replay with DSpark |
| 06 | `06-bounded-replay-prompt-logprobs.patch` | [sgl-project/sglang#42719](https://github.com/sgl-project/sglang/pull/42719) | Reject prompt logprobs at admission under decoder bounded replay |
| 07 | `07-sm12x-candidate-indexer.patch` | [sgl-project/sglang#42371](https://github.com/sgl-project/sglang/pull/42371) | V4.1 candidate indexer on SM12x through the paged decode path |
| 08 | `08-ragged-capture-layout-per-tier.patch` | [sgl-project/sglang#42709](https://github.com/sgl-project/sglang/pull/42709) | One capture layout per ragged-verify token tier across CUDA graphs |
| 09 | `09-dsv41-ragged-verify-engram.patch` | [sgl-project/sglang#42718](https://github.com/sgl-project/sglang/pull/42718) | Packed compact-verify tokens mapped to their requests (replaces #40567) |
| 10 | `10-dspark-forced-budget-verify-all.patch` | [sgl-project/sglang#42711](https://github.com/sgl-project/sglang/pull/42711) | Correct SPS cost tables when profiling without a table |
| 11 | `11-dspark-folded-fp32-corrected-logits.patch` | [sgl-project/sglang#42710](https://github.com/sgl-project/sglang/pull/42710) | Folded-sampling corrected draft logits stay FP32 (lossless acceptance) |
| 12 | `12-flash-mla-sm120-split-buffers.patch` | [sgl-project/sglang#42714](https://github.com/sgl-project/sglang/pull/42714) | SM120 FlashMLA page-split buffers allocated once per size (captured graphs never write freed memory) |
| 13 | `13-dspark-confidence-relay-sync.patch` | [sgl-project/sglang#42717](https://github.com/sgl-project/sglang/pull/42717) | Every TP rank resolves the same relayed DSpark confidence |
| 14 | `14-forced-uniform-capture-token-key.patch` | [sgl-project/sglang#42713](https://github.com/sgl-project/sglang/pull/42713), rebased, on 08 | Forced-uniform compact captures keyed by token count, with a full-width layout |
| 15 | `15-dspark-sts-accept-labels.patch` | [sgl-project/sglang#42712](https://github.com/sgl-project/sglang/pull/42712) | STS calibration labels from the actual acceptance, unverified positions masked |
| 16 | `16-dsv4-fp4-indexer-exp2-free.patch` | [sgl-project/sglang#42861](https://github.com/sgl-project/sglang/pull/42861) | FP4 index keys decoded without `exp2` (bit-identical) |
| 17 | `17-sm120-mxfp8-skinny.patch` | [sgl-project/sglang#42865](https://github.com/sgl-project/sglang/pull/42865) | Split-K MXFP8 GEMM for tuned decode shapes at up to 16 rows |
| 18 | `18-mhc-mix-vec-reduce.patch` | [sgl-project/sglang#42864](https://github.com/sgl-project/sglang/pull/42864) | Vectorized mHC reduce; 160 K slices at SM120 decode |
| 19 | `19-mhc-prefill-stats-overlap.patch` | [sgl-project/sglang#42985](https://github.com/sgl-project/sglang/pull/42985) | Wide prefill overlaps the mHC mix stats with the TP all-reduce again (#42245 removed it; measured prefill regression) |
| 20 | `20-pcie-ipc-gather.patch` | [sgl-project/sglang#42863](https://github.com/sgl-project/sglang/pull/42863) | TP4 logits and DSpark vocab gathers on FlashInfer PCIe-IPC |
| 21 | `21-dsv41-request-window-copy.patch` | [sgl-project/sglang#42715](https://github.com/sgl-project/sglang/pull/42715) | Encoder-replay request-window token copies as one kernel per layer copy |
| 22 | `22-dsv41-request-window-history-check.patch` | [sgl-project/sglang#42921](https://github.com/sgl-project/sglang/pull/42921), on 21 | Request-window history checked once per layout with an async device assert (no prefill host sync) |
| 23 | `23-bf16-skinny-lm-head.patch` | [sgl-project/sglang#42867](https://github.com/sgl-project/sglang/pull/42867) | Opt-in weight-streaming BF16 LM head GEMM at <= 16 rows (`SGLANG_ENABLE_BF16_SKINNY_LM_HEAD`) |
| 24 | `24-serving-coverage-warmup.patch` | [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541) | `--warmups serving_coverage`: real requests through every serving path, including decodes at every power-of-two batch size up to the running-request limit |
| 25 | `25-dspark-sps-autotune.patch` | not yet upstream | `--speculative-dspark-sps-autotune off\|auto\|force`: fit the DSpark SPS cost table at startup from real decode steps per batch size and verify tier plus a per-verify-token context term, cached per configuration |
| 26 | `26-ragged-sub-width-tiers.patch` | [sgl-project/sglang#43011](https://github.com/sgl-project/sglang/pull/43011), rebased (closed upstream) | Opt-in ragged-verify tiers below one request's width (`SGLANG_RAGGED_VERIFY_SUB_WIDTH_TIERS`) |
| 27 | `27-dspark-verify-pad-last-row.patch` | not yet upstream | No top-p renormalization on compact-verify padding rows |
| 28 | `28-compact-verify-committed-seq-lens.patch` | not yet upstream | Compact verify passes committed CPU lengths to self-adding backends |
| 29 | `29-dsv41-candidate-scorer.patch` | not yet upstream | Paged decode consumer layers score only their candidate blocks |
| 30 | `30-dsv41-wo-a-fp8-weight.patch` | [sgl-project/sglang#43008](https://github.com/sgl-project/sglang/pull/43008), rebased | SM120 decode/verify WO-A reads an exact FP8 weight copy (`SGLANG_DSV41_WO_A_FP8_COPY`, default on) |
| 31 | `31-encoder-swa-replay-multimodal.patch` | not yet upstream | Encoder SWA bounded replay accepts image requests (replay carries the request's multimodal inputs) |
| 32 | `32-dsv41-indexer-prefill-tile-budget.patch` | not yet upstream | Prefill indexer score tiles sized from free memory, scratch counted, no padded copy (OOM fix) |
| 33 | `33-dspark-smoothed-budget-sizing.patch` | not yet upstream | Host verify-budget sizing from a per-request running average of lagged survival (opt-in); request-pool generations unique across a clear |

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
