# Changelog

## v0.4.0-rc.1

The 0.3.0 image rebuilt on current upstream sources for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, unchanged). The runtime configuration is unchanged.

### Changes

- SGLang `main` of 2026-10-17 (`4eb606b2ae`) with 27 patches, FlashInfer `main` of 2026-10-17 (`05041022c3`, version 0.7.2) with two patches, sgl-project DeepGEMM `cbe1727eac` (unchanged) and Triton `release/3.8.x` with its backport (unchanged), on the `lmsysorg/sglang` CUDA 13 nightly of 2026-10-16, whose SGLang kernel package (0.4.10) and Transformers (5.19.0) match SGLang `main`.
- Now upstream and no longer patched: the InstantTensor loader (sgl-project/sglang#40453), the complete SWA window for short bounded-replay extends (#41303), HiCache prefix hits under encoder SWA replay (#42716), one capture layout per ragged-verify tier (#42709), the V4.1 tool-result encoding (#44751), the request-window copy kernel (#44555 in place of #42715), FP32 draft step logits (#43790 in place of #42710) and prompt logprobs under decoder SWA bounded replay (#44850 in place of #42719).
- Prompt-logprob requests under decoder SWA bounded replay are now served instead of rejected (#44850).
- New patches: DSpark folded-accept results are copied before the next verify replay, so tensor-parallel ranks commit the same tokens under overlap scheduling with HiCache (sgl-project/sglang#42812); the FlashInfer PCIe-IPC all-reduce tuner adds a barrier before its warm-up, bounds the copy-engine ring depth and alternates the candidate order while ranking (flashinfer-ai/flashinfer#6401).
- The other patches are carried at their current heads, rebased onto the new sources; the mHC decode split and the wide-prefill mHC statistics overlap are ported to SGLang's new layer-boundary transports.

### Upgrade

Start the `v0.4.0-rc.1` image with the same settings as 0.3.0. FlashInfer changed, so the first start compiles its kernels into a new directory of the kernel cache, and the speculative-decoding cost table is measured again.

## v0.3.0 (stable; 2026-10-17)

The 0.2.1 image for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), DeepSeek's update to V4.1-Flash with the same architecture, configuration and tokenizer.

### Changes

- The model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Download it into a new directory; the earlier checkpoint is not modified.
- Prompt encoding follows the 1017 release's `encoding/encoding.py` (backport of sgl-project/sglang#44751): a message that only returns tool results no longer starts a new user turn, so reasoning from earlier in the current turn is kept across tool calls; tool and user content given as a list of blocks is accepted; images in tool results follow the tool results' call order.
- Sources, the other 33 patches and the runtime configuration are unchanged from 0.2.1.

### Upgrade

Download `deepseek-ai/DeepSeek-V4.1-Flash-1017` at the pinned revision into a new directory, point `MODEL_DIR` at it and start the `v0.3.0` image with the same settings as 0.2.1. The 0.2.1 kernel cache can be reused: FlashInfer, DeepGEMM and Triton are rebuilt from the same locked sources.

### Measurements

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with the privately built image of this candidate, three repetitions per cell, compared with 0.2.1 on the earlier checkpoint.

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 186 | 318 | 498 | 667 | 1,020 |
| Accepted tokens per step | 2.25 | 2.15 | 2.18 | 2.03 | 2.13 |

- The engine runs at the same speed as 0.2.1: decode steps per second at 2 to 16 running requests and at 0, 32K and 128K tokens of context are within run-to-run variation. Output tok/s is lower at low concurrency because the 1017 model's speculative acceptance is lower (2.25 against 2.58 accepted tokens per step at C1).
- The 1017 model writes shorter answers: on the same sampled prompts its mean completion was 13% shorter at C4 and 30% shorter at C16, so a typical answer finishes sooner even where tok/s is lower.
- Cold prefill: about 10.0K, 12.6K and 12.9K prompt tok/s at 8K, 32K and 128K tokens, unchanged from 0.2.1.
- Device KV pool: 2,078,208 tokens across requests; one request may use up to 1,048,576.
- Ten minutes at C8: 321 requests, no errors or runaway outputs. A 54K-token prompt evicted from the GPU cache is served back from host RAM.

These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance.

### Known limitations

- DeepSeek's 1017 reference rounds the attention output to FP8 before the WO-A projection; this image computes it in BF16, as 0.2.1 did and as upstream SGLang does for V4.1 checkpoints. Output quality checks pass; matching the reference numerics is planned work.
- The 1017 model card recommends `top_p` 1.0; the startup speculative cost table is measured with `top_p` 0.95 (`SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P`).

## v0.2.1 (stable; 2026-10-16)

First published release of this image for `deepseek-ai/DeepSeek-V4.1-Flash` on four RTX PRO 6000 Blackwell GPUs over PCIe.

### Changes

- SGLang `main` with 33 patches (open upstream pull requests and fixes prepared for upstream), FlashInfer `main` with one patch, sgl-project DeepGEMM `dev` and Triton `release/3.8.x` with one backport, all rebuilt from source on the CUDA 13 `lmsysorg/sglang` nightly.
- TP4 with expert parallelism 2, FP8 KV cache, a 1,048,576-token per-request limit and 16 running requests.
- DSpark speculative decoding with a verify-cost table measured on the serving machine at startup and cached, compact ragged verify with sub-width tiers, and a single request's verify budget sized from a running average of its acceptance.
- Engram tables held in host RAM, one shard per rank, loaded with the InstantTensor loader.
- Decoder and encoder SWA bounded replay with HiCache, so evicted prefixes are restored from host RAM; image input under encoder replay.
- Kernel and serving fixes for SM120, including split-K MXFP8 and BF16 skinny GEMMs for decode-sized shapes, FP8 WO-A weights, PCIe-IPC vocabulary gathers, and prefill overlap of the mHC statistics with the tensor-parallel all-reduce.

### Measurements

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with a separately built image from the same pinned sources and patches, three repetitions per cell. Decode output measured 220, 326, 515, 697 and 1,009 tok/s at C1, C2, C4, C8 and C16, and cold prefill about 10.1K, 12.7K and 12.9K prompt tok/s at 8K, 32K and 128K tokens. A ten-minute C8 run completed 309 requests with no errors or runaway outputs. The device KV pool holds 2,078,208 tokens. These controlled measurements do not necessarily represent real-world performance; output rates include reasoning tokens and depend on speculative acceptance.
