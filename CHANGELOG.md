# Changelog

## v0.4.3 (stable; 2026-10-19)

0.4.2 with the image path warmed before serving. The sources other than patch 17, the stack, the model and the runtime configuration are unchanged.

### Changes

- The `serving_coverage` warmup (patch 17, [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541)) sends its image request under encoder SWA bounded replay on CUDA, which serves image requests since 0.4.2. 0.4.2 skipped the image phase, so the first user image loaded the vision kernels while serving.

### Upgrade

Start the `v0.4.3` image with the 0.4.2 launcher settings; it can share 0.4.2's kernel cache directory. The first start measures the speculative-decoding cost table again (about 7 minutes after warmup).

### Measurements

The serving code is the same as 0.4.2's, so 0.4.2's measurements apply. On the hardware below, the 0.4.3 warmup log shows the image phase, and a first user image request loads no kernels after the server is ready.

## v0.4.2 (stable; 2026-10-19)

0.4.1 rebuilt on current upstream sources, with faster prefill. The model and the runtime configuration are unchanged.

### Changes

- SGLang `main` of 2026-10-19 (`62cee1b15c`) with 32 patches, FlashInfer `main` of 2026-10-19 (`f64b4ba876`, version 0.7.2) with one patch, sgl-project DeepGEMM `release/v0.2.3` (`d6fcb32f19`, the 0.2.3 SGLang `main` requires) and Triton `release/3.8.x` with its backport (unchanged), on the `lmsysorg/sglang` CUDA 13 nightly of 2026-10-19.
- Now upstream and no longer patched: the FlashInfer PCIe-IPC all-reduce tuner fixes (flashinfer-ai/flashinfer#6401).
- The wide-prefill mHC statistics overlap (patch 13, [sgl-project/sglang#42985](https://github.com/sgl-project/sglang/pull/42985)) now also covers the MoE sublayer; since 0.4.0 only the attention sublayer overlapped.
- The prefill indexer's score-tile budget (patch 25, [#43376](https://github.com/sgl-project/sglang/pull/43376)) uses free memory during breakable prefill CUDA-graph replay; only graph capture uses the static budget.
- SGLang `main` carries image inputs through encoder SWA bounded replay on CUDA (#45048); patch 24 ([#43241](https://github.com/sgl-project/sglang/pull/43241)) now only warms up the image path there.
- The carried pull requests that needed it are updated on SGLang `main` (#42712, #42718, #42865, #42867, #42985, #43008, #43241, #43376).

### Upgrade

Start the `v0.4.2` image with the 0.4.1 launcher settings and a new kernel cache directory. The first start compiles kernels for the new FlashInfer, DeepGEMM and SGLang (about 40 minutes on the hardware below) and measures the speculative-decoding cost table (about 7 minutes after warmup); later starts reuse both.

### Measurements

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with the privately built image of this release, at the model card's sampling (temperature 1.0, `top_p` 1.0, thinking with `reasoning_effort` high), three repetitions per cell.

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 202 | 322 | 497 | 674 | 1,051 |
| Accepted tokens per step | 2.38 | 2.22 | 2.18 | 2.04 | 2.14 |

- Cold prefill: about 14.1K, 14.9K and 14.7K prompt tok/s at 8K, 32K and 128K tokens, 10%, 13% and 14% above 0.4.1, with prompts sized to whole 8,192-token prefill chunks.
- Long context at C1: 86, 84-86 and 81-83 decode steps/s at 0, 32K and 128K tokens of context (two runs).
- Device KV pool: 2,931,712 tokens across requests; one request may use up to 1,048,576.
- Quality: teacher-forced log-probabilities within 0.08 nats (worst prompt mean) and 0.58 nats (maximum) of a full prefill. No repetition loops in 64 sampled reasoning streams at 4 and 16 concurrent requests, or in 346 requests over 10 minutes at 8 concurrent requests.

Against 0.4.1 at the same settings, decode and long-context step rates are within the run-to-run noise. At 32K context and one request, output is 6-9% lower in both runs, from fewer accepted draft tokens per step at an unchanged step rate. These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance.

## v0.4.1 (stable; 2026-10-19)

0.4.0 with OpenAI API compatibility fixes, submitted upstream. The sources other than these patches, the model and the runtime configuration are unchanged.

### Changes

- Usage reports reasoning tokens at `usage.completion_tokens_details.reasoning_tokens`, where OpenAI clients and gateways read them, capped at `completion_tokens`; SGLang's top-level `usage.reasoning_tokens` is kept ([sgl-project/sglang#45131](https://github.com/sgl-project/sglang/pull/45131)).
- The OpenAI routes return errors as `{"error": {"message", "type", "param", "code"}}`, and server errors no longer include exception text ([#45130](https://github.com/sgl-project/sglang/pull/45130)).
- Chat and completions never report `finish_reason: "abort"`: an abort with an error status returns that error, and one without is reported as `"stop"` ([#45132](https://github.com/sgl-project/sglang/pull/45132)).
- A chat or completion stream that fails after its first chunk ends with an error frame and `[DONE]` ([#45133](https://github.com/sgl-project/sglang/pull/45133)).
- `reasoning_effort` `minimal` and `medium` map to budgets 50 and 62 instead of the default ([#45134](https://github.com/sgl-project/sglang/pull/45134)).

### Upgrade

Clients that read the old flat error body (`body["message"]`, `body["code"]`) must read `body["error"]["message"]` and the HTTP status. Start the `v0.4.1` image with the 0.4.0 launcher settings and a new kernel cache directory: the first start measures the speculative-decoding cost table again (about 7 minutes after warmup).

### Measurements

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with the privately built image of this release, at the model card's sampling (temperature 1.0, `top_p` 1.0, thinking with `reasoning_effort` high). Decode figures are the median of six repetitions per cell from two runs.

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 205 | 323 | 495 | 677 | 1,023 |
| Accepted tokens per step | 2.39 | 2.26 | 2.16 | 2.07 | 2.09 |

- Cold prefill: about 12.8K, 13.2K and 12.8K prompt tok/s at 8K, 32K and 128K tokens, with prompts sized to whole 8,192-token prefill chunks.
- Long context at C1: 87, 84 and 83 decode steps/s at 0, 32K and 128K tokens of context.
- Device KV pool: 2,934,016 tokens across requests; one request may use up to 1,048,576.
- Quality: teacher-forced log-probabilities within 0.09 nats (worst prompt mean) and 0.75 nats (maximum) of a full prefill. No repetition loops in 64 sampled reasoning streams at 4 and 16 concurrent requests, or in 321 requests over 10 minutes at 8 concurrent requests.

Against 0.4.0 at the same settings, every decode and prefill cell is within the run-to-run noise. These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance.

## v0.4.0 (stable; 2026-10-17)

The 0.3.0 image rebuilt on current upstream sources for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, unchanged). The runtime configuration is unchanged.

### Changes

- SGLang `main` of 2026-10-17 (`4eb606b2ae`) with 27 patches, FlashInfer `main` of 2026-10-17 (`05041022c3`, version 0.7.2) with two patches, sgl-project DeepGEMM `cbe1727eac` (unchanged) and Triton `release/3.8.x` with its backport (unchanged), on the `lmsysorg/sglang` CUDA 13 nightly of 2026-10-16, whose SGLang kernel package (0.4.10) and Transformers (5.19.0) match SGLang `main`.
- Now upstream and no longer patched: the InstantTensor loader (sgl-project/sglang#40453), the complete SWA window for short bounded-replay extends (#41303), HiCache prefix hits under encoder SWA replay (#42716), one capture layout per ragged-verify tier (#42709), the V4.1 tool-result encoding (#44751), the request-window copy kernel (#44555 in place of #42715), FP32 draft step logits (#43790 in place of #42710) and prompt logprobs under decoder SWA bounded replay (#44850 in place of #42719).
- Prompt-logprob requests under decoder SWA bounded replay are now served instead of rejected (#44850).
- New patches: DSpark folded-accept results are copied before the next verify replay, so tensor-parallel ranks commit the same tokens under overlap scheduling with HiCache (sgl-project/sglang#42812, amended so the in-graph accept still writes the static accept buffers after SGLang's accept refactor); the FlashInfer PCIe-IPC all-reduce tuner adds a barrier before its warm-up, bounds the copy-engine ring depth and alternates the candidate order while ranking (flashinfer-ai/flashinfer#6401).
- The other patches are carried at their current heads, rebased onto the new sources; the mHC decode split and the wide-prefill mHC statistics overlap are ported to SGLang's new layer-boundary transports.

- The launcher measures the speculative-decoding cost table at `top_p` 1.0, the DeepSeek-V4.1-Flash-1017 model card's sampling, passes `--tp-size` and no longer sets `SGLANG_DSPARK_FOLDED_SAMPLING`: SGLang now folds a sampling DSpark proposal by default.

### Upgrade

Start the `v0.4.0` image with the updated launcher, or with 0.3.0's settings plus `SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P=1.0`. FlashInfer changed, so use a new kernel cache directory: the first start compiles its kernels and measures the speculative-decoding cost table (about 7 minutes after warmup; its decode log lines show an accept length of 1.00 by design), and later starts reuse both.

### Measurements

Validation runs on four RTX PRO 6000 Blackwell Max-Q GPUs at 250 W each, with the privately built image of this release, at the model card's sampling (temperature 1.0, `top_p` 1.0, thinking with `reasoning_effort` high), three repetitions per cell.

| Concurrency | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| Decode output tok/s | 200 | 319 | 502 | 710 | 1,022 |
| Accepted tokens per step | 2.25 | 2.20 | 2.24 | 2.13 | 2.09 |

- Cold prefill: about 12.9K, 13.4K and 12.9K prompt tok/s at 8K, 32K and 128K tokens, with prompts sized to whole 8,192-token prefill chunks.
- Device KV pool: 2,934,016 tokens across requests (2,078,208 in 0.3.0); one request may use up to 1,048,576.
- Quality: teacher-forced log-probabilities within 0.07 nats (worst prompt mean) and 0.52 nats (maximum) of a full prefill, and no repetition loops in 64 sampled reasoning streams at 4 and 16 concurrent requests.

These are controlled measurements and do not necessarily represent real-world performance. Output rates include reasoning tokens and depend on speculative acceptance. Earlier releases measured decode at `top_p` 0.95 and prefill with prompts that spilled into a short second prefill pass, so their numbers are not directly comparable.

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
