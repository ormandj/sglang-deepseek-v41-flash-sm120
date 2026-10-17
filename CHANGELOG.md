# Changelog

## v0.3.0-rc.1

The 0.2.1 image for `deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), DeepSeek's update to V4.1-Flash with the same architecture, configuration and tokenizer.

### Changes

- The model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Download it into a new directory; the earlier checkpoint is not modified.
- Prompt encoding follows the 1017 release's `encoding/encoding.py` (backport of sgl-project/sglang#44751): a message that only returns tool results no longer starts a new user turn, so reasoning from earlier in the current turn is kept across tool calls; tool and user content given as a list of blocks is accepted; images in tool results follow the tool results' call order.
- Sources, the other 33 patches and the runtime configuration are unchanged from 0.2.1.

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
