# Notices

The container images do not include model weights. The vendor base supplies the pinned CUDA/PyTorch dependency stack; `stack.lock.json` records the SGLang, FlashInfer, DeepGEMM and Triton source inputs and patches.

`deepseek-ai/DeepSeek-V4.1-Flash` is released by DeepSeek under the **MIT License**.

SGLang and FlashInfer are Apache-2.0. DeepGEMM and Triton are MIT. The `instanttensor` wheel installed by the image carries its own license. The `lmsysorg/sglang` base image carries its own upstream notices.
