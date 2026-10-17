# DeepSeek V4.1 Flash on SM120, serving image. Builds FROM the stack image
# (Containerfile.stack: vendor CUDA/PyTorch base plus FlashInfer, DeepGEMM and
# Triton) and installs SGLang (with its Rust extensions) from the source and
# series in stack.lock.json. The build fails unless the stack matches the lock.
ARG STACK_IMAGE
FROM ${STACK_IMAGE} AS runtime
ARG STACK_IMAGE
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION
COPY stack.lock.json Containerfile.stack /opt/dsv41-platform/
COPY build /opt/dsv41-platform/scripts
COPY patches /opt/dsv41-platform/patches
RUN bash /opt/dsv41-platform/scripts/build-stack.sh
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash on SM120" \
      org.opencontainers.image.description="SGLang for DeepSeek-V4.1-Flash on four RTX PRO 6000 Blackwell GPUs over PCIe" \
      org.opencontainers.image.version="0.3.0-rc.1" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.revision="${IMAGE_SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${STACK_IMAGE}" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/dsv41-platform
ENTRYPOINT []
CMD ["sglang", "serve", "--help"]
