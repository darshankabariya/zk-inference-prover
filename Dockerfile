# Both backends of the prover from a clean image.  CUDA 12.9 is the toolkit the
# recorded digests were built with; the vendored fork links its CUDA library on
# every Linux build, so the toolkit is needed even for the SIMD backend.  No GPU
# is needed to build.  The recorded runs' hosts reported drivers 580.105.08 and
# 595.84; any driver that supports CUDA 12.9 runs the CUDA backend.
#
#   docker build -t zk-prover .
#   docker run --rm zk-prover b65-simd compose --s 32 --ctx 64 --bands 2,1 --corner
#   docker run --rm --gpus all zk-prover b65-cuda compose --s 32 --ctx 64 --bands 2,1 --corner
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates git pkg-config python3 && \
    rm -rf /var/lib/apt/lists/*

# nvcc 12.9 accepts gcc 14 or older; Ubuntu 24.04 ships gcc 13.
ENV CUDAHOSTCXX=/usr/bin/g++
# Device architecture of the kernels: sm_90.  Override for another card.
ENV CUDAARCHS=90

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /usr/local/bin/

WORKDIR /src
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY prover prover
COPY nitroo nitroo
RUN cargo build --release --locked && cp target/release/b65 /usr/local/bin/b65-simd && \
    cargo build --release --locked -p b65-pipeline --features cuda && cp target/release/b65 /usr/local/bin/b65-cuda

COPY harness harness
RUN cd harness && uv sync --frozen
