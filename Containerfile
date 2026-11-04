# x0ne-infer, the attested server, on CUDA through x0ne Core, for a confidential
# GPU VM. The CVM image of x0ne-OS (scripts/pkg-infer-cuda.sh) is how it ships.
#
#   podman build --target confidential -t x0ne-infer .
#
# Every input is pinned: base images by digest, crates by Cargo.lock (--locked).
# Bit-for-bit reproducibility of the binary has not been measured.

ARG RUST_IMAGE=docker.io/library/rust:1.98.1-slim-bookworm@sha256:ff521445a372125ed4f76e1453a1f8098f2d05332d1601d30db1c1f62757e730
# nvcc for Core's kernels and the cuBLAS the server opens. 13.3 was the ceiling of
# cudarc, which left with candle's CUDA path on 2026-11-04; the pin stays until measured.
ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13:nonroot@sha256:54df941ed0d06a1bd95ef5e0ce391fd8d9f94b64782dc9a60062727849ee3f97

FROM ${RUST_IMAGE} AS rust

# ── build: CUDA toolkit + the Rust toolchain copied from the pinned image ────
FROM ${CUDA_DEVEL_IMAGE} AS build
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:$PATH \
    CARGO_TARGET_DIR=/build/target CARGO_INCREMENTAL=0
# Core's kernels are cubins for sm_90a, sm_100a and sm_120a, no PTX (x0ne-core/kernels/build.rs).
WORKDIR /src
COPY . .
# Libraries the server opens with dlopen (x0ne-core/runtime/src/cuda.rs): cuBLAS
# and the cuBLASLt it links. libcuda is the driver's, injected at run time by CDI.
ENV CUDA_LIBS="libcublas.so.13 libcublasLt.so.13"

# The attested server, for a confidential GPU VM. Defined, never built or run
# as of 2026-09-25: it also needs configfs-tsm and NVML inside the container.
# The runtime has no ldconfig: the loader's cache for cuBLAS is built here, on /out.
FROM build AS build-confidential
ARG CARGO_BUILD_JOBS
RUN cargo build --release --locked -p x0ne-infer --bin x0ne-infer \
        --features "cuda nvml" \
 && install -D -m 0555 "$CARGO_TARGET_DIR/release/x0ne-infer" /out/usr/local/bin/x0ne-infer \
 && mkdir -p /out/usr/local/cuda/lib64 /out/etc/ld.so.conf.d \
 && for l in $CUDA_LIBS; do cp -L "/usr/local/cuda/lib64/$l" /out/usr/local/cuda/lib64/; done \
 && echo /usr/local/cuda/lib64 > /out/etc/ld.so.conf.d/x0ne-infer.conf \
 && echo 'include /etc/ld.so.conf.d/*.conf' > /out/etc/ld.so.conf \
 && ldconfig -X -r /out

# ── runtime: glibc + libstdc++, no shell, no package manager, uid 65532 ─────
FROM ${RUNTIME_IMAGE} AS runtime
# No LD_LIBRARY_PATH: a release x0ne-infer refuses it (exit 73). libcuda.so.1 is
# mounted by CDI into /usr/lib/x86_64-linux-gnu, which the loader searches by
# default (its --help in this image, 2026-11-04); cuBLAS is found through
# /etc/ld.so.conf.d/x0ne-infer.conf and the cache built from it.
# The server's license (AGPL-3.0-only and its NVIDIA permission), the map of what is
# Apache-2.0, Apache-2.0 §4(d) and the MIT/BSD texts of every crate (scripts/third-party-licenses.sh).
COPY --from=build /src/LICENSE /src/LICENSING.md /src/NOTICE /src/THIRD_PARTY_LICENSES.html /usr/share/doc/x0ne-infer/
USER 65532:65532
EXPOSE 8477

# Last stage: the default target.
FROM runtime AS confidential
COPY --from=build-confidential /out/ /
ENTRYPOINT ["/usr/local/bin/x0ne-infer"]
