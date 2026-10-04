# Known build baseline: CUDA 12.6.3, Ubuntu 24.04, GCC 13.2.
FROM nvidia/cuda@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a
COPY build-packages.lock /tmp/build-packages.lock
RUN test "$(g++-13 -dumpfullversion)" = "13.2.0" \
 && apt-get update -qq \
 && xargs -r -a /tmp/build-packages.lock apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
