# Linux SM75 build

The known baseline is CUDA 12.6.3, GCC 13.2 and Ubuntu 24.04 x86-64. The
container pins its base image, CMake/Ninja package versions and Ubuntu archive
snapshot `20261003T000000Z`. Building needs no GPU or NVIDIA container runtime.
Running GPU tests requires an SM75 CUDA device.

From `ports/cuda-compat`:

```sh
docker build -t opendlss-cuda-compat-build toolchain
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/workspace" \
  opendlss-cuda-compat-build bash -lc \
  'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 && ctest --test-dir build -L host --output-on-failure'
```

Or use the same CMake commands in a compatible Linux CUDA development environment.
`BUILD_TESTING=OFF` builds just the required library; missing sources always fail.
The CMake build produces `build/libnr_cuda.so`, a native source inventory, and a
configuration receipt. `nr_source_sha256()` identifies the listed sources;
`nr_build_info()` also identifies compiler versions and the configuration hash.
These are distinct from the built library's SHA256. No byte-identical build claim
is made across toolchains or hosts.

On the supported GPU host, make the CUDA runtime shared library available
through the system loader or `LD_LIBRARY_PATH`, then run:

```sh
ctest --test-dir build -L gpu --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Host tests do not initialize CUDA, although the executables dynamically link the
CUDA runtime. GPU tests are separately labelled and include canonical numerical
fixtures, observed browser SiLU, synthetic GEMMs, fused/staged windows and context
lifecycle checks. Use `ctest -L host` on a CPU build host; unfiltered CTest also
runs GPU tests. The source distribution does not bundle NVIDIA runtime libraries.

The Ubuntu 24.04 host ABI does not establish support for older glibc systems.
The C interface is internal to its matching Python package at ABI version 1.
The numerical policy is `browser-compat-v1` (ID 6). Window selection is an
explicit create argument: fused windows are the default; staged windows support
operator comparison.
