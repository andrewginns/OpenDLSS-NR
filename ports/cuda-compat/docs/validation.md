# Validation of the submitted CUDA compatibility port

Validation on 2026-10-04 passed for the extracted `browser-compat-v1` port on
an RTX 2060. The results below apply to the identified source and library, with
fused windows. They establish extraction parity against pinned browser outputs
and the previously validated CUDA implementation. **Original/Vulkan full-network
parity remains unavailable.** They do not establish general image-quality acceptance.

## Build and execution environment

The native library was built in the pinned Ubuntu 24.04 x86-64 development
container with CUDA Toolkit 12.6.3 (NVCC 12.6.85), GCC 13.2.0, C++17, SM75 and
`--ftz=false`. Seven host CTests passed. A fresh Python environment installed
the package and ran all 68 CPU tests on Linux with zero skips. The corresponding
Windows CPU run completed 68 tests with one symlink-permission skip; this is
not Windows CUDA support. The GitHub Actions workflow has not run; the local
build, installation and CPU checks were executed separately.

GPU execution used an RTX 2060 with 6 GiB VRAM, Ubuntu 26.04.1 LTS,
kernel 7.0.0-31-generic, glibc 2.43 and NVIDIA driver 595.91.07. The installed package
used Python 3.12.14 and NumPy 2.5.3. The isolated environment supplied
`nvidia-cuda-runtime-cu12==12.6.77` and `nvidia-cublas-cu12==12.6.4.1`.
Their library directories were explicitly configured with `LD_LIBRARY_PATH`.
These NVIDIA libraries are runtime dependencies and are not bundled in the port.
Other GPUs, operating systems and toolchain combinations have not been validated.

## Numerical and lifecycle results

| Check | Result |
| --- | --- |
| Native GPU tests | All seven passed: API, canonical numerics, F13, F24, browser numerics, compact GEMM and fused window tests. |
| Native artifact inspection | SM75 target; inspected PTX contains no native FP8 MMA or conversion instructions. |
| Complete 320×320 browser fixture | All 75 required boundaries and the entire padded four-channel head matched bit for bit. |
| Production versus capture | Ten production replays were identical; a fresh instrumented context matched production and the independent reference. |
| Fixed development corpus | All 168 full heads matched their pinned browser captures exactly. |
| Context lifecycle | Two contexts, ten exact replays each, constant allocated bytes, close/reopen and use-after-close rejection passed. |
| Invalid input and ownership | Invalid dtype/shape, half overflow and NaN were rejected; wrong-thread inference/close was rejected without making the owner context unusable. |
| Capture failure handling | Abort and subsequent context reopening passed. |
| Compute Sanitizer | API and complete 320×320 network runs each reported zero errors and zero leaked bytes/allocations; the network head remained exact. |

The corpus contains 18 cases at 320×320, 18 at 512×512 and 48 at 1280×720
for **each** of `upstream` and `legacy_belvedere`, with seeds 0, 1 and 17
equally represented. Upstream 720p uses a 1344×768 padded field; legacy 720p
uses 1280×768. Exact comparison includes all four channels and padded pixels.
These are existing development inputs, not held-out coverage or a claim about
every supported shape. The second-device ownership test was unavailable because
only one CUDA device was visible.

All 23 main supervised jobs completed without guard or cleanup failures.
Their observed maxima were 2082 MiB VRAM, 73°C and 123.87 W instantaneous
power draw. The unchanged configured power limit was **115 W**, whereas the
driver-reported default was **80 W**. These measurements are not stock-power
performance or power-efficiency results.

## Extraction timing

Five alternating-order pairs compared the prior exact CUDA implementation with
this extraction on the same upstream 1280×720 feature tensor and model. Each
condition used ten warmups and twenty timed public calls. All thirty outputs
were checked for exact equality outside the timed intervals. The public timer
covers input validation, upload, graph replay and host readback; CUDA events
measure device execution separately. Model setup, hashing and report writing
are excluded. This measures the feature-to-head API, not temporal processing.

| Pair | Order | Prior public median (ms) | New public median (ms) | Prior GPU median (ms) | New GPU median (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Prior → new | 1016.510 | 1018.246 | 991.802 | 993.478 |
| 2 | New → prior | 1019.475 | 1017.925 | 994.732 | 993.225 |
| 3 | Prior → new | 1019.389 | 1020.682 | 994.685 | 996.001 |
| 4 | New → prior | 1021.464 | 1020.394 | 996.656 | 995.625 |
| 5 | Prior → new | 1020.683 | 1022.104 | 995.927 | 997.443 |

The median paired new/prior ratio was **1.00127** for public calls and
**1.00132** for GPU time, about 0.13% slower. The predeclared 5% regression
investigation threshold was not triggered. These differences do not support
a speedup claim; they are a check that extraction did not materially regress
this measured workload. They do not compare CUDA against Vulkan or WebGPU.

## Separate short temporal smoke

A 24th supervised job used an external temporal harness for eight consecutive
720p upstream-geometry frames, seed 17, making sixteen real inference calls
across the prior and new implementations. Features, heads, composed RGB,
history and temporal state matched exactly on every frame. Initialization reset
occurred at frame 817; there were no scene cuts. Both contexts closed cleanly.
This is short extraction-parity evidence, not a temporal quality release,
scene-cut test, throughput result or a new temporal API in this port.

## Identities and evidence

All hashes below are SHA256. Native source identity covers the recorded native
source inventory; it is distinct from the shared library and configuration.
The upstream base was `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`.
The pinned browser captures used Chrome 153.0.8010.52, with the recorded
`vulkan_enable_f16_on_nvidia` feature enabled.
The candidate source archive preceded three formatting-only changes (one blank
line and two JSON newline normalizations) and this validation document. Native
and Python implementation bytes used for these tests were unchanged by them.

| Item | SHA256 |
| --- | --- |
| Candidate source archive | `ecc75214a4c8e352d79be7731a66db4f6cc168732f9d871adb7bb69dee3cad5b` |
| Candidate source manifest | `7de4ea5e96e2d15fafce144e166026f9fcfc01ec6223f7e4b9fd52829b44a293` |
| New shared library | `546abd06b6f3429ad752c15c3006fa9e30b3fc722038e181e3e9d8595062ea44` |
| Native source inventory | `83da8407f6c2eceacc986f74ad5c14b2163f05ce1407b67e6f99630505b6cdc2` |
| Native configuration | `9ac6331ca46cfcb20abf31a070fff4d22d8f799ab213a0c88c48341166b7e0ab` |
| Python package source aggregate | `73e07e4ba74b42d6d1f2612ec63dafad5a92b553765b89a9e9c1b95e247595a0` |
| Prior exact CUDA library | `fe482347e693dea8eb578cf42770201f00c1120ca1bf4632021d13888cd87a25` |
| Original model manifest | `11be575b6785a5f262a63448565e86f495e11a51a663175b7e47e0687abfdf20` |
| Converted model manifest | `acca89f88bc02012d6c47482b7a93e62e7e204cc068dfe7f8829f1c506352919` |
| Converted weight blob | `e21af8b6f7f39e23ec58298ce9ba6791e2c6a4f4a28e8f17ac992ad832bad960` |
| Complete 320×320 fixture manifest | `e0f6ffd6975bde9b1997f733281ff8b21570015273050ab59d6c3b8f170eefad` |
| Complete 320×320 head | `7b8dc8fdfd4906dc85b90a17372e6320658c6df2c9fca420e43fc1374f0d900c` |
| Main receipt archive | `0873b53c62267b83de0f85bc516157fa5ec9b6a23961e6d822d9a8839fea4f11` |

The main evidence archive contains 134 JSON/JSONL/log receipts (568,597 bytes)
for the 23 main jobs. Temporal evidence is separate. Full feature/capture arrays,
weights and NVIDIA binaries are not included in that receipt archive or this
source contribution. The numerical fixtures committed here are public, bounded
operator fixtures with [provenance and regeneration instructions](fixtures.md).
They are not a substitute for an independently captured full-network reference.

## Reproduce the available checks

From the repository root, build and install using [the port instructions](../README.md)
or [the pinned toolchain](../toolchain/README.md), then run:

```sh
python -m unittest discover -s ports/cuda-compat/tests -p 'test_*.py' -v
python ports/cuda-compat/tools/generate_numeric_fixtures.py \
  --verify ports/cuda-compat/tests/fixtures
ctest --test-dir ports/cuda-compat/build -L host --output-on-failure
ctest --test-dir ports/cuda-compat/build -L gpu --output-on-failure
```

GPU tests require the supported device and loader dependencies. Full-network
parity additionally requires user-supplied weights and a trusted independent
[capture pack](fixtures.md); the private development corpus is not distributed.
Use fresh output directories and features with the declared padded geometry:

```sh
python -m opendlss_cuda parity \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --fixture /path/to/independent-fixture --replays 10 --out /path/to/new-parity
python -m opendlss_cuda bench \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --features /path/to/features.npy --width 1280 --height 720 \
  --warmups 10 --calls 20 --out /path/to/new-benchmark
```

An unavailable original/Vulkan capture returns `unavailable`, never a passing
parity result. Browser agreement does not establish native FP8 equivalence,
Vulkan/demo integration or NVIDIA image-quality parity.
