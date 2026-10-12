# Validation

Validated on 2026-10-04 with an RTX 2060 (6 GiB), driver 595.91.07 and Ubuntu
26.04.1. The library was built for SM75 using the pinned Ubuntu 24.04 container,
CUDA 12.6.3, GCC 13.2 and `--ftz=false`; its CUDA dependency is the CUDA 12 runtime.

## Results

| Check | Result |
|---|---|
| Installed Python package | All 70 CPU tests passed on Linux Python 3.10.12 / NumPy 2.2.6 and Python 3.12.14 / NumPy 2.5.3. |
| Native operators | Seven host and seven GPU groups passed, including canonical numeric fixtures, browser SiLU, compact GEMM and fused/staged windows. |
| Complete 320×320 browser fixture | All 75 boundaries and the full four-channel head matched exactly; ten production replays and an instrumented run agreed. |
| Fixed development corpus | All 168 full heads matched browser captures across 320×320, 512×512 and 1280×720, both geometry modes and seeds 0, 1 and 17. |
| Lifecycle | Context reuse, stable allocation, ownership rejection, capture abort and cleanup passed. |
| Compute Sanitizer | API and complete 320×320 network checks reported zero memory errors and zero leaks. |

The Windows Python 3.11 CPU suite ran 70 tests with one symlink-permission skip.
[Hosted CI](https://github.com/andrewginns/OpenDLSS-NR/actions/runs/38160084890)
passed at `3ed7ba8`: native compilation, seven host groups and all 70 tests on
Python 3.10 and 3.12. GPU execution was tested on one SM75 device.
An external eight-frame temporal smoke also matched the accepted CUDA backend's
heads, composed output and history exactly; it covered initialization, with no scene cuts.

The 2026-10-12 follow-up passed 73 CPU tests on Linux, including selected-root
and child-symlink cases; Windows passed with four symlink-permission skips.
The offline pinned-toolchain build and eight host CTest groups passed, including
compiler-flag configure regressions. No new GPU checks ran for this follow-up.

The corpus uses existing development inputs. These results establish the
`browser-compat-v1` contract for the recorded inputs. **Original/Vulkan
full-network parity remains unavailable.** Full-network browser agreement alone
does not establish NVIDIA image-quality parity.

## Measured throughput

For upstream 1280×720 features (1344×768 padded), the median of five run medians
was **1019.9 ms per public call**, including input validation, upload and
host readback; CUDA-event execution was **995.1 ms**. Each run used ten
warmups and twenty timed calls, with exact output checks outside the timer.
Setup, hashing and file I/O were excluded. Peak observed GPU memory across
validation was 2,082 MiB. The configured power limit was
115 W; the driver-reported default was 80 W.

## Reference and build identities

The reference used Chrome 153.0.8010.52 with `vulkan_enable_f16_on_nvidia`.
Operator fixture provenance and regeneration are documented in [fixtures.md](fixtures.md).
SHA256 identities for the historical 2026-10-04 tested build (before the later
compiler-flag and fixture-path changes):

- Native source inventory: `33c68618c0a9167b871ff0dbc74d57f91c3ef31a50b2ced1543375a42b3b4a92`
- Shared library: `2733e837989af26c1bb567506fffa063524affa2a9a634b136d6bb1ac5c5c734`
- Python source aggregate: `ff84ee178b525a7797118cf448a9e78f2e4d367bba349d9c1b0bccc9c411b832`

## Reproduce

Build and install using [the port instructions](../README.md) or
[the pinned toolchain](../toolchain/README.md). From the repository root:

```sh
python -m unittest discover -s ports/cuda-compat/tests -p 'test_*.py' -v
python ports/cuda-compat/tools/generate_numeric_fixtures.py \
  --verify ports/cuda-compat/tests/fixtures
ctest --test-dir ports/cuda-compat/build -L host --output-on-failure
ctest --test-dir ports/cuda-compat/build -L gpu --output-on-failure
```

GPU checks require the supported device and CUDA runtime. Full-network parity
also requires weights and an independent [capture pack](fixtures.md):

```sh
python -m opendlss_cuda parity \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --fixture /path/to/independent-fixture --replays 10 --out /path/to/new-parity
python -m opendlss_cuda bench \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --features /path/to/features.npy --width 1280 --height 720 \
  --warmups 10 --calls 20 --out /path/to/new-benchmark
```

The source distribution includes public operator fixtures. Full-network captures,
model weights and private development images are supplied separately. An
unavailable reference produces `unavailable`, never a passing parity result.

For padded-token memory checks, run both tools on the supported GPU host:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 python \
  ports/cuda-compat/tools/sanitizer_smoke.py --model /path/to/cache --library /path/to/libnr_cuda.so
compute-sanitizer --tool initcheck --error-exitcode 99 python \
  ports/cuda-compat/tools/sanitizer_smoke.py --model /path/to/cache --library /path/to/libnr_cuda.so
```

The smoke covers 512x512 and 1280x720 with production/captured contexts and graph
replay. These added sanitizer cases have not run; the GPU host was unreachable
on 2026-10-12. They check memory safety and consistency, not reference parity.
