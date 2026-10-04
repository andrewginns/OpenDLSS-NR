# CUDA SM75 compatibility port

An optional Linux CUDA implementation of the 71-block OpenDLSS-NR network for
SM75 devices, with RTX 2060 as the hardware validation target. It consumes
prepared feature tensors and returns the full four-channel network head: an RGB
residual and a temporal-blend logit. Callers provide preprocessing and composition.

Weights are decoded once into FP16 storage. Inference uses software E4M3
publication and F13/F24 reductions to reproduce the pinned browser behavior.
Native FP8 instructions are not required. The supported arithmetic is
`browser-compat-v1`; see [the numerical contract](docs/arithmetic.md).

The port is independent of the Vulkan executable and Filament demo. Current
validation status and tested configurations are in [validation.md](docs/validation.md).

## Requirements

- Linux x86-64 and an SM75 CUDA device; this contribution targets the 6 GiB RTX 2060.
- CUDA Toolkit 12.6.3, CMake 3.24 or later, a compatible C++17 host compiler and Ninja.
- A compatible NVIDIA driver, CUDA 12 runtime and cuBLAS shared libraries.
- Python 3.10 or later and NumPy 1.24 or later. Python help, model conversion and
  CPU tests do not load CUDA. The `info` command does initialize the device.
- User-supplied 310.8.0 weights in the [upstream model layout](../../README.md#model-directory).

The native C ABI is internal to the matching Python package. A version mismatch
is rejected before creating a device context. Keep each context on its creating
thread and CUDA device. Use it as a context manager or explicitly call `close()`.

## Build and install

From the repository root:

```sh
cmake -S ports/cuda-compat -B ports/cuda-compat/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build ports/cuda-compat/build --parallel 4
python3 -m venv /path/to/new-venv
. /path/to/new-venv/bin/activate
python -m pip install ./ports/cuda-compat
python -m opendlss_cuda info --library "$PWD/ports/cuda-compat/build/libnr_cuda.so"
```

The Python wheel contains the Python interface. Build the native library
separately and pass its absolute path. Make the installed CUDA runtime and
cuBLAS available to the Linux loader using its normal search configuration or
`LD_LIBRARY_PATH`. The source contribution does not redistribute these libraries.

## Convert weights

```sh
python -m opendlss_cuda export-model \
  --model /path/to/original-model --out /path/to/new-fp16-cache
```

Conversion verifies the supported tensor layout, stage hashes and bounds. The
cache retains source identities and hashes each output array. Existing cache
directories are refused. Keep original weights and converted caches outside Git.

## Run prepared features

Use a C-order float32 NPY file of shape `[full_height, full_width, 16]`. Values
must be finite and have absolute value at most 65504. Geometry is explicit:
1280×720 valid pixels use a **1344×768** padded field under the upstream rule.
The default is `upstream`. `legacy_belvedere` is an explicit compatibility option
for existing callers; it uses 1280×768 for the same valid dimensions.

```sh
python -m opendlss_cuda run-features \
  --model /path/to/new-fp16-cache --library /path/to/libnr_cuda.so \
  --features /path/to/features.npy --width 1280 --height 720 \
  --out /path/to/new-run
```

The command writes `head.npy` and `report.json` to a fresh directory. The receipt
records the actual geometry/device, input/model/library identities and timing.
Interrupted work never writes a successful receipt. `head.npy` includes the
entire padded field and all four channels; it is not a finished RGB image.

The supported allocation envelope is valid dimensions up to 1280×720, padded
dimensions up to 1344×768 and 320 padded global-attention tokens. Some smaller
dimensions cannot satisfy the graph's window rules and are rejected. These are
implementation bounds, not a validation claim for every shape in that envelope.

```python
import numpy as np
from opendlss_cuda import Backend

features = np.load("features.npy", allow_pickle=False)
with Backend("/path/to/cache", "/path/to/libnr_cuda.so", 1280, 720) as backend:
    head = backend.run_features(features)
    print(backend.last_metrics)
```

## Check parity and measure calls

```sh
python -m opendlss_cuda parity \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --fixture /path/to/independent-fixture --replays 3 --out /path/to/new-parity
python -m opendlss_cuda bench \
  --model /path/to/cache --library /path/to/libnr_cuda.so \
  --features /path/to/features.npy --width 1280 --height 720 \
  --warmups 10 --calls 20 --out /path/to/new-benchmark
```

Parity requires independently recorded expected values, all declared boundaries,
the full head and consistent production/capture replays. See the
[fixture format](docs/fixtures.md). Exit codes are 0 for success, 1 for a numerical
mismatch, 2 for invalid input or runtime failure, and 3 for an unavailable reference.
Unavailable original-network captures never establish original/Vulkan parity.

Benchmarks report CUDA-event time separately from the public call through host
readback. Startup, warmup, hashing and output writing are identified separately.
The command checks output determinism outside each timed interval.

## Tests

```sh
python -m unittest discover -s ports/cuda-compat/tests -p 'test_*.py' -v
```

CPU tests require no weights or GPU. Native tests have separate host and GPU
labels in CTest; GPU acceptance requires the actual device and independently
prepared model/fixtures. A skipped device test is not a passing GPU result.

See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution and
native dependency terms.
