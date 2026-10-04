# Feature-input parity fixtures

`python -m opendlss_cuda parity --model MODEL --library LIBRARY --fixture FIXTURE --out NEW_REPORT_DIR`
compares the complete padded network head and **all 75 published graph boundaries**.
It does not resize, crop, compose RGB, reconstruct temporal history or accept a
subset as a complete parity result. The native context must report SM75,
`browser-compat-v1` and fused windows. No GPU is opened until the fixture and
canonical model have passed CPU validation.

The default run performs three production replays without boundary captures,
then one run in a fresh context with captures. All production heads must match
the reference and one another bit for bit. The instrumented head must match
both. Signed zeros remain distinct. Runtime failures, missing captures,
nonfinite outputs and cleanup failures cannot produce a passing receipt.

## Reference domains and result codes

| Domain | What an exact result establishes |
| --- | --- |
| `browser-compat-v1` | Agreement with the identified browser implementation and its saved outputs. |
| `original-vulkan` | Agreement with an independently supplied original Vulkan capture, for these inputs and geometry. |

Browser agreement does not establish original Vulkan, native FP8 or NVIDIA
image-quality parity. A manifest's provenance fields are claims supplied by its
author; hashes bind files to those claims and are not signatures authenticating
the author. Use independently captured, trusted reference files. Never create
expected outputs from the candidate backend being tested.

Exit codes are `0` / `passed`, `1` / `mismatch`, `2` / `invalid` or
`runtime_error`, and `3` / `unavailable`. A missing original reference can be
represented honestly by this complete manifest:

```json
{
  "format": "opendlss-cuda-feature-fixture-v1",
  "availability": "unavailable",
  "reference_domain": "original-vulkan",
  "reason": "Original Vulkan head and all-boundary capture not supplied"
}
```

It returns `unavailable`, never success, and does not open CUDA. No full-network
reference pack, pretrained weights or proprietary test images are bundled in
this port. The small numeric fixtures described below are not such a pack.

## Available manifest

Pass either `fixture.json` or its directory. All tensor files are relative to
that directory. Symlinks, absolute paths, parent traversal, duplicate JSON
keys, unknown fields, nonfinite data and incorrect hashes are rejected.
Individual files are limited to 128 MiB and the complete pack to 512 MiB.
Use a small complete capture, such as the 320 by 320 field, for this bounded
all-boundary test. It is not a head-only quality-corpus runner.

The following schematic shows every required top-level field. Replace the
placeholder hashes and include every boundary listed below; the ellipses mark
placeholders that must be replaced with complete tensor descriptors.

```json
{
  "format": "opendlss-cuda-feature-fixture-v1",
  "availability": "available",
  "checks": ["head", "all-75-boundaries"],
  "reference": {
    "domain": "browser-compat-v1",
    "upstream_commit": "<40 lowercase hex characters>",
    "capture_manifest_sha256": "<64 lowercase hex characters>",
    "producer": "independent capture tool and version",
    "producer_sha256": "<capture tool SHA256>",
    "device": "reference device",
    "driver": "reference driver version",
    "browser_version": "reference browser version",
    "browser_binary_sha256": "<reference browser SHA256>",
    "shader_source_sha256": "<reference shader source inventory SHA256>"
  },
  "model": {
    "source_manifest_sha256": "<packed model manifest SHA256>",
    "canonical_manifest_sha256": "<canonical manifest.json SHA256>",
    "canonical_blob_sha256": "<canonical weights.bin SHA256>"
  },
  "geometry": {
    "mode": "upstream",
    "valid_width": 320,
    "valid_height": 320,
    "full_width": 320,
    "full_height": 320
  },
  "features": {"...": "descriptor below"},
  "head": {"...": "descriptor below"},
  "boundaries": {"block-0": {"...": "descriptor below"}, "...": "remaining 74"}
}
```

For `original-vulkan`, omit exactly the three browser-specific fields
(`browser_version`, `browser_binary_sha256`, `shader_source_sha256`) and retain
all common reference fields. Use the original shader's actual full head and
raw E4 boundary bytes, not a browser recapture with a renamed domain.

Each tensor descriptor has exactly these fields:

```json
{
  "file": "features.npy",
  "file_sha256": "<SHA256 of entire file>",
  "bytes": 6553728,
  "array_sha256": "<SHA256 of contiguous array bytes without NPY header>",
  "format": "f32-npy",
  "shape": [320, 320, 16]
}
```

The byte count above is illustrative: use the actual file length. Features
and head are C-order, little-endian float32 NPY v1/v2, respectively
`[full_height, full_width, 16]` and `[full_height, full_width, 4]`. All four head
channels, including the blend logit, are required. Features must remain within
the finite half range. Input NPY headers are shape-checked before allocation;
pickles, NPZ containers and trailing payload bytes are not accepted.

Boundary descriptors instead use `format: "e4"`, raw uint8 files without a
header and shape `[rows, channels]`. For raw E4 files, file and array SHA256
are identical. Each descriptor must refer to a distinct file.

Required names are `block-0` through `block-69` and these five transitions:

```text
transition-0-1
transition-4-5
transition-8-9
transition-14-15
transition-22-23
```

Block 70 is represented by the complete four-channel head. The required
boundary names follow the original graph capture points; there is no
`transition-30-31` capture in this set. `required_boundary_shapes(geometry)`
provides their shapes from the fixed graph schedule. The geometry's valid and
padded dimensions must agree with its named rule (`upstream` or
`legacy_belvedere`). Geometries are never silently substituted.

Native captures contain half values. The comparator first requires each value
to be exactly E4 representable, then independently maps its exact half bits to
the finite E4 code and compares the **original reference bytes**. It also
reports decoded-half agreement separately. This prevents a NaN E4 code in the
reference from passing through a NaN-to-zero decode. Negative zero is preserved
by the independent encoder; it cannot compare equal to positive zero.

The library-level API is:

```python
from opendlss_cuda.fixture import load_fixture, run_parity

validated = load_fixture("fixture.json")  # CPU only
receipt = run_parity(fixture="fixture.json", model="MODEL", library="LIBRARY", replays=3)
```

`run_parity` returns a JSON-serializable receipt and writes nothing. Invalid
inputs raise `ValueError`; execution failures raise `RuntimeError`. Receipts
include fixture/model/library/input hashes, actual context/build identity,
geometry, repeat hashes, the instrumented-head checks and every boundary check.

## Small numeric fixtures

`tests/fixtures` contains about 1.1 MiB of independent arithmetic inputs and
expected results, with no model weights or images:

- `numerics.bin`: upstream browser-port numeric vectors, copied unchanged.
- `upstream_numeric.h` and `LICENSE.upstream`: the independent original CPU
  numeric oracle and its MIT notice.
- `browser-silu-observed.bin` / `.json`: the captured browser SiLU table,
  explicitly scoped to the observed browser behavior. Each input half bit
  pattern has a four-byte record: little-endian output-half bits, E4 code and
  zero reserved byte. Finite-half cases are the supported arithmetic contract.
- `f13/`: 4,096 synthetic K16 groups and six GEMM cases covering residuals,
  multiple partitions, a partial final partition, tails and K=4096. The
  retained report records their original hashes and binary layout.

From the repository root, verify pinned references and independently regenerate
every synthetic F13 file, without a GPU:

```sh
python ports/cuda-compat/tools/generate_numeric_fixtures.py \
  --verify ports/cuda-compat/tests/fixtures
```

To write only the seven synthetic F13 files to a fresh directory:

```sh
python ports/cuda-compat/tools/generate_numeric_fixtures.py --out NEW_FIXTURE_DIRECTORY
```

The generator uses exact dyadic products in FP64 followed by integer grid
truncation; it does not import the CUDA helpers, model exporter or captured
network outputs. Tests add literal cancellation, partition-order and midpoint
witnesses. These checks establish arithmetic consistency, not image quality,
timing, native FP8 parity or a complete-network release decision.
