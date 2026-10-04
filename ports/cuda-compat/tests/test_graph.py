"""Graph wiring/lifetime validation without GPU allocations or model files."""
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda.export_model import array_specs
from opendlss_cuda.graph import NativeGraph


@dataclass
class Tensor:
    shape: tuple
    dtype: str = "float16"
    initialized: bool = False

    @property
    def nbytes(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count * (4 if self.dtype == "float32" else 2)


class FakeModel:
    def __init__(self):
        self.arrays = {s["name"]: Tensor(tuple(s["shape"]), "float32" if s["kind"] == "float32" else "float16", True)
                       for s in array_specs()}

    def upload(self, runtime):
        return {name: runtime.upload(a) for name, a in self.arrays.items()}


class CheckingRuntime:
    def __init__(self):
        self.allocations = 0
        self.gemms = []
        self.attention = []

    def empty(self, shape, dtype="float16"):
        self.allocations += 1
        return Tensor(tuple(shape), dtype)

    def upload(self, array):
        return Tensor(array.shape, array.dtype, True)

    def read(self, *tensors):
        for t in tensors:
            assert t.initialized, f"read before write: {t.shape}"

    def copy(self, source, destination):
        self.read(source)
        assert source.shape == destination.shape
        assert source is not destination
        destination.initialized = True

    def f16gemm(self, source, weights, output, quantized=None):
        self.read(source, weights)
        assert source.shape[1] == weights.shape[0]
        assert output.shape == (source.shape[0], weights.shape[1])
        assert source is not output
        output.initialized = True
        if quantized is not None:
            assert quantized.shape == output.shape
            quantized.initialized = True

    def gemm(self, source, weights, *, rows, k, n, output, raw=None, residual=None, scale=None,
             silu=False, batches=1, broadcast=False, partition=0, quantize=True):
        self.read(source, weights)
        assert source.shape == (rows, k if broadcast else k * batches), (source.shape, rows, k, batches, broadcast)
        assert output.shape == (rows, n * batches)
        assert output is not source
        if residual is not None:
            self.read(residual, scale)
            assert residual.shape == output.shape
            assert scale.shape == (n * batches,)
        if raw is not None:
            assert raw.shape == output.shape
            raw.initialized = True
        output.initialized = True
        self.gemms.append(dict(k=k, n=n, batches=batches, broadcast=broadcast, partition=partition, quantize=quantize, silu=silu))

    def window(self, qkv, prior, scales, out, *, width, height, heads, phase):
        self.read(qkv, prior, scales)
        assert qkv.shape == (width * height, heads * 96)
        assert prior.shape == (heads, 64, 64)
        assert scales.shape == (heads,)
        assert out.shape == (width * height, heads * 32)
        assert phase in (0, 1, 2, 3)
        out.initialized = True
        self.attention.append(("window", phase, heads))

    def vit(self, qkv, scales, out, *, tokens, heads):
        self.read(qkv, scales)
        assert qkv.shape == (tokens, heads * 96)
        assert scales.shape == (heads,)
        assert out.shape == (tokens, heads * 32)
        out.initialized = True
        self.attention.append(("global", None, heads))

    def pool(self, source, output, *, in_width, in_height, out_width, out_height, channels):
        self.read(source)
        assert source.shape == (in_width * in_height, channels)
        assert output.shape == (out_width * out_height, channels)
        output.initialized = True

    def upsample(self, source, skip, scale, output, *, raw_out=None,
                 in_width, in_height, out_width, out_height, channels):
        self.read(source, skip, scale)
        assert source.shape == (in_width * in_height, channels)
        assert output.shape == skip.shape == (out_width * out_height, channels)
        assert scale.shape == (channels,)
        output.initialized = True
        if raw_out is not None:
            assert raw_out.shape == output.shape
            raw_out.initialized = True

    def post_blend(self, source, skip, input_scale, adapter_scale, output, raw_out, **kwargs):
        self.read(adapter_scale)
        self.upsample(source, skip, input_scale, output, raw_out=raw_out, **kwargs)


class GraphTests(unittest.TestCase):
    def test_supported_720p_field_and_rejected_larger_graph(self):
        from unittest.mock import Mock
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 1280, 720, capture_boundaries=True)
        self.assertEqual((graph.geometry["full_width"], graph.geometry["full_height"]), (1344, 768))
        self.assertEqual(graph.geometry["padded_vit_tokens"], 320)
        self.assertLess(graph.allocated_bytes, 4 * 1024 ** 3)
        expected = {f"block-{i}" for i in range(70)}
        expected.update(f"transition-{i}-{i + 1}" for i in (0, 4, 8, 14, 22))
        self.assertEqual(set(graph.boundaries), expected)
        model = FakeModel()
        model.upload = Mock()
        with self.assertRaisesRegex(ValueError, "Supported inference extent"):
            NativeGraph(CheckingRuntime(), model, 1920, 1080)
        model.upload.assert_not_called()

    def test_complete_graph_shapes_initialization_and_memory(self):
        for mode in ("upstream", "legacy_belvedere"):
            rt = CheckingRuntime()
            graph = NativeGraph(rt, FakeModel(), 1280, 720, mode)
            initial_allocations = rt.allocations
            graph.features.initialized = True
            graph.record()
            self.assertTrue(graph.head.initialized)
            self.assertEqual(len(rt.attention), 70)  # block 39 is merge-only
            self.assertEqual(sum(a[0] == "global" for a in rt.attention), 8)
            self.assertEqual(rt.allocations, initial_allocations)
            self.assertLess(graph.allocated_bytes, 4 * 1024 ** 3)
            self.assertEqual(set(graph.skips), {0, 4, 8, 14, 22, 30})
            self.assertEqual(sum(g["partition"] == 1024 for g in rt.gemms), 8)
            self.assertEqual(sum(g["partition"] == 512 for g in rt.gemms), 8)
            self.assertEqual(sum(g["partition"] == 256 for g in rt.gemms), 9)
            graph.record()  # buffer reuse must preserve valid shapes and ordering
            self.assertEqual(rt.allocations, initial_allocations)

    def test_boundary_selection_and_full_capture(self):
        for captures, expected in [(["block-0", "block-39", "transition-4-5"], 3), (True, 75)]:
            graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=captures)
            graph.features.initialized = True
            graph.record()
            self.assertEqual(len(graph.boundaries), expected)
            self.assertTrue(all(t.initialized for t in graph.boundaries.values()))
        with self.assertRaises(ValueError):
            NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=["block-999"])

    def test_raw_residual_paths(self):
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320)
        prologues = {}
        for name, (_, _, kwargs) in zip(graph.operation_names, graph.ops):
            if name in ("block0.w2", "block66.w2", "block70.w2"):
                prologues[name] = kwargs["residual"]
        self.assertIs(prologues["block0.w2"], graph.buffers["level6.input_raw"])
        self.assertIs(prologues["block66.w2"], graph.buffers["level0.input_raw"])
        self.assertIs(prologues["block70.w2"], graph.buffers["level6.input_raw"])

    def test_block0_detail_capture_order_and_sources(self):
        expected = [
            ("input adapter", "block0-input-adapter-half", "level6.input_raw", 32),
            ("input adapter", "block0-input-adapter-e4", "level6.state_a", 32),
            ("block0.w1.preactivation-diagnostic", "block0-ffn-expand-pre-half",
             "level6.expanded_preactivation_diagnostic", 128),
            ("block0.w1", "block0-ffn-expand-e4", "level6.expanded", 128),
            ("block0.w2", "block0-ffn-contract-half", "level6.ffn_raw", 32),
            ("block0.w2", "block0-ffn-contract-e4", "level6.ffn", 32),
            ("block0.qkv", "block0-qkv-half", "level6.qkv", 96),
            ("block0.attention", "block0-attended-e4", "level6.attended", 32),
            ("block0.projection", "block0-projection-half", "level6.output_raw", 32),
            ("block0.projection", "block0-projection-e4", "level6.state_b", 32),
        ]
        names = [item[1] for item in expected]
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=names)
        self.assertEqual(list(graph.boundaries), names)
        graph.features.initialized = True
        graph.record()
        self.assertTrue(all(t.initialized for t in graph.boundaries.values()))
        producers = {}
        producer = None
        for name, (_, args, _) in zip(graph.operation_names, graph.ops):
            if name.startswith("capture "):
                producers[name[8:]] = (producer, args)
            else:
                producer = name
        for operation, boundary, source, channels in expected:
            producer, (captured_source, destination) = producers[boundary]
            self.assertEqual(producer, operation)
            self.assertIs(captured_source, graph.buffers[source])
            self.assertIs(destination, graph.boundaries[boundary])
            self.assertIsNot(destination, captured_source)
            self.assertEqual(destination.shape, (graph.geometry["full_rows"], channels))
        # False and True retain their original allocation and capture behavior;
        # detailed buffers are allocated only for a named diagnostic request.
        for capture in (False, True):
            plain = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=capture)
            self.assertFalse(set(names).intersection(plain.boundaries))
            self.assertNotIn("level6.expanded_preactivation_diagnostic", plain.buffers)
            self.assertNotIn("block0.w1.preactivation-diagnostic", plain.operation_names)
        diagnostic_index = graph.operation_names.index("block0.w1.preactivation-diagnostic")
        original_index = graph.operation_names.index("block0.w1")
        _, diagnostic_args, diagnostic_kwargs = graph.ops[diagnostic_index]
        _, original_args, original_kwargs = graph.ops[original_index]
        self.assertEqual(diagnostic_args, original_args)
        self.assertFalse(diagnostic_kwargs["silu"])
        self.assertFalse(diagnostic_kwargs["quantize"])
        self.assertTrue(original_kwargs["silu"])
        self.assertTrue(original_kwargs["quantize"])
        self.assertLess(diagnostic_index, original_index)
        one = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320,
                          capture_boundaries=["block0-qkv-half"])
        self.assertEqual(list(one.boundaries), ["block0-qkv-half"])

    def test_grouped_split_and_partitioned_detail_captures(self):
        expected = {}
        for block, rows, channels, expanded in ((5, 6400, 64, 256), (23, 144, 512, 2048), (31, 64, 1024, 4096)):
            stages = {"input-state-e4": channels, "ffn-expand-e4": expanded,
                      "ffn-contract-e4": channels, "qkv-half": 3 * channels,
                      "attended-e4": channels, "projection-e4": channels}
            if block in (5, 23):
                stages["ffn-narrow-e4"] = channels
            if block == 23:
                stages["ffn-branch-e4"] = channels
            if block == 31:
                stages.update({"ffn-expand-pre-half": expanded, "ffn-contract-half": channels,
                               "projection-half": channels})
            expected.update({f"block{block}-{name}": (rows, width) for name, width in stages.items()})
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320,
                            mode="legacy_belvedere", capture_boundaries=list(expected))
        graph.features.initialized = True
        graph.record()
        self.assertEqual(set(graph.boundaries), set(expected))
        for name, shape in expected.items():
            self.assertEqual(graph.boundaries[name].shape, shape, name)
            self.assertTrue(graph.boundaries[name].initialized)
        operations = dict(zip(graph.operation_names, graph.ops))
        for name, partition in (("block31.w2", 1024), ("block31.qkv", 512), ("block31.projection", 256)):
            self.assertEqual(operations[name][2]["partition"], partition)
        self.assertIsNotNone(operations["block31.w2"][2]["raw"])
        self.assertIsNotNone(operations["block31.projection"][2]["raw"])
        self.assertFalse(operations["block31.w1.preactivation-diagnostic"][2]["silu"])
        self.assertFalse(operations["block31.w1.preactivation-diagnostic"][2]["quantize"])
        baseline = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, mode="legacy_belvedere")
        all_blocks = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320,
                                 mode="legacy_belvedere", capture_boundaries=True)
        self.assertEqual(len(all_blocks.boundaries), 75)
        self.assertEqual([n for n in graph.operation_names if not n.startswith("capture ") and "preactivation-diagnostic" not in n],
                         baseline.operation_names)
        for plain in (baseline, all_blocks):
            self.assertFalse(set(expected).intersection(plain.boundaries))
            self.assertFalse(any("diagnostic" in name for name in plain.buffers))


if __name__ == "__main__":
    unittest.main()
