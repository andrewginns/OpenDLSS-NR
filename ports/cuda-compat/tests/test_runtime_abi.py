"""Native ABI and ownership contracts without loading CUDA."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from opendlss_cuda import runtime as module


class Function:
    def __init__(self, call): self.call = call
    def __call__(self, *args): return self.call(*args)


class FakeLibrary:
    def __init__(self):
        self.calls = []
        self.functions = {}
        self.missing = set()
        self.abi = 1
        self.source = 'a' * 64
        self.build = {'source_sha256': self.source}
        self.device = (0, 7, 5)
        self.policy = None
        self.fused = None
        self.create_error = False
        self.destroy_error = False
        self.ownership_error = False
        self.capture_end_error = False
        self.capturing = False
        self.destroyed_capture = False
        self.next_pointer = 100
        self.message = b'injected native error'

    def __getattr__(self, name):
        if not name.startswith('nr_') or name in self.missing:
            raise AttributeError(name)
        if name not in self.functions:
            self.functions[name] = Function(lambda *args: self.call(name, *args))
        return self.functions[name]

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == 'nr_abi_version': return self.abi
        if name == 'nr_source_sha256': return self.source.encode()
        if name == 'nr_build_info': return json.dumps(self.build).encode()
        if name == 'nr_error': return self.message
        if name == 'nr_check_context': return -1 if self.ownership_error else 0
        if name == 'nr_create':
            self.policy = args[1] if self.policy is None else self.policy
            self.fused = args[2] if self.fused is None else self.fused
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = 42
            return -1 if self.create_error else 0
        if name == 'nr_device_info':
            for pointer, value in zip(args[1:], self.device):
                C.cast(pointer, C.POINTER(C.c_int))[0] = value
        if name == 'nr_context_options':
            C.cast(args[1], C.POINTER(C.c_int))[0] = self.policy
            C.cast(args[2], C.POINTER(C.c_int))[0] = self.fused
        if name == 'nr_alloc':
            self.next_pointer += 100
            C.cast(args[1], C.POINTER(C.c_void_p))[0] = self.next_pointer
        if name == 'nr_allocated': C.cast(args[1], C.POINTER(C.c_size_t))[0] = 1234
        if name == 'nr_capture_begin': self.capturing = True
        if name == 'nr_capture_end':
            self.capturing = False
            return -1 if self.capture_end_error else 0
        if name == 'nr_destroy':
            self.destroyed_capture = self.capturing
            self.capturing = False
            return -1 if self.destroy_error else 0
        return 0


class RuntimeABITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'libnr_cuda.so'
        self.path.write_bytes(b'fake library identity')

    def make(self, fake=None, **options):
        fake = FakeLibrary() if fake is None else fake
        with mock.patch.object(module.C, 'CDLL', return_value=fake):
            result = module.Runtime(self.path, **options)
        self.addCleanup(result.close)
        return result, fake

    def test_only_public_policy_and_explicit_window(self):
        with mock.patch.object(module.C, 'CDLL') as loader:
            for policy in ('f13browser', 'f13', 'group16', 'fp32browser', None, ''):
                with self.assertRaisesRegex(ValueError, 'Only browser-compat-v1'):
                    module.Runtime(self.path, policy=policy)
            with self.assertRaisesRegex(ValueError, 'window_impl'):
                module.Runtime(self.path, window_impl='auto')
            loader.assert_not_called()
        with mock.patch.dict(os.environ, {'NR_WINDOW_IMPL': 'invalid', 'NR_FP16_LIBRARY': 'unrelated'}):
            runtime, fake = self.make(window_impl='staged')
        self.assertEqual(runtime.context_options, {'policy': 'browser-compat-v1', 'native_policy_id': 6,
                         'abi_version': 1, 'window_impl': 'staged'})
        self.assertEqual(next(args[1:] for name, args in fake.calls if name == 'nr_create'), (6, 0))

    def test_abi_and_identity_rejection_precede_creation(self):
        cases = []
        fake = FakeLibrary(); fake.abi = 2; cases.append(fake)
        fake = FakeLibrary(); fake.missing.add('nr_abi_version'); cases.append(fake)
        fake = FakeLibrary(); fake.source = ''; cases.append(fake)
        fake = FakeLibrary(); fake.build = {}; cases.append(fake)
        fake = FakeLibrary(); fake.missing.add('nr_context_options'); cases.append(fake)
        for fake in cases:
            with self.subTest(fake=fake), mock.patch.object(module.C, 'CDLL', return_value=fake):
                with self.assertRaises(RuntimeError): module.Runtime(self.path)
            self.assertNotIn('nr_create', [name for name, _ in fake.calls])

    def test_receipts_describe_actual_file_and_context(self):
        fake = FakeLibrary(); fake.device = (2, 7, 5)
        runtime, _ = self.make(fake)
        self.assertEqual(runtime.library_path, self.path.resolve())
        self.assertEqual(runtime.library_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(runtime.source_sha256, fake.source)
        self.assertEqual(runtime.allocated_bytes, 1234)
        receipt = runtime.device_info; receipt['compute_capability'][0] = 9
        runtime.close()
        self.assertEqual(runtime.device_info, {'ordinal': 2, 'compute_capability': [7, 5]})

    def test_creation_identity_mismatch_closes_context(self):
        for attribute, value in [('device', (0, 8, 9)), ('policy', 7), ('fused', 0)]:
            fake = FakeLibrary(); setattr(fake, attribute, value)
            with mock.patch.object(module.C, 'CDLL', return_value=fake):
                with self.assertRaises(RuntimeError): module.Runtime(self.path)
            self.assertEqual(sum(name == 'nr_destroy' for name, _ in fake.calls), 1)

    def test_context_thread_rejection_keeps_pointer_and_never_calls_ffi(self):
        runtime, fake = self.make()
        before = len(fake.calls); errors = []
        def another_thread():
            for operation in (runtime.sync, runtime.close, lambda: runtime.empty((1,))):
                try: operation()
                except RuntimeError as error: errors.append(str(error))
        thread = threading.Thread(target=another_thread); thread.start(); thread.join()
        self.assertEqual(len(errors), 3)
        self.assertTrue(all('another host thread' in e for e in errors))
        self.assertEqual(len(fake.calls), before)
        self.assertEqual(runtime.context.value, 42)
        runtime.sync()

    def test_device_ownership_rejection_keeps_pointer(self):
        runtime, fake = self.make(); fake.ownership_error = True
        with self.assertRaisesRegex(RuntimeError, 'injected'): runtime.close()
        self.assertEqual(runtime.context.value, 42)
        self.assertNotIn('nr_destroy', [name for name, _ in fake.calls])
        fake.ownership_error = False
        runtime.close()

    def test_cleanup_failure_consumes_pointer_exactly_once(self):
        runtime, fake = self.make(); fake.destroy_error = True
        with self.assertRaisesRegex(RuntimeError, 'injected'): runtime.close()
        self.assertIsNone(runtime.context.value)
        runtime.close()
        self.assertEqual(sum(name == 'nr_destroy' for name, _ in fake.calls), 1)

    def test_partial_creation_and_capture_cleanup_preserve_first_error(self):
        fake = FakeLibrary(); fake.create_error = fake.destroy_error = True
        with mock.patch.object(module.C, 'CDLL', return_value=fake):
            with self.assertRaises(RuntimeError) as caught: module.Runtime(self.path)
        self.assertEqual(sum(name == 'nr_destroy' for name, _ in fake.calls), 1)
        self.assertTrue(caught.exception.__notes__)
        runtime, fake = self.make(); fake.destroy_error = True
        original = ValueError('operator failed')
        with self.assertRaises(ValueError) as caught:
            with runtime:
                runtime.capture_begin()
                raise original
        self.assertIs(caught.exception, original)
        self.assertTrue(fake.destroyed_capture)
        self.assertIsNone(runtime.context.value)

    def test_capture_host_access_and_failure_state(self):
        runtime, fake = self.make(); tensor = runtime.empty((2,))
        runtime.capture_begin()
        for call in (lambda: runtime.empty((2,)), lambda: runtime.download(tensor),
                     lambda: runtime.copy_from_host(np.zeros(2), tensor)):
            with self.assertRaisesRegex(RuntimeError, 'capture'): call()
        fake.capture_end_error = True
        with self.assertRaises(RuntimeError): runtime.capture_end()
        self.assertFalse(runtime._capturing)

    def test_shape_validation_precedes_native_allocation(self):
        runtime, fake = self.make(); before = len(fake.calls)
        for shape in ((), (0,), (-1,), (True,), (1.5,), (2**62, 2**62)):
            with self.assertRaises(ValueError): runtime.empty(shape)
        with self.assertRaises(ValueError): runtime.empty((1,), 'int32')
        self.assertEqual(len(fake.calls), before)

    def test_closed_and_foreign_tensor_rejected_before_ffi(self):
        runtime, fake = self.make(); tensor = runtime.empty((16, 16))
        other, _ = self.make()
        with self.assertRaisesRegex(ValueError, 'another runtime'): other.download(tensor)
        runtime.close(); before = len(fake.calls)
        for operation in (runtime.sync, runtime.capture_begin, runtime.capture_end, runtime.replay,
                          runtime.timer_start, runtime.timer_stop, lambda: runtime.allocated_bytes,
                          lambda: runtime.empty((1,)), lambda: runtime.download(tensor),
                          lambda: runtime.gemm(tensor, tensor, 16, 16, 16, tensor)):
            with self.assertRaisesRegex(RuntimeError, 'closed'): operation()
        runtime.close()
        self.assertEqual(len(fake.calls), before)


if __name__ == '__main__': unittest.main()
