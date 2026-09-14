"""Minimal TensorRT engine runner: load engine, allocate I/O buffers, run
inference, return outputs plus device-side timing.

Kept deliberately dependency-light (raw pycuda, no wrapper library) so it's
obvious exactly what's happening at each step — this is the part of the repo
meant to demonstrate you understand the mechanics, not just call a helper.

NOTE on TensorRT API version: this uses the "legacy" binding-index API
(get_binding_name / binding_is_input / execute_async_v2), which is what
ships with older JetPack releases. TensorRT 8.5+ deprecates these in favor
of the tensor-name API (get_tensor_name / get_tensor_mode / execute_async_v3).
Check `python3 -c "import tensorrt; print(tensorrt.__version__)"` on your
Jetson — if you're on 10.x you'll likely need to port this to the newer API;
the logic (allocate host+device buffers per binding, copy in, execute, copy
out) stays identical either way.
"""
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInfer:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            entry = {"name": name, "shape": shape, "dtype": dtype,
                     "host": host_mem, "device": device_mem}
            if self.engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def infer(self, input_array: np.ndarray, timing_event_pair=None):
        """Run one forward pass. `input_array` must already be preprocessed
        and shaped to match the engine's expected input (NCHW, float32,
        normalized 0-1). Returns a list of output numpy arrays, in the same
        order as `self.outputs`.

        If `timing_event_pair` is a (start, end) tuple of cuda.Event, they're
        recorded around just the GPU execution (excludes H2D/D2H copy), which
        is what benchmark.py uses for the "pure inference" latency number.
        """
        np.copyto(self.inputs[0]["host"], input_array.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        if timing_event_pair:
            timing_event_pair[0].record(self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        if timing_event_pair:
            timing_event_pair[1].record(self.stream)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [out["host"].reshape(out["shape"]).copy() for out in self.outputs]

    def input_shape(self):
        return self.inputs[0]["shape"]
