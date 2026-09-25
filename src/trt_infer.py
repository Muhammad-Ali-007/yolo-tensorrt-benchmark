"""Minimal TensorRT engine runner: load engine, allocate I/O buffers, run
inference, return outputs plus device-side timing.
Kept deliberately dependency-light (raw pycuda, no wrapper library) so it's
obvious exactly what's happening at each step — this is the part of the repo
meant to demonstrate you understand the mechanics, not just call a helper.

NOTE on TensorRT API version: this targets the tensor-name API introduced in
TensorRT 8.5 and mandatory from 10.x on (get_tensor_name / get_tensor_mode /
set_tensor_address / execute_async_v3). The older binding-index API
(get_binding_name / binding_is_input / execute_async_v2) was *removed* in
TensorRT 10 — it is not merely deprecated — so this file will not run on a
JetPack old enough to predate 8.5. Check with:
    python3 -c "import tensorrt; print(tensorrt.__version__)"
The logic is identical either way: allocate host+device buffers per I/O
tensor, copy in, execute, copy out.
"""

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInfer:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]

        # Pass 1: pin down every input shape. Engines exported with
        # dynamic=False are already static, but if a dimension is -1 we fall
        # back to the optimization profile's max so the buffers we allocate
        # are always big enough. Output shapes are only resolvable *after*
        # this, which is why it's split into two passes.
        for name in names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
            self.context.set_input_shape(name, shape)

        # Pass 2: allocate host+device memory per tensor and hand TensorRT the
        # device pointers once. Addresses stay valid for the object's life, so
        # infer() never has to re-bind anything.
        self.inputs, self.outputs = [], []
        for name in names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.context.set_tensor_address(name, int(device_mem))

            entry = {"name": name, "shape": shape, "dtype": dtype,
                     "host": host_mem, "device": device_mem}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

        if not self.inputs:
            raise RuntimeError(f"engine {engine_path} exposes no input tensors")

    def infer(self, input_array: np.ndarray, timing_event_pair=None):
        """Run one forward pass. `input_array` must already be preprocessed
        and shaped to match the engine's expected input (NCHW, float32,
        normalized 0-1). Returns a list of output numpy arrays, in the same
        order as `self.outputs`.

        If `timing_event_pair` is a (start, end) tuple of cuda.Event, they're
        recorded around just the GPU execution (excludes H2D/D2H copy), which
        is what benchmark.py uses for the "pure inference" latency number.
        """
        inp = self.inputs[0]
        if input_array.size != inp["host"].size:
            raise ValueError(
                f"input has {input_array.size} elements, engine expects "
                f"{inp['host'].size} (shape {inp['shape']})"
            )
        np.copyto(inp["host"], input_array.ravel().astype(inp["dtype"], copy=False))
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        if timing_event_pair:
            timing_event_pair[0].record(self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        if timing_event_pair:
            timing_event_pair[1].record(self.stream)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return [out["host"].reshape(out["shape"]).copy() for out in self.outputs]

    def input_shape(self):
        return self.inputs[0]["shape"]
