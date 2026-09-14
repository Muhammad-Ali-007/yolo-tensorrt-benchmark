"""INT8 entropy calibrator used by build_engine.py.

Feeds batches of preprocessed calibration images to TensorRT's builder so it
can compute per-tensor dynamic ranges for INT8 quantization. Caches the
resulting calibration table so re-building an engine later doesn't require
re-running calibration from scratch.
"""
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

try:
    import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
    import pycuda.driver as cuda
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pycuda is required for INT8 calibration on-device. "
        "Install it on the Jetson with `pip install pycuda`."
    ) from e


def preprocess(image_path: str, input_size: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read {image_path}")
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(img)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir: str, input_size: int, batch_size: int,
                 cache_path: str, max_images: int = 500):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.cache_path = cache_path

        exts = (".jpg", ".jpeg", ".png")
        self.image_paths = [
            str(p) for p in sorted(Path(calib_dir).iterdir())
            if p.suffix.lower() in exts
        ][:max_images]
        if not self.image_paths:
            raise ValueError(f"no calibration images found in {calib_dir}")

        self.batch_idx = 0
        self.n_batches = len(self.image_paths) // batch_size
        nbytes = batch_size * 3 * input_size * input_size * np.float32().itemsize
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.n_batches:
            return None

        start = self.batch_idx * self.batch_size
        paths = self.image_paths[start:start + self.batch_size]
        batch = np.stack([preprocess(p, self.input_size) for p in paths])
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))

        self.batch_idx += 1
        print(f"  calibration batch {self.batch_idx}/{self.n_batches}")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_path).exists():
            return Path(self.cache_path).read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        Path(self.cache_path).write_bytes(cache)
