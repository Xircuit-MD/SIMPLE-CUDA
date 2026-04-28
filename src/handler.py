"""RunPod serverless handler for CUDA vector-add.

Loads the nvcc-compiled shared library at module import (so subsequent
invocations on the same worker are "warm") and exposes a single handler
that runs vector addition on the GPU and returns timing info.
"""

import ctypes
import os
import time

import numpy as np
import runpod

LIB_PATH = os.environ.get("VECADD_LIB", "/app/libvecadd.so")

_lib = ctypes.CDLL(LIB_PATH)
_lib.run_vecadd.restype = None
_lib.run_vecadd.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]

WORKER_BOOT_TS = time.time()
_first_job_done = False


def _run_vecadd(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    n = a.size
    c = np.empty(n, dtype=np.float32)
    kernel_ms = ctypes.c_float(-1.0)

    a_p = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_p = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_p = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _lib.run_vecadd(a_p, b_p, c_p, ctypes.c_int(n), ctypes.byref(kernel_ms))
    return c, float(kernel_ms.value)


def handler(job):
    global _first_job_done

    job_input = job.get("input", {}) or {}
    n = int(job_input.get("n", 1 << 20))
    seed = int(job_input.get("seed", 0))

    if n <= 0 or n > (1 << 28):
        return {"error": f"invalid n={n}; must be in (0, 2^28]"}

    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)

    t0 = time.perf_counter()
    c, kernel_ms = _run_vecadd(a, b)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    cold = not _first_job_done
    _first_job_done = True

    return {
        "n": n,
        "seed": seed,
        "kernel_ms": kernel_ms,
        "wall_ms": wall_ms,
        "cold": cold,
        "worker_uptime_s": time.time() - WORKER_BOOT_TS,
        "checksum": float(c.sum()),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
