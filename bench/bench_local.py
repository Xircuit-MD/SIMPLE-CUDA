"""Benchmark the handler locally (no RunPod, no network).

Calls ``src.handler.handler`` in-process with the same payloads as
``bench_client.py`` and prints cold vs warm stats for kernel_ms, wall_ms,
and client-side round-trip time.

Runtime:
    ``rt_ms`` wraps the full ``handler()`` call (NumPy setup + GPU work).
    ``wall_ms`` / ``kernel_ms`` come from the handler response (GPU path only).

Cost:
    There is no cloud meter locally. See printed note; optional --usd-per-hour
    gives a rough *illustrative* USD estimate if you assign a nominal GPU $/hr.

Usage (from repo root, after ``nvcc ... -o libvecadd.so``):

    python bench/bench_local.py --n 1048576 --runs 10

If ``libvecadd.so`` is not next to ``cuda/``, set ``VECADD_LIB`` to its path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HANDLER_PATH = os.path.join(_REPO_ROOT, "src", "handler.py")


def _load_handler_module():
    spec = importlib.util.spec_from_file_location("bench_local_handler", _HANDLER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_HANDLER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1 << 20, help="vector length")
    p.add_argument("--runs", type=int, default=10, help="total handler calls")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--usd-per-hour",
        type=float,
        default=None,
        metavar="USD",
        help="optional nominal $/GPU-hour for rough cost estimate (local only; no billing)",
    )
    return p.parse_args()


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    median = statistics.median(values)
    p95 = statistics.quantiles(values, n=20)[-1] if len(values) >= 2 else values[0]
    return f"mean={mean:8.2f}ms  median={median:8.2f}ms  p95={p95:8.2f}ms  n={len(values)}"


def main() -> int:
    lib = os.environ.get("VECADD_LIB", os.path.join(_REPO_ROOT, "libvecadd.so"))
    if not os.path.isfile(lib):
        print(
            f"ERROR: VECADD_LIB not found: {lib}\n"
            "Build: nvcc -O3 -Xcompiler -fPIC -shared cuda/vec_add.cu -o libvecadd.so\n"
            "Then: export VECADD_LIB=\"$PWD/libvecadd.so\"",
            file=sys.stderr,
        )
        return 2

    args = parse_args()

    # Handler reads VECADD_LIB at import time; set it before loading.
    os.environ["VECADD_LIB"] = os.path.abspath(lib)
    handler_mod = _load_handler_module()
    handler = handler_mod.handler

    cold_rt: list[float] = []
    cold_kernel: list[float] = []
    cold_wall: list[float] = []
    warm_rt: list[float] = []
    warm_kernel: list[float] = []
    warm_wall: list[float] = []

    print(f">> Local bench: {args.runs} calls (n={args.n}), VECADD_LIB={lib}")
    print(f"{'i':>3} {'cold':>5} {'kernel_ms':>11} {'wall_ms':>10} {'rt_ms':>10} {'uptime_s':>10}")

    for i in range(args.runs):
        job = {"input": {"n": args.n, "seed": args.seed + i}}
        t0 = time.perf_counter()
        out = handler(job)
        rt_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(out, dict) or "kernel_ms" not in out:
            print(f"{i:>3}  unexpected response: {out!r}")
            continue

        cold = bool(out.get("cold", False))
        kernel_ms = float(out["kernel_ms"])
        wall_ms = float(out["wall_ms"])
        uptime = float(out.get("worker_uptime_s", 0.0))

        bucket_rt, bucket_kernel, bucket_wall = (
            (cold_rt, cold_kernel, cold_wall)
            if cold
            else (warm_rt, warm_kernel, warm_wall)
        )
        bucket_rt.append(rt_ms)
        bucket_kernel.append(kernel_ms)
        bucket_wall.append(wall_ms)

        print(
            f"{i:>3} {str(cold):>5} {kernel_ms:>11.3f} {wall_ms:>10.2f} "
            f"{rt_ms:>10.2f} {uptime:>10.2f}"
        )

    print()
    print("=== Summary (local) ===")
    print(f"COLD  round-trip : {fmt_ms(cold_rt)}")
    print(f"COLD  kernel     : {fmt_ms(cold_kernel)}")
    print(f"COLD  handler    : {fmt_ms(cold_wall)}")
    print(f"WARM  round-trip : {fmt_ms(warm_rt)}")
    print(f"WARM  kernel     : {fmt_ms(warm_kernel)}")
    print(f"WARM  handler    : {fmt_ms(warm_wall)}")
    print()
    print(
        "Cost: no cloud billing on a local PC. "
        "RunPod cost needs the deployed endpoint + console billing."
    )
    if args.usd_per_hour is not None and args.usd_per_hour > 0:
        # Illustrative: assume billed time ~= mean warm rt per call (no idle idle tax).
        warm_mean_s = statistics.mean(warm_rt) / 1000.0 if warm_rt else 0.0
        per_call = warm_mean_s * (args.usd_per_hour / 3600.0)
        print(
            f"Illustrative (optional --usd-per-hour={args.usd_per_hour:g}): "
            f"~${per_call:.6f} per warm call if GPU time ≈ warm rt_ms only."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
