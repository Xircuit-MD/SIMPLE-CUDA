"""Benchmark the deployed RunPod serverless endpoint.

Sends N synchronous requests to the endpoint and reports cold-start vs
warm-call statistics (kernel_ms, wall_ms inside the handler, and client
round-trip time).

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python bench/bench_client.py --n 1048576 --runs 10 --wait-cold 60

The first request right after `--wait-cold` seconds of idle is treated as
the cold sample; the rest are warm.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import runpod


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs into environment without overriding existing vars."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1 << 20, help="vector length")
    p.add_argument("--runs", type=int, default=10, help="total requests")
    p.add_argument(
        "--wait-cold",
        type=int,
        default=0,
        help="seconds to sleep before the first request to force a cold start "
             "(set to slightly more than the endpoint's idle timeout)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    median = statistics.median(values)
    p95 = (
        statistics.quantiles(values, n=20)[-1]
        if len(values) >= 2
        else values[0]
    )
    return f"mean={mean:8.2f}ms  median={median:8.2f}ms  p95={p95:8.2f}ms  n={len(values)}"


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        print("ERROR: set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID", file=sys.stderr)
        return 2

    runpod.api_key = api_key
    endpoint = runpod.Endpoint(endpoint_id)

    if args.wait_cold > 0:
        print(f">> Sleeping {args.wait_cold}s to let workers idle out (force cold)...")
        time.sleep(args.wait_cold)

    cold_rt: list[float] = []
    cold_kernel: list[float] = []
    cold_wall: list[float] = []
    warm_rt: list[float] = []
    warm_kernel: list[float] = []
    warm_wall: list[float] = []

    print(f">> Sending {args.runs} requests (n={args.n}) to endpoint {endpoint_id}")
    print(f"{'i':>3} {'cold':>5} {'kernel_ms':>11} {'wall_ms':>10} {'rt_ms':>10} {'uptime_s':>10}")

    for i in range(args.runs):
        payload = {"input": {"n": args.n, "seed": args.seed + i}}
        t0 = time.perf_counter()
        try:
            out = endpoint.run_sync(payload, timeout=600)
        except Exception as exc:
            print(f"{i:>3}  ERROR: {exc}")
            continue
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
    print("=== Summary ===")
    print(f"COLD  round-trip : {fmt_ms(cold_rt)}")
    print(f"COLD  kernel     : {fmt_ms(cold_kernel)}")
    print(f"COLD  handler    : {fmt_ms(cold_wall)}")
    print(f"WARM  round-trip : {fmt_ms(warm_rt)}")
    print(f"WARM  kernel     : {fmt_ms(warm_kernel)}")
    print(f"WARM  handler    : {fmt_ms(warm_wall)}")
    print()
    print("After this run, open RunPod console -> Serverless -> your endpoint -> ")
    print("Billing tab to read total billed seconds and $/hr for the chosen GPU.")
    print("Then: cost_per_call = billed_seconds * (price_per_hr / 3600)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
