# CUDA Vector-Add on RunPod Serverless

A minimal end-to-end example: a hand-written CUDA `vecAdd` kernel, compiled
with `nvcc` into a shared library, called from a RunPod serverless Python
handler via `ctypes`, packaged in Docker, and benchmarked for cold-start /
warm-call latency and cost.

```
SIMPLE-CUDA/
  cuda/vec_add.cu        # __global__ vecAdd + extern C wrapper, CUDA-event timing
  src/handler.py         # RunPod handler, loads libvecadd.so via ctypes
  bench/bench_client.py  # hits endpoint N times, splits cold vs warm stats
  Dockerfile             # nvidia/cuda:12.4.1-devel, builds libvecadd.so at image build
  requirements.txt
  build.sh               # docker build + push helper
```

## Architecture

```
client (bench_client.py)
   |  HTTPS run_sync
   v
RunPod Serverless endpoint
   |  schedules onto a worker
   v
Docker container
   |  python3 handler.py
   v
ctypes -> libvecadd.so -> CUDA kernel on GPU
```

`handler.py` loads `libvecadd.so` at *module import* time, so that work happens
once per worker process. The first call on a worker is reported as
`cold: true`; later calls reuse the same loaded library and are `cold: false`.

## 1. Local build + push

You need Docker + a Docker Hub account (or any registry RunPod can pull from).
RunPod serverless GPUs are `linux/amd64`, so we force that platform on
Apple Silicon hosts.

```bash
docker login

DOCKERHUB_USER=youruser ./build.sh
# -> youruser/cuda-vecadd:latest
```

Or manually:

```bash
docker build --platform linux/amd64 -t youruser/cuda-vecadd:latest .
docker push youruser/cuda-vecadd:latest
```

You do *not* need a GPU on the build machine; `nvcc` only compiles, it does
not run the kernel during the build.

## 2. Create the RunPod Serverless endpoint (console)

1. Open <https://console.runpod.io/serverless>.
2. Click **New Endpoint**.
3. Source:
   - Select **Custom Source** -> **Docker Image**.
   - Container image: `youruser/cuda-vecadd:latest`.
   - Container start command: leave blank (the Dockerfile `CMD` runs the handler).
4. GPU configuration:
   - Pick the cheapest GPU available, e.g. **RTX A4000**, **L4**, or **A5000**.
     Vector-add is memory-bandwidth bound on tiny inputs, so the cheapest
     card is fine.
   - Workers: **Min 0**, **Max 1**.
   - Idle timeout: **5 seconds** (low value forces cold starts so we can
     measure them).
   - Container disk: **5 GB** is plenty.
   - Execution timeout: **60 seconds**.
5. Hit **Deploy**.
6. Copy the **Endpoint ID** and create an API key under
   *Settings -> API Keys*.

## 3. Run the benchmark

From your local machine:

```bash
python -m venv .venv && source .venv/bin/activate
pip install runpod numpy

export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...

# Force a cold start (idle timeout is 5s, so 30s of sleep is more than enough),
# then run 10 requests back-to-back.
python bench/bench_client.py --n 1048576 --runs 10 --wait-cold 30
```

You can also create a local `.env` file in the repo root:

```bash
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
```

`bench/bench_client.py` auto-loads `.env` if present.

You'll get a per-request log plus a summary like:

```
=== Summary ===
COLD  round-trip : mean=  ...ms  median=  ...ms  p95=  ...ms  n=1
COLD  kernel     : mean=  ...ms  ...
COLD  handler    : mean=  ...ms  ...
WARM  round-trip : mean=  ...ms  ...
WARM  kernel     : mean=  ...ms  ...
WARM  handler    : mean=  ...ms  ...
```

What each metric means:

| metric | source | what it captures |
|---|---|---|
| `kernel_ms` | `cudaEventRecord` inside `run_vecadd` | pure GPU kernel time |
| `wall_ms` | `time.perf_counter` around the ctypes call in the handler | kernel + H2D/D2H memcpy + cudaMalloc/Free |
| round-trip (`rt_ms`) | client clock around `endpoint.run_sync` | network + RunPod queue + worker spin-up + handler |
| `cold` | module-level flag in handler | first job on this worker process |

## 4. Reading runtime + cost from RunPod

After the bench run, in the RunPod console open your endpoint and look at:

- **Metrics / Requests** tab: per-request execution time (this is what RunPod
  bills you for) and number of cold starts.
- **Billing** tab (or the workspace-level *Billing*): total GPU-seconds used by
  this endpoint, plus the `$/hr` rate for the GPU type you selected.

Then compute per-call cost:

```
cost_per_call_usd = billed_seconds_per_call * (price_per_hr_usd / 3600)
```

Worked example (numbers are illustrative; replace with what your dashboard
shows):

| GPU | $/hr | warm exec time | warm cost/call | cold exec time | cold cost/call |
|---|---|---|---|---|---|
| RTX A4000 | $0.17 | 0.20 s | $0.0000094 | 4.5 s | $0.000213 |
| L4        | $0.43 | 0.18 s | $0.0000215 | 5.0 s | $0.000597 |

So for tiny workloads like vector-add, **cold starts dominate cost**, not the
kernel itself. To make per-call cost cheaper, raise `Min workers` to 1 (you
pay idle time instead) or raise `Idle timeout` to keep the worker alive
between calls.

## 5. Local sanity check (optional)

If you have a CUDA-capable Linux box with `nvcc`, you can compile and call
the library directly without RunPod:

```bash
nvcc -O3 -Xcompiler -fPIC -shared cuda/vec_add.cu -o libvecadd.so
VECADD_LIB=$PWD/libvecadd.so python -c "
import os, src.handler as h
print(h.handler({'input': {'n': 1<<20}}))
"
```

## Notes / scope

- No PyCUDA / CuPy / Torch — pure `nvcc` + `ctypes`.
- No CI, no GitHub auto-deploy — endpoint is created once via the console.
- Single GPU, synchronous handler only.
