#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,   \
                    __LINE__, cudaGetErrorString(_err));                       \
            return;                                                            \
        }                                                                      \
    } while (0)

__global__ void vecAdd(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c,
                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void run_vecadd(const float* a,
                           const float* b,
                           float* c,
                           int n,
                           float* kernel_ms_out) {
    if (kernel_ms_out) *kernel_ms_out = -1.0f;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = (size_t)n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (kernel_ms_out) *kernel_ms_out = ms;

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
