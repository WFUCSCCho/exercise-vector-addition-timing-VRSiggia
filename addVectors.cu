/*
 * Sample program that uses CUDA to perform element-wise add of two
 * vectors.  Each element is the responsibility of a separate thread.
 *
 * compile with:
 *    nvcc -o addVectors addVectors.cu
 * run with:
 *    ./addVectors
 */

#include <stdio.h>

// Helper macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define N 10

__global__ void kernel(int* res, int* a, int* b) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id < N) {
        res[thread_id] = a[thread_id] + b[thread_id];
    }
}

int main() {
    int *a, *b, *res;
    int *dev_a, *dev_b, *dev_res;
    size_t size = N * sizeof(int);

    // Host Allocation
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    res = (int*)malloc(size);

    // Device Allocation with Error Checking
    CHECK_CUDA(cudaMalloc((void**)&dev_a, size));
    CHECK_CUDA(cudaMalloc((void**)&dev_b, size));
    CHECK_CUDA(cudaMalloc((void**)&dev_res, size));

    for (int i = 0; i < N; i++) a[i] = b[i] = i;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Transfer data to GPU
    CHECK_CUDA(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

    // Measure only the kernel execution
    cudaEventRecord(start);
    int threads = 512;
    int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(dev_res, dev_a, dev_b);
    cudaEventRecord(stop);

    // Transfer result back
    CHECK_CUDA(cudaMemcpy(res, dev_res, size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // verify results
    for(int i=0; i < N; i++)
      printf("%d ", res[i]);
    printf("\n");

    // free the memory (because I care)
    free(a);
    free(b);
    free(res);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    return 0;
}