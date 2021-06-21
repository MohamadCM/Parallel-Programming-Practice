#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define N 50000
#define block_size 32

cudaError_t reduceWithCuda(int *input, int *output, unsigned int size);

__global__ void reduce0(int* g_idata, int* g_odata) {
    extern __shared__ int sdata[block_size];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    int a[N];
    for (size_t i = 0; i < N; i++) {
        a[i] = i + 1;
    }
    int c[N] = { 0 };

    // Reduce vector in parallel.
    cudaError_t cudaStatus = reduceWithCuda(a, c, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduce0Kernel failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t reduceWithCuda(int *input, int *output, unsigned int size)
{
    unsigned __int64 startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    int *dev_i = 0;
    int *dev_o = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_o, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_i, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_i, input, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    const int num_blocks = (size / block_size) + ((size % block_size) ? 1 : 0);
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    // Launch a kernel on the GPU with one thread for each element.
    reduce0<<<num_blocks, block_size >>>(dev_i, dev_o);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduce0Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduce0Kernel!\n", cudaStatus);
        goto Error;
    }
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    // float msecTotal = 0.0f;
    // cudaEventElapsedTime(&msecTotal, start, stop);
    

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_o, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    int sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += output[i];
    }
    printf("Reduce calculation result = %d\n", sum);
    // printf("Kernel Elapsed time in milliseconds = %f\n", msecTotal);
    unsigned __int64 endTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("Full elapsed time in milliseconds = %d\n", endTime - startTime);
Error:
    cudaFree(dev_o);
    cudaFree(dev_i);
    
    return cudaStatus;
}
