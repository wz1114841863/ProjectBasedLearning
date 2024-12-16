#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define NUM_BLOCKS 64
#define NUM_THREADS 256

__global__ static void timedReduction(const float *input, float *output, clock_t *timer) {
    // 动态共享内存(大小由启动配置决定)
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (tid == 0) timer[bid] = clock();

    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    for (int d = blockDim.x; d > 0; d /= 2) {
        // 归约操作
        __syncthreads();
        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0) {
                shared[tid] = f1;
            }
        }
    }

    if (tid == 0) output[bid] = shared[0];
    __syncthreads();
    if (tid == 0) timer[bid + gridDim.x] = clock();
}

int main(int argc, char **argv) {
    int dev_id = findCudaDevice(argc, (const char **)argv);
    float *dev_input = NULL;
    float *dev_output = NULL;
    clock_t *dev_timer = NULL;
    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    checkCudaErrors(cudaMalloc((void **)&dev_input, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cudaMalloc((void **)&dev_output, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cudaMalloc((void **)&dev_timer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cudaMemcpy(dev_input, input,
        sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));

    timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dev_input, dev_output, dev_timer);

    checkCudaErrors(cudaMemcpy(input, dev_input,
        sizeof(float) * NUM_THREADS * 2, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(timer, dev_timer, sizeof(clock_t) * NUM_BLOCKS * 2,
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_input));
    checkCudaErrors(cudaFree(dev_output));
    checkCudaErrors(cudaFree(dev_timer));

    long double avg_elapsed_clocks = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        avg_elapsed_clocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }

    avg_elapsed_clocks = avg_elapsed_clocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avg_elapsed_clocks);

    return EXIT_SUCCESS;
}
