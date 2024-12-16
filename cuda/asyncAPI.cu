#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void increment_kernel(int *g_data, int inc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x) {
    for (int i = 0; i < n; ++i) {
        if (data[i] != x) {
                printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
                return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int dev_id;
    cudaDeviceProp device_props;
    printf("[%s] - Starting... \n", argv[0]);

    dev_id = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_props, dev_id));
    printf("CUDA device [%s]\n", device_props.name);

    int n = 16 * 1024 * 1024;
    int num_bytes = n * sizeof(int);
    int value = 26;

    int *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, num_bytes));
    memset(a, 0, num_bytes);

    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, num_bytes));
    checkCudaErrors(cudaMemset(d_a, 255, num_bytes));

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(num_bytes / threads.x, 1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // profiler 用于对代码进行性能分析
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    // 在默认流中依次注册: start -> memcpy -> kernel -> memcpy -> stop
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, num_bytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, num_bytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        ++counter;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
            counter);

    // check the output for correctness
    bool results = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(results ? EXIT_SUCCESS : EXIT_FAILURE);
}
