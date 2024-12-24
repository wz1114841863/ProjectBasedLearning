#include <stdio.h>
#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <helper_functions.h>

namespace cg = cooperative_groups;

__global__ void clock_block(clock_t *d_o, clock_t clock_count) {
    unsigned int start_clock = (unsigned int)clock();
    clock_t clock_offset = 0;

    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int)clock();
        clock_offset = (clock_t)(end_clock - start_clock);
    }
    d_o[0] = clock_offset;
}

__global__ void sum(clock_t *d_clocks, int N) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    cg::sync(cta);

    for (int i = 16; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }
        cg::sync(cta);
    }
    d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
    int nkernels = 8;             // number of concurrent kernels
    int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
    int nbytes = nkernels * sizeof(clock_t);  // number of data bytes
    float kernel_time = 10;                   // time the kernel should run in ms
    float elapsed_time;                       // timing variables
    int cuda_device = 0;

    printf("[%s] - Starting...\n", argv[0]);

    // get number of kernels if overridden on the command line
    if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
        nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
        nstreams = nkernels + 1;
    }

    cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp device_prop;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, cuda_device));
    if ((device_prop.concurrentKernels == 0)) {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
        device_prop.major, device_prop.minor, device_prop.multiProcessorCount);

    clock_t *a = 0;
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    clock_t *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    cudaEvent_t *kernel_event;
    kernel_event = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));
    for (int i = 0; i < nkernels; ++i) {
        checkCudaErrors(cudaEventCreateWithFlags(&(kernel_event[i]), cudaEventDisableTiming));
    }

    // time execution with nkernels streams
    clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
    // the kernel takes more time than the channel reset time on arm archs, so to
    // prevent hangs reduce time_clocks.
    clock_t time_clocks = (clock_t)(kernel_time * (device_prop.clockRate / 100));
#else
    clock_t time_clocks = (clock_t)(kernel_time * device_prop.clockRate);
#endif

    cudaEventRecord(start_event, 0);
    // queue nkernels in separate streams and record when they are done.
    for (int i = 0; i < nkernels; ++i) {
        clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
        total_clocks += time_clocks;
        checkCudaErrors(cudaEventRecord(kernel_event[i], streams[i]));
        checkCudaErrors(cudaStreamWaitEvent(streams[nstreams - 1], kernel_event[i], 0));
    }

    sum<<<1, 32, 0, streams[nstreams - 1]>>>(d_a, nkernels);
    checkCudaErrors(cudaMemcpyAsync(
        a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams - 1]));

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
            nkernels * kernel_time / 1000.0f);
    printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
            nkernels, kernel_time / 1000.0f);
    printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

    bool test_result = (a[0] > total_clocks);

    for (int i = 0; i < nkernels; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernel_event[i]);
    }

    free(streams);
    free(kernel_event);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(a);
    cudaFree(d_a);

    if (!test_result) {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
