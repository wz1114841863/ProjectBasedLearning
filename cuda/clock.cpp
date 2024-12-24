#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <nvrtc_helper.h>
#include <helper_functions.h>

#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(int argc, char **argv) {
    typedef long clock_t;
    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];
    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    char *cubin, *kernel_file;
    size_t cubin_size;
    kernel_file = sdkFindFilePath("clock_kernel.cu", argv[0]);
    compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubin_size, 0);

    CUmodule module = loadCUBIN(cubin, argc, argv);
    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "timedReduction"));

    dim3 cuda_block_size(NUM_THREADS, 1, 1);
    dim3 cuda_grid_size(NUM_BLOCKS, 1, 1);

    CUdeviceptr dev_input, dev_output, dev_timer;
    checkCudaErrors(cuMemAlloc(&dev_input, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cuMemAlloc(&dev_output, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cuMemAlloc(&dev_timer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cuMemcpyHtoD(dev_input, input, sizeof(float) * NUM_THREADS * 2));

    void *arr[] = {
        (void *)&dev_input,
        (void *)&dev_output,
        (void *)&dev_timer
    };

    checkCudaErrors(cuLaunchKernel(
        kernel_addr,
        cuda_grid_size.x, cuda_grid_size.y, cuda_grid_size.z,
        cuda_block_size.x, cuda_block_size.y, cuda_block_size.y,
        sizeof(float) * 2 * NUM_THREADS,
        0,
        &arr[0],
        0
    ));

    checkCudaErrors(cuCtxSynchronize());
    checkCudaErrors(cuMemcpyDtoH(timer, dev_timer, sizeof(clock_t) * NUM_BLOCKS * 2));
    checkCudaErrors(cuMemFree(dev_input));
    checkCudaErrors(cuMemFree(dev_output));
    checkCudaErrors(cuMemFree(dev_timer));

    long double avg_elapsed_clocks = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        avg_elapsed_clocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }

    avg_elapsed_clocks = avg_elapsed_clocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avg_elapsed_clocks);

    return EXIT_SUCCESS;
}
