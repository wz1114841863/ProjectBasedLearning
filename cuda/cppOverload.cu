#include <stdio.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_math.h>

#define THREAD_N 256
#define N 1024
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

#include "cppOverload_kernel.cuh"

const char *sample_name = "C++ Function Overloading";

#define OUTPUT_ATTR(attr)                                           \
    printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);       \
    printf("Constant Size: %d\n", (int)attr.constSizeBytes);        \
    printf("Local Size:    %d\n", (int)attr.localSizeBytes);        \
    printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock); \
    printf("Number of Registers: %d\n", attr.numRegs);              \
    printf("PTX Version: %d\n", attr.ptxVersion);                   \
    printf("Binary Version: %d\n", attr.binaryVersion);

bool check_func1(int *h_input, int *h_output, int a) {
    for (int i = 0; i < N; ++i) {
        int cpu_res = h_input[i] * a + i;
        if (h_output[i] != cpu_res) {
            return false;
        }
    }
    return true;
}

bool check_func2(int2 *h_input, int *h_output, int a) {
    for (int i = 0; i < N; i++) {
        int cpu_res = (h_input[i].x + h_input[i].y) * a + i;

        if (h_output[i] != cpu_res) {
            return false;
        }
    }
    return true;
}

bool check_func3(int *h_input1, int *h_input2, int *h_output, int a) {
    for (int i = 0; i < N; i++) {
        if (h_output[i] != (h_input1[i] + h_input2[i]) * a + i) {
            return false;
        }
    }
    return true;
}

int main(int argc, const char *argv[]) {
    int *h_input = NULL;
    int *h_output = NULL;
    int *d_input = NULL;
    int *d_output = NULL;

    printf("%s starting...\n", sample_name);

    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    printf("Device Count: %d\n", device_count);

    int device_id = findCudaDevice(argc, argv);
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));
    if (prop.major < 2) {
        printf(
            "ERROR: cppOverload requires GPU devices with compute SM 2.0 or "
            "higher.\n");
        printf("Current GPU device has compute SM%d.%d, Exiting...", prop.major,
            prop.minor);
        exit(EXIT_WAIVED);
    }

    checkCudaErrors(cudaSetDevice(device_id));

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_input, sizeof(int) * N * 2));
    checkCudaErrors(cudaMalloc(&d_output, sizeof(int) * N));

    // Allocate host memory
    checkCudaErrors(cudaMallocHost(&h_input, sizeof(int) * N * 2));
    checkCudaErrors(cudaMallocHost(&h_output, sizeof(int) * N));

    for (int i = 0; i < N * 2; i++) {
        h_input[i] = i;
    }

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(int) * N * 2, cudaMemcpyHostToDevice));

    // Test C++ overloading
    bool test_result = true;
    bool func_result = true;
    int a = 1;

    void (*func1)(const int *, int *, int);
    void (*func2)(const int2 *, int *, int);
    void (*func3)(const int *, const int *, int *, int);
    // cudaFuncAttributes 是 CUDA 提供的一个结构体
    // 用于描述核函数的各种属性,例如共享内存大小/寄存器使用数量等.
    struct cudaFuncAttributes attr;

    // overload function 1
    func1 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    checkCudaErrors(cudaFuncSetCacheConfig(*func1, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func1));
    OUTPUT_ATTR(attr);

    (*func1)<<<DIV_UP(N, THREAD_N), THREAD_N>>>(d_input, d_output, a);
    checkCudaErrors(cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost));

    func_result = check_func1(h_input, h_output, a);
    printf("simple_kernel(const int *pIn, int *pOut, int a) %s\n\n", func_result ? "PASSED" : "FAILED");
    test_result &= func_result;

    // overload function 2
    func2 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    checkCudaErrors(cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func2));
    OUTPUT_ATTR(attr);
    (*func2)<<<DIV_UP(N, THREAD_N), THREAD_N>>>((int2 *)d_input, d_output, a);
    checkCudaErrors(
        cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost));
    func_result = check_func2(reinterpret_cast<int2 *>(h_input), h_output, a);
    printf("simple_kernel(const int2 *pIn, int *pOut, int a) %s\n\n",
            func_result ? "PASSED" : "FAILED");
    test_result &= func_result;

    // overload function 3
    func3 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    checkCudaErrors(cudaFuncSetCacheConfig(*func3, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func3));
    OUTPUT_ATTR(attr);
    (*func3)<<<DIV_UP(N, THREAD_N), THREAD_N>>>(d_input, d_input + N, d_output, a);
    checkCudaErrors(
        cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost));
    func_result = check_func3(&h_input[0], &h_input[N], h_output, a);
    printf(
        "simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) "
        "%s\n\n",
        func_result ? "PASSED" : "FAILED");
    test_result &= func_result;

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFreeHost(h_output));
    checkCudaErrors(cudaFreeHost(h_input));

    checkCudaErrors(cudaDeviceSynchronize());

    exit(test_result ? EXIT_SUCCESS : EXIT_FAILURE);
}
