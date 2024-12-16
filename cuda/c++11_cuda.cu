#include <iostream>
#include <thrust/device_ptr.h>  // Thrust 是一个与 CUDA 兼容的 C++ 模板库
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <helper_cuda.h>
#include "range.hpp"

using namespace util::lang;

template <typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__ step_range<T> grid_stride_range(T begin, T end) {
    /*
        每个线程通过 全局索引 确定初始位置,以 网格总线程数 为步长循环处理数据.
        在线程数 << 处理数据长度时, 能够避免重复处理
    */
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template <typename T, typename Predicate>
__device__ void count_if(int *count, T *data, int n, Predicate p) {
    for (auto i : grid_stride_range(0, n)) {
        // 获取当前线程应该处理的对应数据位置: 线程索引 + 网格步长
        if (p(data[i])) {
            atomicAdd(count, 1);
        }
    }
}

__global__ void xyzw_frequency(int *count, char *text, int n) {
    const char letters[]{'x', 'y', 'z', 'w'};
    count_if(count, text, n, [&](char c){
        for (const auto x : letters) {
            if (c == x) return true;
        }
        return false;
    });
}

__global__ void xyzw_frequency_device(int *count, char *text, int n) {
    const char letters[]{'x', 'y', 'z', 'w'};
    *count = thrust::count_if(thrust::device, text, text + n, [=](char c){
        for (const auto x : letters) {
            if (c == x) return true;
        }
        return false;
    });
}

int main(int argc, char **argv) {
    const char *filename = sdkFindFilePath("warandpeace.txt", argv[0]);
    int dev_id = findCudaDevice(argc, (const char **)argv);

    int num_bytes = 16 * 1048576;
    char *h_text = (char *)malloc(num_bytes);

    char *d_text;
    checkCudaErrors(cudaMalloc((void **)&d_text, num_bytes));

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Cannot find the input text file\n. Exiting..\n");
        return EXIT_FAILURE;
    }
    int len = (int)fread(h_text, sizeof(char), num_bytes, fp);
    fclose(fp);
    std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

    checkCudaErrors(cudaMemcpy(d_text, h_text, len, cudaMemcpyHostToDevice));

    int count = 0;
    int *d_count;
    checkCudaErrors(cudaMalloc(&d_count, sizeof(int)));
    checkCudaErrors(cudaMemset(d_count, 0, sizeof(int)));

    xyzw_frequency<<<8, 256>>>(d_count, d_text, len);
    // Thrust 算法在内部自动划分网格和线程块, 并将任务并行执行在 GPU 上.
    xyzw_frequency_device<<<1, 1>>>(d_count, d_text, len);
    checkCudaErrors(
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "counted " << count
            << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\""
            << std::endl;

    checkCudaErrors(cudaFree(d_count));
    checkCudaErrors(cudaFree(d_text));

    return EXIT_SUCCESS;
}
