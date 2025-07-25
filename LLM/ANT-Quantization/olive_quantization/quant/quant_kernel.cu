#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
using namespace std;

// 这个 CUDA 函数的作用是 量化,即:
// 给定一个输入张量 x(例如 [1.2, 3.4, 5.6, ...])
// 和一个目标量化值集合 y(例如 [1.0, 2.0, 3.0, 4.0, 5.0, ...])
// 对 x 的每个元素,找到 y 中最接近的值(L1 距离最小),并输出量化后的张量 z.
namespace {
template <typename scalar_t>
__global__ void quant_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> y,
    size_t x_size,
    size_t y_size,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> z,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> tensor_idx)
    {
        __shared__ float y_shared[256];
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadIdx.x < y_size) y_shared[threadIdx.x] = y[threadIdx.x];
        __syncthreads();
        float sub_min = 102400.0;
        float z_min = 0.0;
        if(idx < x_size) {
            float x_v = x[idx];
            for(int i = 0; i < y_size; i++){
                float sub_v = fabsf(x_v - y_shared[i]);
                if(sub_v <= sub_min)
                {
                    sub_min = sub_v;
                    z_min = y_shared[i];
                }
            }
            z[idx] = z_min;
        }
    }
} // namespace

std::tuple<torch::Tensor, torch::Tensor>  quant_forward_cuda(
    torch::Tensor x,
    torch::Tensor y)
{
    const int threads = 1024;
    const dim3 blocks((x.size(0) + threads - 1) / threads);
    auto z   = torch::zeros_like(x);
    auto idx = torch::zeros_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "quant_forward_cuda", ([&] {
        quant_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            y.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            x.size(0),
            y.size(0),
            z.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            idx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));

    return std::make_tuple(z,idx);
}
