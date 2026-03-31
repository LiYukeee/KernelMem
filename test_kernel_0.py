import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ModelNew optimization header:
# Chosen granularity: (A) optimize a single hotspot op (ReLU)
# Ops replaced: torch.relu -> custom CUDA kernel
# Remaining in PyTorch: None (single op model)

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void relu_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    auto num_elements = input.numel();
    
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    auto stream = at::cuda::getDefaultCUDAStream();
    relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}
"""

cpp_src = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_custom = load_inline(
    name="relu_custom",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=["-O3", "-std=c++17"],
     extra_cuda_cflags=[
         "-O3",
         "-std=c++17",
         "--expt-relaxed-constexpr",
         "-lineinfo",
         "-gencode=arch=compute_80,code=sm_80"
     ]
)

class ModelNew(nn.Module):
     def __init__(self):
         super().__init__()
     
     def forward(self, x):
         return relu_custom.relu_cuda(x)
