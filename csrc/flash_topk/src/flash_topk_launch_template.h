/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo and Kasra Mazaheri from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_topk_launch_template.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "flash_topk_static_switch.h"
#include "hardware_info.h"
#include "flash_topk.h"
#include "flash_topk_kernel.h"

namespace FLASH_TOPK_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_TOPK_KERNEL(kernelName) \
template<typename Kernel_traits> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Fused_topk_params params)

#define DEFINE_FLASH_TOPK_KERNEL_V(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Fused_topk_params params)


DEFINE_FLASH_TOPK_KERNEL_V(fused_topk_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_TOPK_NAMESPACE::compute_fused_topk<Kernel_traits, Is_causal, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_TOPK_KERNEL(topk_epilogue_offset_kernel) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_TOPK_NAMESPACE::compute_topk_offset<Kernel_traits>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_TOPK_KERNEL_V(topk_epilogue_kernel, bool Is_even_M, bool Is_even_TopK) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_TOPK_NAMESPACE::compute_topk_epilogue<Kernel_traits, Is_even_M, Is_even_TopK>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal>
void run_fused_topk(Fused_topk_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_km == nullptr && params.seqlen_km % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            auto kernel = &fused_topk_kernel<Kernel_traits, Is_causal, IsEvenMNConst, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
void run_topk_epilogue(Fused_topk_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    dim3 offset_grid(params.b, params.h);

    auto offsets_kernel = &topk_epilogue_offset_kernel<Kernel_traits>;

    offsets_kernel<<<offset_grid, Kernel_traits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    const bool is_even_M = params.cu_seqlens_q == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_TopK = params.moba_topk == Kernel_traits::kTopk;

    BOOL_SWITCH(is_even_M, IsEvenMConst, [&] {
        EVENK_SWITCH(is_even_TopK, IsEvenTopKConst, [&] {
            auto kernel = &topk_epilogue_kernel<Kernel_traits, IsEvenMConst, IsEvenTopKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool Is_causal>
void run_fused_topk16_hdim64(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kTopk = 16;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 64, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk32_hdim64(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kTopk = 32;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 64, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk64_hdim64(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kTopk = 64;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 64, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}


template<typename T, bool Is_causal>
void run_fused_topk16_hdim128(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kTopk = 16;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk32_hdim128(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kTopk = 32;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk64_hdim128(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kTopk = 64;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 128, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk16_hdim256(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    constexpr static int kTopk = 16;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 64, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk32_hdim256(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    constexpr static int kTopk = 32;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 64, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

template<typename T, bool Is_causal>
void run_fused_topk64_hdim256(Fused_topk_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    constexpr static int kTopk = 64;
    run_fused_topk<Fused_topk_kernel_traits<Headdim, 64, 32, 4, kTopk, true, T>, Is_causal>(params, stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128, 4, kTopk, T>>(params, stream);
}

}  // namespace FLASH_TOPK_NAMESPACE
