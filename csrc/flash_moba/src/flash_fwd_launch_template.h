/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "flash_moba_static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace FLASH_MOBA_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashMoBA requires building with sm version sm80-sm90, but was built for < 8.0!");


#define DEFINE_FLASH_MOBA_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_moba_fwd_params params)


DEFINE_FLASH_MOBA_FORWARD_KERNEL(flash_moba_fwd_kernel, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_MOBA_NAMESPACE::compute_moba_attn<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_moba_fwd(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_lg_m_block = (params.seqlen_q + params.m_lg_block_dim - 1) / params.m_lg_block_dim;
    dim3 grid(num_lg_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                auto kernel = &flash_moba_fwd_kernel<Kernel_traits, Is_dropout /*&& !Is_softcap*/, Is_causal, /*Has_alibi*/false, IsEvenMNConst && IsEvenKConst /*&& !Has_alibi*/ && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst /*&& !Has_alibi*/, /*Is_softcap*/ false, ReturnSoftmaxConst && Is_dropout /*&& !Is_softcap*/>;
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}


template<typename T, bool Is_causal>
void run_moba_fwd_hdim32(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim64(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 256) is 27% slower for seqlen=2k
            // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
            // print("to 64x64 \n");
            if (params.n_lg_block_dim % 128 == 0) {
                run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 128, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim96(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // These two are always slower
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim128(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 1st ones are good for H100, A100
            // 2nd one is good for A6000 bc we get slightly better occupancy
        } else {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim192(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream); // add by JXGuo: warps = 8 cause problem for moba, check later
        } else {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_moba_fwd_hdim256(Flash_moba_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For A100, we want to run with 128 x 64 (128KB smem).
        // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream); // add by JXGuo: exceed smem limit for A100
        } else {
            run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 64, 4, 1024, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // 64 KB
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // 96 KB
        // run_flash_moba_fwd<Flash_moba_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

}  // namespace FLASH_MOBA_NAMESPACE
