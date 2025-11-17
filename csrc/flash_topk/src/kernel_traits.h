/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo and Kasra Mazaheri from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<int kNWarps_, typename elem_type=cutlass::half_t>
struct Base_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;

    /*  add by JXGuo: for learning purpose
        For example, the Volta section below will refer to the SM70_8x8x4_F32F16F16F32_NT Operation struct defined in include/cute/arch/mma_sm70.hpp.

    "SM70" refers to Volta.

    "8x8x4" refers to M = 8, N = 8, and K = 4, the dimensions of the MMA operation that the quadpair performs (see below). This is reflected in the PTX as .m8n8k4..

    "F32F16F16F32" refers to the element types of the four matrix operands A, B, C, and D. An MMA computes D = C + A * B, so we read the types from left to right: D is F32 (float), A is F16 (half), B is F16 (half), and C is F32 (float). This is reflected in the PTX instruction name as .f32.f16.f16.f32.

    "NT" means that the PTX instruction is designed for inputs A as M-major (not transposed, column-major) and inputs B as N-major (transposed, row-major). This is reflected in the PTX instruction name as .col.row..
    */
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};


// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kTopk_, bool Is_Q_in_regs_=false, typename elem_type=cutlass::half_t,
         typename Base=Base_kernel_traits<kNWarps_, elem_type> >
struct Fused_topk_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    static constexpr bool Share_Q_K_smem = false;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kTopk = kTopk_;
    static_assert(kTopk % 16 == 0);
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutTopK = Layout<Shape<Int<kBlockM>, Int<kTopk>>,
                                  Stride<Int<kTopk>, _1>>;

    using SmemLayoutTopKReduce = Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                                  Stride<Int<kBlockN>, _1>>;

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomR = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemTopKSize = size(SmemLayoutTopK{}) * 2 * sizeof(ElementAccum);
    static constexpr int kSmemTopKReduceSize = size(SmemLayoutTopKReduce{}) * sizeof(ElementAccum);
    static constexpr int kSmemSize = kSmemKSize + kSmemTopKSize + (Is_Q_in_regs ? std::max(kSmemQSize, kSmemTopKReduceSize) : kSmemQSize + kSmemTopKReduceSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    
    static constexpr int kGmemTopKPerLoad = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowTopK = kTopk / kGmemTopKPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRowTopK == 0, "kNThreads must be a multiple of kGmemThreadsPerRowTopK");
    using TopKLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRowTopK>, Int<kGmemThreadsPerRowTopK>>,
                                  Stride<Int<kGmemThreadsPerRowTopK>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemTiledCopyTopKIdx = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, int>{},
                        TopKLayoutAtom{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read
    using GmemTiledCopyTopKVal = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        TopKLayoutAtom{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};


template<int kBlockM_, int kNWarps_, int kTopk_, typename elem_type=cutlass::half_t,
         typename Base=Base_kernel_traits<kNWarps_, elem_type> >
struct Topk_epilogue_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kTopk = kTopk_;
    static_assert(kTopk % 16 == 0);

    using SmemLayoutTopK = Layout<Shape<Int<kBlockM>, Int<kTopk>>,
                                  Stride<Int<kTopk>, _1>>;

    static constexpr int kSmemTopKSize = size(SmemLayoutTopK{}) * sizeof(int32_t);
    static constexpr int kSmemSize = kSmemTopKSize;

    static constexpr int kGmemTopKPerLoad = sizeof(cute::uint128_t) / sizeof(int32_t);
    static constexpr int kGmemThreadsPerRowTopK = kTopk / kGmemTopKPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRowTopK == 0, "kNThreads must be a multiple of kGmemThreadsPerRowTopK");
    using TopKLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRowTopK>, Int<kGmemThreadsPerRowTopK>>,
                                  Stride<Int<kGmemThreadsPerRowTopK>, _1>>;

    
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemTiledCopyTopKIdx = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, int>{},
                        TopKLayoutAtom{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per read
};



////////////////////////////////////////////////////////////////////////////////////////////////////
