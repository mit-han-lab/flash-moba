/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_bwd_preprocess_kernel.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"

namespace FLASH_MOBA_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_ROW, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void dot_do_o(Tensor<Engine0, Layout0> const &do_, Tensor<Engine0, Layout0> const &o,
                                Tensor<Engine1, Layout1> &dP_sum, const int gdP_col_stride, const float scale) {
    static_assert(Layout0::rank == 3, "Only support 3D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(do_.layout() == o.layout());
    // Reshape do_ and o from (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, 8 * kHeadDim / 64)
    // The last coordinate is the "page".
    Tensor do_reshaped = make_tensor(do_.data(), make_layout(get<1>(do_.layout()),
                                                             make_layout(get<0>(do_.layout()),
                                                                         get<2>(do_.layout()))));
    Tensor o_reshaped = make_tensor(o.data(), do_reshaped.layout());
    Tensor do_fp32 = FLASH_MOBA_NAMESPACE::convert_type<float>(do_reshaped);
    Tensor o_fp32 = FLASH_MOBA_NAMESPACE::convert_type<float>(o_reshaped);
    #pragma unroll
    for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
        float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
        #pragma unroll
        for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
            dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
        }
        FLASH_MOBA_NAMESPACE::SumOp<float> sum_op;
        dP_sum_cur = FLASH_MOBA_NAMESPACE::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
        if (threadIdx.x % THREADS_PER_ROW == 0) {
            dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<bool Clear_dQaccum=true, typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse ? (bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb): (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;

    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    Tensor dP_sum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    // TODO: careful, we're zeroing out dQaccum with type float4, but when
    // we do atomicAdds, we use type float. The layouts are different. Check this.
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    Tensor cdO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdOcdO = gmem_thr_copy_dO.partition_S(cdO);

    // Allocate predicate tensors for k
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOgdO)));
    // Set predicates for k bounds
    #pragma unroll
    for (int k = 0; k < size(tdOpdO); ++k) {tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d;}

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgO, tdOrO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // By right we need to scale dP up by 1/p_dropout, but instead we don't and only scale the final
    // results (dQ and dK) by 1/p_dropout. So we need to keep dP_sum scaled down by p_dropout here,
    // so that (dP - dP_sum) is on the same scale.
    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, dP_sum,
                                                Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    if (Clear_dQaccum) {
        // We're actually not zero'ing out all of dQaccum, but only the part that we're going to
        // do atomicAdds on.
        Tensor zero = make_fragment_like(tdQgdQaccum);
        clear(zero);
        cute::copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert dQ from dQaccum (in float) to fp16/bf16.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dQ(const Params &params, const int nsplits) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;

    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    // typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    // Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);
    Tensor tdQgdQaccum = thr_mma_dq.partition_C(gdQaccum);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));

    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    clear(acc_dq);
    for (int s = 0; s < nsplits; ++s) {
        cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) += tdQrdQaccum(i); }
        tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = FLASH_MOBA_NAMESPACE::convert_type<Element>(acc_dq);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    #pragma unroll
    for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

} // namespace flash
