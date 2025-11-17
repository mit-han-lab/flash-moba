#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash_topk.h"
#include "flash_topk_static_switch.h"


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_TOPK_NAMESPACE {

void set_params_fused_topk(
    Fused_topk_params &params,
    /*-------------- tensors ---------------*/
    const at::Tensor q,
    const at::Tensor km,
    at::Tensor       indices,
    at::Tensor       values,
    at::Tensor       col_offsets,
    at::Tensor       col_nnz,
    at::Tensor       flat_indices,
    /*--------- sequence length ptrs -------*/
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *cu_seqlens_km_d,
    /*---------------- sizes ----------------*/
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_km,
    const size_t seqlen_q_rounded,
    const size_t seqlen_km_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    const size_t moba_topk,
    const size_t moba_topk_rounded,
    const size_t moba_chunk_size,
    const size_t max_lg_col_num,
    const bool   is_causal) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    /**************** Pointers & strides ****************/
    params.q_ptr   = q.data_ptr();
    params.km_ptr  = km.data_ptr();
    params.topk_idx_ptr = indices.data_ptr();
    params.topk_val_ptr = values.data_ptr();

    // Element-based strides (not bytes)
    params.q_row_stride   = q.stride(-3);
    params.q_head_stride  = q.stride(-2);

    params.km_row_stride  = km.stride(-3);
    params.km_head_stride = km.stride(-2);

    params.idx_row_stride  = indices.stride(-3);
    params.idx_head_stride = indices.stride(-2);

    params.val_row_stride  = values.stride(-3);
    params.val_head_stride = values.stride(-2);

    params.col_offsets_ptr = col_offsets.data_ptr();
    params.col_nnz_ptr     = col_nnz.data_ptr();
    params.indices_ptr     = flat_indices.data_ptr();

    // Batch strides (only used when cu_seqlens is nullptr)
    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride  = q.stride(0);
        params.km_batch_stride = km.stride(0);
        params.idx_batch_stride = indices.stride(0);
    }

    /*************** Sequence length ptrs ***************/
    params.cu_seqlens_q  = static_cast<int*>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
    params.cu_seqlens_km = static_cast<int*>(cu_seqlens_km_d);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_km = seqlen_km;
    params.seqlen_q_rounded  = seqlen_q_rounded;
    params.seqlen_km_rounded = seqlen_km_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set MoBA-specific parameters
    params.moba_topk = moba_topk;
    params.moba_topk_rounded = moba_topk_rounded;
    params.moba_chunk_size = moba_chunk_size;
    params.max_lg_col_num = max_lg_col_num;

    params.is_causal = is_causal; // simple heuristic, adjust if needed
}

void run_fused_topk(Fused_topk_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            TOPK_SWITCH(params.moba_topk, [&] {
                BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                    run_fused_topk_<elem_type, kHeadDim, kTopk, Is_causal>(params, stream);
                });
            });
        });
    });
}


std::vector<at::Tensor> moba_fused_topk(
    const at::Tensor &q,                 // [total_q, num_heads, head_size]
    const at::Tensor &km,                // [total_km, num_heads_k, head_size]
    const at::Tensor &cu_seqlens_q,      // [b + 1]
    const at::Tensor &cu_seqlens_k,     // [b + 1]
    const at::Tensor &cu_seqlens_km,    // [b + 1]
    const int          max_seqlen_q,
    const int          max_seqlen_k,
    const int          moba_topk,
    const int          moba_chunk_size,
    const bool         is_causal) {

    /**********************
     * 1. Sanity checks   *
     *********************/
    at::cuda::CUDAGuard device_guard{q.device()};

    // Hardware capability
    {
        auto [cc_major, _] = get_compute_capability(get_current_device());
        TORCH_CHECK(cc_major >= 8, "FlashMoBA only supports Ampere GPUs or newer.");
    }

    // Dtype / device checks
    auto dtype = q.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "FlashMoBA only supports fp16 and bf16 data types");
    TORCH_CHECK(km.dtype() == dtype, "q and km must have the same dtype");

    CHECK_DEVICE(q);  CHECK_DEVICE(km);
    CHECK_DEVICE(cu_seqlens_q);  
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(cu_seqlens_km);

    // Layout / contiguity checks
    TORCH_CHECK(q.stride(-1)  == 1, "q must be contiguous on the last dim");
    TORCH_CHECK(km.stride(-1) == 1, "km must be contiguous on the last dim");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_km);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");
    TORCH_CHECK(cu_seqlens_km.dtype() == torch::kInt32, "cu_seqlens_km must be int32");

    TORCH_CHECK(moba_topk <= 64, "moba_topk must be <= 64");

    /**********************
     * 2. Dimension logic *
     *********************/
    const int batch_size   = cu_seqlens_q.numel() - 1;
    const int total_q      = q.size(0);
    const int total_km     = km.size(0);
    const int num_heads    = q.size(1);
    const int num_heads_km = km.size(1);
    const int head_size    = q.size(2);

    TORCH_CHECK(batch_size > 0,        "batch size must be positive");
    TORCH_CHECK(head_size  <= 256,     "head_size must be <= 256");
    TORCH_CHECK(head_size % 8 == 0,    "head_size must be multiple of 8");
    TORCH_CHECK(num_heads % num_heads_km == 0,
                "#heads in key/value must divide #heads in query");

    CHECK_SHAPE(q,  total_q,  num_heads,      head_size);
    CHECK_SHAPE(km, total_km, num_heads_km,   head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_km, batch_size + 1);

    /**********************
     * 4. Derived sizes   *
     *********************/
    auto ceil_div = [](int x, int m){ return (x + m - 1) / m; };
    auto round_up = [](int x, int m){ return (x + m - 1) / m * m; };
    const int max_seqlen_km       = ceil_div(max_seqlen_k, moba_chunk_size);
    const int head_size_rounded   = round_up(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded    = round_up(max_seqlen_q, 128);
    const int seqlen_km_rounded   = round_up(max_seqlen_km, 128);
    const int moba_topk_rounded   = round_up(moba_topk, 16);

    /**********************
     * 3. Output buffers  *
     *********************/
    const int  max_col_num   = max_seqlen_km;
    at::Tensor indices       = torch::empty({total_q, num_heads, moba_topk_rounded}, q.options().dtype(torch::kInt32));
    at::Tensor values        = torch::empty({total_q, num_heads, moba_topk_rounded}, q.options().dtype(torch::kFloat));
    at::Tensor col_offsets   = torch::zeros({batch_size, num_heads, max_col_num}, km.options().dtype(torch::kInt64));
    at::Tensor col_nnz       = torch::zeros({batch_size, num_heads, max_col_num}, km.options().dtype(torch::kInt32));
    at::Tensor flat_indices  = torch::zeros({total_q * num_heads * moba_topk},   km.options().dtype(torch::kInt32));

    /**********************
     * 5. Param struct    *
     *********************/
    Fused_topk_params params;
    set_params_fused_topk(
        params,
        /*-------------- tensors ---------------*/
        q, km,
        indices, values, col_offsets, col_nnz, flat_indices,
        /*--------- sequence length ptrs -------*/
        cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(), cu_seqlens_km.data_ptr(),
        /*---------------- sizes ----------------*/
        batch_size,
        max_seqlen_q, max_seqlen_k, max_seqlen_km,
        seqlen_q_rounded, seqlen_km_rounded,
        num_heads, num_heads_km,
        head_size, head_size_rounded,
        moba_topk, moba_topk_rounded,
        moba_chunk_size,
        max_col_num,
        is_causal);
    params.total_q = total_q;

    /**********************
     * 6. Kernel launch   *
     *********************/
    // Always initialise indices; optionally flat_indices
    indices.fill_(-1);
    values.fill_(-std::numeric_limits<float>::infinity());
    // if (zero_tensors) { flat_indices.fill_(-1); }

    if (max_seqlen_k > 0) {
        run_fused_topk(params, at::cuda::getCurrentCUDAStream().stream());
    } else {
        // No keys – outputs remain -1.
        flat_indices.fill_(-1);
    }

    return {col_offsets, col_nnz, flat_indices, values, indices};
}
} // namespace FLASH_TOPK_NAMESPACE

// Expose init function to be called from main extension module.
void init_flash_topk(pybind11::module &m) {
    m.def("varlen_sort", &FLASH_TOPK_NAMESPACE::varlen_sort, "");
    m.def("moba_fused_topk", &FLASH_TOPK_NAMESPACE::moba_fused_topk,
          "MoBA top-k indices (variable length)");
}