// Copyright (c) 2024, Tri Dao.
// Copyright (c) 2025, FlashMoBA Team.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_topk_launch_template.h"

namespace FLASH_TOPK_NAMESPACE {

template<>
void run_fused_topk_<cutlass::half_t, 128, 64, false>(Fused_topk_params &params, cudaStream_t stream) {
    run_fused_topk64_hdim128<cutlass::half_t, false>(params, stream);
}

} // namespace FLASH_TOPK_NAMESPACE