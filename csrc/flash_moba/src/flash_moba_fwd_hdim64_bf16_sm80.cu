// Copyright (c) 2024, Tri Dao.
// Copyright (c) 2025, FlashMoBA Team.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_MOBA_NAMESPACE {

template<>
void run_moba_fwd_<cutlass::bfloat16_t, 64, false>(Flash_moba_fwd_params &params, cudaStream_t stream) {
    run_moba_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_MOBA_NAMESPACE