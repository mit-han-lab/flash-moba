# Copyright (c) 2025, FlashMoBA Team.
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        # triton.Config({'kBlockN': 16}, num_warps=2, num_stages=3),
        triton.Config({'kBlockN': 32}, num_warps=2, num_stages=3),
        triton.Config({'kBlockN': 32}, num_warps=4, num_stages=3),
        triton.Config({'kBlockN': 32}, num_warps=4, num_stages=4),
        triton.Config({'kBlockN': 64}, num_warps=2, num_stages=3),
        triton.Config({'kBlockN': 64}, num_warps=4, num_stages=3),
        triton.Config({'kBlockN': 64}, num_warps=4, num_stages=4),
        triton.Config({'kBlockN': 64}, num_warps=8, num_stages=3),
        triton.Config({'kBlockN': 128}, num_warps=2, num_stages=3),
        triton.Config({'kBlockN': 128}, num_warps=4, num_stages=3),
        triton.Config({'kBlockN': 128}, num_warps=4, num_stages=4),
        triton.Config({'kBlockN': 128}, num_warps=8, num_stages=3),
        triton.Config({'kBlockN': 128}, num_warps=8, num_stages=4),
        # triton.Config({'kBlockN': 256}, num_warps=4, num_stages=3),
        # triton.Config({'kBlockN': 256}, num_warps=8, num_stages=3),
        # triton.Config({'kBlockN': 256}, num_warps=8, num_stages=4),
        # triton.Config({'kBlockN': 256}, num_warps=16, num_stages=2),
        # triton.Config({'kBlockN': 512}, num_warps=8, num_stages=2),
        # triton.Config({'kBlockN': 512}, num_warps=16, num_stages=2),
        # triton.Config({'kBlockN': 512}, num_warps=16, num_stages=3),
        # triton.Config({'kBlockN': 1024}, num_warps=16, num_stages=2),
    ],
    key=['HEAD_DIM', 'POOL_BLOCK_SIZE'],
)
@triton.jit
def mean_pool_kernel(
        # Pointers to matrices
        input_ptr,
        output_ptr,
        # Matrix dimensions
        HEAD_DIM: tl.constexpr,
        POOL_BLOCK_SIZE: tl.constexpr,
        cu_seqlens_input,
        cu_seqlens_output,
        input_stride_row, input_stride_head,
        output_stride_row, output_stride_head,
        # Meta-parameters
        kBlockN: tl.constexpr,
):
    """
    Triton kernel for mean pooling over variable-length sequences.

    This kernel computes the mean of non-overlapping blocks of size `POOL_BLOCK_SIZE`
    for each sequence in a batch. It is designed to handle variable sequence lengths.

    Args:
        input_ptr: Pointer to the input tensor of shape (total_seqlen, num_heads, head_dim).
        output_ptr: Pointer to the output tensor of shape (total_blocks, num_heads, head_dim).
        HEAD_DIM: The dimension of each head.
        POOL_BLOCK_SIZE: The size of the pooling window.
        cu_seqlens_input: Cumulative sequence lengths of the input tensor, shape (batch_size + 1,).
        cu_seqlens_output: Cumulative sequence lengths of the output tensor, shape (batch_size + 1,).
        input_stride_row: Stride of the input tensor along the sequence dimension.
        input_stride_head: Stride of the input tensor along the head dimension.
        output_stride_row: Stride of the output tensor along the sequence dimension.
        output_stride_head: Stride of the output tensor along the head dimension.
        kBlockN: Block size for the sequence dimension, a meta-parameter for tuning.
    """
    n_block = tl.program_id(0)
    bidb = tl.program_id(1)
    bidh = tl.program_id(2)
    
    seq_start = tl.load(cu_seqlens_input + bidb)
    seq_end = tl.load(cu_seqlens_input + bidb + 1)
    
    block_start_row = seq_start + n_block * POOL_BLOCK_SIZE
    
    if seq_end <= block_start_row:
        return
    
    actual_block_size = tl.minimum(POOL_BLOCK_SIZE, seq_end - block_start_row)
    
    offsets_d = tl.arange(0, HEAD_DIM)
    # mask_d = offsets_d < HEAD_DIM

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    for block_k_start in range(0, actual_block_size, kBlockN):
        offsets_k = block_k_start + tl.arange(0, kBlockN)
        mask_k = offsets_k < actual_block_size
        
        row_indices = block_start_row + offsets_k
        
        input_offset = row_indices[:, None] * input_stride_row.to(tl.int64) + bidh * input_stride_head.to(tl.int64) + offsets_d[None, :]
        
        inp = tl.load(input_ptr + input_offset, mask=mask_k[:, None], other=0.0)
        acc += tl.sum(inp, axis=0)
    
    # safe division
    mean_val = acc / actual_block_size
    
    output_start = tl.load(cu_seqlens_output + bidb)
    output_offset = (output_start + n_block) * output_stride_row.to(tl.int64) + bidh * output_stride_head.to(tl.int64) + offsets_d
    tl.store(output_ptr + output_offset, mean_val)


def flash_topk_mean_pool(input, cu_seqlens_input, max_seqlen_input, pool_block_size):
    """
    Performs mean pooling on variable-length sequences using a Triton kernel.

    This function takes a tensor of packed sequences and applies mean pooling over
    fixed-size blocks.

    Args:
        input (torch.Tensor): The input tensor of shape (total_seqlen, num_heads, head_dim).
        cu_seqlens_input (torch.Tensor): Cumulative sequence lengths for the input, shape (batch_size + 1,).
        max_seqlen_input (int): The maximum sequence length in the input batch.
        pool_block_size (int): The size of the pooling window.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
            - output (torch.Tensor): The pooled output tensor of shape (total_blocks, num_heads, head_dim).
            - cu_seqlens_output (torch.Tensor): Cumulative sequence lengths for the output.
            - max_seqlen_output (int): The maximum number of blocks for any sequence in the batch.
    """
    total_seqlen, head_num, head_dim = input.shape
    batch_size = cu_seqlens_input.shape[0] - 1

    max_seqlen_output = (max_seqlen_input + pool_block_size - 1) // pool_block_size
    
    actual_input_seqlens = cu_seqlens_input[1:] - cu_seqlens_input[:-1]
    actual_output_seqlens = (actual_input_seqlens + pool_block_size - 1) // pool_block_size
    cu_seqlens_output = F.pad(torch.cumsum(actual_output_seqlens, dim=0), (1, 0)).to(torch.int32)

    total_blocks = cu_seqlens_output[-1].item()
    
    output = torch.zeros((total_blocks, head_num, head_dim), dtype=input.dtype, device=input.device)

    grid = (max_seqlen_output, batch_size, head_num)
    
    mean_pool_kernel[grid](
        input, 
        output,
        head_dim,
        pool_block_size,
        cu_seqlens_input,
        cu_seqlens_output,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output, cu_seqlens_output, max_seqlen_output
    