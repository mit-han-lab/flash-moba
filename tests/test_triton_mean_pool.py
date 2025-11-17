import torch
import pytest
from flash_moba.triton_mean_pool import flash_topk_mean_pool


def mean_pool_torch_gappy(input_tensor, cu_seqlens, pool_block_size):
    _, head_num, head_dim = input_tensor.shape
    batch_size = cu_seqlens.shape[0] - 1
    
    output_blocks = []
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i+1].item()
        seq_len = end_idx - start_idx

        if seq_len > 0:
            for block_start in range(0, seq_len, pool_block_size):
                block_end = min(block_start + pool_block_size, seq_len)
                block = input_tensor[start_idx + block_start : start_idx + block_end]
                pooled_block = block.to(torch.float32).mean(dim=0, keepdim=True).to(input_tensor.dtype)
                output_blocks.append(pooled_block)

    if not output_blocks:
        return torch.empty((0, head_num, head_dim), dtype=input_tensor.dtype, device=input_tensor.device)
    
    return torch.cat(output_blocks, dim=0)


DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)

@pytest.mark.parametrize('batch_size, max_seqlen, head_num, head_dim, pool_block_size, dtype', 
    [ (b, s, h, d, p, dt) 
      for b in [1, 8]
      for s in [512, 2048]
      for h in [4, 12]
      for d in [64, 128, 256]
      for p in [32, 64, 128, 256, 512] # Test with pool_block_size <, =, > head_dim
      for dt in DTYPES
    ] + [
        (4, 128, 4, 32, 256, torch.float16), # pool_block_size > max_seqlen
        (16, 8, 4, 32, 4, torch.float16),   # Very short sequences
    ]
)
def test_mean_pool(batch_size, max_seqlen, head_num, head_dim, pool_block_size, dtype):
    device = 'cuda'
    random_seed = 42
    torch.manual_seed(random_seed)

    # Allow zero-length sequences to merge edge case test
    seqlens = torch.randint(0, max_seqlen, (batch_size,), device=device)
    cu_seqlens = torch.cat([torch.tensor([0], device=device), torch.cumsum(seqlens, dim=0)]).to(torch.int32)
    total_seqlen = seqlens.sum().item()

    input_tensor = torch.randn(total_seqlen, head_num, head_dim, device=device, dtype=dtype)

    triton_output, cu_seqlens_output, max_seqlen_output = flash_topk_mean_pool(input_tensor, cu_seqlens, max_seqlen, pool_block_size)
    torch_output = mean_pool_torch_gappy(input_tensor, cu_seqlens, pool_block_size)

    assert triton_output.shape == torch_output.shape, f"Shape mismatch: Triton is {triton_output.shape}, Torch is {torch_output.shape}"
    
    atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
    assert torch.allclose(triton_output, torch_output, atol=atol, rtol=0), "Value mismatch"


@pytest.mark.parametrize('dtype', DTYPES)
def test_mean_pool_all_zero_seqlen(dtype):
    batch_size = 8
    head_num = 12
    head_dim = 128
    pool_block_size = 128
    device = 'cuda'

    # Case: All sequences have zero length
    seqlens = torch.zeros(batch_size, device=device, dtype=torch.long)
    cu_seqlens = torch.cat([torch.tensor([0], device=device), torch.cumsum(seqlens, dim=0)]).to(torch.int32)
    total_seqlen = seqlens.sum().item()
    assert total_seqlen == 0
    input_tensor = torch.randn(total_seqlen, head_num, head_dim, device=device, dtype=dtype)

    triton_output, cu_seqlens_output, max_seqlen_output = flash_topk_mean_pool(input_tensor, cu_seqlens, 0, pool_block_size)
    torch_output = mean_pool_torch_gappy(input_tensor, cu_seqlens, pool_block_size)
    
    expected_shape = (0, head_num, head_dim)
    assert triton_output.shape == expected_shape, f"Shape mismatch (all zero): {triton_output.shape} vs {expected_shape}"
    assert triton_output.shape == torch_output.shape, f"Shape mismatch (all zero): {triton_output.shape} vs {torch_output.shape}"
    
    
if __name__ == "__main__":
    test_mean_pool(batch_size=8, max_seqlen=512, head_num=4, head_dim=64, pool_block_size=256, dtype=torch.float16)