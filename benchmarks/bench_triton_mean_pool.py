import torch
import triton
import triton.testing
from flash_moba.triton_mean_pool import flash_topk_mean_pool

def mean_pool_torch(input_tensor: torch.Tensor, cu_seqlens: torch.Tensor, pool_block_size: int):
    """Vectorised reference implementation used for benchmarking against the Triton kernel.

    Args:
        input_tensor: (total_seqlen, head, head_dim)
        cu_seqlens:    (batch+1) cumulative sequence lengths
        pool_block_size: window size for mean-pooling

    This version avoids Python-side loops and leverages `index_add_` for O(N) complexity
    with minimal kernel launches.  It is 10-100× faster than the naïve loop for typical
    transformer sequence lengths.
    """
    total_seqlen, head_num, head_dim = input_tensor.shape
    device = input_tensor.device

    batch_size = cu_seqlens.numel() - 1
    if total_seqlen == 0:
        return torch.empty((0, head_num, head_dim), dtype=input_tensor.dtype, device=device)

    # Sequence lengths and number of output blocks per sequence
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]                # (batch,)
    num_blocks_per_seq = (seqlens + pool_block_size - 1) // pool_block_size  # (batch,)
    total_blocks = int(num_blocks_per_seq.sum().item())

    if total_blocks == 0:
        return torch.empty((0, head_num, head_dim), dtype=input_tensor.dtype, device=device)

    # Prefix-sum of blocks so we can map each token row to its global block id
    cu_blocks = torch.cumsum(
        torch.cat([input_tensor.new_zeros(1, dtype=num_blocks_per_seq.dtype), num_blocks_per_seq]),
        dim=0,
    )  # (batch+1,)

    # Map every row to its batch index (seq_id)               ────────────────╮
    seq_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), seqlens)  # (total_seqlen,)

    # Row offset within its sequence ➜ block offset within sequence
    row_offsets = torch.arange(total_seqlen, device=device) - cu_seqlens[seq_ids]
    block_idx_in_seq = row_offsets.div(pool_block_size, rounding_mode="floor")  # (total_seqlen,)

    # Global block id for every row
    global_block_idx = cu_blocks[seq_ids] + block_idx_in_seq  # (total_seqlen,)

    # Accumulate sums per block
    flat_inp = input_tensor.reshape(total_seqlen, -1)  # (total_seqlen, head*dim)
    out_sum = torch.zeros(total_blocks, flat_inp.shape[1], dtype=input_tensor.dtype, device=device)
    out_sum.index_add_(0, global_block_idx, flat_inp)

    # Accumulate counts to compute mean (using same dtype for numerical parity)
    counts = torch.zeros(total_blocks, dtype=input_tensor.dtype, device=device)
    ones = torch.ones(total_seqlen, dtype=input_tensor.dtype, device=device)
    counts.index_add_(0, global_block_idx, ones)

    # Normalize and reshape back
    out_mean = out_sum / counts.unsqueeze(1).clamp_min(1.0)  # avoid div-by-zero
    return out_mean.view(total_blocks, head_num, head_dim)


configs = [
    triton.testing.Benchmark(
        x_names=['total_seqlen'],
        x_vals=[1024 * i for i in range(2, 17, 2)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='GB/s',
        plot_name='mean-pool-performance',
        args={'batch_size': 16, 'head_num': 12, 'head_dim': 128, 'pool_block_size': 64, 'dtype': torch.float16}
    )
]

@triton.testing.perf_report(configs)
def benchmark(total_seqlen, batch_size, head_num, head_dim, pool_block_size, provider, dtype):
    device = 'cuda'
    # setup inputs
    seqlens = (torch.ones(batch_size, dtype=torch.int32) * (total_seqlen // batch_size))
    cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(seqlens, dim=0)]).to(device).to(torch.int32)
    max_seqlen = seqlens.max().item()
    current_total_seqlen = seqlens.sum().item()
    input_tensor = torch.randn(current_total_seqlen, head_num, head_dim, device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: mean_pool_torch(input_tensor, cu_seqlens, pool_block_size), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_topk_mean_pool(input_tensor, cu_seqlens, max_seqlen, pool_block_size), quantiles=quantiles)
        
    # calculate total_blocks to compute bytes
    num_blocks_per_seq = (seqlens.to(device) + pool_block_size - 1) // pool_block_size
    total_blocks = num_blocks_per_seq.sum().item()

    # perf calculation
    total_bytes = (current_total_seqlen + total_blocks) * head_num * head_dim * input_tensor.element_size()
    gbps = lambda ms: total_bytes / (ms * 1e-3) / 1e9 if ms > 0 else 0
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)