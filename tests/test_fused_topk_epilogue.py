import torch
import pytest


try:
    import flash_moba_cuda  # Python wrapper that re-exports the CUDA op
except ImportError as exc:
    pytest.skip(reason=f"CUDA extension not built: {exc}", allow_module_level=True)

def _compute_key_gate_weight_ref(k: torch.Tensor, moba_chunk_size: int):
    """Reference implementation of `_compute_key_gate_weight`."""
    chunks = k.split(moba_chunk_size, dim=0)
    pooled_chunks = [c.mean(dim=0, keepdim=True) for c in chunks]
    return torch.cat(pooled_chunks, dim=0)

# === New parallel implementation ===
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_single_bh(bidb, bidh, col_offsets, o_indices, cu_seqlens_q, moba_topk, device):
    """Helper executed within a worker thread for a single (batch, head)."""
    # Local tensors to accumulate results for this (i, h)
    max_lg_col_num = col_offsets.size(-1)
    col_nnz_local = torch.zeros(max_lg_col_num, device=device, dtype=torch.int32)
    # We store updates as a python list to avoid write-time races. Each item is (index, value)
    updates: list[tuple[int, int]] = []

    start_q, end_q = cu_seqlens_q[bidb].item(), cu_seqlens_q[bidb + 1].item()
    for j in range(start_q, end_q):
        for k in range(moba_topk):
            col_idx = o_indices[j, bidh, k].item()
            if col_idx >= 0:
                col_offset = col_offsets[bidb, bidh, col_idx].item()
                flat_idx = col_offset + col_nnz_local[col_idx].item()
                updates.append((flat_idx, j - start_q))
                col_nnz_local[col_idx] += 1

    return bidb, bidh, updates, col_nnz_local

def indices_epilogue_thread_pool(col_offsets, col_nnz, flat_indices, o_indices, cu_seqlens_q, moba_topk, max_workers: int | None = None):
    """Thread-pooled version of :func:`indices_epilogue_ref`.

    The computation for each (batch, head) pair is independent, so we can safely
    parallelise across these two dimensions. A shared output tensor is returned
    after gathering per-thread partial results.
    """
    # Allocate outputs
    flat_indices_ref = torch.zeros_like(flat_indices)
    col_nnz_ref = torch.zeros_like(col_nnz)

    batch_size, num_heads, _ = col_offsets.shape
    device = flat_indices.device

    # Submit a task for every (batch, head) pair
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_single_bh, bidb, bidh, col_offsets, o_indices, cu_seqlens_q, moba_topk, device
            )
            for bidb in range(batch_size) for bidh in range(num_heads)
        ]

        # Gather results
        for fut in as_completed(futures):
            bidb, bidh, updates, col_nnz_local = fut.result()
            # Write back results for this (i, h)
            if updates:
                update_index, update_value = zip(*updates)
                update_index_tensor = torch.tensor(update_index, device=device, dtype=torch.int64)
                update_value_tensor = torch.tensor(update_value, device=device, dtype=torch.int32)
                flat_indices_ref.scatter_(0, update_index_tensor, update_value_tensor)
            col_nnz_ref[bidb, bidh] = col_nnz_local

    return flat_indices_ref, col_nnz_ref

@pytest.mark.parametrize("moba_topk", [1, 2, 4, 8, 16, 24, 32, 33, 64])
@pytest.mark.parametrize("moba_chunk_size", [1, 16, 32])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize(
    "batch_size, num_heads, seqlen",
    [(1, 4, 128), (2, 8, 256), (4, 8, 512), (1, 2, 1024)]
)
def test_moba_fused_topk_epilogue(
    batch_size, num_heads, seqlen, head_size, dtype, causal, moba_chunk_size, moba_topk
):
    torch.random.manual_seed(0)
    q_seqlens = torch.randint(
        low=seqlen, high=seqlen + 1, size=(batch_size,), device="cuda"
    )
    # Ensure not all seqlens are 0, which would lead to empty q.
    if torch.all(q_seqlens == 0):
        q_seqlens[0] = seqlen // 2
    k_seqlens = q_seqlens
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"),
                              q_seqlens.cumsum(dim=0).to(torch.int32)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"),
                              k_seqlens.cumsum(dim=0).to(torch.int32)])
    max_seqlen_q = q_seqlens.max().item() if batch_size > 0 and q_seqlens.numel() > 0 else 0
    max_seqlen_k = k_seqlens.max().item() if batch_size > 0 and k_seqlens.numel() > 0 else 0

    q = torch.randn(
        (cu_seqlens_q[-1], num_heads, head_size),
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )
    k = torch.randn(
        (cu_seqlens_k[-1], num_heads, head_size),
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )

    # Compute km and cu_seqlens_km for the kernel
    km_list = []
    seqlens_km = []
    for i in range(batch_size):
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        k_ = k[start_k:end_k]
        if k_.numel() > 0:
            key_gate_weight = _compute_key_gate_weight_ref(k_, moba_chunk_size)
            km_list.append(key_gate_weight)
            seqlens_km.append(key_gate_weight.shape[0])
        else:
            seqlens_km.append(0)
    
    if len(km_list) > 0:
        km = torch.cat(km_list, dim=0)
    else:
        km = torch.empty((0, num_heads, head_size), device='cuda', dtype=dtype)
        
    cu_seqlens_km = torch.tensor([0] + seqlens_km, device='cuda', dtype=torch.int32).cumsum(dim=0)
    cu_seqlens_km = cu_seqlens_km.to(torch.int32)

    # Our optimised kernel (the one we want to test)
    col_offsets, col_nnz, flat_indices, o_values, o_indices = flash_moba_cuda.moba_fused_topk(
        q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km, max_seqlen_q, max_seqlen_k,
        moba_topk, moba_chunk_size, causal
    )
    end_offsets = col_offsets + col_nnz
    flat_indices = flash_moba_cuda.varlen_sort(col_offsets.view(-1), end_offsets.view(-1), flat_indices)
    flat_indices_ref, col_nnz_ref = indices_epilogue_thread_pool(col_offsets, col_nnz, flat_indices, o_indices, cu_seqlens_q, moba_topk)
    assert torch.all(flat_indices == flat_indices_ref)
    assert torch.all(col_nnz == col_nnz_ref)



if __name__ == "__main__":
    test_moba_fused_topk_epilogue(
    batch_size=2, num_heads=1, seqlen=256, head_size=64, dtype=torch.float16, causal=True, moba_chunk_size=1, moba_topk=1
)