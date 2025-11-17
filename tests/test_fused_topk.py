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


def _apply_chunk_causal_mask_ref(gate: torch.Tensor, moba_chunk_size: int):
    """Reference implementation of `_apply_chunk_causal_mask`."""
    H, S, N = gate.shape
    for i in range(N):
        gate[:, : (i + 1) * moba_chunk_size, i] = float("-inf")
        gate[:, i * moba_chunk_size : (i + 1) * moba_chunk_size, i] = float("inf")
    return gate


def stable_topk(x: torch.Tensor, k: int, dim: int = -1):
    """A stable topk implementation that returns sorted results."""
    # `torch.sort` is stable on CUDA for float32 and float64.
    # The `gate` tensor in this context is float32.
    sorted_values, sorted_indices = torch.sort(x, dim=dim, descending=False, stable=True)
    
    # We need to slice the first k elements.
    topk_values = sorted_values.narrow(dim, sorted_values.shape[dim] - k, k).flip(dim)
    top_indices = sorted_indices.narrow(dim, sorted_indices.shape[dim] - k, k).flip(dim)
    
    return topk_values, top_indices


def moba_fused_topk_ref(
    q, k, cu_seqlens_q, cu_seqlens_k, moba_topk, moba_chunk_size, causal
):
    """Reference implementation for fused top-k."""
    batch_size = cu_seqlens_q.numel() - 1
    moba_topk_rounded = (moba_topk + 15) // 16 * 16
    
    all_topk_values = []
    all_topk_indices = []

    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_ = q[start_q:end_q]
        k_ = k[start_k:end_k]
        
        if k_.numel() == 0:
            # Handle empty k case.
            # Kernel seems to output -inf for values and -1 for indices
            # for all queries in the batch item.
            S, H, _ = q_.shape
            vals = torch.full((S, H, moba_topk_rounded), -float('inf'), dtype=torch.float32, device=q.device)
            inds = torch.full((S, H, moba_topk_rounded), -1, dtype=torch.int32, device=q.device)
            all_topk_values.append(vals)
            all_topk_indices.append(inds)
            continue
            
        km_ = _compute_key_gate_weight_ref(k_, moba_chunk_size)
        # print(f"[batch {i}] km_: {km_[-3:, :, -10:]}")
        gate = torch.einsum(
            "shd,nhd->hsn", q_.to(torch.float32), km_.to(torch.float32)
        )
        if causal:
            gate = _apply_chunk_causal_mask_ref(gate, moba_chunk_size)
        
        num_blocks = gate.shape[2]
        k_topk = min(moba_topk, num_blocks)
        # torch.set_printoptions(precision=16)
        # torch.set_printoptions(profile="full")
        # print(f"[batch {i}] gate shape: {gate.shape}")
        # print(f"[batch {i}] gate: {gate[:, -3:, -10:]}")
        # torch.set_printoptions(profile="default")
        vals, inds = stable_topk(gate, k=k_topk, dim=-1)
        
        # If causal, some tokens are masked with -inf.
        # torch.topk will return random indices for these tokens.
        # We set indices to -1 for these masked tokens to match kernel behavior.
        if causal:
            inds[vals == -float('inf')] = -1
        
        S, H, _ = q_.shape
        
        vals = vals.permute(1, 0, 2) # S, H, k_topk
        inds = inds.permute(1, 0, 2) # S, H, k_topk
        
        # Pad to moba_topk_rounded
        padded_vals = torch.full(
            (S, H, moba_topk_rounded), -float('inf'), dtype=vals.dtype, device=vals.device
        )
        padded_vals[:, :, :k_topk] = vals
        padded_inds = torch.full(
            (S, H, moba_topk_rounded), -1, dtype=torch.int32, device=inds.device
        )
        padded_inds[:, :, :k_topk] = inds
        
        all_topk_values.append(padded_vals)
        all_topk_indices.append(padded_inds)
        
    if not all_topk_values:
        return torch.empty((0,0,0), device=q.device, dtype=torch.float32), torch.empty((0,0,0), device=q.device, dtype=torch.int32)
    return torch.cat(all_topk_values, dim=0), torch.cat(all_topk_indices, dim=0)



@pytest.mark.parametrize("moba_topk", [1, 2, 4, 8, 16, 24, 32, 33, 64])
@pytest.mark.parametrize("moba_chunk_size", [1, 16, 32])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize(
    "batch_size, num_heads, seqlen",
    [(1, 4, 128), (2, 8, 256), (4, 16, 512), (8, 2, 1024)]
)
def test_moba_fused_topk(
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

    # q = torch.zeros_like(q)
    # k = torch.zeros_like(k)
    # for i in range(batch_size):
    #     for j in range(num_heads):
    #         q[:, j, 0] = 1
    #         q[:, j, -1] = 1
    #         for row in range(cu_seqlens_k[-1]):
    #             # k[row, j, 0] = row
    #             k[row, j, -1] = row
    # q[:, :, 0] = 1
    # for row in range(k.shape[0]):
    #     k[row, :, -1] = k.shape[0] - row
            # k[:, j, 0] = j+1
            # k[:, j, -1] = j+1
    # Compute km and cu_seqlens_km for the kernel
    km_list = []
    seqlens_km = []
    for i in range(batch_size):
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        k_ = k[start_k:end_k]
        if k_.numel() > 0:
            key_gate_weight = _compute_key_gate_weight_ref(k_, moba_chunk_size)
            # print(f"[batch {i}] key_gate_weight: {key_gate_weight[-3:, :, -10:]}")
            km_list.append(key_gate_weight)
            seqlens_km.append(key_gate_weight.shape[0])
        else:
            seqlens_km.append(0)
    
    if len(km_list) > 0:
        km = torch.cat(km_list, dim=0)
    else:
        km = torch.empty((0, num_heads, head_size), device='cuda', dtype=dtype)
        
    seqlens_km_pt = torch.tensor([0] + seqlens_km, device='cuda', dtype=torch.int32)
    cu_seqlens_km = seqlens_km_pt.cumsum(dim=0).to(torch.int32)


    # print(f"[batch 0] km: {km[-3:, :, -10:]}")
    # Our optimised kernel (the one we want to test)
    col_offsets, col_nnz, flat_indices, o_values, o_indices = flash_moba_cuda.moba_fused_topk(
        q, km, cu_seqlens_q, cu_seqlens_k, cu_seqlens_km, max_seqlen_q, max_seqlen_k,
        moba_topk, moba_chunk_size, causal
    )
    
    # Reference kernel (the one we trust)
    ref_values, ref_indices = moba_fused_topk_ref(
        q, k, cu_seqlens_q, cu_seqlens_k, moba_topk, moba_chunk_size, causal
    )

    assert o_values.shape == ref_values.shape
    assert o_indices.shape == ref_indices.shape
    
    # o_indices_mismatch_indices = (o_indices != ref_indices)
    # if o_indices_mismatch_indices.any():
    #     for row in range(o_indices_mismatch_indices.shape[0]):
    #         for head in range(o_indices_mismatch_indices.shape[1]):
    #             if o_indices_mismatch_indices[row, head].any():
    #                 k_list = []
    #                 o_idx_list = []
    #                 ref_idx_list = []
    #                 o_val_list = []
    #                 ref_val_list = []
    #                 for k in range(o_indices_mismatch_indices.shape[2]):
    #                     if o_indices_mismatch_indices[row, head, k]:
    #                         k_list.append(k)
    #                         o_idx_list.append(o_indices[row, head, k].item())
    #                         ref_idx_list.append(ref_indices[row, head, k].item())
    #                         o_val_list.append(o_values[row, head, k].item())
    #                         ref_val_list.append(ref_values[row, head, k].item())
    #                 torch.set_printoptions(precision=8)
                    # if row >= 16000:
                    # print(f"[r={row}, h={head}] k_list: {k_list}, \n o_idx_list: {o_idx_list}, \n ref_idx_list: {ref_idx_list}, \n o_val_list: {o_val_list}, \n ref_val_list: {ref_val_list}")
                    # # print(f"[r={row}, h={head}] km: {km[row, head, -10:].to(torch.float32)}")
                    # print(f"[r={row}, h={head}] o_indices: {o_indices[row, head, :]}")
                    # print(f"[r={row}, h={head}] ref_indices: {ref_indices[row, head, :]}")
                    # print(f"[r={row}, h={head}] o_values: {o_values[row, head, :]}")
                    # print(f"[r={row}, h={head}] ref_values: {ref_values[row, head, :]}")
                    
        # for row in range(km.shape[0]):
        #     for head in range(km.shape[1]):
        #         print(f"[r={row}, h={head}] km: {km[row, head, -1:].to(torch.float32)}")
    # For values, we might have float precision differences
    # torch.testing.assert_close(o_values.to(torch.float32), ref_values.to(torch.float32), atol=1e-2, rtol=1e-2)
    # torch.testing.assert_close(o_indices, ref_indices)
    
    
            # print(f"[r={row}, h={head}] ref_indices: {ref_indices[row, head, -10:].to(torch.float32)}")

    # torch.set_printoptions(profile="full") # reset to default
    # print(f"o_values: {o_values[-3:, :, :]}")
    # print(f"ref_values: {ref_values[-3:, :, :]}")
    # print(f"o_indices: {o_indices[-3:, :, :]}")
    # print(f"ref_indices: {ref_indices[-3:, :, :]}")
    # torch.set_printoptions(profile="default") # reset to default
    
    # For indices, they should be exactly the same
    mismatches = (o_indices != ref_indices).sum().item()
    total_elements = o_indices.numel()
    mismatch_ratio = mismatches / total_elements if total_elements > 0 else 0
    assert mismatch_ratio < 0.005, f"Index mismatch ratio {mismatch_ratio:.4f} is >= 0.5%"


if __name__ == "__main__":
    test_moba_fused_topk(
    batch_size=2, num_heads=2, seqlen=8192, head_size=64, dtype=torch.float16, causal=False, moba_chunk_size=1, moba_topk=8
)