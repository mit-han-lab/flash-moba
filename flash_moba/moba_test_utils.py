import torch
from typing import Tuple
from math import ceil
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from flash_moba.bert_padding import pad_input, unpad_input


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def decide_lg_block_m(top_k: int, lg_block_n: int, seqlen: int, causal: bool = False) -> int:
    sparsity = 0.0
    budget = top_k * lg_block_n
    if causal:
        density = (2*(budget * seqlen) - budget**2) / (seqlen**2)
    else:
        density = budget / seqlen
    
    sparsity = 1 - density
    
    if sparsity <= 0.5:
        lg_block_m = 128
    elif sparsity <= 0.7:
        lg_block_m = 256
    elif sparsity <= 0.8:
        lg_block_m = 512
    elif sparsity <= 0.9:
        lg_block_m = 768
    else:
        lg_block_m = 1024

    return lg_block_m



def generate_moba_sparse_mask_topk(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int,
    max_seqlen_k: int,
    lg_block_n: int,
    num_heads: int,
    top_k: int,
    causal: bool = False,
    device: str = "cuda"
) -> torch.Tensor:
    """Generate sparse mask for MOBA attention pattern."""
    batch_size = cu_seqlens_q.shape[0] - 1
    ncol = ceil(max_seqlen_k / lg_block_n)
    base_mask = torch.zeros(batch_size, num_heads, max_seqlen_q, ncol, device=device, dtype=torch.bool)
    
    # Process each batch
    for batch in range(batch_size):
        q_len = cu_seqlens_q[batch + 1] - cu_seqlens_q[batch]
        k_len = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch]
        
        for head in range(num_heads):
            for row in range(q_len):
                # Calculate available column blocks
                end_col = k_len - (q_len - row - 1) if causal else k_len
                col_blocks = max(0, ceil(end_col / lg_block_n))
                
                if col_blocks > 0:
                    # Select blocks randomly and ensure last block is included
                    num_selected = min(top_k - 1, col_blocks - 1)
                    if num_selected > 0:
                        indices = torch.randperm(col_blocks - 1, device=device)[:num_selected]
                        base_mask[batch, head, row, indices] = True
                    base_mask[batch, head, row, col_blocks - 1] = True
                
    return base_mask


def generate_moba_sparse_mask(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int,
    max_seqlen_k: int,
    lg_block_n: int,
    num_heads: int,
    sparsity: float,
    causal: bool = False,
    device: str = "cuda"
) -> torch.Tensor:
    """Generate sparse mask for MOBA attention pattern."""
    batch_size = cu_seqlens_q.shape[0] - 1
    ncol = ceil(max_seqlen_k / lg_block_n)
    base_mask = torch.zeros(batch_size, num_heads, max_seqlen_q, ncol, device=device, dtype=torch.bool)
    density = 1 - sparsity
    # Handle special cases
    if density == 0.0:
        return base_mask
    if density == 1.0:
        return torch.ones_like(base_mask)
    
    # Process each batch
    for batch in range(batch_size):
        q_len = cu_seqlens_q[batch + 1] - cu_seqlens_q[batch]
        k_len = cu_seqlens_k[batch + 1] - cu_seqlens_k[batch]
        
        for head in range(num_heads):
            for row in range(q_len):
                # Calculate available column blocks
                end_col = k_len - (q_len - row - 1) if causal else k_len
                col_blocks = max(0, ceil(end_col / lg_block_n))
                
                if col_blocks > 0:
                    # Select blocks randomly and ensure last block is included
                    num_selected = max(1, int(density * col_blocks))
                    indices = torch.randperm(col_blocks, device=device)[:num_selected]
                    base_mask[batch, head, row, indices] = True
                    base_mask[batch, head, row, col_blocks - 1] = True
                
    return base_mask


def generate_moba_params_from_sparse_mask(
    sparse_mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    lg_block_n: int,
    max_seqlen_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate MOBA sparse pattern parameters from a sparse mask.
    
    Args:
        sparse_mask: Boolean mask of shape (batch_size, num_heads, max_seqlen_q, n_blocks_k)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        lg_block_n: Logical block size for K dimension
        max_seqlen_k: Maximum key sequence length
        
    Returns:
        moba_col_offsets: Column offsets tensor (batch_size, num_heads, max_cols)
        moba_col_nnz: Non-zero counts per column (batch_size, num_heads, max_cols)
        moba_row_indices: Row indices tensor (flattened) (total_selected_rows)
    """
    batch_size, num_heads = sparse_mask.shape[:2]
    device = sparse_mask.device
    max_cols = ceil(max_seqlen_k / lg_block_n)
    
    # Initialize outputs
    col_offsets = torch.zeros((batch_size, num_heads, max_cols), dtype=torch.int64, device=device)
    col_nnz = torch.zeros((batch_size, num_heads, max_cols), dtype=torch.int32, device=device)
    row_indices = []
    offset = 0
    
    # Process each batch and head
    for b in range(batch_size):
        seq_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
        seq_k = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
        n_blocks = ceil(seq_k / lg_block_n)
        
        for h in range(num_heads):
            pattern = sparse_mask[b, h, :seq_q, :n_blocks]
            
            # Process each column
            for c in range(n_blocks):
                rows = pattern[:, c].nonzero(as_tuple=True)[0]
                col_offsets[b, h, c] = offset
                col_nnz[b, h, c] = len(rows)
                
                if len(rows) > 0:
                    row_indices.extend(rows.sort()[0].tolist())
                    offset += len(rows)
    
    # Convert to tensor
    row_indices_tensor = torch.tensor(row_indices or [0], dtype=torch.int32, device=device)[:len(row_indices)]
    
    return col_offsets, col_nnz, row_indices_tensor


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False, key_leftpad=None
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()


def tailor_mixedmask_for_test(spanded_base_mixedmask, seqlen_q, seqlen_k):
    batch_size = spanded_base_mixedmask.shape[0]
    nheads = spanded_base_mixedmask.shape[1]
    spanded_base_mixedmask = spanded_base_mixedmask[:, :, :seqlen_q, :seqlen_k]
    pad_blockmask = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device = spanded_base_mixedmask.device)
    pad_blockmask[:, :, :spanded_base_mixedmask.shape[2], :spanded_base_mixedmask.shape[3]] = spanded_base_mixedmask
    spanded_base_mixedmask = pad_blockmask
    spanded_base_mixedmask = spanded_base_mixedmask.contiguous()
    return spanded_base_mixedmask


def prepare_moba_ref_mask(moba_sparse_mask, lg_block_n, actual_seqlen_q, actual_seqlen_k, device="cuda"):

    moba_sparse_mask = repeat(moba_sparse_mask, "b h s_m s_n -> b h s_m (s_n d_n)", d_n=lg_block_n)
    moba_sparse_mask = tailor_mixedmask_for_test(moba_sparse_mask, actual_seqlen_q, actual_seqlen_k)
    moba_sparse_mask = ~moba_sparse_mask
    
    return moba_sparse_mask


def attention_moba_sparse_ref(
    q,
    k,
    v,
    moba_sparse_mask,
    lg_block_n, 
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    
    scores.masked_fill_(rearrange(moba_sparse_mask, "b h t s -> b h t s"), float("-inf"))
    
    # print("processed blockmask: ", rearrange(~base_blockmask, "h t s -> 1 h t s"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(torch.bitwise_or(local_mask, rearrange(moba_sparse_mask, "b h t s -> b h t s")), dim=-1, keepdim=True), 0.0)

    attention = attention.masked_fill(rearrange(moba_sparse_mask, "b h t s -> b h t s"), 0.0) 
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def moba_verifier(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    # MOBA sparse pattern parameters
    moba_topk,
    moba_kblock_size,
    causal,

    col_offsets, 
    col_nnz, 
    flat_indices,

    eps=1e-3,
):
    """
    Asserts that the output of flash_moba_topk_varlen is correct, given an epsilon tolerance.
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of value tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), cumulative sequence lengths of the queries in the batch, 
           used to index into q.
        cu_seqlens_k: (batch_size + 1,), cumulative sequence lengths of the keys/values in the batch, 
           used to index into k and v.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        moba_topk: int. Number of top-k blocks to attend to per query block.
        moba_kblock_size: int. Size of each key block.
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        col_offsets: (batch_size, nheads, max_lg_col_num), where max_lg_col_num denotes the maximum 
           number of column blocks (of size moba_kblock_size) in a batch.
           The start offset within the flat list of indices that contain the list of queries that
           attend to this key-block.
        col_nnz: (batch_size, nheads, max_lg_col_num). The number of non-zero entries for each
           key-block, i.e., the number of query tokens that attend to this key-block.
        indices: (total_q * nheads * moba_topk,). The flat list of indices of query tokens that attend
           to each key-block, as specified by col_offsets and col_nnz.
    Raises:
        AssertionError: if the output is incorrect.
    """
    batch = cu_seqlens_q.numel() - 1
    
    num_heads = q.shape[1]
    num_heads_k = k.shape[1]
    h_h_k_ratio = num_heads // num_heads_k
    max_lg_col_num = (max_seqlen_k + moba_kblock_size - 1) // moba_kblock_size

    for batch_idx in range(batch):
        for h in range(num_heads):
            batch_start_q = cu_seqlens_q[batch_idx].item()
            batch_end_q = cu_seqlens_q[batch_idx + 1].item()
            batch_start_k = cu_seqlens_k[batch_idx].item()
            batch_end_k = cu_seqlens_k[batch_idx + 1].item()

            # get qkv of this batch
            q_ = q[batch_start_q:batch_end_q, h]
            k_ = k[batch_start_k:batch_end_k, h // h_h_k_ratio]
            # calc key gate weight
            key_gate_weight = []
            batch_size = batch_end_k - batch_start_k
            num_block = math.ceil(batch_size / moba_kblock_size)
            for block_idx in range(0, num_block):
                block_start = block_idx * moba_kblock_size
                block_end = min(batch_size, block_start + moba_kblock_size)
                key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
            key_gate_weight = torch.cat(key_gate_weight, dim=0)  # [ N, D ]
            # calc & mask gate
            gate = torch.einsum("sd,nd->sn", q_, key_gate_weight)  # [ S, N ]
            
            for i in range(num_block):
                # select the future Qs that can attend to KV chunk i
                if causal:
                    gate[: (i + 1) * moba_kblock_size, i] = float("-inf")
                    gate[i * moba_kblock_size : (i + 1) * moba_kblock_size, i] = float("inf")
            # gate_top_k_idx = gate_top_k_val = [ S K ]
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=min(moba_topk, num_block), dim=-1, largest=True, sorted=True
            )

            for i in range(0, max_lg_col_num):
                le = col_offsets[batch_idx, h, i].item()
                ri = col_offsets[batch_idx, h, i].item() + col_nnz[batch_idx, h, i].item()
                q_min_gate_values = gate_top_k_val[flat_indices[le : ri], -1]
                q_gate_value = gate[flat_indices[le : ri], i]
                print("q_min_gate_values: ", q_min_gate_values)
                print("q_gate_value: ", q_gate_value)

                assert torch.all(q_gate_value >= q_min_gate_values - eps), f"mismatch at batch {batch_idx}, head {h}, key-block {i}"

