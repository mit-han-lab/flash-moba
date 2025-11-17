import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_moba import (
    flash_moba_attn_varlen_func,
)
from flash_moba.bert_padding import pad_input, unpad_input

from flash_moba.moba_test_utils import (
    generate_moba_sparse_mask,
    generate_moba_params_from_sparse_mask,
    generate_moba_sparse_mask_topk,
    prepare_moba_ref_mask,
    attn_bias_from_alibi_slopes,
    generate_random_padding_mask,
    generate_qkv,
    attention_moba_sparse_ref,
    convert_flash_attn_S_to_softmax,
    normalize_flash_attn_S,
    get_dropout_fraction,
    attention_ref,
)


MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('mha_type', ["mqa"])
@pytest.mark.parametrize("deterministic", [False]) #, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("alibi", [False]) #, True])
# @pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [False]) #, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("softcap", [0.0])#, 50.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_moba_attn_varlen_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, softcap
):
    print(f"[Args] seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, d: {d}, dropout_p: {dropout_p}, causal: {causal}, local: {local}, alibi: {alibi}, deterministic: {deterministic}, mha_type: {mha_type}, dtype: {dtype}, softcap: {softcap} \n")
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if softcap > 0.0 and dropout_p > 0.0:
        pytest.skip("Softcap and dropout not supported together")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap

    k = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
    )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    
    # # MOBA parameters
    # lg_block_m = 64
    # lg_block_n = 64
    
    # moba_col_offsets, moba_col_nnz, moba_row_indices = generate_moba_sparse_pattern(
    #     q=q_unpad,
    #     k=k_unpad,
    #     cu_seqlens_q=cu_seqlens_q,
    #     cu_seqlens_k=cu_seqlens_k,
    #     lg_block_m=lg_block_m,
    #     lg_block_n=lg_block_n,
    #     topk_per_row=1,
    #     max_seqlen_q=seqlen_q,
    #     max_seqlen_k=seqlen_k,
    # )
    lg_block_m = 128
    lg_block_n = 64
    topk = 4
    print(f"topk: {topk}")
    moba_sparse_mask = generate_moba_sparse_mask_topk(
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        lg_block_n,
        nheads,
        top_k=topk,
        causal=causal,
    )
    # print(f"moba_sparse_mask.shape: {moba_sparse_mask.shape}")
    # print(f"moba_sparse_mask: {moba_sparse_mask.to(torch.int32)}")
    moba_col_offsets, moba_col_nnz, moba_row_indices = generate_moba_params_from_sparse_mask(
        sparse_mask=moba_sparse_mask,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        lg_block_n=lg_block_n,
        max_seqlen_k=max_seqlen_k,
    )
    # print(f"moba_col_offsets: {moba_col_offsets}")
    # print(f"moba_col_nnz: {moba_col_nnz}")
    # print(f"moba_row_indices: {moba_row_indices}")
    
    moba_sparse_mask_ref = prepare_moba_ref_mask(moba_sparse_mask, lg_block_n, seqlen_q, seqlen_k)
    
    
    
    
    out_unpad, sm_lse, S_dmask = flash_moba_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        moba_col_offsets,
        moba_col_nnz,
        moba_row_indices,
        lg_block_m,
        lg_block_n,
        dropout_p,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()

        k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_moba_sparse_ref(
        q,
        k,
        v,
        moba_sparse_mask_ref,
        lg_block_n,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
    )
    out_pt, attn_pt = attention_moba_sparse_ref(
        q,
        k,
        v,
        moba_sparse_mask_ref,
        lg_block_n,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    g = torch.randn_like(out)
    if ((d <= MAX_HEADDIM_SM8x or dropout_p == 0) or (is_sm80 or is_sm90)):
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        
        # print(f"dK: {dk}")
        # print(f"dk_ref: {dk_ref}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    # if dropout_p > 0.0:
    #     assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    #     # With alibi, many of the prob values are 0.0 & -0.0 so dropout_fraction isn't accurate
    #     if not alibi:
    #         assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.04)

    if (d <= MAX_HEADDIM_SM8x or dropout_p == 0) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()



# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("causal", [False, True])
# # @pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize("d", [16, 32, 64])
# # @pytest.mark.parametrize('d', [16])
# def test_flash_attn_bwd_varlen_overflow(d, causal, dtype):
#     """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
#     in the case where seqlen % 128 != 0 or varlen.
#     """
#     device = "cuda"
#     # set seed
#     torch.random.manual_seed(0)
#     nheads = 5
#     q_cuseqlen = torch.tensor([0, 76, 110, 256], device=device, dtype=torch.int32)
#     k_cuseqlen = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
#     Mq = 256
#     Mk = 3

#     q = torch.randn([Mq, nheads, d], dtype=dtype, device=device) * 3
#     k, v = [torch.randn([Mk, nheads, d], dtype=dtype, device=device) * 3 for _ in range(2)]
#     q.requires_grad_(True)
#     k.requires_grad_(True)
#     v.requires_grad_(True)

#     out = flash_moba_attn_varlen_func(q, k, v, q_cuseqlen, k_cuseqlen, Mq, Mk, causal=causal)
#     g = torch.randn_like(out)
#     out.backward(g)

#     assert not q.grad.isnan().any()
#     assert not k.grad.isnan().any()
#     assert not v.grad.isnan().any()



# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# # @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("local", [False, True])
# # @pytest.mark.parametrize("local", [True])
# @pytest.mark.parametrize("causal", [False, True])
# # @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# # @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# # @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# # @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# # @pytest.mark.parametrize('d', [56, 80])
# # @pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize("swap_sq_sk", [False, True])
# # @pytest.mark.parametrize("swap_sq_sk", [True])
# @pytest.mark.parametrize(
#     "seqlen_q,seqlen_k",
#     [
#         (1, 239),
#         (3, 799),
#         (127, 512),
#         (127, 513),
#         (113, 203),
#         (128, 217),
#         (113, 211),
#         (108, 256),
#         (256, 512),
#         (1023, 1024),
#     ],
# )
# # @pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 128)])
# def test_flash_attn_varlen_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
#     if (
#         max(seqlen_q, seqlen_k) >= 2048
#         and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
#     ):
#         pytest.skip()  # Reference implementation OOM
#     if swap_sq_sk:
#         seqlen_q, seqlen_k = seqlen_k, seqlen_q
#     device = "cuda"
#     # set seed
#     torch.random.manual_seed(0)
#     batch_size = 2
#     nheads = 9
#     window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
#     q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
#     k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
#     v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
#     query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
#     key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
#     (
#         q_unpad,
#         k_unpad,
#         v_unpad,
#         cu_seqlens_q,
#         cu_seqlens_k,
#         max_seqlen_q,
#         max_seqlen_k,
#         q,
#         k,
#         v,
#         output_pad_fn,
#         dq_pad_fn,
#         dk_pad_fn,
#     ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
#     out = flash_moba_attn_varlen_func(
#         q_unpad,
#         k_unpad,
#         v_unpad,
#         cu_seqlens_q,
#         cu_seqlens_k,
#         max_seqlen_q,
#         max_seqlen_k,
#         0.0,
#         causal=causal,
#         window_size=window_size,
#         deterministic=True,
#     )

#     g = torch.randn_like(out)
#     dq0, dk0, dv0 = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
#     for _ in range(50):
#         dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
#         assert torch.equal(dv, dv0)
#         assert torch.equal(dk, dk0)
#         assert torch.equal(dq, dq0)



if __name__ == "__main__":
    test_flash_moba_attn_varlen_output(
        seqlen_q=1,
        seqlen_k=147,
        d=32,
        dropout_p=0.0,
        causal=False,
        local=False,
        alibi=False,
        deterministic=False,
        mha_type="mha",
        dtype=torch.float16,
        softcap=0.0,
    )
    
# def test_flash_moba_attn_varlen_output(
#     seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, softcap
# )