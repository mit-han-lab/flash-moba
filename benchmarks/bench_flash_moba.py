# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_moba.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_moba.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_moba import flash_moba_varlen_func
from flash_attn import flash_attn_varlen_func
from flash_moba.moba_test_utils import (
    generate_moba_sparse_mask_topk,
    generate_moba_params_from_sparse_mask,
    decide_lg_block_m
)


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(2, 4096), (2, 8192), (2, 16384), (2, 32768), (2, 65536), (2, 131072), (2, 262144), (2, 524288)]
causal_vals = [True]
headdim_vals = [128]
dim = 2048
dropout_p = 0.0
top_k = 8
# MOBA parameters
# lg_block_m = 512
lg_block_n = 128
print(f"lg_block_n = {lg_block_n}, top_k = {top_k}")

methods = (["FlashMoba", "Flash2"])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            if seqlen >= 262144:
                repeats = 2
            elif seqlen >= 65536:
                repeats = 10
            else:
                repeats = 200

            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            q, k, v = qkv.unbind(dim=2)
            q = rearrange(q, 'b s h d -> (b s) h d')
            k = rearrange(k, 'b s h d -> (b s) h d')
            v = rearrange(v, 'b s h d -> (b s) h d')
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
            max_seqlen = seqlen
            
            

            # lg_block_m = decide_lg_block_m(top_k, lg_block_n, seqlen, causal)
            
            f, b = time_fwd_bwd(
                flash_moba_varlen_func, 
                q, k, v, 
                cu_seqlens, cu_seqlens, 
                max_seqlen, max_seqlen, 
                lg_block_n, top_k,
                causal=causal, 
                repeats=repeats, 
                verbose=False
            )
            time_f[config, "FlashMoba"] = f
            time_b[config, "FlashMoba"] = b
            
            qkv = qkv.detach().requires_grad_(True)
            q, k, v = qkv.unbind(dim=2)
            q = rearrange(q, 'b s h d -> (b s) h d')
            k = rearrange(k, 'b s h d -> (b s) h d')
            v = rearrange(v, 'b s h d -> (b s) h d')
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
            max_seqlen = seqlen
            f2, b2 = time_fwd_bwd(
                flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "Flash2"] = f2
            time_b[config, "Flash2"] = b2
            
            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, time: {(time_f[config, method] * 1000):.2f} ms, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, time: {(time_b[config, method] * 1000):.2f} ms, "
                        f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s, time: {time_f_b[config, method] * 1000:.2f} ms"
                    )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
