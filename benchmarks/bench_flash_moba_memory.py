"""
Peak memory analysis of FlashMoBA components.

This script measures the peak memory usage of three different attention implementations:
1. Flash Attention (Baseline)
2. Flash MoBA (Our efficient implementation)
3. MoBA (Official implementation)

The goal is to compare the memory footprint of these methods under various configurations.

CUDA Synchronization:
torch.cuda.synchronize() is used to ensure accurate measurement of memory usage after
all asynchronous GPU operations are complete.
"""

import torch
import random
import itertools
import numpy as np
from einops import rearrange
import pandas as pd
from datetime import datetime
import math

# from flash_attn import flash_attn_varlen_func
from flash_attn import flash_attn_varlen_func
from flash_moba import flash_moba_varlen_func

try:
    from moba.moba_efficient import moba_attn_varlen
except ImportError:
    print("Warning: moba is not installed, MoBA comparison is skipped.")
    moba_attn_varlen = None


def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    """Generate test data for timing experiments."""
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # # gen qkv
    # q = torch.randn(
    #     (seqlen, num_q_head, headdim), dtype=dtype, device=device, requires_grad=True
    # )
    # k = torch.randn(
    #     (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    # )
    # v = torch.randn(
    #     (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    # )

    # # gen cu seqlen
    # cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
    # cu_seqlen.sort()
    # cu_seqlen = [0] + cu_seqlen + [seqlen]
    # cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)

    # # max_seqlen
    # max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])

    # return q, k, v, cu_seqlen, max_seqlen.item()
    assert num_q_head == num_kv_head, "For simplicity, only support num_q_head == num_kv_head"
    qkv = torch.randn(batch, seqlen, 3, num_q_head, headdim, device=device, dtype=dtype, requires_grad=True)
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b s h d -> (b s) h d')
    k = rearrange(k, 'b s h d -> (b s) h d')
    v = rearrange(v, 'b s h d -> (b s) h d')
    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
    max_seqlen = seqlen
    return q, k, v, cu_seqlens, max_seqlen


def test_moba_peak_memory(batch, seqlen, head, head_dim, moba_chunk_size, moba_topk, dtype=torch.bfloat16, warmup_iters=1, test_iters=1):
    """
    Peak memory usage comparison for Flash Attention vs FlashMoBA.
    
    Args:
        batch: Batch size
        head: Number of attention heads  
        seqlen: Sequence length
        head_dim: Head dimension
        moba_chunk_size: Chunk size for MoBA
        moba_topk: Top-k chunks to select
        dtype: Data type for tensors
        warmup_iters: Number of warmup iterations
        test_iters: Number of test iterations for measurement
    """
    print(f"\n{'='*80}")
    print(f"FlashMoBA Peak Memory Analysis")
    print(f"{'='*80}")
    print(f"Config: batch={batch}, heads={head}, seqlen={seqlen}, chunk_size={moba_chunk_size}, topk={moba_topk}")
    print(f"Warmup iterations: {warmup_iters}, Test iterations: {test_iters}")
    print(f"{'='*80}")
    
    
    # =========================
    # Flash Attention Baseline
    # =========================
    
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    vo_grad = torch.randn_like(q)
    
    # Warmup
    for _ in range(warmup_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad, retain_graph=True)
    
    # Measurement
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(test_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        torch.autograd.backward(o, vo_grad, retain_graph=True)
        
    torch.cuda.synchronize()
    peak_memory_flash = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  Flash Attention Peak Memory: {peak_memory_flash:.2f}MB")
    
    # =========================
    # Flash MoBA Comparison
    # =========================
    
    del q, k, v, cu_seqlen, max_seqlen, vo_grad
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
    vo_grad = torch.randn_like(q)
    
    
    # Warmup
    for _ in range(warmup_iters):
        o = flash_moba_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, 
                                        moba_chunk_size,
                                        moba_topk,
                                        causal=True)
        torch.autograd.backward(o, vo_grad, retain_graph=True)
        
    
    # Measurement
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(test_iters):
        o = flash_moba_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, 
                                        moba_chunk_size,
                                        moba_topk,
                                        causal=True)
        torch.autograd.backward(o, vo_grad, retain_graph=True)
        
    torch.cuda.synchronize()
    peak_memory_flash_moba = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  FlashMoBA Peak Memory: {peak_memory_flash_moba:.2f}MB")
    
    
    # =========================
    # MoBA Comparison
    # =========================
    try:
        del q, k, v, cu_seqlen, max_seqlen, vo_grad
        q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype)
        vo_grad = torch.randn_like(q)
        
        
        # Warmup
        for _ in range(warmup_iters):
            o = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, 
                                            moba_chunk_size,
                                            moba_topk)
            torch.autograd.backward(o, vo_grad, retain_graph=True)
            
        
        # Measurement
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(test_iters):
            o = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, 
                                            moba_chunk_size,
                                            moba_topk)
            torch.autograd.backward(o, vo_grad, retain_graph=True)
            
        torch.cuda.synchronize()
        peak_memory_moba = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  MoBA Peak Memory: {peak_memory_moba:.2f}MB")
    except Exception as e:
        print(f"\tError with config {batch, seqlen, head, head_dim, moba_chunk_size, moba_topk}: {e}")
        peak_memory_moba = float('nan')
    
    results_data = {
        'batch': batch,
        'seqlen': seqlen,
        'head': head,
        'head_dim': head_dim,
        'moba_chunk_size': moba_chunk_size,
        'moba_topk': moba_topk,
        'dtype': str(dtype),
        'flash_peak_memory_mb': round(peak_memory_flash, 4),
        'flash_peak_memory_gb': round(peak_memory_flash / 1024, 4),
        'flash_moba_peak_memory_mb': round(peak_memory_flash_moba, 4),
        'flash_moba_peak_memory_gb': round(peak_memory_flash_moba / 1024, 4),
        'moba_peak_memory_mb': round(peak_memory_moba, 4) if peak_memory_moba is not float('nan') else float('nan'),
        'moba_peak_memory_gb': round(peak_memory_moba / 1024, 4) if peak_memory_moba is not float('nan') else float('nan'),
    }

    return results_data


def run_breakdown_experiments():
    # Test configurations
    num_heads = [16]
    headdim_vals = [128]
    bs_seqlen_vals = [(2, 4096), (2, 8192), (2, 16384), (2, 32768), (2, 65536), (2, 131072), (2, 262144), (2, 524288)]
    kblock_size_vals = [128]#, 256, 512]
    topk_vals = [8]

    configs = itertools.product(bs_seqlen_vals, num_heads, headdim_vals, kblock_size_vals, topk_vals)
    
    all_results = []
    for config in configs:
        (batch, seqlen), num_heads, head_dim, kblock_size, topk = config

        # if seqlen <= 2**13:  # up to 8192
        #     warmup_iters = 10
        #     test_iters = 20
        # elif seqlen <= 2**15:  # up to 32768
        #     warmup_iters = 5
        #     test_iters = 10
        # else:  # for 65536 and longer
        #     warmup_iters = 3
        #     test_iters = 5
        warmup_iters = 1
        test_iters = 2
            
        try:
            results = test_moba_peak_memory(batch, seqlen, num_heads, head_dim, kblock_size, topk, warmup_iters=warmup_iters, test_iters=test_iters)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"\tError with config {config}: {e}")
            continue

    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moba_peak_memory_profile_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\nSuccessfully saved performance breakdown to {filename}")


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    run_breakdown_experiments()
