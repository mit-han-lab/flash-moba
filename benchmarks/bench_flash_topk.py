import math
import torch
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
import torch.utils.benchmark as benchmark

from flash_moba import flash_topk_varlen_func


def benchmark_forward(
    fn, *inputs, repeats=200, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def main():
    """
    Main benchmarking function.
    """
    device = 'cuda'
    dtype = torch.float16
    
    configs = []
    for causal in [True]:
        for headdim in [128]:
            for moba_chunk_size in [128]:
                for seqlen in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
                    batch_size = 2 #64 * 1024 // seqlen
                    configs.append(
                        (causal, headdim, batch_size, seqlen, moba_chunk_size, 8)
                    )
    
    results = []
    
    for causal, headdim, batch_size, seqlen, moba_chunk_size, moba_topk in configs:
        print(
            f"Causal: {causal}, HeadDim: {headdim}, BS: {batch_size}, "
            f"SeqLen: {seqlen}, ChunkSize: {moba_chunk_size}, TopK: {moba_topk}"
        )
        
        nheads = 16 #2048 // headdim
        q = torch.randn(
            batch_size * seqlen, nheads, headdim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size * seqlen, nheads, headdim, device=device, dtype=dtype
        )
        
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
        )
        
        args = [
            q, k, cu_seqlens, cu_seqlens, seqlen, seqlen,
            moba_topk, moba_chunk_size, causal
        ]
        
        t_flash = benchmark_forward(flash_topk_varlen_func, *args, verbose=False)[1].mean
        
        results.append({
            "causal": causal,
            "headdim": headdim,
            "batch_size": batch_size,
            "seqlen": seqlen,
            "moba_chunk_size": moba_chunk_size,
            "moba_topk": moba_topk,
            "time (ms)": t_flash * 1000,
        })

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":
    main()