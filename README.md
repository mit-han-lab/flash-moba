# FlashMoBA

FlashMoBA is a memory-efficient sparse attention mechanism designed to accelerate the training and inference of long-sequence models. This repository provides the official implementation of FlashMoBA from the following paper, which is implemented based on [FlashAttention](https://github.com/Dao-AILab/flash-attention) 2.8.3 and is inspired by [MoBA](https://arxiv.org/abs/2502.13189).

![FlashMoBA](assets/flash_moba_attention.png)

**Optimizing Mixture of Block Attention**  
Guangxuan Xiao<sup>&#42;</sup>, Junxian Guo<sup>&#42;</sup>, Kasra Mazaheri, Song Han

Paper: [Optimizing Mixture of Block Attention]()

## News

- [2025/11] We release the implementation of FlashMoBA 2.0.0, which is implemented based on FlashAttention 2.8.3.

## Installation
**Requirements:**
- CUDA toolkit 12.9 and above (we need the support of 64-bit offset in `DeviceSegmentedSort` from [NVIDIA/cccl#3308](https://github.com/NVIDIA/cccl/pull/3308)).
- PyTorch 2.8 and above.
- `packaging` Python package (`pip install packaging`)
- `ninja` Python package (`pip install ninja`)
- Linux.

\* Make sure `ninja` is installed and discoverable. You can verify this by running `ninja --version` and checking its exit code. If it returns a non-zero exit code, we recommend reinstalling it (`pip uninstall -y ninja && pip install ninja`). Without `ninja`, compilation can be very slow as it won't parallelize across CPU cores. With `ninja`, compilation typically takes 3-5 minutes on a multi-core machine.

**To install:**
```sh
MAX_JOBS=32 python setup.py install
```

**Interface:** `flash_moba/flash_moba_interface.py`

## How to use FlashMoBA

FlashMoBA operates on variable-length sequences packed into single tensors. This format avoids wasting computation on padding tokens and is highly efficient.

The primary entry point is `flash_moba_varlen_func`. This function is a convenient wrapper that first computes the sparse attention pattern based on the Mixture of Block Attention (MoBA) algorithm, and then performs the attention computation.

```python
from flash_moba import flash_moba_varlen_func

def flash_moba_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    moba_chunk_size,
    moba_topk,
    causal=True,
):
    """
    This is the main entry point for FlashMoBA. It first computes the sparse
    attention pattern based on Mixture-of-Blocks Attention (MOBA) and then performs
    the attention computation. This function is designed for variable length sequences.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        moba_chunk_size: int. The chunk size for MOBA.
        moba_topk: int. The number of top-k key blocks to select for each query block.
        causal: bool. Whether to apply causal attention mask. Default to True.

    Return:
        out: (total_q, nheads, headdim).
    """
```

For more advanced use cases that require finer control, FlashMoBA also exposes the two underlying steps as separate functions:

1.  `flash_topk_varlen_func`: Computes the sparse attention pattern (top-k indices).
2.  `flash_moba_attn_varlen_func`: Performs attention using a pre-computed pattern.

This two-step approach is useful if you want to inspect the sparse indices or reuse a pattern.

```python
from flash_moba import flash_topk_varlen_func, flash_moba_attn_varlen_func

def flash_topk_varlen_func(
    q,
    k,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    # MOBA sparse pattern parameters
    moba_topk,
    moba_chunk_size,
    causal=False,
):
    """
    Computes the top-k indices for Mixture-of-Blocks Attention (MOBA).
    This function handles variable length sequences.

    Args:
        q (torch.Tensor): Query tensor of shape (total_q, num_heads, head_size).
        k (torch.Tensor): Key tensor of shape (total_k, num_heads, head_size).
        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries, shape (batch_size + 1,).
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys, shape (batch_size + 1,).
        max_seqlen_q (int): Maximum sequence length for queries.
        max_seqlen_k (int): Maximum sequence length for keys.
        moba_topk (int): The number of top-k elements to select.
        moba_chunk_size (int): The chunk size for MOBA.
        causal (bool): Whether to apply causal masking.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - col_offsets (torch.Tensor): Column offsets for the sparse matrix.
            - col_nnz (torch.Tensor): Number of non-zero elements per column block.
            - indices (torch.Tensor): The top-k indices.
    """

def flash_moba_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    # MOBA sparse pattern parameters
    moba_col_offsets,
    moba_col_nnz,
    moba_row_indices,
    lg_block_m=64,
    lg_block_n=64,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Performs scaled dot product attention with a pre-computed MOBA sparse pattern.
    This function handles variable length sequences.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        moba_col_offsets: Column offsets from flash_topk_varlen_func.
        moba_col_nnz: Non-zero counts from flash_topk_varlen_func.
        moba_row_indices: Row indices from flash_topk_varlen_func.
        ...

    Return:
        out: (total, nheads, headdim).
    """

    Note: currently the 'dropout_p', 'softcap', 'alibi_slopes' are not supported.
```


## Performance

The following benchmarks demonstrate the speedup and memory savings of FlashMoBA compared to official implementation of MoBA and FlashAttention-2.

We currently have benchmarks for these GPUs:
* [H100](#h100)

### H100

Benchmarks were run on an H100 GPU with the following parameters:
* Head dimension 128, hidden dimension 2048 (i.e. 16 heads).
* Sequence length from 8k up to 512k.
* Batch size of 2.

#### Speedup and Memory

![FlashMoBA speedup and memory savings on H100 80GB SXM5 with FP16/BF16](assets/efficiency.png)


## Team

<div align="center">

<table width="100%">
  <tr>
    <td align="center"><a href="https://guangxuanx.com/">Guangxuan Xiao</a><br/>MIT</td>
    <td align="center"><a href="https://github.com/JerryGJX">Junxian Guo</a><br/>MIT</td>
    <td align="center"><a href="https://github.com/KasraMazaheri">Kasra Mazaheri</a><br/>MIT</td>
    <td align="center"><a href="https://hanlab.mit.edu/songhan">Song Han</a><br/>Nvidia, MIT</td>
  </tr>
</table>

</div>

## Acknowledgments

- [FlashAttention](https://github.com/Dao-AILab/flash-attention): the codebase we built upon. Thanks for their wonderful work.
- [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [MoBA](https://arxiv.org/abs/2502.13189): get the idea of mixture of block attention and how it can be implemented.



## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{xiao2025optimizing,
  title={Optimizing Mixture of Block Attention},
  author={Xiao, Guangxuan and Guo, Junxian and Mazaheri, Kasra and Han, Song},
  booktitle={},
  year={2025}
}
```
