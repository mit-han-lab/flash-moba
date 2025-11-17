import torch
import pytest


try:
    import flash_moba_cuda  # Python wrapper that re-exports the CUDA op
except ImportError as exc:
    pytest.skip(reason=f"CUDA extension not built: {exc}", allow_module_level=True)


def cpu_segmented_sort(values: torch.Tensor, starts: torch.Tensor, lens: torch.Tensor):
    """Reference segmented radix-sort on CPU (ascending)."""
    if not starts.numel():
        return torch.empty_like(values)
    sorted_segments = [
        torch.sort(values[s : s + l])[0] for s, l in zip(starts.tolist(), lens.tolist())
    ]
    return torch.cat(sorted_segments, 0)


# ------------------------ basic correctness ---------------------------------

@pytest.mark.parametrize("segments,seed", [
    (1, 3),      # single segment
    (17, 11),    # moderate segment count
    (128, 47),   # larger batch but still lightweight
    (1024, 97),  # larger batch
    (4096, 233),  # larger batch
    # (16384, 233),  # larger batch
])
def test_varlen_sort_smoke(segments, seed):
    """Light-weight randomized smoke test covering a few segment counts."""
    torch.manual_seed(seed)
    device = "cuda"

    # Keep individual segment length small to avoid heavy memory usage.
    lens = torch.randint(low=0, high=2**18, size=(segments,), dtype=torch.int32)
    
    cum_lens = torch.cumsum(lens, 0, dtype=torch.int64)
    starts = torch.cat([torch.tensor([0], dtype=torch.int64), cum_lens[:-1]], 0)
    ends = cum_lens
    total = lens.sum().item()

    src = torch.randint(
        low=0,
        high=torch.iinfo(torch.int32).max,
        size=(total,),
        dtype=torch.int32,
        device=device,
    )
    
    sorted_gpu = flash_moba_cuda.varlen_sort(starts.to(device), ends.to(device), src)
    sorted_cpu = cpu_segmented_sort(src.cpu(), starts, lens)

    assert torch.equal(sorted_gpu.cpu(), sorted_cpu)


# ------------------------ edge-case: max / min int32 ------------------------


def test_varlen_sort_extreme_values(device: str = "cuda"):
    lens = torch.tensor([16] * 4, dtype=torch.int32)
    
    cum_lens = torch.cumsum(lens, 0, dtype=torch.int64)
    starts = torch.cat([torch.tensor([0], dtype=torch.int64), cum_lens[:-1]], 0)
    ends = cum_lens
    total = lens.sum().item()

    # interleave INT32 MAX/MIN to stress radix boundaries
    vals = torch.tensor(
        [
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).min,
        ]
        * (total // 2),
        dtype=torch.int32,
        device=device,
    )

    sorted_gpu = flash_moba_cuda.varlen_sort(starts.to(device), ends.to(device), vals)
    sorted_cpu = cpu_segmented_sort(vals.cpu(), starts, lens)

    assert torch.equal(sorted_gpu.cpu(), sorted_cpu)


if __name__ == "__main__":
    test_varlen_sort_smoke(segments = 1, seed = 3)