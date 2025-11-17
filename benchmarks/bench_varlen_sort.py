import argparse, time
import torch

try:
    import flash_moba_cuda
except ImportError as exc:
    raise RuntimeError(f"CUDA extension not built: {exc}")

def generate_data(seg_num: int, total_elems: int, device):
    torch.manual_seed(233)
    lens = torch.randint(low=1, high=min(total_elems // seg_num * 2, 2**31-1), size=(seg_num,), dtype=torch.int64)
    lens = (lens.float() / lens.sum() * total_elems).long().clamp_min(1)
    diff = total_elems - lens.sum()
    lens[0] += diff
    starts = torch.zeros_like(lens)
    starts[1:] = torch.cumsum(lens[:-1], dim=0)
    src = torch.randint(low=-2 ** 31, high=2 ** 31 - 1, size=(total_elems,), dtype=torch.int32, device=device)
    return starts.to(device), lens.to(torch.int32).to(device), src

def cpu_sort(src, starts, lens):
    res = []
    for s, l in zip(starts.tolist(), lens.tolist()):
        res.append(torch.sort(src[s:s+l])[0])
    return torch.cat(res, 0)

def benchmark(args):
    """Benchmark segmented varlen_sort vs. whole-tensor torch.sort on GPU (and optional CPU)."""

    starts, lens, src = generate_data(args.segments, args.total, "cuda")
    starts = starts.to(torch.int64)
    ends = starts + lens
    ends = ends.to(torch.int64)

    # ---------------- GPU segmented sort ----------------
    for _ in range(5):  # warm-up
        flash_moba_cuda.varlen_sort(starts, ends, src)
    torch.cuda.synchronize()

    begin = time.time()
    for _ in range(args.iters):
        flash_moba_cuda.varlen_sort(starts, ends, src)
    torch.cuda.synchronize()
    elapsed = time.time() - begin
    print(
        f"[GPU] varlen_sort: {(args.total * args.iters / elapsed) / 1e6:.2f} M items/s"
        f"  (elapsed {elapsed:.3f}s / {args.iters} iters)"
    )

    # ---------------- GPU whole-tensor sort baseline ----------------
    for _ in range(5):
        torch.sort(src)
    torch.cuda.synchronize()

    begin = time.time()
    for _ in range(args.iters):
        torch.sort(src)
    torch.cuda.synchronize()
    elapsed = time.time() - begin
    print(
        f"[GPU] torch.sort (full tensor): {(args.total * args.iters / elapsed) / 1e6:.2f} M items/s"
        f"  (elapsed {elapsed:.3f}s / {args.iters} iters)"
    )

    # ---------------- Optional CPU reference ----------------
    if args.cpu:
        v_cpu = src.cpu()
        torch.cuda.synchronize()
        begin = time.time()
        torch.sort(v_cpu)  # full sort on CPU
        elapsed_cpu = time.time() - begin
        print(f"[CPU] torch.sort: {elapsed_cpu:.3f}s (single pass)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', type=int, default=8192)
    parser.add_argument('--total', type=int, default=1<<22)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--cpu', action='store_true')
    benchmark(parser.parse_args())
