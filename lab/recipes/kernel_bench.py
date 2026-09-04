"""Attention kernels, FLOP accounting and KV-cache arithmetic (Lab 11/13: architecture and kernels).

What this teaches
  * the exact FLOP count of attention. For batch B, heads H, sequence S and
    head dim D, the forward pass does two matmuls per head: Q K^T (S x D by
    D x S) and P V (S x S by S x D), each 2 S^2 D flops, so
        FLOPs_forward = 4 * B * H * S^2 * D
    (the softmax and the scale are O(S^2) and are left out, as everyone does).
    Time and this number give achieved TFLOP/s, which you can compare with the
    card's peak to see how far from the roofline you are.
  * naive attention (materializes the S x S scores) against
    torch.nn.functional.scaled_dot_product_attention with its backends forced
    one at a time (math, efficient, flash) via torch.nn.attention.sdpa_kernel,
    so you can see which backend fires and what it buys at each sequence length.
  * KV-cache bytes per token from a model config:
        bytes/token = 2 (K and V) * layers * kv_heads * head_dim * bytes(dtype)
    with a worked example printed for --layers --kv-heads --kv-head-dim --dtype
    and a total for --context tokens at --cache-batch sequences. Grouped-query
    attention shrinks kv_heads and that is the whole reason it exists.
  * a Triton fused softmax (one program per row, the row stays in registers)
    compared with torch.softmax; guarded by `try: import triton` and only run
    on cuda.

How to run
  smoke (CPU, tiny sizes, no Triton):   python lab/recipes/kernel_bench.py --smoke
  real (RTX 5090):                       python lab/recipes/kernel_bench.py --seqs 512,1024,2048,4096,8192 --heads 16 --head-dim 128
  KV cache for a Llama-3-8B-shaped config: python lab/recipes/kernel_bench.py --smoke --layers 32 --kv-heads 8 --kv-head-dim 128 --dtype bf16 --context 8192
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


def build_parser():
    p = C.base_parser("kernel_bench", __doc__.split("\n")[0])
    p.add_argument("--seqs", default=None, help="comma-separated sequence lengths")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--layers", type=int, default=32, help="KV-cache worked example")
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--kv-head-dim", type=int, default=128, help="head dim for the KV-cache worked example")
    p.add_argument("--dtype", choices=list(DTYPE_BYTES), default="bf16")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--cache-batch", type=int, default=1)
    p.add_argument("--no-triton", action="store_true")
    return p


# --------------------------------------------------------------------------- attention implementations


def naive_attention(q, k, v, causal):
    s = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if causal:
        S = q.shape[-2]
        s = s.masked_fill(torch.ones(S, S, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v


def sdpa_with_backend(backend_name):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}[backend_name]

    def fn(q, k, v, causal):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return fn


def bench(fn, q, k, v, causal, reps, device) -> float:
    """Median milliseconds over reps after 2 warmups."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(2):
        fn(q, k, v, causal)
    sync()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(q, k, v, causal)
        sync()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def attention_flops(B, H, S, D, causal) -> float:
    f = 4.0 * B * H * S * S * D
    return f / 2 if causal else f      # a causal kernel that skips masked blocks does about half the work


# --------------------------------------------------------------------------- KV cache


def kv_cache_report(layers, kv_heads, head_dim, dtype, context, batch) -> dict:
    per_token = 2 * layers * kv_heads * head_dim * DTYPE_BYTES[dtype]
    total = per_token * context * batch
    C.log("KV cache worked example:")
    C.log(f"  bytes/token = 2 (K,V) * {layers} layers * {kv_heads} kv_heads * {head_dim} head_dim * {DTYPE_BYTES[dtype]} bytes ({dtype}) = {per_token:,} bytes")
    C.log(f"  {context:,} context tokens x {batch} sequences = {total / 2**30:.3f} GiB")
    return {"kv_bytes_per_token": per_token, "kv_bytes_total": total, "kv_gib_total": total / 2**30}


# --------------------------------------------------------------------------- Triton fused softmax

TRITON_OK = False
try:
    import triton
    import triton.language as tl

    TRITON_OK = True

    @triton.jit
    def _softmax_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(in_ptr + row * stride + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        y = num / tl.sum(num, axis=0)
        tl.store(out_ptr + row * stride + offs, y, mask=mask)

    def triton_softmax(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(cols)
        _softmax_kernel[(rows,)](out, x, x.stride(0), cols, BLOCK=BLOCK)
        return out
except ImportError:
    pass


def main():
    args = build_parser().parse_args()
    d = dict(seqs="64,128", batch=1, heads=2, head_dim=32, reps=5) if args.smoke else dict(seqs="512,1024,2048,4096", batch=4, heads=16, head_dim=128, reps=20)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    device = C.pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    seqs = [int(s) for s in args.seqs.split(",")]
    impls = {"naive": naive_attention, "sdpa_math": sdpa_with_backend("math"), "sdpa_efficient": sdpa_with_backend("efficient"),
             "sdpa_flash": sdpa_with_backend("flash")}
    C.status("attention", f"device={device} dtype={dtype} B={args.batch} H={args.heads} D={args.head_dim} causal={args.causal}")
    rows, step = [], 0
    for S in seqs:
        q, k, v = (torch.randn(args.batch, args.heads, S, args.head_dim, device=device, dtype=dtype) for _ in range(3))
        ref = naive_attention(q.float(), k.float(), v.float(), args.causal)
        flops = attention_flops(args.batch, args.heads, S, args.head_dim, args.causal)
        for name, fn in impls.items():
            try:
                out = fn(q, k, v, args.causal)
                err = (out.float() - ref).abs().max().item()
                ms = bench(fn, q, k, v, args.causal, args.reps, device)
            except RuntimeError as e:
                C.log(f"  S={S:<6} {name:<15} unsupported here ({str(e).splitlines()[0][:70]})")
                continue
            tflops = flops / (ms / 1000) / 1e12
            C.metric(step, seq=S, impl=name, ms=ms, tflops=tflops, max_abs_err=err)
            C.log(f"  S={S:<6} {name:<15} {ms:9.3f} ms  {tflops:8.3f} TFLOP/s  max|err| vs fp32 naive {err:.2e}")
            rows.append({"seq": S, "impl": name, "ms": ms, "tflops": tflops})
            step += 1
    kv = kv_cache_report(args.layers, args.kv_heads, args.kv_head_dim, args.dtype, args.context, args.cache_batch)

    tri = {}
    if device.type == "cuda" and TRITON_OK and not args.no_triton and not args.smoke:
        C.status("triton", "fused softmax vs torch.softmax")
        for cols in (1024, 4096):
            x = torch.randn(4096, cols, device=device, dtype=torch.float32)
            ok = torch.allclose(triton_softmax(x), torch.softmax(x, -1), atol=1e-5)
            ms_t = bench(lambda a, b, c, d: triton_softmax(x), x, x, x, False, args.reps, device)
            ms_p = bench(lambda a, b, c, d: torch.softmax(x, -1), x, x, x, False, args.reps, device)
            C.metric(step, softmax_cols=cols, triton_ms=ms_t, torch_ms=ms_p, correct=int(ok))
            C.log(f"  softmax rows=4096 cols={cols}: triton {ms_t:.3f} ms, torch {ms_p:.3f} ms, allclose={ok}")
            tri[f"softmax_{cols}_triton_ms"] = ms_t
            tri[f"softmax_{cols}_torch_ms"] = ms_p
            step += 1
    else:
        C.log(f"triton softmax skipped (cuda={device.type == 'cuda'}, triton_installed={TRITON_OK}, smoke={args.smoke})")
    C.status("done", "")
    best = {f"best_ms_S{r['seq']}": min(x["ms"] for x in rows if x["seq"] == r["seq"]) for r in rows}
    C.result(device=str(device), seqs=seqs, **best, **kv, **tri)


if __name__ == "__main__":
    main()
