---
title: "Lab 13: GPUs, kernels, and the KV cache"
kind: permanent
topics: [lab]
chapter: 13
station: none
recipe: recipes/kernel_bench.py
reading_time: 45 min
---

# Lab 13: GPUs, kernels, and the KV cache

## What you will be able to do

1. Place any operation on a roofline for your RTX 5090: compute its arithmetic intensity from shapes and dtypes, compare it with the ridge point, and predict whether it is bound by compute or by memory bandwidth before you run it.
2. Derive the online softmax and write an exact blocked attention forward pass that never materializes the score matrix, then explain in two sentences why FlashAttention is faster despite doing more arithmetic.
3. Write a fused row softmax in Triton that runs within about 15 percent of the card's memory bandwidth, and read the numbers that tell you when a Triton kernel is spilling registers.
4. Size a KV cache from a model's config, explain why decode is bandwidth-bound, and estimate tokens per second for a given batch and context length from first principles.
5. Run `recipes/kernel_bench.py` on the 5090 and check its four measurements (GEMM peak, copy bandwidth, attention TFLOP/s versus sequence length, decode GB/s versus context length) against the formulas in this chapter.

## The idea in one paragraph

A GPU is a machine that can do arithmetic far faster than it can fetch the operands. On your card the ratio is roughly 130 floating point operations per byte moved from memory, so any operation that does fewer operations than that per byte it touches is waiting on memory, not on the ALUs. Attention as written in a textbook does about 64 operations per byte at head dimension 128, so it is memory-bound, and the fix is not fewer operations but fewer bytes: keep a block of queries in fast on-chip memory, stream the keys and values past it, and maintain the softmax incrementally with a running maximum so the answer is exact. The same accounting explains inference. Prefill (processing the prompt) is a big matrix multiply and runs near compute peak; decode (one token at a time) reads every weight and every cached key and value to produce a single token, so it is a bandwidth test, and the KV cache is the term in that byte count that grows with context. Everything in this chapter is one calculation done repeatedly: count the bytes, count the operations, divide.

## The math

### The memory hierarchy, with your card's numbers

A kernel is a function that runs on the GPU. It is launched as a grid of thread blocks; each block runs on one streaming multiprocessor (SM), and threads within a block are scheduled in groups of 32 called warps. The memory levels, from slow and large to fast and small, are:

High bandwidth memory (HBM, or on a consumer card GDDR). This is the 32 GB you see in `nvidia-smi`. Everything that persists between kernels lives here. Bandwidth is the number that matters. The 5090 spec sheet gives a 512-bit GDDR7 bus at 28 Gbit/s per pin, which is $512/8 \times 28 \times 10^9 = 1{,}792$ GB/s. Treat that as an assumption; a 1 GiB `clone()` measured on 2026-09-03 with PyTorch 2.11 reached 1,525 GB/s (read plus write), about 85 percent of the theoretical figure, which is typical of what a real kernel can get.

L2 cache. Shared by all SMs, on-die. `torch.cuda.get_device_properties(0).L2_cache_size` reports 96 MiB on the 5090. Its bandwidth is several times HBM. A working set that fits in L2 makes a benchmark lie about HBM bandwidth (see How it goes wrong).

Shared memory (SRAM) and L1. Per SM, on-chip, explicitly managed by the kernel. The 5090 reports 100 KB per SM with up to 99 KB usable by one block. FlashAttention's whole design is about what fits here.

Registers. The fastest storage, private to a thread. 65,536 32-bit registers per SM, at most 255 per thread. A Triton kernel that asks for more than the register file can hold spills to local memory (which is really HBM with a cache in front) and its performance falls off a cliff.

The card has 170 SMs. Dense bf16 tensor-core throughput with fp32 accumulation is quoted around 210 TFLOP/s on the spec sheet (the 419 figure counts 2:4 structured sparsity and does not apply to your dense matmuls). An $8192^3$ bf16 GEMM measured 239 TFLOP/s on the card, so the spec sheet number is conservative; use the measured value as your compute peak $\pi$.

### The roofline

Define for any kernel:

$$
W = \text{floating point operations performed}, \qquad Q = \text{bytes moved to or from HBM}, \qquad I = \frac{W}{Q}
$$

$I$ is the arithmetic intensity in FLOP per byte. With compute peak $\pi$ (FLOP/s) and bandwidth $\beta$ (bytes/s), the time the kernel needs is at least the larger of the compute time and the memory time:

$$
T \ge \max\left(\frac{W}{\pi}, \frac{Q}{\beta}\right)
\quad\Longrightarrow\quad
\text{attainable FLOP/s} = \frac{W}{T} \le \min(\pi,\; \beta I).
$$

Plotted against $I$ on log axes this is a flat roof at $\pi$ and a sloped roof $\beta I$; they meet at the ridge point $I^* = \pi / \beta$. With the measured peaks, $I^* = 239 \times 10^{12} / 1.525 \times 10^{12} \approx 157$ FLOP/byte; with spec-sheet numbers, $210/1.792 \approx 117$. Either way, a kernel below about 120 to 160 FLOP/byte is memory-bound on this card. The roofline is an upper bound, not a prediction: latency, launch overhead, and poor occupancy can keep you well under both roofs.

### Intensity of a matmul

For $C = AB$ with $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, each output element needs $K$ multiply-adds, so $W = 2MNK$. If each element is $b$ bytes and each matrix is read or written exactly once, $Q = b(MK + KN + MN)$. For square matrices of side $n$ in bf16 ($b = 2$):

$$
I_{\text{matmul}} = \frac{2n^3}{6n^2} = \frac{n}{3}.
$$

So a $4096^2$ matmul has $I \approx 1365$, far above the ridge, and is compute-bound; the measured 188 TFLOP/s at $n = 4096$ and 239 at $n = 8192$ confirm it. Now set $M = 1$, which is what a decode step does when it multiplies one token's hidden state by a weight matrix:

$$
I_{\text{GEMV}} = \frac{2KN}{2(K + KN + N)} \approx 1 \text{ FLOP/byte}.
$$

Every weight byte is used for exactly one multiply-add. That is about a hundred times below the ridge. Batching $M$ token rows through the same weights raises the intensity to roughly $M$ FLOP/byte (the weight read is amortized), so you need a batch of about 120 to 160 rows before a projection layer becomes compute-bound. This one line is the reason serving systems batch aggressively.

### Intensity of attention, naive and fused

Take one head with $N$ queries and keys of dimension $d$, everything in bf16. The textbook computation is $S = QK^\top / \sqrt{d}$, $P = \mathrm{softmax}(S)$ row-wise, $O = PV$. Operations: $2N^2 d$ for $QK^\top$ and $2N^2 d$ for $PV$, so $W = 4N^2 d$ (the exponentials are $N^2$ and are usually ignored in the FLOP count, though they are not free). Bytes, if each intermediate goes to HBM: write $S$, read $S$, write $P$, read $P$, each $2N^2$ bytes, plus $Q, K, V, O$ at $2Nd$ each:

$$
I_{\text{naive}} = \frac{4N^2 d}{8N^2 + 8Nd} \;\xrightarrow{N \gg d}\; \frac{d}{2}.
$$

With $d = 128$ that is 64 FLOP/byte, below the ridge: naive attention is memory-bound on this card, and it gets worse if $S$ is kept in fp32 ($d/4$). Note the intensity does not grow with $N$; longer sequences do not help, they only make the $N^2$ intermediate bigger.

FlashAttention removes the $N^2$ traffic. Split queries into blocks of $B_q$ rows and keys into blocks of $B_k$ rows. For each query block, held in SRAM, stream every key and value block through SRAM, computing the score block, its contribution to the softmax, and its contribution to the output, without ever writing scores to HBM. $K$ and $V$ are read once per query block, so

$$
Q_{\text{flash}} \approx 2Nd \cdot 4 + \frac{N}{B_q} \cdot 2 \cdot 2Nd = 8Nd + \frac{4N^2 d}{B_q},
\qquad
I_{\text{flash}} \;\xrightarrow{N \gg B_q}\; B_q.
$$

With $B_q = 128$ the intensity is around 128 FLOP/byte, at the ridge, and with the tile sizes real kernels use on Blackwell it is above it. The constraint is that a query block plus a key block plus a value block plus the score block must fit in the roughly 100 KB of shared memory per SM; that is what sets $B_q$ and $B_k$, and it is why FlashAttention's HBM traffic is $O(N^2 d^2 / M)$ for SRAM size $M$: bigger SRAM means bigger blocks means fewer passes over $K, V$. The measured curve on the 5090 (PyTorch's flash backend, causal, 32 heads, $d = 128$) climbs from 104 TFLOP/s at $N = 1024$ to 209 TFLOP/s at $N = 16384$; the low end is launch overhead and too few blocks to fill 170 SMs, not memory.

### The online softmax, derived

The obstacle to streaming keys is the softmax denominator, which needs all $N$ scores before any probability is known. The stable softmax for one query row with scores $s_1, \dots, s_N$ is

$$
p_j = \frac{e^{s_j - m}}{\ell}, \qquad m = \max_j s_j, \qquad \ell = \sum_{j} e^{s_j - m},
$$

and the output row is $o = \sum_j p_j v_j$. Subtracting $m$ is not optional: bf16 shares fp32's exponent range but fp16 overflows at 65504, and even in fp32 a score of 100 gives $e^{100} \approx 2.7 \times 10^{43}$, past the fp32 maximum.

Now suppose you have processed the first $t$ keys and hold three quantities that depend only on them:

$$
m_t = \max_{j \le t} s_j, \qquad \ell_t = \sum_{j \le t} e^{s_j - m_t}, \qquad a_t = \sum_{j \le t} e^{s_j - m_t} v_j.
$$

Then $o = a_N / \ell_N$. The question is whether $(m_t, \ell_t, a_t)$ can be updated to $(m_{t+1}, \ell_{t+1}, a_{t+1})$ using only the new score $s_{t+1}$. Set $m_{t+1} = \max(m_t, s_{t+1})$. Every existing term was scaled by $e^{-m_t}$ but should now be scaled by $e^{-m_{t+1}}$, and the ratio is the same for all of them:

$$
e^{s_j - m_{t+1}} = e^{s_j - m_t} \cdot e^{m_t - m_{t+1}}.
$$

Define the rescale factor $\alpha = e^{m_t - m_{t+1}} \le 1$. Then

$$
\ell_{t+1} = \alpha\,\ell_t + e^{s_{t+1} - m_{t+1}}, \qquad a_{t+1} = \alpha\, a_t + e^{s_{t+1} - m_{t+1}} v_{t+1}.
$$

By induction on $t$ the invariant holds at every step, so the final $a_N / \ell_N$ equals the exact softmax-weighted sum. Nothing was approximated. The same argument works when a whole block of keys arrives at once: take the block maximum, rescale the old state by $\alpha$, and add the block's sums. This is the entire trick; FlashAttention is this recurrence, tiled so that $Q$ blocks stay in SRAM, with the extra observation that you can initialize $m_0 = -\infty$, $\ell_0 = 0$, $a_0 = 0$ and the first update is then exact because $\alpha = e^{-\infty - m_1} = 0$.

For the backward pass you need $P$ again. Storing it costs the $N^2$ you just avoided, so FlashAttention stores only the per-row log-sum-exp $L = m + \log \ell$ (one number per query) and recomputes $P = e^{S - L}$ block by block in the backward kernel. This is more FLOPs, not fewer, and it is still faster because the FLOPs were never the bottleneck.

### The KV cache

A decoder-only transformer at decode time has already computed keys and values for every earlier position, and causal attention means those never change, so they are cached. Per token, per layer, you store one key and one value vector for each key-value head:

$$
\text{bytes per token} = L_{\text{layers}} \times 2 \times H_{kv} \times d_{\text{head}} \times b,
$$

where the 2 counts K and V and $b$ is bytes per element. Llama-3 8B has 32 layers, 32 query heads but 8 key-value heads (grouped-query attention, GQA, where 4 query heads share one KV head), $d_{\text{head}} = 128$, and in bf16 $b = 2$:

$$
32 \times 2 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KiB per token}.
$$

An 8k-token context is therefore 1 GiB of cache; 32 concurrent 4k sequences is 16 GiB. Without GQA (Llama-2 7B, 32 KV heads) the figure is 512 KiB per token, four times larger, which is the practical reason GQA exists. The weights of the 8B model in bf16 are about 16.1 GB, so on a 32 GB card you have roughly 13 GB for cache plus activations, or about 100k tokens of cache in total across all sequences.

### Why decode is bandwidth-bound

Per decode step the model reads every weight once (a GEMV, intensity about 1) and, in each layer, one query attends to $t$ cached tokens: $W = 4 t d$ per head-pair and the bytes read are the $K$ and $V$ for those tokens, $4 t d$ bytes in bf16, so the intensity is again about 1 FLOP/byte. The time for one step at batch size $B$ and context $t$ is then bounded below by bytes over bandwidth:

$$
T_{\text{step}} \ge \frac{\text{weight bytes} + B \times t \times \text{bytes per token}}{\beta}.
$$

At $B = 1$, $t = 4096$ for Llama-3 8B: $(16.1 + 0.54) \times 10^9 / 1.525 \times 10^{12} \approx 10.9$ ms per token, about 92 tokens per second, and no amount of kernel cleverness gets you past that without reading fewer bytes (quantized weights, quantized cache, or shorter context). At $B = 32$ the weight read is amortized: $(16.1 + 32 \times 0.54) / 1525 \approx 21.9$ ms per step, but that step produces 32 tokens, so aggregate throughput is about 1,460 tokens per second at the cost of a per-stream latency that roughly doubled. Prefill is the opposite: the prompt's $N$ tokens go through every projection as an $N \times K$ by $K \times N'$ matmul with intensity of order $N$, so a 2k prompt is compute-bound and runs near peak.

The measured decode curve on the 5090 (32 layers of single-query attention against a Llama-3 8B shaped cache) shows the second regime you must know about: at $t = 65536$ the cache read reached 1,517 GB/s, right at the practical bandwidth, but at $t = 1024$ it reached only 385 GB/s, because the 32 kernel launches cost about 11 microseconds each and dominate the 0.35 ms step. Short-context decode is launch-bound, and CUDA graphs (which replay a recorded sequence of launches without CPU involvement) are the standard fix; every serving engine uses them.

### Paged attention, quantization, eviction

Paged attention (vLLM) addresses a different waste. If each sequence's cache is a contiguous buffer sized for the maximum length, a 1,000-token sequence in an 8,192-token slot wastes 7,192 tokens of memory, and slots cannot be reused across sequences of different lengths. Instead, the cache is stored in fixed blocks of, say, 16 tokens, and each sequence holds a block table mapping its logical positions to physical blocks, exactly like virtual memory pages. Internal fragmentation drops to at most 15 tokens per sequence, blocks are allocated on demand as the sequence grows, and sequences that share a prefix (parallel samples, beam search, a shared system prompt) share physical blocks with copy-on-write. The attention kernel takes the block table and gathers K and V through it, which costs a little indirection and is why paged kernels are slightly slower than contiguous ones at small batch.

KV quantization shrinks the bytes-per-token term. Storing the cache in int8 or fp8 halves it; KIVI shows that keys and values want different treatment, keys quantized per channel (because a few channels carry large-magnitude outliers after rotary embedding) and values per token, and gets to 2 bits with small quality loss. Eviction reduces $t$ instead: keep only tokens that receive substantial attention mass (H2O's heavy hitters), or keep a few initial tokens plus a sliding window (StreamingLLM's attention sinks, which found that the first tokens absorb a large share of attention mass regardless of content), or select which tokens to keep by looking at the prompt's own attention pattern (SnapKV). Each is a trade of accuracy for bytes; measure on your task, because a method that keeps perplexity flat can still break retrieval of a fact that lived in an evicted token.

### Speculative decoding in outline

Because a decode step is bandwidth-bound with intensity about 1, verifying $\gamma$ candidate tokens in one forward pass costs almost the same as generating one: the weights are read once either way and the intensity rises to about $\gamma$. Speculative decoding exploits this. A small draft model proposes $\gamma$ tokens $x_1, \dots, x_\gamma$ autoregressively with probabilities $q(x_i)$; the target model then computes $p(\cdot)$ at all $\gamma$ positions in a single pass. Token $i$ is accepted with probability $\min(1, p(x_i)/q(x_i))$; at the first rejection, a replacement is drawn from the residual distribution $\mathrm{norm}(\max(0, p - q))$ and the rest of the draft is discarded. This acceptance rule makes the sequence of accepted tokens an exact sample from $p$, so the output distribution is unchanged. If each token is accepted independently with probability $\alpha$, the expected number of tokens produced per target pass is

$$
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha},
$$

counting the guaranteed one from the residual. The speedup is that expectation divided by the relative cost of one target pass plus $\gamma$ draft passes. It only helps when the target is bandwidth-bound; at large batch the verify pass is no longer nearly free, and the gain vanishes.

## Build it small

The snippet below implements blocked attention with the online softmax in plain PyTorch on the CPU and checks it against the reference, with and without a causal mask. The reference materializes an $N \times N$ score matrix; the blocked version never holds more than a $B_q \times B_k$ tile. Run it with any PyTorch 2.x.

```python
import torch

torch.manual_seed(0)


def attention_reference(q, k, v, causal=False):
    n = q.shape[0]
    s = (q @ k.T) * q.shape[-1] ** -0.5
    if causal:
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def flash_forward(q, k, v, bq=64, bk=64, causal=False):
    """Blocked attention with an online softmax. Never materializes the n x n score matrix."""
    n, d = q.shape
    scale = d ** -0.5
    out = torch.empty_like(q)
    for i in range(0, n, bq):
        qi = q[i:i + bq]
        rows = qi.shape[0]
        m = torch.full((rows,), float("-inf"))      # running row max
        l = torch.zeros(rows)                        # running row sum of exp
        acc = torch.zeros(rows, d)                   # running unnormalized output
        for j in range(0, n, bk):
            if causal and j > i + rows - 1:
                break                                # whole key block is in the future
            kj, vj = k[j:j + bk], v[j:j + bk]
            s = (qi @ kj.T) * scale                  # (rows, bk) block of scores
            if causal:
                qpos = torch.arange(i, i + rows)[:, None]
                kpos = torch.arange(j, j + kj.shape[0])[None, :]
                s = s.masked_fill(kpos > qpos, float("-inf"))
            m_new = torch.maximum(m, s.max(dim=-1).values)
            alpha = torch.exp(m - m_new)             # rescale factor for old state
            p = torch.exp(s - m_new[:, None])        # unnormalized probabilities
            l = alpha * l + p.sum(dim=-1)
            acc = alpha[:, None] * acc + p @ vj
            m = m_new
        out[i:i + rows] = acc / l[:, None]
    return out


n, d = 512, 64
q, k, v = (torch.randn(n, d) for _ in range(3))
for causal in (False, True):
    ref = attention_reference(q, k, v, causal)
    got = flash_forward(q, k, v, causal=causal)
    print(f"causal={causal}  max|err|={(ref - got).abs().max().item():.2e}")
print(f"reference score matrix: {n * n * 4 / 1024:.0f} KiB; flash working set per block: "
      f"{(64 * 64 + 3 * 64 * d) * 4 / 1024:.0f} KiB")
```

Expected output (fp32, so the differences are rounding):

```
causal=False  max|err|=2.53e-07
causal=True  max|err|=4.17e-07
reference score matrix: 1024 KiB; flash working set per block: 64 KiB
```

Two things to notice. First, the causal branch breaks out of the key loop early, which is where the factor of two in causal FLOPs comes from; a real kernel also skips the fully masked blocks and only applies the elementwise mask on the diagonal blocks. Second, the causal mask in this code never masks an entire row within a processed block, because the diagonal block always contains the query's own position; if you write a variant with arbitrary masks, a fully masked block gives $m_{\text{new}} = -\infty$ and $\alpha = e^{-\infty - (-\infty)} = \text{NaN}$. That failure is real and is in the list below.

## Build it real

This chapter has two real pieces: a fused softmax in Triton, which you should write yourself once, and the benchmark recipe.

### A fused softmax in Triton

Triton is a Python-embedded language in which you write the body of one program (one thread block) over tiles of a tensor, and the compiler handles threads, shared memory, and vectorization. The unit of thought is a block-shaped tensor in registers, not a thread. `tl.program_id(0)` tells you which program you are; `tl.arange(0, BLOCK)` gives a vector of offsets; `tl.load` and `tl.store` move tiles with a mask for the ragged edge; reductions like `tl.max` and `tl.sum` run across the tile. Block sizes are `tl.constexpr`, fixed at compile time, and must be powers of two.

A row softmax in PyTorch is three kernels (max, exp-and-sum, divide), each reading and writing the whole tensor, so it moves about six times the tensor size through HBM. Fused, it reads once and writes once:

```python
import torch, triton, triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                       # one program per row
    offs = tl.arange(0, BLOCK)                   # BLOCK must be a power of 2
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)                    # subtract the row max: no overflow
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(y_ptr + row * stride + offs, num / den, mask=mask)


def fused_softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK=BLOCK)
    return y


x = torch.randn(4096, 2048, device="cuda")
y = fused_softmax(x)
print("max|err| vs torch:", (y - torch.softmax(x, dim=-1)).abs().max().item())
for f, name in ((fused_softmax, "triton"), (lambda t: torch.softmax(t, -1), "torch")):
    ms = triton.testing.do_bench(lambda: f(x))
    gbps = 2 * x.numel() * x.element_size() / ms * 1e-6
    print(f"{name}: {ms:.3f} ms, {gbps:.0f} GB/s")
```

On the 5090 with Triton 3.6 this printed a maximum error of $3.7 \times 10^{-9}$, 0.044 ms and 1,513 GB/s for the Triton kernel against 0.048 ms and 1,410 GB/s for `torch.softmax` (which is itself already fused for this shape). Both are at the practical bandwidth ceiling, which is the point: a correct memory-bound kernel is finished when it reaches bandwidth, and the way to check is to compute achieved GB/s, not to compare against another library. The masked load with `other=-inf` is what makes the ragged last block harmless: masked lanes contribute $e^{-\infty} = 0$ to the sum.

To go from this to attention you add a second loop over key blocks inside the program and carry `m`, `l`, and `acc` exactly as in the PyTorch snippet, using `tl.dot` for the tile matmuls so they hit the tensor cores. The Triton tutorials ship a complete fused attention kernel; read it after this chapter and you will recognize every line.

### The benchmark recipe

`recipes/kernel_bench.py` runs four measurements on the 5090 and prints the roofline quantities next to each. Arguments:

`--gemm N` measures a bf16 $N \times N$ matmul and reports TFLOP/s (default 8192; this is your compute peak $\pi$).

`--copy GiB` measures a device-to-device clone of that many GiB and reports GB/s counting read plus write (default 1; this is your practical bandwidth $\beta$).

`--attn N1,N2,...` runs causal `scaled_dot_product_attention` with the flash backend at each sequence length, 32 heads, head dim 128, batch 1, and reports ms and TFLOP/s using $W = 4 N^2 d H / 2$.

`--decode L1,L2,...` builds a Llama-3 8B shaped cache (32 layers, 8 KV heads, 128 dim) of each length, runs one query through all 32 layers with `enable_gqa=True`, and reports cache GiB, ms, and achieved GB/s using the bytes-per-token formula. `--layers`, `--kv-heads`, `--heads`, `--head-dim`, and `--dtype` change the model shape.

`--graph` wraps the decode step in a CUDA graph so you can see the launch-bound regime disappear at short context.

Every timing does three warm-up calls, then `torch.cuda.synchronize()`, then twenty timed calls, then synchronize again. Total run time is under a minute. The core of the decode measurement, which you can paste into a notebook, is:

```python
import time, torch, torch.nn.functional as F


def bench(fn, iters=20):
    for _ in range(3):
        fn()                                   # warm-up: compile, allocator, clocks
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()                   # never time without this
    return (time.perf_counter() - t) / iters


layers, H, H_kv, D = 32, 32, 8, 128            # Llama-3 8B
per_token = 2 * layers * H_kv * D * 2          # bf16
for L in (1024, 4096, 16384, 65536):
    k = torch.randn(layers, 1, H_kv, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(layers, 1, H, 1, D, device="cuda", dtype=torch.bfloat16)

    def step():
        for i in range(layers):
            F.scaled_dot_product_attention(q[i], k[i], v[i], enable_gqa=True)

    s = bench(step)
    print(f"L={L:6d}  cache {per_token * L / 2**30:5.2f} GiB  {s * 1e3:7.3f} ms  "
          f"{per_token * L / s / 1e9:5.0f} GB/s")
    del k, v, q
```

What to watch in the output: the GEMM number should be above 200 TFLOP/s and the copy above 1,400 GB/s, or the card is throttled or shared. Attention TFLOP/s should rise with $N$ and flatten near the GEMM number; if it flattens far below, the flash backend was not selected (check with `torch.nn.attention.sdpa_kernel`). Decode GB/s should rise with $L$ toward the copy number; the low value at short $L$ is launch overhead, and `--graph` should lift it.

## How it goes wrong

Timings that are too good to be true. You timed a kernel without `torch.cuda.synchronize()`; the CPU returned before the GPU finished. Symptom: microsecond timings for work that should take milliseconds, and timings that do not change with problem size. Fix: synchronize before starting and after stopping the clock, and warm up first so compilation and allocator growth are excluded.

Bandwidth above the spec sheet. A $4096 \times 4096$ bf16 weight is 32 MiB, which fits in the 96 MiB L2, so a benchmark that reuses it every iteration reads from L2 and reports 1,783 GB/s, as one of this chapter's own runs did at $M = 16$. Symptom: achieved GB/s above the theoretical HBM figure, or a decode benchmark that looks faster than the weight-read bound. Fix: use working sets larger than L2 (rotate through several copies of the weight, or use the real model), and treat any number above $\beta$ as an L2 artifact.

NaN from a fully masked block. In an online softmax, a block whose scores are all $-\infty$ gives $m_{\text{new}} = -\infty$, and $\alpha = e^{m - m_{\text{new}}}$ is $e^{-\infty - (-\infty)}$, which is NaN, poisoning the accumulator. Symptom: NaN outputs only for some sequence positions or only with padding. Fix: skip fully masked blocks before computing the max, or use a large finite negative like $-10^{30}$ so the rescale stays finite and the masked terms still underflow to zero.

Register spills in Triton. `BLOCK = next_power_of_2(n_cols)` with 100k columns asks for a 131,072-element tile per program, which does not fit in the register file. Symptom: the kernel compiles and is correct but runs ten times slower than the bandwidth bound, and `softmax_kernel.cache` or the compiled kernel object reports `n_spills > 0`. Fix: loop over column chunks inside the program (two passes: max and sum, then normalize) or, for attention-like reductions, use the online recurrence so the tile is fixed-size.

Wrong FLOP count. Forgetting the causal halving overstates attention FLOP/s by two; forgetting that attention has two matmuls (the factor 4 in $4N^2d$) understates it by two; comparing against the sparse tensor-core figure understates utilization by two. Symptom: utilization numbers above 100 percent or suspiciously near 50. Fix: write the FLOP formula in the benchmark next to the timing and derive it once from the shapes.

fp16 overflow, or bf16 precision loss, in scores. Without max subtraction, fp16 scores above about 11 overflow the exponential. With max subtraction, bf16 accumulation of the softmax denominator loses precision at long context because bf16 has 8 bits of mantissa. Symptom: NaN in fp16, or a slow drift of attention outputs from the fp32 reference as $N$ grows. Fix: always subtract the running max, and keep $m$, $\ell$, and the accumulator in fp32 while the tiles of $Q, K, V$ stay in bf16, which is what every production kernel does.

Out of memory during decode at moderate context. The cache formula says 32 concurrent 4k sequences need 16 GiB, but the run fails at 24. Cause: contiguous per-sequence buffers sized for the maximum length, plus allocator fragmentation as sequences of different lengths finish. Fix: paged allocation, or at least `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for experiments, and compute the budget from bytes per token times the actual token count, not the maximum.

Speculative decoding that slows things down. Symptom: tokens per second drop after enabling a draft model. Causes: the target was already compute-bound (large batch), the draft's acceptance rate on your distribution is far below the rate measured on easy text, or the draft is not small enough relative to the target. Fix: measure $\alpha$ on your traffic, compute the expected tokens per pass from the formula, and only enable it when the ratio beats the draft's cost.

## Measure it

For a memory-bound kernel the metric is achieved bandwidth as a fraction of $\beta$; 80 to 90 percent of the measured copy bandwidth is finished, and chasing the rest is rarely worth it. For a compute-bound kernel the metric is achieved TFLOP/s as a fraction of $\pi$; PyTorch's flash backend reaching 87 percent of the GEMM peak at $N = 16384$ is a good number, and a hand-written kernel that reaches 60 percent is respectable. The first diagnostic for any kernel is the arithmetic intensity from the shapes: if it is below the ridge, stop optimizing arithmetic and count bytes.

For inference the metrics are per-stream latency (ms per token at batch 1, bounded below by weight bytes over $\beta$, about 10.5 ms for a bf16 8B model on this card) and aggregate throughput (tokens per second across the batch), and you should report both together with the batch size and context length, since one is traded for the other. A serving configuration is good when its per-token time is within 1.5 times the bandwidth bound at its batch size; the gap is launch overhead, attention at long context, and sampling.

For KV-cache compression the metric is task accuracy at a fixed bytes-per-token budget on a task that needs the evicted or quantized information, such as needle-in-a-haystack retrieval or long-document QA, not perplexity, which is dominated by local context and hides retrieval failures.

## Exercises

1. Derive the arithmetic intensity of the SwiGLU MLP block of Llama-3 8B (hidden 4096, intermediate 14336, three weight matrices) at prefill with $N$ tokens and at decode with batch $B$. Check: at decode, intensity is about $B$ FLOP/byte, and the block reads $3 \times 4096 \times 14336 \times 2 \approx 352$ MB of weights per step.

2. Extend `flash_forward` to return the per-row log-sum-exp $L = m + \log \ell$ and write a backward pass that recomputes $P$ from $L$ block by block. Check: gradients with respect to $Q, K, V$ match `torch.autograd` on the reference to within $10^{-5}$.

3. Modify the Triton softmax to handle rows longer than the register budget by looping over column chunks with the online max and sum. Check: correct on a $256 \times 100{,}000$ input, and achieved GB/s within a factor of 1.3 of the single-tile version on $4096 \times 2048$.

4. Compute the KV-cache bytes per token for Llama-3 70B (80 layers, 8 KV heads, head dim 128) and for a hypothetical 70B model with 64 KV heads. Check: 320 KiB and 2.5 MiB. How many tokens of bf16 cache fit in the 5090's memory alongside the 70B model's weights? (Trick question; answer in the Test yourself section.)

5. Using the bandwidth bound, plot predicted per-stream tokens per second for Llama-3 8B in bf16 against batch size from 1 to 128 at context 4096, and overlay the aggregate throughput. Check: per-stream drops monotonically, aggregate rises then flattens once $B \times 0.54$ GB exceeds the 16.1 GB weight term.

6. Implement speculative decoding's acceptance rule for two categorical distributions $p$ and $q$ in NumPy and verify empirically that the accepted-token distribution equals $p$. Check: over $10^5$ trials the total variation distance between the empirical distribution and $p$ is below 0.01 for any $q$ you choose, including one with $q(x) = 0$ where $p(x) > 0$.

## Test yourself

1. A colleague says FlashAttention is faster because it reduces the number of floating point operations. What is wrong with that, and what does it reduce?

<details><summary>Answer</summary>
It does slightly more arithmetic: the rescaling by $\alpha$ on every block, and in the backward pass a full recomputation of $P$ from the stored log-sum-exp. It reduces HBM traffic from $O(N^2)$ to $O(N^2 d^2 / M)$ by never writing the score matrix, which raises the intensity from about $d/2$ to about $B_q$ and moves attention from the memory-bound side of the roofline to the compute-bound side. Speed came from bytes, not FLOPs.
</details>

2. Using $\beta = 1.525$ TB/s and $\pi = 239$ TFLOP/s, is a bf16 matmul of shape $64 \times 4096 \times 4096$ compute-bound or memory-bound, and how many rows would you need to cross the ridge?

<details><summary>Answer</summary>
$W = 2 \times 64 \times 4096^2 \approx 2.15$ GFLOP; $Q \approx 2(4096^2 + 2 \times 64 \times 4096) \approx 34.6$ MB; $I \approx 62$ FLOP/byte, below the ridge of about 157, so memory-bound. Intensity is roughly $M$ for $M \ll K, N$, so about 160 rows are needed. In the measured run $M = 128$ reached 171 TFLOP/s, consistent with being near the ridge.
</details>

3. Spot the bug in this online-softmax update:

```python
m_new = torch.maximum(m, s.max(dim=-1).values)
p = torch.exp(s - m[:, None])
alpha = torch.exp(m - m_new)
l = alpha * l + p.sum(dim=-1)
acc = alpha[:, None] * acc + p @ vj
m = m_new
```

<details><summary>Answer</summary>
`p` is computed with the old maximum `m` instead of `m_new`. The old state is correctly rescaled to the new reference $m_{\text{new}}$, but the new block's terms are referenced to $m$, so the two are added at inconsistent scales and the result is wrong whenever the block raised the max. It also reintroduces overflow risk, since $s - m$ can be large and positive. On the first block, where $m = -\infty$, it gives $e^{+\infty}$ immediately.
</details>

4. In the fp32 reference implementation, the score matrix for $N = 16384$ and 32 heads is how large, and would the naive computation fit on the card at batch 1?

<details><summary>Answer</summary>
$32 \times 16384^2 \times 4$ bytes $= 32$ GiB for $S$ alone, and the same again for $P$ unless computed in place. It does not fit in the 31.4 GiB available, which is why naive attention at 16k context is not only slow but impossible here without chunking, and why the flash version at that length took 10.5 ms with no intermediate at all.
</details>

5. Exercise 4 asked how many tokens of cache fit beside Llama-3 70B on the 5090. Answer it properly.

<details><summary>Answer</summary>
None: the bf16 weights are about 141 GB, more than four times the card's memory, so the question is malformed until you fix the weight format. At 4-bit weights (about 35 to 40 GB with overhead) it still does not fit in 31.4 GiB usable. At 3-bit or with layer offloading the arithmetic starts, but then decode is bounded by PCIe bandwidth for the offloaded layers, not GDDR7, and the roofline must be redrawn with that $\beta$. The lesson is to check the weight term before the cache term.
</details>

6. With acceptance rate $\alpha = 0.8$ and draft length $\gamma = 4$, a target pass costs 1.1 decode-step units and each draft step costs 0.1. What is the expected speedup, and name two situations where the real speedup is far lower.

<details><summary>Answer</summary>
Expected tokens per pass $= (1 - 0.8^5)/(1 - 0.8) = (1 - 0.32768)/0.2 \approx 3.36$. Cost per pass $= 1.1 + 4 \times 0.1 = 1.5$. Speedup $\approx 3.36 / 1.5 \approx 2.24$. Lower in practice when: the target is at large batch and no longer bandwidth-bound, so the verify pass costs closer to $\gamma$ steps; the acceptance rate measured on easy text does not hold on the real distribution (code, low-temperature sampling, and repetitive text accept well; creative sampling at high temperature does not); the draft model's own decode is launch-bound so 0.1 is optimistic at short context.
</details>

7. Why does grouped-query attention reduce decode time at long context but hardly at all at short context, even though it cuts the KV cache by 4x?

<details><summary>Answer</summary>
Per-step bytes are weight bytes plus cache bytes. GQA changes only the cache term (and slightly reduces the K and V projection weights). At short context the 16 GB weight read dominates and the cache is a rounding error, so the step time barely moves. At 32k context the bf16 MHA cache for an 8B model would be 16 GiB, equal to the weights, and GQA cuts that to 4 GiB, which is a large change in the byte count. The benefit is proportional to context times batch.
</details>

8. A paged attention system uses 16-token blocks. A request has a 3,000-token prompt and generates 500 tokens. How many blocks does it hold at the end, how many token slots are wasted, and what happens to that count if you sample four completions in parallel from the same prompt?

<details><summary>Answer</summary>
$\lceil 3500 / 16 \rceil = 219$ blocks, $219 \times 16 - 3500 = 4$ wasted slots. With four parallel samples, the 187 full prompt blocks ($3000 / 16 = 187.5$, so 187 full and one partial) are shared by reference; the partial block containing the last 8 prompt tokens is copied on write when each sample appends its first token, and each sample then holds its own generation blocks. Total physical blocks are about $187 + 4 \times 32 = 315$ rather than $4 \times 219 = 876$ for four contiguous copies.
</details>

9. You quantize the K cache to int8 with one scale per tensor and see a large accuracy drop; the V cache quantized the same way is fine. What is the likely cause and the fix?

<details><summary>Answer</summary>
Keys have a few channels with magnitudes far above the rest, a consequence of rotary embeddings and how attention concentrates on specific dimensions; a single per-tensor scale spends its range on those channels and rounds everything else to a few levels. Values do not have this structure. Quantize keys with a scale per channel (per head dimension) and values with a scale per token, which is the asymmetry KIVI describes; the attention scores $q^\top k$ then see accurate outlier channels, and the value mixture stays accurate per token.
</details>

10. A decode benchmark at context 1,024 reports 385 GB/s while at context 65,536 it reports 1,517 GB/s. A colleague concludes the attention kernel is inefficient at short context. Give a different explanation and a test that distinguishes the two.

<details><summary>Answer</summary>
At 1,024 tokens the per-layer cache read is 4 MiB, about 3 microseconds at bandwidth, but each kernel launch costs on the order of 10 microseconds of CPU and driver time, so the 32-layer step is launch-bound, not kernel-bound. The kernel may be perfectly efficient. Test: capture the 32 launches in a CUDA graph and replay it; if achieved GB/s rises sharply at short context and is unchanged at long context, the bottleneck was launch overhead. A second test is the Nsight timeline: gaps between kernels rather than long kernels.
</details>

## What will change, what will not

The roofline will not change. It is a statement about two rates and a ratio, and every accelerator you will use in the next decade, GPU or otherwise, has a compute peak, a memory bandwidth, and a ridge point between them. The numbers move every generation (Blackwell's bandwidth and peak are already different from Ada's, and the next card will differ again), but the habit of writing $W$, $Q$, and $I$ from the shapes before touching code is the durable skill. The same is true of the bytes-per-token formula for the cache and the bandwidth bound on decode: they follow from the architecture of a causal transformer, and any model with cached attention state obeys them.

The online softmax is mathematics and will outlive every kernel that uses it. The invariant (a running max, a running sum, a running unnormalized output, all rescaled by $e^{m - m_{\text{new}}}$) is the reason exact attention can stream, and it will be inside whatever attention implementation is fastest in five years. What will change is the tiling: block sizes, the way tiles are moved (Hopper's TMA and warp specialization, datacenter Blackwell's tcgen05 tensor memory, which the consumer sm_120 part in your box does not have), the precision of the tiles (fp8 today, fp4 for some paths), and which library exposes it.

Triton's syntax and its compiler will change, and the specific numbers for register spills and shared memory limits will change with each card. The idea that a kernel is a program over tiles, that memory movement is explicit, and that a memory-bound kernel is done when it reaches bandwidth is not tied to Triton. Learn it there because the feedback loop is short; expect to rewrite the kernels.

The KV cache's byte count is an invariant of the architecture; the mechanisms that reduce it are tooling. Paged allocation is the right abstraction and will persist under some name because it is virtual memory, which has been the right abstraction for sixty years. Which quantization format wins (int8, fp8, 2-bit with per-channel keys) and which eviction heuristic survives will be decided by measurement on real workloads, and the current answers are provisional. Architectures that store less per token (multi-head latent attention, hybrid state-space layers, sliding windows) attack the same term and may make some of this moot for some models; the accounting still applies to whatever state remains.

Speculative decoding's acceptance rule is exact and will not change. The draft mechanism will: separate draft models, extra heads on the target, and n-gram lookups are all current, and the winner depends on the batch regime the serving system runs in. The invariant is that verification is cheap only when the target is bandwidth-bound, which brings you back to the roofline.

## Read next

1. "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Williams, 2009. The original model; short, and the diagrams in this chapter are its diagrams.
2. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", Dao, 2022. The IO-complexity argument, the tiling, and the backward recomputation, with the proof that HBM traffic is $O(N^2 d^2 / M)$.
3. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", Dao, 2023. What changed between the algorithm and a kernel that reaches a large fraction of peak: work partitioning across warps and fewer non-matmul operations.
4. "Online normalizer calculation for softmax", Milakov, 2018. The online softmax recurrence on its own, before it was used in attention.
5. "Efficient Memory Management for Large Language Model Serving with PagedAttention", Kwon, 2023. The vLLM paper; block tables, copy-on-write, and the memory-waste measurements that motivated it.
6. "Fast Inference from Transformers via Speculative Decoding", Leviathan, 2023. The acceptance rule and its proof of exactness, with the expected-tokens formula.
7. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", Ainslie, 2023. Why the KV-head count is a free parameter and how to convert a trained model.
8. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", Tillet, 2019. The tile-program model behind the language you wrote the softmax in.
