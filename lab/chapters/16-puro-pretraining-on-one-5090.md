---
title: "Lab 16: Pretraining a 1.5B model on one RTX 5090 (the Puro-2B recipe)"
kind: permanent
topics: [lab]
chapter: 16
station: pretrain
recipe: recipes/pretrain_nano.py
reading_time: 65 min
---

## What you will be able to do

- Price a pretraining run on consumer hardware from first principles: FLOPs from $6ND$, GPU-hours from a measured throughput, dollars from an amortized rate, and say which of those three numbers is an assumption.
- Explain FP8 training precisely: the E4M3 and E5M2 formats, why one scale per tensor fails on real activations, what per-block scaling with power-of-two scales does about it, which tensors stay in BF16 or FP32, and how it differs from the bf16 autocast you have been using since Lab 02.
- Write down the MuonH (Muon with Hyperball) update, derive why its learning rate is the effective learning rate, and say what ordinary Muon with weight decay does instead.
- Describe curriculum model averaging as Puro-2B implements it (component-local ordering, a constant-rate continuation, an equal-weight average of six checkpoints) and read the ablation that shows which of the three parts carries the gain.
- Read the Puro cost scaling law, $P = a + b \log_2(C - C_{P1})$, without over-reading it, and know exactly which fifteen benchmarks the "comparable to Qwen2.5-1.5B" claim rests on.
- Plan a weekend reproduction of the recipe's shape at a size one 5090 can afford, and list which pieces `recipes/pretrain_nano.py` already has and which it does not.

## The idea in one paragraph

Puro-2B is a report from Tsinghua (Luo, Cui, Yin, Chen, Yang, Gao, Wang, Zhang, Wen, Lyu, and Chen, arXiv 2608.27370, August 2026) that trains a dense 2B-parameter decoder in the Qwen3-1.7B shape from scratch on 1.4 trillion tokens using only RTX 5090 cards, and reports a marginal accelerator cost of about $6.9K for the best model. The cost comes from five decisions stacked on top of each other. The card: a 5090 has roughly a fifth of an H200's peak throughput but costs about a thirteenth as much per hour under their accounting, so peak compute per dollar is about 2.7 times higher. The arithmetic: the linear-layer matmuls run in FP8 with one scale per 128-element block, which they measure as 1.36 times faster at the 1.7B scale for a loss penalty of a few thousandths of a nat. The optimizer: Muon wrapped in a Hyperball constraint, which pins each hidden matrix to a sphere so that the learning rate you schedule is the angular step you get. The late stage: instead of decaying the learning rate to zero over uniformly shuffled data, they order each data source from its lowest-scored to its highest-scored examples, hold the rate constant for the final 29B tokens, and average six checkpoints. The data: public datasets selected by training small proxy models on slices and reading the resulting benchmark vectors. Everything is released under Apache 2.0. In the pretrain station you have been watching a 2-layer, width-48 character model fall from $\ln V$; this chapter is about what changes when the same loss is run for 1.4T tokens on a budget you could actually spend.

## The math

### The cost model: from tokens to dollars

Lab 02 gave you $C \approx 6ND$ FLOPs for $N$ matmul parameters over $D$ tokens. Puro-2B uses the same accounting (their Equation 12) with $N \approx 2 \times 10^9$ and $D = 1.4 \times 10^{12}$, giving $C = 1.68 \times 10^{22}$ FLOPs, a number the paper states. To turn FLOPs into GPU-hours they use (their Equation 15)

$$H = \frac{C}{3600 \times 10^{12} \times \bar p},$$

where $\bar p$ is the median per-GPU throughput in TFLOP/s over the step history, so $H$ is aggregate GPU-hours and the number of GPUs cancels. The dollar cost is $H$ times a rental-equivalent rate $r$. All three inputs deserve scrutiny.

$N$ is the least contentious but note the model is about 2B parameters, not 1.5B: the Qwen3-1.7B configuration (28 layers, hidden 2048, MLP width 6144, 16 query heads, 8 KV heads, head dimension 128, from their scaling-ladder Table 10) with the input embedding untied from the output head. The "1.5B" in the comparison is Qwen2-1.5B and Qwen2.5-1.5B, the models it is measured against; the paper never states a vocabulary size in the text, only that the vocabulary is large enough that the LM head costs as much as several transformer layers.

$\bar p$ is measured, and it is the number you should copy. Phase 1 (24 GPUs, three nodes) sustained a median 238 TFLOP/s per card; Phase 2 (96 GPUs) sustained 192, because gradient synchronization over PCIe and InfiniBand takes a larger share of each step at that world size. To call 238 an MFU you need a peak, and their peak is precision-weighted: 72 percent of the theoretical tensor-core work runs in FP8 at a 419 TFLOP/s peak and 28 percent in BF16 at 209.5, so

$$P_{\text{eff}} = \left(\frac{0.72}{419} + \frac{0.28}{209.5}\right)^{-1} \approx 327 \text{ TFLOP/s}, \qquad \text{MFU} = \frac{238}{327} \approx 0.73.$$

This is a harmonic mean, because the two kinds of work are done in sequence and time adds. Read the 73 percent with its convention attached: it is not 73 percent of the BF16 peak (that would be 114 percent, impossible), and it is not comparable to an MFU quoted against a single dense peak.

$r$ is the assumption. There is no public rental market for 5090s (NVIDIA's EULA forbids data-center use of consumer cards), so they amortize: an 8-GPU node costs them about 12,000 CNY ($1{,}763$) per month including electricity, spread over five years, which is $1763 / (8 \times 730) \approx \$0.30$ per GPU-hour, reported as $0.31. Every dollar figure in the paper is GPU-hours times this rate, and the accounting boundary is narrow by design: it includes only the two production runs, and excludes data processing, proxy experiments, ablations, failed runs, post-training, evaluation, and the averaging step itself. Their own word for it is a marginal accelerator cost, and you should use the same care when you quote it.

Now check the headline. Phase 1: $438.84$B tokens, $6 \times 2 \times 10^9 \times 4.388 \times 10^{11} = 5.27 \times 10^{21}$ FLOPs, at 238 TFLOP/s is $6{,}150$ GPU-hours; wall clock was $10.43$ days on 24 cards, which is $6{,}008$ GPU-hours. Phase 2: $960$B tokens, $1.15 \times 10^{22}$ FLOPs at 192 TFLOP/s is $16{,}700$ GPU-hours; $7.16$ days on 96 cards is $16{,}497$. The measured total is $22{,}514$ GPU-hours, and $22{,}514 \times 0.306 \approx \$6{,}890$, the reported $6.89K. The FLOP accounting and the wall clock agree to within 3 percent, which tells you the $N \approx 2$B and the $6ND$ approximation are both fine at this scale.

The comparison that motivates the whole paper is compute per dollar at peak (their Table 2): a 5090 delivers $419 \times 10^{12} \times 3600 / 0.31 = 4.87 \times 10^{18}$ FP8 FLOPs per dollar, an H200 at $4.00/h delivers $1979 \times 10^{12} \times 3600 / 4 = 1.78 \times 10^{18}$, a ratio of 2.74. In BF16 the ratio is 2.43 to 0.89, again 2.7. Tokens per second per dollar is the same number divided by the FLOPs per token: at 238 TFLOP/s and $6N = 1.2 \times 10^{10}$ FLOPs per token, one card processes about $19{,}800$ tokens per second, which is $2.3 \times 10^8$ tokens per dollar-hour at $0.31. The price of that ratio is 32 GB of memory and no NVLink; Section 3.1.2 of the paper describes the driver modifications they used to enable PCIe peer-to-peer and GPUDirect RDMA on consumer cards, and warns that they are unsupported.

Worked example, one card. Suppose you wanted the paper's $D = 1.4$T tokens at a nominal $N = 1.5 \times 10^9$ on the single 5090 you have. $C = 6 \times 1.5 \times 10^9 \times 1.4 \times 10^{12} = 1.26 \times 10^{22}$. At their 238 TFLOP/s, $H = 1.26 \times 10^{22} / (3600 \times 10^{12} \times 238) = 14{,}700$ hours, which is 613 days. At the rate a single-GPU bf16 nanoGPT-style trainer plausibly sustains, say 40 percent of the 209.5 BF16 peak or 84 TFLOP/s (an assumption; the recipe prints the measured value), it is $41{,}700$ hours, nearly five years. The conclusion is not that the paper is out of reach; it is that the paper's unit is a node of eight cards and the reproduction is 22.5K GPU-hours however you slice them. What one card affords in a weekend is worked out in "Build it real".

### FP8 training

Lab 02 told you to use bf16 and never fp16 without loss scaling, and Lab 13 mentioned fp8 tiles inside attention kernels. Here is the whole story. A floating-point format with $e$ exponent bits and $m$ mantissa bits represents $\pm 2^{E}(1 + f)$ with $f$ on a grid of $2^{-m}$, so its relative rounding error is at most $2^{-(m+1)}$ and its dynamic range is set by $e$. bf16 has $e = 8, m = 7$: the fp32 exponent range and a relative precision of $2^{-8} \approx 0.4$ percent. The two 8-bit formats (Micikevicius et al., 2022) split the remaining bits differently:

$$\text{E4M3: } e = 4, m = 3, \quad \max = 448, \quad \text{smallest normal } 2^{-6}, \quad \text{precision } 2^{-4} = 6.25\%,$$
$$\text{E5M2: } e = 5, m = 2, \quad \max = 57{,}344, \quad \text{smallest normal } 2^{-14}, \quad \text{precision } 2^{-3} = 12.5\%.$$

(E4M3 as implemented by NVIDIA and PyTorch's `float8_e4m3fn` spends its would-be infinity encodings on the value 448, hence the odd maximum.) The standard recipe uses E4M3 for anything you multiply and reserves E5M2, if it is used at all, for gradients whose range is wider than their precision needs. Puro-2B uses E4M3 for all three GEMMs of each linear layer (forward, gradient with respect to input, gradient with respect to weight) and does not use E5M2; the paper does not report an E5M2 comparison.

The precision is coarse, so the whole design is about placing the tensor's values where the format has them. Given a tensor $X$, choose a scale $s$ and store $Q(sX)$, then compute $Y = (Q(sX) Q(s' W)) / (s s')$. The obvious choice $s = 448 / \max |X|$ puts the largest entry at the top of the range. The problem is what it does to the smallest entries. Transformer activations are heavy-tailed: a handful of channels in the residual stream carry values hundreds or thousands of times the median. If $\max |X| / \text{median} |X| = 10^4$, then with one scale per tensor the median entry lands at $448 / 10^4 = 0.045$, which is below the smallest normal $2^{-6} = 0.0156$ times a few, and entries a further factor of ten smaller are subnormal (absolute, not relative, precision) or flushed to zero entirely. The outlier has spent the format's range on itself. This is the failure per-tensor scaling has on real models, and it is why "FP8 works" and "FP8 diverges" both appear in the literature: it depends on the scaling granularity.

Per-block scaling gives each small tile its own $s$. DeepSeek-V3 introduced the arrangement Puro-2B copies: activations and activation gradients get one scale per 128 consecutive elements along the reduction dimension of the GEMM (a $1 \times 128$ strip), weights get one scale per $128 \times 128$ block, and the scales are computed online from each block's current maximum, not carried over from a previous step (delayed scaling, the older Transformer Engine default, uses a stale maximum and is the classic cause of an FP8 overflow after a spike). A block that contains an outlier is quantized coarsely; every other block is not. The dot product in the GEMM then accumulates $\sum_k Q(x_k) Q(w_k) / (s_{\text{blk}(k)} s'_{\text{blk}(k)})$, so the kernel must apply the scale pair per block inside the reduction, which is exactly the feature the hardware path provides.

One detail is specific to the card you own. The 5090 is compute capability SM 120, and its block-scaled FP8 path is MXFP8, in which each block scale is an E8M0 value: eight exponent bits, no mantissa, so a power of two. Puro-2B therefore keeps DeepSeek-V3's logical block sizes but rounds each scale down to a power of two, which costs up to one bit of headroom in a block (the block maximum lands somewhere between 224 and 448 rather than at 448). The snippet below measures what that costs.

What stays in higher precision, in their Figure 14: master weights and optimizer states in FP32 and BF16; the BF16 copy of the weights is what the checkpoints hold; core attention (the FlashAttention or SDPA call between the projections) in BF16; embeddings, norms, and the LM head in BF16; every residual-stream tensor, all communication, and gradient accumulation in BF16 or FP32. FP8 exists only inside a linear layer: it quantizes its BF16 input and weight immediately before the GEMM, keeps the quantized input for the weight-gradient GEMM (which halves the dominant saved-activation footprint), and hands a BF16 output back. They run this from step zero, with no BF16 warm-up and no switch, and they report that 72 percent of the total computation runs on the FP8 path.

Contrast with bf16 autocast. Autocast changes the storage and matmul type of activations to bf16 and keeps fp32 master weights and fp32 reductions, but it uses no scales, because bf16 has the fp32 exponent range and nothing underflows. FP8 adds a per-block statistic (the absmax) and a scale multiply on every operand of every GEMM, in both directions, every step. That overhead is why the paper notes that FP8 kernels "become memory-bound more readily than BF16" at small hidden dimension, and why they tuned the micro-batch to the knee of the throughput curve per GEMM shape. FP8 pays off when the GEMMs are large enough that the arithmetic saved exceeds the quantization traffic added; at 124M parameters it usually does not, and you should not expect the 1.36 times on the weekend model.

The measured quality cost, from their five-model scaling ladder (0.17B to 1.7B, 20 tokens per parameter, same data, MuonH in both arms): blockwise FP8 sits 0.0031 to 0.0039 nats above BF16 in validation loss at every size. Fitting both arms to a shared-shape curve $L(C) = L_\infty + A_g (C / C_0)^{-\alpha}$ and reading the horizontal gap turns that vertical gap into a compute retention of 98.0 percent (97.6 to 98.1 leaving one size out), so FP8 needs 2 percent more nominal compute to match BF16. Multiplying the 1.36 times throughput gain at 1.7B by 0.98 gives their headline 1.34 times, or 25 percent fewer GPU-hours at matched quality. The throughput ratio is measured only at 1.7B and used as a proxy for the 2B production model; they say so.

### MuonH: Muon on a sphere

Lab 12 built Muon: keep a momentum buffer $M_t = \mu M_{t-1} + G_t$ of the gradient of a hidden weight matrix, orthogonalize it with a few Newton-Schulz iterations to get $u_t \approx U V^\top$ (all singular values pushed toward one), and step $W_{t+1} = W_t - \eta_t u_t$, with decoupled weight decay added at scale. Puro-2B uses the Hyperball wrapper of Wen et al. (2026), which they call MuonH, on the attention and MLP matrices, and AdamW on everything else (embeddings, norms, the LM head). The update (their Equation 1) is

$$\hat u_t = \frac{u_t}{\|u_t\|_F}, \qquad \widetilde W_{t+1} = W_t - \eta_t R \, \hat u_t, \qquad W_{t+1} = R \, \frac{\widetilde W_{t+1}}{\|\widetilde W_{t+1}\|_F}, \qquad R = \|W_0\|_F.$$

Read it as two operations. The update is normalized to unit Frobenius norm and then scaled by the sphere's radius, so the displacement has norm exactly $\eta_t R$ regardless of how large or small the Muon update happened to be. Then the result is projected back onto the sphere of radius $R$, the norm the matrix had at initialization, so $\|W_t\|_F = R$ for all $t$. MuonH-wrapped matrices use no weight decay (Table 8); the sphere replaces it.

Why a sphere. Call a matrix approximately scale-invariant when $\mathcal L(cW) \approx \mathcal L(W)$ for $c > 0$ with everything else fixed (their Equation 2). In a pre-norm transformer this holds well for a matrix whose output passes through a normalization before it affects anything, which covers most of the attention and MLP weights. Differentiate the invariance at $c = 1$: $\frac{d}{dc} \mathcal L(cW)|_{c=1} = \langle \nabla \mathcal L(W), W \rangle = 0$, so the gradient is orthogonal to the weight. For such a matrix, the only thing an update can do is rotate it, and the meaningful size of a step is the angle, which is the displacement relative to the norm. That is the effective learning rate (their Equation 4):

$$\rho_t(W) = \frac{\eta_t \|u_t\|_F}{\|W_t\|_F}.$$

For ordinary Muon with decay, $W_{t+1} = (1 - \lambda \eta_t) W_t - \eta_t u_t$ (their Equation 3), and $\rho_t$ depends on $\|W_t\|_F$ and $\|u_t\|_F$, both of which drift. You can see how by squaring the update when $u_t \perp W_t$: $\|W_{t+1}\|^2 \approx (1 - \lambda\eta)^2 \|W_t\|^2 + \eta^2 \|u_t\|^2$, which has a fixed point at $\|W\|^2 \approx \eta \|u\|^2 / (2\lambda)$, so at equilibrium $\rho \approx \sqrt{2 \lambda \eta}$ (this derivation is mine, not the paper's; the paper simply observes the drift). The effective rate is a function of the decay and the scalar rate, not the scalar rate alone, and before equilibrium is reached it is whatever the current norms make it. In their 170M diagnostic (Figure 5), ordinary Muon fed the same base schedule as MuonH produced an effective-rate trace that fell fast early and finished near zero, and ended at validation loss 3.073 against MuonH's 3.029.

For MuonH, the pre-projection displacement has norm $\eta_t R$ and the matrix has norm $R$, so $\rho_t = \eta_t R / R = \eta_t$: the number in the schedule is the effective rate. Geometrically, if $\hat u_t \perp W_t$ then $\|\widetilde W_{t+1}\|^2 = R^2(1 + \eta_t^2)$, the projection shrinks by $1/\sqrt{1 + \eta_t^2}$, and the rotation angle is $\arctan \eta_t$, about $\eta_t$ for small rates; with their production Hyperball rate $10 \eta^{\text{base}}$ peaking at $5 \times 10^{-2}$, that is an angle of about 2.9 degrees per step at peak. The decisive control in the paper is the third run of Figure 5: ordinary Muon with its scalar rate adjusted online every step as $\eta_t = \rho^H_t \|W_t\|_F / \|u_t\|_F$ (their Equation 5) to follow MuonH's effective-rate schedule, without any projection, reached 3.030. The sphere is not the point; the explicit effective rate is. Appendix F.3 pushes this further: a scalar schedule that rises during the stable phase (a "hill") is what ordinary Muon needs to hold a constant effective rate while its norms grow.

Two learning rates, then. The AdamW groups follow a base schedule $\eta^{\text{base}}_t$; the MuonH groups use $\eta^H_t = m \, \eta^{\text{base}}_t$ with $m = 10$ in production (the scaling ladder used $m = 2$ with a base of $10^{-2}$). Phase 1's base schedule (their Equation 10), with $k$ the optimizer step:

$$\eta^{\text{base}}(k) = \begin{cases} 5 \times 10^{-3} \, k / 1000, & k \le 1000, \\ 5 \times 10^{-4} + 4.5 \times 10^{-3} \left(1 + \dfrac{k - 1000}{1000}\right)^{-1/2}, & k > 1000. \end{cases}$$

That is a 1,000-step linear warmup to $5 \times 10^{-3}$ followed by a power decay (Shen et al., 2024) toward an asymptotic floor of $5 \times 10^{-4}$, chosen because it never commits to an end date: a power law is open-ended, so Phase 1 can be extended or continued without a decision. Check the endpoint: Phase 1 is $438.84$B tokens at $1536 \times 4096 = 6.29$M tokens per step, about $69{,}750$ steps, and $(1 + 68750/1000)^{-1/2} = 0.120$, giving $5 \times 10^{-4} + 5.4 \times 10^{-4} = 1.04 \times 10^{-3}$, the value the paper reports as Phase 1's terminal rate. Phase 2 restarts a local clock at that checkpoint and decays linearly (their Equation 11), $\eta(s) = \eta_0 + (\eta_{\min} - \eta_0) \, s / S_j$, from $\eta_0 = 1.04 \times 10^{-3}$ to $\eta_{\min} = 10^{-5}$ over a budget $S_j$ of samples. Their WSD sweeps on a 0.6B model (Section 3.3.2) motivated the long linear decay: at higher effective peaks and at longer horizons, the region of decay ratios within 0.01 nats of the best shifted toward longer decays, and at peaks of 0.020 and 0.024 a fully linear decay was within 0.005 nats of the best grid point.

What the paper does not give: $\beta$ values or $\epsilon$ for the AdamW groups, the Muon momentum, the number of Newton-Schulz iterations, and the gradient clipping threshold. Do not fill those in from memory when you write about this recipe; take them from the released Puro-Megatron configuration when you need them.

### Curriculum model averaging

Phase 2 is 960B tokens on a mixture that shifts toward mathematics, code, and instruction-formatted data (Figure 8: English 73.2 percent of Phase 1 versus 59.4 percent of Phase 2, math 7.2 versus 18.3, code 7.9 versus 11.5, Chinese 11.7 versus 9.4, instruction data 0 versus 1.3 percent). Three variants of Phase 2 share this pool and differ only in order and endpoint. UD (uniform data with decay) globally reshuffles the pool and decays the rate to $10^{-5}$. CD (curriculum with decay) keeps the mixture but orders each source from its least to its most preferred quality score. CMA adds the late constant-rate continuation and the six-checkpoint average.

The ordering is component-local, which is the part to get right if you reimplement it. Within each source that ships a usable sample-level score (many public datasets do), examples are sorted from lower to higher score and assigned a normalized rank by cumulative token mass, so rank 0.25 means a quarter of that source's tokens come earlier. Sources without scores get a fixed random order and the same ranks. The rank range is cut into 376 intervals, and curriculum bucket $k$ takes interval $k$ from every source, so each bucket holds about 2.5B tokens, draws roughly $1/376$ of every component, and preserves the mixture weights exactly while its quality rises. Scores are never compared across sources; this is explicitly not a global quality ranking. The Phase 2 stream is materialized once into shards read sequentially (seeds are recorded), so the same pool yields the curriculum stream or the reshuffled UD stream by changing only the materialization order.

Why a curriculum conflicts with a decay: the best data arrives last, exactly when the learning rate, and so the size of the update each example can make, has gone to nearly zero. CMA (Luo et al., 2026) resolves the conflict by holding the rate constant through the end and removing the noise ball by averaging instead of by decaying. Lab 03 derived the noise ball: at constant $\eta$ the iterate fluctuates around the minimum with a spread proportional to $\eta$, and both a cooldown and an average of iterates remove that excess. The difference is that a cooldown removes it by making the last examples nearly irrelevant, and an average removes it while every averaged checkpoint saw the late data at full rate.

The production numbers. They followed the linear decay along the curriculum to step 218,000, resumed from that checkpoint with the base rate frozen at its value there, $4.08 \times 10^{-5}$ (Hyperball rate $4.08 \times 10^{-4}$, which is the effective rate of the MuonH matrices), trained the final 29B tokens at that constant rate, and averaged the parameters saved at steps 222,100, 222,200, 222,300, 222,400, 222,500, and 222,569 with equal weights. The window is 469 steps, about 2.95B tokens; only parameters are averaged, not optimizer states; the released model is the average, not the last iterate. Check the resume rate: the decay runs from $1.04 \times 10^{-3}$ at Phase 2's start (global step about 69,750) to $10^{-5}$ at its scheduled end near step 222,570, and step 218,000 is 97 percent of the way, giving $1.04 \times 10^{-3} - 0.97 \times 1.03 \times 10^{-3} \approx 4 \times 10^{-5}$, consistent with the reported value.

The ablation (Section 3.4, all numbers the 15-benchmark average defined below) is worth reading as a teacher, because each pair isolates one part. Ordering: UD 55.99 versus CD 57.17, so the curriculum alone is worth 1.18 points. Averaging on the decayed trajectory: UD 55.57 (a loss of 0.42) and CD 57.18 (a gain of 0.01), so averaging checkpoints whose noise has already been removed by a decay does nothing, which is the noise-ball theory saying the same thing twice. Constant-rate continuation without averaging (their CDC controls): 55.64 resuming at step 215,000 and 57.12 at 218,000, below CD, because the continuation reintroduces the noise ball. Continuation plus averaging: 56.80 and 57.81. The gain of averaging is therefore conditional on the rate being nonzero, and the whole CMA endpoint beats CD by 0.64 and UD by 1.82 points. The paper also notes the honest caveat that the two resume points were not a controlled sweep, and that some of CD's edge over UD could be benchmark contamination in the late, high-scored data, which its post-training experiments (below) argue against but do not rule out.

### The data recipe, briefly

The corpus is assembled from public datasets (Appendix I lists every family with its Hugging Face identifier, tokens per phase, and declared license): Nemotron-CC-v2 high-quality and synthetic partitions (the largest, 107.5B and 97.7B tokens in Phase 1, 448.0B and 39.6B in Phase 2), FineWeb-Edu, DCLM-baseline (a top-5-percent-by-score slice of 33.8B tokens in Phase 2 only), Cosmopedia-v2, ArXiv and FineWiki, the Chinese FineWeb-Edu variants, MegaMath (web-pro and code partitions), UltraData-Math (118.3B, Phase 2 only), OpenWebMath, FineMath, SwallowMath and Swallow-Code, Nemotron synthetic code, and a 12.7B-token Phase 2 tail of instruction-formatted data (Nemotron-Terminal-Corpus at 6.3B, JiuZhang3.0 chain-of-thought, Tulu-3 SFT, ToolMind, and others). The release warns that upstream terms are component-specific and that the NVIDIA data agreements permit training but forbid redistribution of the raw data, which is why the preprocessing framework (Kaiyuan-Spark) is released alongside.

Selection is by proxy profile rather than by a universal quality score. Each candidate slice (four score-quantile slices for a scored source over 50B tokens, one random 4B-token slice for a source between 5B and 50B) is used to continue a Qwen3-0.6B checkpoint, pretrained on 86B tokens of the base mixture, for 2,000 steps of about 8.4B tokens with the candidate's share ramping to 80 percent, and the resulting checkpoint is scored on the 15-benchmark suite. The vector of scores is the slice's capability profile; sources are then weighted by hand from those profiles, with a PCA (Figure 20) used to see which capabilities trade against which (code against general knowledge more than math does). The within-source quantile slices are what made the DCLM filter: the top slice ranked third of 39 candidates while the second quartile ranked eleventh. Web deduplication is within-source only (MinHash, in native C++ through Kaiyuan-Spark); they found cross-source deduplication removed little.

### The Puro cost scaling law

Starting from the same Phase 1 checkpoint, they ran Phase 2 with the UD recipe at five budgets (60, 120, 240, 480, and 960B tokens; total costs $2.16K, $2.47K, $3.10K, $4.37K, and $6.89K; Table 17) and fit the 15-benchmark average $P$ against cost $C$ in dollars as

$$P = a + b \log_2 (C - C_{P1}), \qquad C_{P1} = \$1.84\text{K}.$$

Read the form. Subtracting the fixed Phase 1 cost and taking a log of the increment says that each doubling of Phase 2 spend buys a constant number of points; the shift matters (in-sample RMSE 0.209 with it, 0.452 without). The fitted $a$ and $b$ are not printed in the text, only drawn in Figure 2(b); from the UD points you can read (51.98, 53.24, 54.01, 55.54, 55.99 at Phase 2 spends of $0.32K, $0.63K, $1.26K, $2.53K, $5.05K) the slope is roughly one point per doubling, which is a reading of the table, not the paper's fit. Three uses and three limits. First, it places the $4.4K UD checkpoint (55.54) above Qwen2-1.5B (55.14), the basis of the abstract's "less than $5,090" claim. Second, inverting it at the CD score (57.17) gives a UD-equivalent cost of $11.36K, so the curriculum is worth 1.65 times the money; inverting at the CMA score (57.81) gives $16.55K, or 2.40 times. Those multipliers are how "cost-efficiency gains" in Figure 2 are defined: the UD spend that would have reached the same average. Third, it is explicitly a scale-down curve for this one model at this one recipe; it says nothing about a larger model or a different pipeline, the authors do not claim a model-size law, and it is fitted on five points whose top end is the production run itself.

### What "comparable to Qwen2.5-1.5B" was measured on

Everything is run through OpenCompass with fixed prompts, shots, and postprocessors, and every baseline is re-evaluated under the same pipeline rather than quoted from its report. Fifteen benchmarks. Generation-based, greedy decoding: GSM8K (4-shot), MATH (4-shot), sanitized MBPP (3-shot), HumanEval (3-shot), MMLU-Pro (5-shot with chain of thought), BBH (4-shot). Log-likelihood ranking of the given choices, no sampling: MMLU, ARC-Challenge, ARC-Easy, BoolQ, CommonsenseQA, HellaSwag, PIQA, SocialIQA, WinoGrande, all 5-shot; where a task supports both cloze and multiple-choice formulations, the better one is used per model (the OLMES convention). $P$ is the unweighted mean of the fifteen.

The numbers (Tables 3, 4, and 6): on the four math and code tasks Puro-2B averages 43.50 against Qwen2-1.5B's 40.29 and Qwen2.5-1.5B's 47.52 (GSM8K 59.67, MATH 30.30, MBPP 52.92, HumanEval 31.10). On the fifteen, Puro-2B (the CMA endpoint) scores 57.81, Qwen2-1.5B 55.14, Qwen2.5-1.5B 60.73, Qwen3-1.7B-Base 65.27, Gemma-2-2B 48.60, SmolLM3-3B-Base 65.85. So "surpasses Qwen2-1.5B" is 2.7 points and "approaches Qwen2.5-1.5B" is 2.9 points short, on this suite, under this protocol; the paper's own reproduction-cost estimates for those Qwen models are $84K and $217K under an H100-equivalent $6ND$ accounting, which is the frontier plot of Figure 1. Chinese benchmarks are deliberately omitted from the headline tables. Note what the phrase does not mean: nothing here is a chat evaluation, and the base model is 700 tokens per parameter overtrained, a regime the authors call useful for a compact deployable model and "not presented as compute-optimal".

The post-training case study is the part only an open pipeline can do. Taking the UD and CMA checkpoints through identical SFT (MuonH, cosine from $10^{-5}$ to $10^{-7}$, batch 160, three seeds each), the CMA-initialized model scores higher on GSM8K after GSM8K-focused SFT (68.66 versus 66.89, a 1.77-point gap), higher after a longer math-and-code SFT with replay (76.12 versus 74.10), and 1.17 points higher on the 15-benchmark suite and 1.36 on IFEval after Tulu-3 mixed-domain SFT, better on 10 of 15 tasks. That persistence is their argument that the curriculum's edge is capability rather than a formatting or contamination artifact, offered with the caveat that a corpus-wide contamination audit was not done.

### Released artifacts

Under Apache 2.0 unless a component says otherwise: the model weights at `thu-pacman/Puro-2B-Base` with ten checkpoint variants (the UD scaling points, CD, and CMA), the data at `thu-pacman/Puro-2B` (with the component-specific license caveats above), the training code at `github.com/thu-pacman/Puro-Megatron` (Megatron Core v0.16.0 with Transformer Engine, adapted for PCIe-only nodes: pipeline plus data parallelism, no tensor parallelism, a (18|10) layer split over two pipeline stages in Phase 1 and (9|9|9|1) over four in Phase 2 so that the LM head's stage is not the bottleneck, and a memory-aware placement of Muon and Adam states to fit 32 GB), and the preprocessing framework at `github.com/thu-pacman/Kaiyuan-Spark`. The checkpoint ledger (Tables 17 and 18) and the SFT datasets are in the appendices.

## Build it small

A simulated FP8 matmul: both operands are quantized to E4M3 or E5M2 with one scale per tensor or one scale per block (a $1 \times 128$ strip for the activation, $128 \times 128$ for the weight, as in the paper), optionally with the scale rounded down to a power of two as the SM 120 path requires, then dequantized and multiplied in fp32. The activations are heavy-tailed with two outlier channels, which is what a residual stream looks like. CPU, about a second. It measures the quantization error only; the accumulation inside a real FP8 GEMM is fp32 and is not simulated.

```python
# Lab 16, build it small: a simulated FP8 matmul. Quantize both operands to E4M3 (or E5M2)
# with one scale per tensor or one scale per block, optionally rounding the scale to a power
# of two as the SM 120 path requires, dequantize, multiply in fp32, and measure the error.
import torch
torch.manual_seed(0)
E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
FMAX = {E4M3: torch.finfo(E4M3).max, E5M2: torch.finfo(E5M2).max}   # 448 and 57344

def fake_quant(x, fmt, block, pow2):
    """Return x after a round trip through fmt with one scale per (block[0] x block[1]) tile."""
    r, c = x.shape; br, bc = block
    xp = torch.nn.functional.pad(x, (0, (-c) % bc, 0, (-r) % br))          # pad to whole tiles
    R, C = xp.shape
    t = xp.view(R // br, br, C // bc, bc).permute(0, 2, 1, 3)              # (tiles_r, tiles_c, br, bc)
    amax = t.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    scale = FMAX[fmt] / amax                                                # tile max lands on the format max
    if pow2:                                                                # E8M0-style scale: a power of two,
        scale = torch.exp2(torch.floor(torch.log2(scale)))                  # rounded down so nothing overflows
    q = (t * scale).to(fmt).float() / scale                                 # quantize, dequantize
    return q.permute(0, 2, 1, 3).reshape(R, C)[:r, :c]

def rel(a, b): return ((a - b).norm() / b.norm()).item()

M, K, N = 256, 1024, 512
# heavy-tailed activations (log-normal magnitudes) with a few outlier channels, as a post-norm
# residual stream really looks; Gaussian weights at 1/sqrt(K)
x = torch.randn(M, K) * torch.exp(1.0 * torch.randn(M, K))
x[:, [3, 500]] *= 300.0                                                     # two outlier channels
w = torch.randn(K, N) / K ** 0.5
ref = x @ w
print(f"amax(x) = {x.abs().max():.0f}, median |x| = {x.abs().median():.3f}, ratio {x.abs().max() / x.abs().median():.0f}")
print(f"{'operands':46s} rel err of y   fraction of x flushed to 0")
print(f"{'bf16, no scaling':46s} {rel(x.bfloat16().float() @ w.bfloat16().float(), ref):.5f}")
for name, fmt, bx, bw, p2 in [
    ("E5M2 per-tensor",                        E5M2, (M, K),   (K, N),     False),
    ("E4M3 per-tensor",                        E4M3, (M, K),   (K, N),     False),
    ("E4M3 x:1x128  w:128x128",                E4M3, (1, 128), (128, 128), False),
    ("E4M3 x:1x128  w:128x128, pow2 scales",   E4M3, (1, 128), (128, 128), True),
    ("E4M3 x:1x32   w:32x32,   pow2 scales",   E4M3, (1, 32),  (32, 32),   True),
]:
    xq, wq = fake_quant(x, fmt, bx, p2), fake_quant(w, fmt, bw, p2)
    print(f"{name:46s} {rel(xq @ wq, ref):.5f}        {((xq == 0) & (x != 0)).float().mean():.4f}")

# the mechanism: the tensor-wide scale is set by the outlier, so ordinary values land in the
# subnormal range (absolute, not relative, precision) or underflow entirely
small = x.abs() < 0.05
for bx in ((M, K), (1, 128)):
    xq = fake_quant(x, E4M3, bx, False)
    print(f"x block {str(bx):12s} err on |x|<0.05 entries {rel(xq[small], x[small]):.3f}   "
          f"err on the outlier channels {rel(xq[:, [3, 500]], x[:, [3, 500]]):.4f}")

# E5M2 fails in the other direction: range it cannot use, precision it does not have
u = torch.linspace(1.0, 2.0, 9)
print("E4M3 on [1,2]:", u.to(E4M3).float().tolist())
print("E5M2 on [1,2]:", u.to(E5M2).float().tolist())
```

What I observed when I ran it (PyTorch 2.11, CPU). The activation tensor has a maximum of about 6,000 against a median of 0.6, a ratio of $10^4$. bf16 operands give a relative error of $0.0025$ on the product. E5M2 per tensor gives $0.069$ and E4M3 per tensor $0.038$, and per-tensor E4M3 flushes 1.75 percent of the activation entries to exactly zero. Block scaling at $1 \times 128$ and $128 \times 128$ brings the error to $0.027$ and the flushed fraction to 0.04 percent; rounding the scales down to powers of two costs some of that back ($0.035$), and shrinking the blocks to 32 changes nothing further at this granularity. The mechanism line is the one to look at: on the entries with $|x| < 0.05$, per-tensor scaling has a relative error of 27 percent, per-block 3 percent, while the outlier channels themselves are fine either way. The last two lines show the two formats' grids on $[1, 2]$: E4M3 has eight steps, E5M2 has four and maps 1.125 to 1.0 and 1.625 to 1.5. Two things to notice against your intuition. The best FP8 setting is still ten times worse than bf16 per matmul, and training tolerates it because the rounding errors are unbiased and average out over steps, which is what the paper's 0.003-nat gap says. And if you replace the heavy-tailed `x` with plain Gaussian noise and a single outlier, per-tensor and per-block come out nearly the same, because floating point keeps its relative precision until values underflow; block scaling is a fix for underflow on wide distributions, not for the precision of the large values.

## Build it real

`recipes/pretrain_nano.py` is the Lab 02 trainer: a pre-norm decoder from `common.py` with RMSNorm, rotary positions, SwiGLU at width $8d/3$ rounded to a multiple of 8, tied embeddings, `F.scaled_dot_product_attention`, fused AdamW with $\beta_2 = 0.95$ and decay on matrices only, bf16 autocast on CUDA, `--compile`, and `--schedule cosine|wsd|constant` with `--warmup`, `--lr`, `--min-lr-ratio`, and `--cooldown-frac`. Model size is `--n-layer --d-model --n-head --seq-len`, the batch is `--batch` sequences per step with no gradient accumulation, and the data is `--dataset` (streamed rows from the Hub, capped by `--max-samples`) or `--data-dir` with the `train.bin` and `val.bin` shards from Lab 01's `curate.py`, tokenized with GPT-2 BPE. It logs `METRIC` lines with `loss`, `lr`, `grad_norm`, `tokens_per_s`, `tflops`, and `val_loss` every `--eval-every` steps, and writes one checkpoint at the end holding weights, config, tokenizer, and step (not the optimizer state). `recipes/midtrain.py` loads that checkpoint with `--ckpt` and continues on a two-domain mixture with `--mix`, `--lr`, `--warmup`, and `--cooldown-frac`; the midtrain station in the browser is the same loop on the character model.

The weekend plan is the recipe's shape at a size the card affords. Budget first. Assume the trainer sustains 84 TFLOP/s in bf16 (40 percent of the 209.5 peak; read the real value from the `tflops` field after compilation settles and rescale everything below). A 12-layer, width-768, 12-head model at $T = 1024$ with the GPT-2 vocabulary is about 124M parameters and costs $c = 6N + 6 \, n_{\text{layer}} T d \approx 8.0 \times 10^8$ FLOPs per token (Lab 02), so the card processes about $105{,}000$ tokens per second and 10B tokens take $9.5 \times 10^4$ seconds, about 27 hours. That is 80 tokens per parameter, far from the paper's 700, and a deliberate choice: the recipe's ingredients are about how tokens are ordered and how the rate ends, and 10B tokens is enough to see both. Data: 10B GPT-2 tokens of a FineWeb-Edu sample through `curate.py`, plus a math or code shard for the Phase 2 shift (Lab 01 and Lab 03 cover the sources). Phase 1: 7B tokens, `--schedule wsd --cooldown-frac 0` as the stand-in for an open-ended schedule, `--lr 6e-4 --warmup 700`, batch 32 sequences of 1,024 (32,768 tokens per step, about 214,000 steps). Phase 2: 3B tokens from that checkpoint through `midtrain.py` with the mixture shifted toward the specialist shard, linear decay to a tenth of the rate, then the CMA variant as a second Phase 2 from the same checkpoint. Compare the two endpoints on held-out per-domain loss and the small `lm-eval-harness` set from Lab 09; at 124M expect the differences to be tenths of a nat on the specialist domain, and run at least two seeds before believing a difference in the third decimal.

What the recipe does not have, listed honestly, so you know what you would be writing.

1. Resumption with a fresh schedule. `midtrain.py` loads weights and starts a new schedule, which is what Phase 2 needs, but the checkpoint carries no optimizer state, so the Adam moments restart from zero; the paper resumes the full state. You would add optimizer state to `save_checkpoint` and a `--resume` flag.
2. A power decay for Phase 1. The schedule choices are cosine, WSD, and constant; Equation 10's $\eta_{\min} + A(1 + (k - k_w)/\tau)^{-1/2}$ is a ten-line addition to `lr_at`.
3. MuonH. `optim_bench.py` (Lab 12) has Muon; neither script has the Hyperball wrapper. You would add per-matrix $R = \|W_0\|_F$ at construction, the normalize-then-project step, a `--muonh-mult` for the factor of 10, zero weight decay on those groups, and the name-based routing that keeps embeddings, norms, and the head on AdamW.
4. FP8. There is no `--fp8`; bf16 autocast is the only mixed precision. The route on a 5090 is Transformer Engine's MXFP8 recipe or `torchao`'s float8 linear with row-wise scaling, applied to the linear layers only. At 124M the GEMMs are small enough that the paper's own memory-bound warning applies, so measure `tokens_per_s` with and without before assuming a gain.
5. An untied head and grouped-query attention. `GPTConfig.tie_embeddings` exists but has no flag, and there is no `n_kv_head`; the Qwen3 shape needs both (Lab 11 has the GQA shapes).
6. Ordered data. `random_windows` samples uniformly with a generator; a curriculum needs sequential shards written in bucket order by `curate.py`, and a `--data-order sequential` reader.
7. Periodic checkpoints and an averaging script. The recipe saves once at the end; CMA needs `--ckpt-every 100` during the constant-rate tail and a script that loads $k$ checkpoints and averages `model` parameters only, which the exercise below writes.
8. Gradient accumulation and a memory-mapped loader. `--batch` is the whole step, and both loaders hold the token tensor in memory as int64, which for 10B tokens is 80 GB; you need `np.memmap` on the `uint16` shard and a `--grad-accum` for anything approaching the paper's 6.3M-token steps.

The log line to watch is `tflops`; it is the $\bar p$ of this chapter's cost formula and the number that decides whether 27 hours is 27 or 60. Then `grad_norm` in the first thousand steps, for the reasons in Lab 02, and `val_loss` on the specialist domain across the two Phase 2 variants, which is where any CMA effect will appear first.

## How it goes wrong

1. FP8 loss spikes and then diverges after thousands of clean steps. With delayed scaling the scale is computed from a previous step's maximum, so the first step after a gradient surge overflows to the format maximum or to NaN; with a single tensor-wide scale, a growing outlier channel pushes ordinary activations into subnormals and the gradient signal through those layers silently vanishes. Compute scales online from the current block, use $1 \times 128$ and $128 \times 128$ blocks, and keep the master weights and optimizer states in fp32; check the flushed-to-zero fraction of a quantized activation the way the snippet does.

2. FP8 is on and the run is no faster, or slower. The GEMMs are memory-bound: at width 768 the arithmetic per byte moved is too low for the tensor cores to be the bottleneck, and the per-block absmax and scale multiplies added to every operand are pure overhead. The paper's fix was to raise the micro-batch to the knee of the throughput curve for each GEMM shape; on a 32 GB card that knee may be beyond the memory you have at a small model, in which case bf16 is the correct choice and FP8 belongs to the 1B-plus regime.

3. MuonH or Muon applied to the embedding, the head, or a norm gain. The orthogonalized update ignores per-row scale, so token rows with rare gradients get updates as large as common ones and the embedding never settles; on the sphere, the head's norm is pinned at its random-init value and the logits cannot grow. Route by parameter name and print the groups; the paper's Table 8 puts exactly the attention and MLP matrices on MuonH and everything else on AdamW at decay 0.1.

4. Ordinary Muon with weight decay looks good early and stalls late. Its effective rate $\eta \|u\| / \|W\|$ is falling as the norms drift, so the run behaves as if the schedule had decayed long before it did; in the paper's diagnostic this cost 0.044 nats at 170M. Either log $\|W\|_F$ and $\|u\|_F$ per group and read the induced effective rate alongside the scalar one, or use MuonH so the two coincide.

5. Averaging the last checkpoints of a decayed run changes nothing, or hurts. The decay already removed the noise ball, so the checkpoints are nearly identical and the average is the endpoint plus interpolation error; UD lost 0.42 points this way. Averaging only pays over a stretch trained at a nonzero constant rate, which is why CMA resumes at a fixed rate before it saves the six checkpoints.

6. The curriculum shows a gain in the base evaluation that disappears after SFT. Two causes to separate: the late data was benchmark-adjacent (contamination that the fine-tuning washes out), or the gain was a transient of the final distribution shift. The paper's test is the right one: put both endpoints through identical SFT with several seeds and see whether the gap persists; a 13-gram overlap filter against the evaluation sets, applied to the late buckets, is the minimum contamination check.

7. Cost figures that cannot be compared. Reporting "$X per run" without the accounting boundary (production only versus everything), the rate assumption ($0.31 amortized versus a cloud rental), and the MFU convention (precision-weighted versus dense BF16) makes two papers' numbers incomparable and your own irreproducible. Report GPU-hours and measured TFLOP/s first, dollars second, with the rate stated.

8. A pipeline stage that holds the LM head runs at a fraction of the others' speed. FP8 accelerates the transformer layers but the head's vocabulary-sized GEMM, in BF16, does not shrink, so with a naive even split the stage with the head is the straggler and every other stage idles. The fix is an uneven split with fewer layers on the head's stage, (18|10) and (9|9|9|1) in the paper; on one card the same imbalance appears as the head being a larger share of step time than its parameter count suggests, which is a reason to keep the vocabulary small at small width (Lab 02, test question 10).

## Measure it

The primary number is still held-out loss with its token count. The paper's validation set is a Nemotron-CC subset; the production run ends Phase 1 at 2.730 and Phase 2 at 2.488 nats, numbers that belong to that corpus and tokenizer and cannot be compared to your TinyStories or FineWeb runs (Lab 02). What transfers is the shape: the drop across Phase 2 comes from the mixture shift plus the decay, and in the pretrain station you can watch the second of those as the cooldown kink.

For an FP8 run, the number that matters is the loss gap against a bf16 twin at matched tokens, measured at two or three sizes so you can see whether it is constant; the paper's 0.0031 to 0.0039 nats across a 10-times range of sizes is what "FP8 is safe here" looks like, and a gap that grows with size is a warning. Convert it to compute the way they do, by fitting both arms to a shared-floor, shared-exponent power law and reading the horizontal ratio; the ratio is the retention (98.0 percent for them), and it is what you multiply the throughput gain by.

For the optimizer, compare against a tuned baseline, not a default one. Their Muon baseline had its rate and decay searched per size (Table 9), and the resulting $\kappa = 1.19$ (1.17 to 1.28 leaving one size out) is a compute-equivalent multiplier from the same shared-shape fit: MuonH reaches the baseline's fitted loss with 16 percent less compute. Plot the induced effective rate for both runs; if the curves match, so will the losses, and the multiplier is telling you about the schedule, not the sphere.

For the endpoint, report the unweighted mean over a fixed benchmark list with the shots and the formulation rule stated, and evaluate every baseline yourself under the same harness; a number copied from another paper's table is a different protocol. On the paper's 15-task suite the useful anchors are 55.14 (Qwen2-1.5B), 57.81 (Puro-2B), 60.73 (Qwen2.5-1.5B), and 65.27 (Qwen3-1.7B-Base, the same architecture with a very different data budget).

For cost, the honest report is three numbers: measured GPU-hours, the median TFLOP/s per card with the peak convention that turns it into an MFU, and the rate assumption. From those anyone can recompute the dollars under their own rate. A 73 percent effective MFU on PCIe-only consumer cards is the paper's systems result; for a single-card nano trainer, 30 to 50 percent of the BF16 peak is healthy (Lab 02) and the FP8 fraction is zero unless you added it.

## Exercises

1. Recompute the two-phase GPU-hours from $6ND$ and the two throughputs, and from the world sizes and wall-clock days, and reconcile them. Check: $6{,}150 + 16{,}700 = 22{,}850$ from FLOPs, $6{,}008 + 16{,}497 = 22{,}505$ from the clock, within 2 percent of the reported 22,514; the residual is the difference between $N \approx 2 \times 10^9$ and the exact matmul parameter count.

2. Fit $P = a + b \log_2 (C - 1.84)$ (cost in thousands of dollars) to the five UD points in the text by least squares and predict the cost at which the fit crosses 55.14. Check: the slope is close to one point per doubling and the crossing is near $4.1K to $4.4K; then compute the same fit without the $C_{P1}$ shift and note the worse residuals, which is the paper's RMSE comparison.

3. Modify the snippet to quantize the weight gradient path: make `w` heavy-tailed instead of `x`, keep `x` Gaussian, and compare $128 \times 128$ blocks against $1 \times 128$ strips on `w`. Check: for a matrix whose outliers are whole columns, strips along the reduction dimension isolate them and blocks do not, which is why the paper gives activations strips and weights squares (the weight outliers are not aligned with one axis).

4. Implement MuonH in the Lab 12 toy: after `muon_step`, normalize the update, scale by $R$, and project back to radius $R$ (store $R$ per matrix at construction). Log $\eta_t \|u_t\| / \|W_t\|$ for plain Muon with decay 0.1 and confirm it equals $\eta_t$ for MuonH. Check: for plain Muon the logged effective rate drifts toward $\sqrt{2 \lambda \eta}$ over the run; for MuonH it is the schedule to floating-point precision.

5. Write `average_ckpts.py`: load $k$ `pretrain_nano` checkpoints, average the `model` tensors with equal weights, keep the last checkpoint's config and tokenizer, and save. Run `--schedule constant` for 600 extra steps from a cooled `char-tiny` checkpoint saving every 100, average the last six, and evaluate the average against the last iterate. Check: the average is lower by an amount comparable to the noise-ball drop from a cooldown at the same rate (Lab 03); repeat from a cosine-decayed run without the constant continuation and see no gain.

6. Build a two-source curriculum for the weekend plan: score TinyStories rows by a cheap proxy (length, or a small classifier from Lab 01), sort within each source, cut both into 20 rank buckets, and write bucket-interleaved sequential shards. Train UD (reshuffled) and CD (ordered) Phase 2 runs with the same linear decay. Check: the per-domain validation losses of the two runs cross during the second half, with CD lower on the domain whose best examples arrive last, and the gap at the endpoint is a few hundredths of a nat at this scale, which is why the paper needs 15 benchmarks and three SFT seeds to see it.

## Test yourself

1. A vendor quotes the 5090 at 419 TFLOP/s FP8 and you measure 238 TFLOP/s in a run that is 72 percent FP8 work by FLOPs. A colleague computes MFU as $238 / 419 = 57$ percent. What is wrong, and what is the right number?

<details><summary>Answer</summary>
The 28 percent of work that runs in BF16 is charged against the FP8 peak. Time adds, so the effective peak is the harmonic combination $(0.72 / 419 + 0.28 / 209.5)^{-1} \approx 327$ TFLOP/s and the MFU is $238 / 327 \approx 73$ percent. Both numbers are correct under their own convention; only the second is comparable to a dense-BF16 MFU after you state the convention.
</details>

2. Why does one outlier in a Gaussian activation tensor barely hurt per-tensor E4M3, while the same outlier in a heavy-tailed tensor does?

<details><summary>Answer</summary>
FP8 is floating point: every normal value gets the same 3-bit relative precision regardless of the scale. A tensor-wide scale only causes damage when it pushes values below the smallest normal, $2^{-6}$ after scaling, into subnormals or zero. Gaussian values have a narrow magnitude range and stay normal; a heavy-tailed tensor has a wide range of small values that underflow. Block scaling is a cure for underflow, not for the precision of large values.
</details>

3. State the MuonH update in words and say why the paper's aligned-Muon control matters for interpreting it.

<details><summary>Answer</summary>
Normalize the orthogonalized momentum to unit Frobenius norm, step by $\eta_t R$ in that direction, and project the result back to the sphere of radius $R = \|W_0\|_F$. The control runs ordinary Muon with its scalar rate adjusted every step so that its effective rate equals MuonH's schedule, and lands at the same loss (3.030 versus 3.029). So the gain over ordinary Muon is explained by the effective-rate schedule, not by the projection itself; the sphere is the cheapest way to make the schedule explicit.
</details>

4. Under decoupled weight decay $\lambda$ and rate $\eta$, ordinary Muon's effective rate settles near $\sqrt{2 \lambda \eta}$. What effective rate does a decay of 0.1 and a rate of $10^{-3}$ imply, and how does it compare with the paper's production Hyperball peak?

<details><summary>Answer</summary>
$\sqrt{2 \times 0.1 \times 10^{-3}} = 0.014$, versus a Hyperball peak of $10 \times 5 \times 10^{-3} = 0.05$. The equilibrium argument is a steady-state approximation with the update orthogonal to the weight; the point is that the scalar rate you typed is not the angular step you got, and that the Hyperball rate is directly the angular step.
</details>

5. Phase 2's linear schedule decays from $1.04 \times 10^{-3}$ to $10^{-5}$ over about 152,800 local steps. At local step 148,250 what is the base rate, and what is the effective rate of a MuonH matrix there?

<details><summary>Answer</summary>
$1.04 \times 10^{-3} + (10^{-5} - 1.04 \times 10^{-3}) \times 148250 / 152800 = 1.04 \times 10^{-3} - 1.03 \times 10^{-3} \times 0.970 \approx 4.1 \times 10^{-5}$, matching the reported $4.08 \times 10^{-5}$ at global step 218,000. The MuonH effective rate is ten times that, $4.1 \times 10^{-4}$, and it is what the continuation holds constant.
</details>

6. The ablation gives UD 55.99, UD averaged 55.57, CD 57.17, CD averaged 57.18, CDC-218k 57.12, CMA-218k 57.81. Which two comparisons isolate the averaging effect, and what do they say?

<details><summary>Answer</summary>
Averaging on the decayed trajectories (UD versus UD averaged, CD versus CD averaged) gives $-0.42$ and $+0.01$: no gain when the noise ball has already been removed by the decay. Averaging on the constant-rate continuation (CDC-218k versus CMA-218k) gives $+0.69$: the average removes the noise the continuation reintroduced while keeping the late data's updates at full size. Averaging is conditional on the rate being nonzero.
</details>

7. Why is a curriculum bucket built from the same normalized-rank interval of every source, rather than by a global sort on score?

<details><summary>Answer</summary>
Scores from different sources are not comparable (a 0.9 from one classifier is not a 0.9 from another), and a global sort would let one source dominate a stretch of training and destroy the mixture weights. Taking interval $k$ from every source keeps each bucket at the target mixture to within the observed 1 to 2 percent drift while the within-source quality rises monotonically.
</details>

8. Your FP8 run at 124M is 0.004 nats worse than bf16 and 5 percent slower. Is the recipe broken?

<details><summary>Answer</summary>
No. The loss gap matches the paper's ladder at every size, so the numerics are fine; the speed is the memory-bound regime the paper warns about at small hidden size, where the per-block scaling traffic exceeds the arithmetic saved. The fix is a larger micro-batch if memory allows, and otherwise the honest conclusion that FP8 does not pay at this width. The 1.36 times figure was measured at 1.7B.
</details>

9. The cost scaling law says $4.4K reaches Qwen2-1.5B. Name three things that number does not tell you.

<details><summary>Answer</summary>
It is a marginal accelerator cost at an amortized $0.31/h that excludes data processing, proxy runs, ablations, failed runs, and evaluation; it is a fit for this 2B model on this recipe, with no model-size law claimed; and "reaches" means the unweighted mean of fifteen base-model benchmarks under one harness, not a chat evaluation and not a claim about every task (Chinese is excluded, and Qwen2.5-1.5B is still 2.9 points ahead).
</details>

10. Spot the bug: `scale = FMAX / amax; scale = torch.exp2(torch.ceil(torch.log2(scale)))`.

<details><summary>Answer</summary>
Rounding the power-of-two scale up can make `amax * scale` exceed the format maximum, and E4M3 has no infinity: the largest values in the block saturate to 448 with a large relative error, or become NaN, in exactly the blocks that hold the outliers the scheme exists to protect. Round down, as the snippet does; the price is at most one bit of headroom.
</details>

## What will change, what will not

The cost arithmetic is a chain of definitions and will survive: $C = 6ND$ (with $N$ the active matmul parameters), GPU-hours as $C$ over measured throughput, dollars as hours times a stated rate, and an MFU that only means something once its peak convention is written down. The rates will move every quarter; the 5090's 2.7-times advantage in peak compute per dollar is a property of one vendor's price list in August 2026 and of an EULA that prevents renting the card, and either can change. The driver modifications that enable peer-to-peer over PCIe on consumer cards are unsupported and may stop working with a driver update.

The FP8 story has a durable core and a moving shell. The core: low-precision formats trade mantissa for exponent, tensors with heavy tails must be scaled at a granularity finer than the tensor, scales must be current rather than delayed, master weights and reductions stay in high precision, and the honest quality metric is a loss gap at matched compute converted to a retention through a shared-shape fit. The shell: E4M3 and E5M2, the 128-element block, the E8M0 power-of-two scale of MXFP8 on this generation, and the 72 percent FP8 fraction of this model. NVFP4 with 16-element blocks and a second-level scale is already on the same card, at four times the FP8 peak, and the same core reasoning will decide when it is safe.

The effective learning rate is the durable idea in the optimizer section: for a scale-invariant matrix the angular step is the only step, and any optimizer that controls it directly, whether by a sphere, by a rotational-equilibrium argument, or by a norm-aware schedule, will behave like MuonH did here. Whether the mechanism is Muon's orthogonalization or a successor is secondary, as the paper's own aligned control shows.

Curriculum model averaging joins two things that will each outlast the combination: iterate averaging as a substitute for decay, which is as old as stochastic approximation, and the observation that when data quality is ordered the schedule and the ordering must be designed together. The 376 buckets, six checkpoints, and 29B-token tail are this run's numbers.

The Puro cost scaling law is the least durable object in the chapter and the paper says so: five points, one model, one recipe. What to keep is the method, fitting capability against incremental spend from a shared checkpoint to decide a target before committing a budget.

What will not change at all is that the interesting experiments (does the pretraining curriculum survive SFT) require the whole pipeline, not the weights, which is the argument for open recipes and the reason this chapter exists.

## Read next

- "Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090", Luo, 2026. The source for this chapter; read Sections 3.2 to 3.4 and Appendices C, D, and H before running anything.
- "DeepSeek-V3 Technical Report", DeepSeek-AI, 2024. The blockwise FP8 recipe ($1 \times 128$ activations, $128 \times 128$ weights, online scales) that Puro-2B adopts, validated at 671B parameters.
- "FP8 Formats for Deep Learning", Micikevicius, 2022. Defines E4M3 and E5M2 and the reasoning behind giving up the infinity encodings for range.
- "Muon: An optimizer for hidden layers in neural networks", Jordan, 2024. The orthogonalized-momentum update inside MuonH; Lab 12 builds it.
- "Muon is Scalable for LLM Training", Liu, 2025. Adds decoupled weight decay to Muon at scale and documents the norm growth that motivates controlling the effective rate.
- "Training Compute-Optimal Large Language Models", Hoffmann, 2022. The 20-tokens-per-parameter ladder the paper's ablations use, and the shared-shape fitting form.
- "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", Hägele, 2024. Warmup-stable-decay schedules and cooldown length, the testbed for Puro-2B's decay-ratio sweeps.
- "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", Wortsman, 2022. The empirical case for equal-weight parameter averaging that CMA's six-checkpoint average relies on.
