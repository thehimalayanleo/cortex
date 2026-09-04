---
title: "Lab 11: Model architecture, with the parameter counts"
kind: permanent
topics: [lab]
chapter: 11
station: pretrain
recipe: none
reading_time: 60 min
---

## What you will be able to do

1. Write down every tensor shape in a decoder block (attention, GQA, RoPE, RMSNorm, SwiGLU) from memory and implement it in under 60 lines of PyTorch.
2. Derive why rotary embeddings make attention scores depend on relative position, and explain what happens to that property when you change the base or scale the positions.
3. Count the parameters of Llama-3 8B, nomic-embed-text-v1.5, and Mixtral 8x7B from their configs and get the published totals exactly.
4. Size a KV cache in bytes for any (heads, kv heads, head dim, layers, context, dtype) and explain the GQA and MQA trade.
5. Build a looped (recurrent-depth) transformer, say what changes in the parameter count, the FLOP count, and the KV cache, and train it with a random loop count.

## The idea in one paragraph

A transformer keeps one vector per token, the residual stream, and passes it through a stack of blocks that each read from the stream, compute something, and add the result back. Attention is the only place tokens exchange information: each token forms a query, compares it with every earlier token's key, and pulls in a weighted mix of their values. The MLP is where a token processes what it has gathered. Everything else in the architecture (norms, rotary positions, grouped key/value heads, gated activations, expert routing) is a modification of one of those two operations chosen to make training stable, inference cheap, or parameters better spent. Because every piece is a matrix multiply of known shape, you can count the parameters exactly, and the count is the first thing to check before you believe any config.

## The math

### The residual stream

Let a sequence have $T$ tokens, and let the model width be $d$. The embedding table $E \in \mathbb{R}^{V \times d}$ maps token ids to rows, giving $X_0 \in \mathbb{R}^{T \times d}$. A decoder with $L$ blocks computes

$$X_{\ell+1} = X_\ell + f_\ell(X_\ell), \qquad \ell = 0, \dots, L-1,$$

where $f_\ell$ is an attention sublayer followed by an MLP sublayer, each with its own residual add. The final state passes through a norm and the unembedding $W_U \in \mathbb{R}^{d \times V}$ to give logits. Reading the model this way (Elhage et al., 2021) tells you two things you will use repeatedly: every sublayer sees the sum of everything before it, and a sublayer can only write to the stream additively. Nothing is overwritten, which is why late blocks can undo early ones, and why the norm on the way out matters: the stream's scale grows with depth.

### Attention with the shapes written out

Fix a block and drop the subscript. Let $H$ be the number of query heads and $d_h = d / H$ the head dimension. Three projections produce queries, keys, and values:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \qquad W_Q \in \mathbb{R}^{d \times H d_h},\; W_K, W_V \in \mathbb{R}^{d \times H_{kv} d_h}.$$

For ordinary multi-head attention, $H_{kv} = H$. Reshape $Q$ to $H$ slices $Q_h \in \mathbb{R}^{T \times d_h}$, likewise $K_h, V_h$. Per head,

$$S_h = \frac{Q_h K_h^\top}{\sqrt{d_h}} \in \mathbb{R}^{T \times T}, \qquad A_h = \mathrm{softmax}_{\text{row}}(S_h + M), \qquad O_h = A_h V_h \in \mathbb{R}^{T \times d_h}.$$

$M$ is the causal mask, $M_{ij} = 0$ for $j \le i$ and $-\infty$ otherwise, so row $i$ of $A_h$ is a distribution over positions $0..i$. Concatenate the heads to $O \in \mathbb{R}^{T \times H d_h}$ and project back with $W_O \in \mathbb{R}^{H d_h \times d}$.

Why $\sqrt{d_h}$: if the entries of a query and a key are independent with zero mean and unit variance, their dot product has variance $d_h$. Without the scale, logits at initialization have standard deviation $\sqrt{d_h}$ (about 11 for $d_h = 128$), the softmax saturates, and gradients through it vanish. Dividing by $\sqrt{d_h}$ brings the logits back to unit variance.

Parameter count of the attention sublayer, no biases:

$$P_{\text{attn}} = d \cdot H d_h + 2 \cdot d \cdot H_{kv} d_h + H d_h \cdot d.$$

FLOPs per token per layer, counting a multiply-add as two FLOPs: the projections cost $2 P_{\text{attn}}$; the score and value products cost $2 \cdot T \cdot H d_h$ each, so $4 T d$ in total when $H d_h = d$. The second term is the quadratic-in-context cost. For a dense MHA layer it equals the projection cost when $4Td = 2 \cdot 4d^2$, that is $T = 2d$; for $d = 4096$ that is a context of 8192. Below that, projections dominate; above, attention does.

### Multi-query and grouped-query attention, and the cache that motivates them

At decode time you generate one token at a time. The new token's query must attend to every previous key and value, and recomputing them from the residual stream each step would cost $O(T)$ projections per token. So you cache $K$ and $V$ for every layer. Per token, per layer, in bytes:

$$\text{cache} = 2 \cdot H_{kv} \cdot d_h \cdot b,$$

with $b$ the bytes per element (2 for bf16). The factor 2 is for $K$ and $V$. Multiply by $L$ layers and by context length.

Llama-3 8B: $H_{kv} = 8$, $d_h = 128$, $L = 32$, bf16. Per token: $2 \cdot 8 \cdot 128 \cdot 2 = 4096$ bytes per layer, times 32 layers is 128 KiB. At 8192 tokens of context that is exactly 1 GiB; at 128k context it is 16 GiB. If the same model used full multi-head attention with $H_{kv} = 32$, multiply every number by 4: 4 GiB at 8k. Batch size multiplies again. The cache, not the weights, is what limits how many concurrent sequences you can serve.

Multi-query attention (Shazeer, 2019) sets $H_{kv} = 1$: all query heads share one key head and one value head. Grouped-query attention (Ainslie et al., 2023) sets $1 < H_{kv} < H$ and assigns query head $h$ to key/value head $\lfloor h / (H / H_{kv}) \rfloor$. In code, you produce $H_{kv}$ key heads and expand each with `repeat_interleave` by the group size $H / H_{kv}$ so that consecutive query heads share a group. The order matters when loading a checkpoint: `repeat` instead of `repeat_interleave` silently pairs the wrong heads and the model still runs.

The quality argument is empirical: the GQA paper reports that a handful of groups recovers most of the MHA quality at a fraction of the cache, and that an MHA checkpoint can be converted by mean-pooling its key and value heads within each group and briefly continuing training.

### Rotary position embedding, derived

Attention as written has no notion of position: permuting the tokens permutes the rows of $A$ but changes nothing else. The original transformer added a position vector to $X_0$. RoPE (Su et al., 2021) instead rotates queries and keys by an angle proportional to their position, and the point is what happens to the dot product.

Start in two dimensions. Write a query at position $m$ as $q \in \mathbb{R}^2$ and rotate it by angle $m\theta$:

$$R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}, \qquad \tilde q_m = R(m\theta) q, \quad \tilde k_n = R(n\theta) k.$$

Rotation matrices satisfy $R(\alpha)^\top = R(-\alpha)$ and $R(\alpha) R(\beta) = R(\alpha + \beta)$. So

$$\tilde q_m^\top \tilde k_n = q^\top R(m\theta)^\top R(n\theta) k = q^\top R((n - m)\theta)\, k.$$

The score depends on the content $q, k$ and on the offset $n - m$, never on $m$ or $n$ separately. That is the whole property. A token at position 1000 attending to position 997 computes the same score as a token at 10 attending to 7, given the same content.

For a head of dimension $d_h$, split the coordinates into $d_h / 2$ pairs and give pair $i$ its own frequency

$$\theta_i = \beta^{-2i / d_h}, \qquad i = 0, \dots, d_h/2 - 1,$$

with base $\beta$ (10000 in the original, 500000 in Llama 3). Pair 0 rotates fastest (one radian per position); the last pair rotates with wavelength about $2\pi\beta$ positions. Writing pair $i$ as a complex number $q_i = q_{2i} + \mathrm{i}\, q_{2i+1}$, the rotation is multiplication by $e^{\mathrm{i} m \theta_i}$, and the full score is

$$\tilde q_m^\top \tilde k_n = \mathrm{Re}\left[\sum_{i=0}^{d_h/2 - 1} q_i \bar k_i\, e^{\mathrm{i}(m - n)\theta_i}\right].$$

Two consequences you should be able to reason about. First, because the offset enters only through the phases, the model has no direct way to know absolute position; the causal mask leaks it weakly (the first token has nothing to attend to). Second, if you want a longer context than you trained on, you can either scale positions ($m \to m \cdot L_{\text{train}} / L_{\text{new}}$, position interpolation) or raise the base $\beta$ so the low-frequency pairs never see angles they did not see in training. nomic-embed-text-v1.5 trains at 2048 tokens and serves 8192 by scaling the rotary frequencies, which is why the embedding model in the encoder station has no learned position table to run out of.

RoPE is applied to $Q$ and $K$ only, never to $V$, and after the per-head reshape. Applying it before the reshape rotates the wrong pairs, and the model still trains, which is the dangerous part (see How it goes wrong).

### RMSNorm versus LayerNorm, pre-norm versus post-norm

LayerNorm on a vector $x \in \mathbb{R}^d$:

$$\mu = \frac{1}{d}\sum_j x_j, \quad \sigma^2 = \frac{1}{d}\sum_j (x_j - \mu)^2, \quad \mathrm{LN}(x) = g \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b,$$

with learned gain $g$ and bias $b$, $2d$ parameters. RMSNorm (Zhang and Sennrich, 2019) drops the centering and the bias:

$$\mathrm{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}},$$

$d$ parameters, one reduction instead of two. The empirical claim is that the re-centering was not doing useful work; the scale invariance is what stabilizes training. Every Llama-family model uses RMSNorm. BERT and nomic-bert use LayerNorm.

Where the norm sits changes the gradient path. Post-norm (the 2017 transformer, BERT):

$$X_{\ell+1} = \mathrm{Norm}(X_\ell + f_\ell(X_\ell)).$$

Pre-norm (GPT-2 onward, Llama):

$$X_{\ell+1} = X_\ell + f_\ell(\mathrm{Norm}(X_\ell)).$$

In pre-norm the residual stream itself is never normalized inside the stack, so there is an identity path from the loss to the embeddings: $\partial X_L / \partial X_0 = I + (\text{terms})$. In post-norm every block's output passes through a norm whose Jacobian rescales the gradient, and Xiong et al. (2020) show that at initialization the expected gradient norm of the last layers is large relative to the first, which is why post-norm needs learning-rate warmup and pre-norm tolerates skipping it. The price of pre-norm is that the stream's norm grows with depth (each block adds to an un-normalized sum), so late blocks contribute relatively less; the final norm before the head absorbs the growth.

### SwiGLU and the parameter-parity choice

The 2017 MLP is

$$\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1) W_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\; W_2 \in \mathbb{R}^{d_{ff} \times d},$$

with $2 d\, d_{ff}$ parameters and the convention $d_{ff} = 4d$, so $8d^2$ per layer. A gated linear unit (Shazeer, 2020) replaces the single hidden projection by two, one of which passes through a nonlinearity and gates the other:

$$\mathrm{SwiGLU}(x) = \left(\mathrm{SiLU}(x W_g) \odot x W_u\right) W_d, \qquad \mathrm{SiLU}(z) = z\, \sigma(z),$$

with $W_g, W_u \in \mathbb{R}^{d \times d_{ff}}$ and $W_d \in \mathbb{R}^{d_{ff} \times d}$, so $3 d\, d_{ff}$ parameters. To compare against the ReLU MLP at equal parameters you solve $3 d\, d_{ff} = 8 d^2$, giving $d_{ff} = 8d/3 \approx 2.67d$. Llama-2 7B took this literally: $8 \cdot 4096 / 3 = 10922.7$, rounded up to a multiple of 256, gives 11008. Llama-3 8B chose $d_{ff} = 14336 = 3.5d$ instead, spending more on the MLP than parity. nomic-embed-text-v1.5 kept BERT's $d_{ff} = 3072 = 4d$ and switched to SwiGLU, so its MLP has $12d^2$ parameters against BERT-base's $8d^2$, which is where most of its extra 27M parameters come from (worked out below).

The gate is why it helps: the product lets one projection decide which hidden units are relevant to this token while the other carries the content, and the multiplicative interaction is something a ReLU MLP has to approximate with more width.

### Mixture of experts

Replace the one MLP in a block by $E$ MLPs (experts) and a router. The router is a linear map $W_r \in \mathbb{R}^{d \times E}$:

$$r = \mathrm{softmax}(x W_r) \in \mathbb{R}^E, \qquad \mathcal{T} = \mathrm{TopK}(r, k), \qquad y = \sum_{e \in \mathcal{T}} \frac{r_e}{\sum_{e' \in \mathcal{T}} r_{e'}} \, \mathrm{FFN}_e(x).$$

Mixtral uses $E = 8$, $k = 2$, and renormalizes over the selected two. Each token runs $k$ experts, so the per-token FLOPs are those of $k$ MLPs plus a tiny router, while the parameter count holds all $E$. That is the active versus total distinction: FLOPs, and therefore training and inference compute, scale with active parameters; memory, and the number of things the model can know, scale with total.

Left alone, a router collapses: whichever experts happen to be slightly better early get more tokens, learn faster, and win everything. The Switch Transformer fix (Fedus et al., 2021) is an auxiliary loss. Let $f_e$ be the fraction of tokens in the batch routed to expert $e$ and $P_e$ the mean router probability assigned to $e$. Then

$$\mathcal{L}_{\text{aux}} = \alpha\, E \sum_{e=1}^{E} f_e P_e.$$

If routing is uniform, $f_e = P_e = 1/E$ and the sum is $E \cdot E \cdot (1/E^2) = 1$, so $\mathcal{L}_{\text{aux}} = \alpha$; any imbalance raises it. $f_e$ is not differentiable (it counts), $P_e$ is, and the product pushes probability away from overloaded experts. Typical $\alpha$ is around $10^{-2}$; too large and the router ignores content.

Expert parallelism places different experts on different devices. Each MoE layer then needs an all-to-all to send each token's activation to the device that holds its expert, and another to bring the outputs back. Imbalance becomes a straggler problem: the layer finishes when the busiest device does. Implementations bound this with a capacity factor, the maximum tokens an expert accepts per batch, and drop or reroute the overflow.

### Worked parameter counts

Llama-3 8B config: $V = 128256$, $d = 4096$, $L = 32$, $H = 32$, $H_{kv} = 8$, $d_h = 128$, $d_{ff} = 14336$, RMSNorm, SwiGLU, no biases, untied embeddings.

Embedding: $128256 \times 4096 = 525{,}336{,}576$. Unembedding: the same, $525{,}336{,}576$.

Per layer, attention: $W_Q$ is $4096 \times 4096 = 16{,}777{,}216$; $W_K$ and $W_V$ are each $4096 \times 1024 = 4{,}194{,}304$; $W_O$ is $16{,}777{,}216$. Sum $41{,}943{,}040$.

Per layer, MLP: three matrices of $4096 \times 14336 = 58{,}720{,}256$ each, sum $176{,}160{,}768$.

Per layer, norms: two RMSNorms, $2 \times 4096 = 8{,}192$.

Per layer total: $41{,}943{,}040 + 176{,}160{,}768 + 8{,}192 = 218{,}112{,}000$. Times 32: $6{,}979{,}584{,}000$. Final norm: $4{,}096$.

Total: $525{,}336{,}576 \times 2 + 6{,}979{,}584{,}000 + 4{,}096 = 8{,}030{,}261{,}248$. That is the published 8.03B. Notice the split: 13 percent of the parameters are the two vocabulary matrices, 81 percent are MLPs, and only 5 percent are attention. GQA is a big part of why attention is so small; with $H_{kv} = 32$ each layer would carry 25M more.

nomic-embed-text-v1.5 config (the nomic-bert-2048 backbone): $d = 768$, $L = 12$, $H = 12$, $d_h = 64$, $d_{ff} = 3072$, vocabulary 30528 (BERT's 30522 padded to a multiple of 64), token-type table of size 2, no position table (RoPE), no biases on the linear layers, LayerNorm with gain and bias, post-norm.

Embeddings: $30528 \times 768 = 23{,}445{,}504$, plus token types $2 \times 768 = 1{,}536$, plus the embedding LayerNorm $1{,}536$.

Per layer, attention: fused $W_{QKV}$ is $768 \times 2304 = 1{,}769{,}472$, $W_O$ is $768 \times 768 = 589{,}824$, sum $2{,}359{,}296$.

Per layer, SwiGLU MLP: three matrices of $768 \times 3072 = 2{,}359{,}296$, sum $7{,}077{,}888$.

Per layer, two LayerNorms: $2 \times 1{,}536 = 3{,}072$.

Per layer total: $9{,}440{,}256$. Times 12: $113{,}283{,}072$. Add embeddings: $136{,}731{,}648$. Add the output-projection biases if the checkpoint carries them (a few thousand at most) and you are at the published 137M. As a check on the reasoning, BERT-base is 110M; nomic added $4d^2 = 2.36$M per layer of MLP (28.3M over 12 layers) and removed the $512 \times 768 = 0.39$M position table and the linear biases, and $110 + 28.3 - 0.4 - 0.1 \approx 137.8$M, consistent with the total within the rounding of the "110M" figure.

Mixtral 8x7B config: $V = 32000$, $d = 4096$, $L = 32$, $H = 32$, $H_{kv} = 8$, $d_h = 128$, $d_{ff} = 14336$, $E = 8$, $k = 2$.

Attention per layer is the same as Llama-3 8B: $41{,}943{,}040$. One expert is one Llama MLP: $176{,}160{,}768$; eight experts $1{,}409{,}286{,}144$. Router: $4096 \times 8 = 32{,}768$. Norms $8{,}192$. Per layer: $1{,}451{,}270{,}144$. Times 32: $46{,}440{,}644{,}608$. Embeddings and head: $2 \times 32000 \times 4096 = 262{,}144{,}000$, plus final norm $4{,}096$. Total $46{,}702{,}792{,}704$, the published 46.7B.

Active per token: attention plus two experts plus router plus norms, $41{,}943{,}040 + 352{,}321{,}536 + 32{,}768 + 8{,}192 = 394{,}305{,}536$ per layer; times 32 plus the embedding side gives $12{,}879{,}925{,}248$, the published 12.9B. So Mixtral costs about the FLOPs of a 13B dense model per token and the memory of a 47B one.

### What changes between an encoder, a decoder, and an MoE decoder

| Aspect | Encoder (BERT, nomic-embed) | Dense decoder (Llama 3) | MoE decoder (Mixtral) |
|---|---|---|---|
| Attention mask | none, every token sees every token | causal | causal |
| Training objective | masked-token prediction, then contrastive for embeddings | next-token prediction | next-token prediction plus routing auxiliary loss |
| Positions | learned table (BERT) or RoPE (nomic) | RoPE | RoPE |
| Norm | LayerNorm, post-norm | RMSNorm, pre-norm | RMSNorm, pre-norm |
| MLP | ReLU/GELU, or SwiGLU at $4d$ (nomic) | SwiGLU at $3.5d$ | 8 SwiGLU experts, 2 active |
| Output | pooled vector (mean or CLS) or per-token head | logits over $V$ at every position | logits over $V$ at every position |
| KV cache | irrelevant, one pass | central to serving cost | same as dense |
| Parameters active per token | all | all | attention plus $k$ of $E$ experts |
| Key parameter | context length, pooling, embedding dim | $H_{kv}$, $d_{ff}$, vocabulary | $E$, $k$, capacity factor |

The encoder station in the browser (MLM then InfoNCE) and the pretrain station (next-token) differ in exactly the first two rows; the same 2-layer, width-48, 3-head block serves both once you change the mask and the loss.

## Looped (recurrent-depth) transformers

A standard decoder has $L$ distinct blocks. A looped transformer keeps one block (or a small group) and applies it $T$ times to the same stream. Parameters stay at one block's worth; FLOPs scale linearly with $T$. That is the entire trade: you buy depth with compute instead of memory.

Formally, with a weight-tied block $f$ and an input-injection map $\phi$,

$$s_0 = \text{init}, \qquad s_{t+1} = f\big(\phi(s_t, e)\big), \quad t = 0, \dots, T - 1,$$

where $e$ is the embedded input (possibly after a non-looped prelude block) and $\phi$ combines the state with the input, for example $\phi(s, e) = [s; e] W_{\text{in}}$ or simply $s + e$. Re-injecting $e$ at every iteration matters: without it, the loop can drift away from the input, and the model has no way to look back at what the question was once $s$ has been overwritten. The Universal Transformer (Dehghani et al., 2018) is the early instance; the recurrent-depth model of Geiping et al. (2025) is a 3.5B-parameter version trained at scale with a prelude, a looped core, and a coda.

Why looping helps on algorithmic and reasoning tasks: many such tasks are naturally iterative (propagate a carry, follow a pointer, apply a rule until nothing changes), and a fixed-depth model has to learn a separate implementation of each iteration in each layer. A looped model learns the iteration once and runs it as many times as the input needs. The fixed-point view makes this precise: if $f \circ \phi(\cdot, e)$ is a contraction in $s$, iterating converges to a fixed point $s^\star(e)$, and a deep equilibrium model (Bai et al., 2019) solves for $s^\star$ directly and differentiates through the fixed-point condition with the implicit function theorem. A looped transformer is the unrolled, truncated version of the same object, and in practice you watch $\|s_{t+1} - s_t\| / \|s_t\|$ shrink with $t$ as a diagnostic of whether the model has learned something like a contraction.

Adaptive depth and halting: since different inputs need different numbers of iterations, you can let the model choose $T$ per token. Adaptive Computation Time (Graves, 2016) adds a halting unit whose sigmoid outputs accumulate until they cross a threshold, with a ponder-cost penalty so the model does not loop forever; the Universal Transformer applied it per position. A cheaper alternative used in recurrent-depth models is to fix $T$ at inference and pick it by validation, or to stop when the relative change in $s$ drops below a tolerance, which needs no extra parameters.

Training tricks. If you always train at the same $T$, the model overfits that depth and degrades when you change it. Sample $T$ per step from a distribution with a long tail (a random or curriculum schedule), so the block learns to be a reusable step. Backpropagating through many loops is expensive in memory (activations for every iteration) and unstable; truncated backpropagation keeps gradients only through the last $k$ iterations and runs the earlier ones under `no_grad`, on the argument that the early iterations only set up the state. Initialize the state randomly (or to zero) rather than to the embedding, so the injection path is the only path by which the input enters and the model is forced to use it.

The KV cache changes in a way that surprises people. Each loop iteration is a fresh attention call on a different state, so it produces its own keys and values; a 1-block model looped $T$ times caches as much as a $T$-block model: $2 \cdot T \cdot H_{kv} \cdot d_h \cdot b$ bytes per token. Parameters shrink by $T$, the cache does not. With adaptive depth the cache size per token becomes variable, which complicates paged allocation. Two mitigations exist: cap the iterations whose keys are stored and let later iterations attend to the most recent cached set, or share keys and values across iterations by computing them once from the injected input and rotating only the query, which trades some expressivity for a constant cache.

A small model, looped, with per-loop injection:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):                        # one pre-norm transformer block
    def __init__(self, d, H, dff):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, H, bias=False, batch_first=True)
        self.wg, self.wu, self.wd = nn.Linear(d, dff, bias=False), nn.Linear(d, dff, bias=False), nn.Linear(dff, d, bias=False)
    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.n2(x)
        return x + self.wd(F.silu(self.wg(h)) * self.wu(h))

class LoopedLM(nn.Module):
    def __init__(self, V, d, H, dff):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.prelude = Block(d, H, dff)         # runs once
        self.core = Block(d, H, dff)            # weight-tied, runs T times
        self.inject = nn.Linear(2 * d, d, bias=False)   # input injection: [state, embedding] -> state
        self.coda = Block(d, H, dff)            # runs once
        self.norm, self.head = nn.RMSNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, idx, T, trace=False):
        B, L = idx.shape
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)   # True = masked (causal)
        e = self.prelude(self.emb(idx), mask)   # e is re-injected at every loop
        s = torch.randn_like(e) * 0.02          # random initial latent state
        deltas = []
        for t in range(T):
            s_next = self.core(self.inject(torch.cat([s, e], -1)), mask)
            deltas.append((s_next - s).norm() / s.norm())
            s = s_next
        if trace: print("relative change per loop:", [round(x.item(), 3) for x in deltas])
        return self.head(self.norm(self.coda(s, mask)))

if __name__ == "__main__":
    torch.manual_seed(0)
    m = LoopedLM(V=256, d=64, H=4, dff=176)
    idx = torch.randint(0, 256, (2, 16))
    print("params:", sum(p.numel() for p in m.parameters()))   # fixed, independent of T
    for T in (1, 4, 16):
        print(T, tuple(m(idx, T).shape))        # same shape for every T
    m(idx, 8, trace=True)
    T = int(torch.randint(4, 12, ()))           # random T per training step
    loss = F.cross_entropy(m(idx, T).flatten(0, 1), idx.flatten())
    loss.backward(); print("loss", round(loss.item(), 3), "T", T)
```

Expected output: `params: 191936`, then `1 (2, 16, 256)`, `4 (2, 16, 256)`, `16 (2, 16, 256)`, a list of eight relative changes, and a loss near $\ln 256 \approx 5.55$ (5.709 with this seed). The parameter count is three blocks of 50304, the injection's 8192, two vocabulary matrices of 16384, and a final norm, and it does not move when $T$ does. The relative changes fall quickly even before training (the first is large because the state starts near zero, then the residual block at initialization is close to the identity, so the iteration is nearly a contraction); the diagnostic becomes informative once training could have made the block expansive, which is when you want to see it still contract. To add truncated backprop, wrap the first $T - k$ iterations of the loop in `torch.no_grad()` and detach `s` before the last $k$.

## Build it small

The block below is the dense decoder block of the Llama family: pre-norm, RMSNorm, GQA with `repeat_interleave`, RoPE on $Q$ and $K$ after the head reshape, SwiGLU. The `count` function reproduces the Llama-3 8B total, and the last line checks the relative-position property numerically: the same content at offsets (1, 3) and (2, 4) gives the same score.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def rope(x, base=10000.0):                      # x: (B, H, T, dh), dh even
    B, H, T, dh = x.shape
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)   # (dh/2,)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]  # (T, dh/2)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class Block(nn.Module):                         # pre-norm, GQA, RoPE, SwiGLU
    def __init__(self, d, H, Hkv, dff):
        super().__init__()
        self.H, self.Hkv, self.dh = H, Hkv, d // H
        self.wq = nn.Linear(d, H * self.dh, bias=False)
        self.wk = nn.Linear(d, Hkv * self.dh, bias=False)
        self.wv = nn.Linear(d, Hkv * self.dh, bias=False)
        self.wo = nn.Linear(H * self.dh, d, bias=False)
        self.wg = nn.Linear(d, dff, bias=False)
        self.wu = nn.Linear(d, dff, bias=False)
        self.wd = nn.Linear(dff, d, bias=False)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
    def attn(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.H, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        q, k = rope(q), rope(k)
        rep = self.H // self.Hkv                # each kv head serves rep query heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        return x + self.wd(F.silu(self.wg(h)) * self.wu(h))

def count(V, d, L, H, Hkv, dff, tied=False):   # dense decoder, RMSNorm, SwiGLU
    dh = d // H
    attn = d * H * dh + 2 * d * Hkv * dh + H * dh * d
    per_layer = attn + 3 * d * dff + 2 * d
    return V * d * (1 if tied else 2) + L * per_layer + d

if __name__ == "__main__":
    torch.manual_seed(0)
    blk = Block(d=64, H=4, Hkv=2, dff=160)
    print(blk(torch.randn(2, 10, 64)).shape)                 # torch.Size([2, 10, 64])
    print(sum(p.numel() for p in blk.parameters()))          # 43136
    print(count(128256, 4096, 32, 32, 8, 14336))             # 8030261248 (Llama-3 8B)
    q0, k0 = torch.randn(8), torch.randn(8)                  # RoPE relative-position check
    Q = rope(q0.expand(1, 1, 6, 8).clone()); K = rope(k0.expand(1, 1, 6, 8).clone())
    print(torch.allclose(Q[0, 0, 1] @ K[0, 0, 3], Q[0, 0, 2] @ K[0, 0, 4], atol=1e-5))  # True
```

Expected output, in order: `torch.Size([2, 10, 64])`, `43136`, `8030261248`, `True`. The small block's 43136 decomposes as attention $4096 + 2048 + 2048 + 4096 = 12288$, MLP $3 \times 64 \times 160 = 30720$, norms 128. Try `count(128256, 4096, 32, 32, 32, 14336)` to see what MHA would have cost, and `count(32000, 4096, 32, 32, 32, 11008)` for Llama-2 7B (6{,}738{,}415{,}616, the published 6.74B).

## Build it real

This chapter has no recipe of its own; the architecture is the thing every other recipe instantiates. What you do on the 5090 here is size and verify, so that Lab 12's `recipes/optim_bench.py` and the post-training recipes (see Lab 04) start from a config you understand.

In the browser, the pretrain station trains the 2-layer, width-48, 3-head character model. Everything in this chapter is present there at toy scale: $d = 48$, $H = 3$, $d_h = 16$, and you can read the parameter count off the station and reproduce it with `count` after adjusting for whatever norm and MLP variant the station uses.

On the 5090, build configs with `transformers` and check three numbers before training anything. First, the parameter count: `LlamaConfig(vocab_size=..., hidden_size=..., intermediate_size=..., num_hidden_layers=..., num_attention_heads=..., num_key_value_heads=...)`, then `LlamaForCausalLM(config)` and `sum(p.numel() for p in model.parameters())`, and compare with `count`. If they disagree, one of you has a bias term or a tied embedding the other does not. Second, training memory. With AdamW in mixed precision you hold bf16 weights (2 bytes), fp32 master weights (4), fp32 gradients or bf16 gradients (4 or 2), and two fp32 moments (8): about 16 to 18 bytes per parameter before activations. On 32 GB that caps from-scratch training at roughly 1.5B parameters with almost no room for activations; in practice you pretrain at 124M to 350M on this card and use LoRA (see Lab 04) above that. Third, activation memory, which scales with batch tokens times $d$ times layers; use `torch.cuda.max_memory_allocated()` after one step at your intended batch and back off if it exceeds about 28 GB, leaving headroom for fragmentation.

For time, use the standard estimate: training FLOPs $\approx 6 N D$ for $N$ non-embedding parameters and $D$ tokens (forward is $2ND$, backward twice that). Suppose you train a 124M-parameter model on 2B tokens: $6 \times 1.24 \times 10^8 \times 2 \times 10^9 = 1.5 \times 10^{18}$ FLOPs. If you assume the card sustains $10^{14}$ FLOP/s in bf16 at 40 percent utilization (an assumption to replace with a measurement from Lab 12's benchmark), that is $1.5 \times 10^{18} / (0.4 \times 10^{14}) = 3.7 \times 10^4$ seconds, about ten hours. The same formula tells you why MoE is attractive: swap $N$ for the active count.

## How it goes wrong

RoPE applied before the head reshape. Symptom: the model trains and its loss falls, but it is weak at anything that depends on precise position (copying, counting, long-range retrieval), and `torch.allclose` checks like the one above fail. Cause: the rotation pairs coordinate $2i$ with $2i+1$ of the flattened $H d_h$ vector, mixing heads. Fix: reshape to $(B, H, T, d_h)$ first, rotate, and make sure $Q$ and $K$ use the same pairing convention (interleaved pairs versus first-half/second-half; the Hugging Face Llama implementation uses halves, the snippet above uses interleaved, and a checkpoint trained with one will not load into the other without permuting $W_Q$ and $W_K$).

Wrong GQA expansion. Symptom: a converted checkpoint produces fluent but wrong text, or perplexity is 10 times worse than reported. Cause: `k.repeat(1, rep, 1, 1)` cycles heads (0,1,2,...,0,1,2,...) while training used `repeat_interleave` (0,0,0,...,1,1,1,...), so query heads attend with the wrong keys. Fix: match the convention of the checkpoint and test by loading and scoring a known sentence.

Missing or doubled $1/\sqrt{d_h}$. Symptom: attention entropy near zero from step one (missing scale) or attention that never sharpens (scaled twice, once by you and once inside `scaled_dot_product_attention`). Fix: pass `scale=` explicitly or not at all, never both.

Post-norm without warmup. Symptom: loss goes to NaN within the first few hundred steps at a learning rate that pre-norm models handle. Cause: the gradient imbalance across depth described above. Fix: linear warmup over a few thousand steps, or switch to pre-norm (see Lab 12 for the schedule).

Norm computed in bf16. Symptom: intermittent NaNs or a slow quality gap that closes when you run in fp32. Cause: $\sum x_j^2$ over $d = 4096$ elements overflows or loses precision in bf16, and $\epsilon = 10^{-5}$ is below bf16 resolution near 1. Fix: upcast to fp32 inside the norm and cast back, which is what every production implementation does.

Router collapse in MoE. Symptom: the auxiliary loss climbs, a histogram of tokens per expert shows two experts taking most of the traffic, and the model behaves like a dense model with 2 of 8 experts' worth of capacity. Cause: $\alpha$ too small, or the router initialized with large weights so early softmaxes are peaked. Fix: raise $\alpha$, initialize $W_r$ near zero, and log per-expert load every step from the start.

KV cache out of memory at serving time. Symptom: the model loads fine, then a batch of long prompts fails. Cause: the cache formula above was never multiplied by batch size and context. Fix: compute $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b \cdot T_{\text{ctx}} \cdot B$ before choosing a batch, and prefer GQA models or quantized caches when serving long contexts on 32 GB.

Padding on the wrong side for generation. Symptom: batched generation gives different outputs than single-sequence generation for the same prompt. Cause: right padding puts pad tokens between the prompt and the generated token, and with RoPE the generated token's position is then wrong. Fix: left-pad decoder inputs and pass the attention mask so positions are computed from real tokens.

## Measure it

Parameter count: your `count` must match `sum(p.numel())` exactly. Not approximately; a mismatch means a bias or tie you did not account for.

FLOPs per token and throughput: compute $6N$ FLOPs per training token, measure tokens per second, and report model FLOP utilization, $\mathrm{MFU} = 6 N \cdot (\text{tokens/s}) / P_{\text{peak}}$, with $P_{\text{peak}}$ the card's bf16 tensor-core peak from the spec sheet. There is no fixed good number; the useful comparison is across your own configs. A small model at short context is usually bound by memory bandwidth and kernel launch overhead, not by FLOPs, so MFU rises with $d$ and with batch tokens until you run out of memory. If it does not rise, the bottleneck is in the data loader or in a per-step host sync.

KV cache: bytes per token from the formula, and the maximum batch size at your target context that fits after weights. Verify with `torch.cuda.max_memory_allocated()` during a generation of the target length.

Architecture comparisons at matched compute: train MHA against GQA with $H_{kv} = H / 4$ at the same $N$ and $D$ and compare validation loss; the GQA paper's claim is that the gap is small. Do the same for parity SwiGLU against $4d$ ReLU. Report the loss difference with a seed-to-seed spread (three seeds each) so a 0.01 nat gap is not mistaken for a result.

MoE health: entropy of the per-expert load distribution, ideally near $\ln E$; fraction of tokens dropped by the capacity limit; and validation loss against the dense model with the same active count, which is the fair baseline.

Looped models: validation loss as a function of inference $T$, which should improve then plateau; the plateau's location tells you the effective depth the task needs.

## Exercises

1. Compute the KV cache per token for Mixtral 8x7B in bf16, and the total for a batch of 16 sequences at 32k context. Check: per token $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131072$ bytes, so $16 \times 32768 \times 128\,\text{KiB} = 64$ GiB, which does not fit on any single consumer card; this is why GQA alone is not enough at long context and cache quantization exists.

2. Take Llama-3 8B and replace SwiGLU with a parity ReLU MLP ($d_{ff} = 4d$, two matrices). By how much does the parameter count change? Check: MLP per layer goes from $3 \cdot 4096 \cdot 14336 = 176.2$M to $2 \cdot 4096 \cdot 16384 = 134.2$M, a drop of 42.0M per layer and 1.34B in total, to about 6.69B.

3. Show that RoPE scores are invariant to a common shift of all positions but not to reversing the sequence. Check: shifting both $m$ and $n$ by $c$ leaves $m - n$ unchanged; reversing maps $m - n$ to $n - m$, and $\mathrm{Re}[z e^{\mathrm{i}\phi}] \ne \mathrm{Re}[z e^{-\mathrm{i}\phi}]$ in general unless $z$ is real.

4. Convert an MHA checkpoint to GQA with 4 groups by mean-pooling key and value heads. Write the shapes of the pooling: $W_K \in \mathbb{R}^{d \times 32 \cdot 128}$ reshaped to $(d, 4, 8, 128)$, averaged over the axis of size 8, giving $(d, 4 \cdot 128)$. Then explain why the model needs continued training afterward. Answer: the queries were trained against 32 distinct key heads and now see averages; the average is the least-squares best single key but not what any query expected.

5. Derive the load-balancing loss gradient with respect to a router logit for one token and show it pushes probability away from experts with high $f_e$. Check: $\partial \mathcal{L}_{\text{aux}} / \partial P_e = \alpha E f_e$ (treating $f_e$ as constant), so the gradient on the softmax input is largest for the most-loaded expert and the update lowers its logit.

6. For the looped model in the snippet, write the FLOP count per token as a function of $T$ for the core block, using $d = 64$, $d_{ff} = 176$, ignoring attention's quadratic term. Check: core block has $4 d^2 + 3 d\, d_{ff} = 16384 + 33792 = 50176$ weights plus the injection's $2 d^2 = 8192$, so about $2 \times 58368 \times T \approx 117\text{k}\, T$ FLOPs per token, against a fixed cost for the prelude, coda, embedding, and head.

## Test yourself

1. A colleague says "GQA with 8 kv heads cuts attention FLOPs by 4x." Is that right?

<details><summary>Answer</summary>
No. GQA cuts the key and value projection FLOPs and the cache by $H / H_{kv}$, but the score computation $Q K^\top$ and the value mixing $A V$ still run once per query head, because each query head attends with its own query against the (shared) keys. In Llama-3 8B the attention projections fall from $4d^2$ to $2.5 d^2$ per layer, a 37 percent cut in the projection term, and the quadratic term is unchanged. The real win is the cache and the memory traffic at decode time, not FLOPs.
</details>

2. In the `rope` function above, what breaks if `dh` is odd?

<details><summary>Answer</summary>
`x[..., 0::2]` and `x[..., 1::2]` have different lengths, the elementwise products fail with a shape error at best; if `dh` were padded to make it run, the last coordinate would be rotated against a coordinate from the next head. Head dimensions are even by construction, and most implementations assert it.
</details>

3. Estimate, with stated assumptions, the largest dense Llama-style model you can pretrain from scratch on one 32 GB card with AdamW, bf16 activations, and a 4096-token batch, without activation checkpointing.

<details><summary>Answer</summary>
Assume 16 bytes per parameter for weights, master weights, gradients, and moments, and reserve 10 GB for activations and workspace. Then $22\,\text{GB} / 16 \approx 1.4$B parameters is the ceiling on states alone. With activations at 4096 tokens, a 1.4B model with $d = 2048$ and 24 layers stores per layer roughly the residual stream plus the $d_{ff}$ intermediates in bf16, a few tens of MB per layer per thousand tokens, so a few GB in total, which fits. The honest answer is around 1B without checkpointing, and you would rather train 350M for longer given the $6ND$ budget.
</details>

4. Why does the final RMSNorm before the unembedding matter more in a pre-norm model than in a post-norm one?

<details><summary>Answer</summary>
In pre-norm the residual stream is never normalized inside the stack, so its norm grows with depth (each block adds a term of order its output scale). Without a final norm the logits would scale with depth and with training time, and the softmax temperature would drift. In post-norm every block output is already normalized, so the final norm is nearly redundant.
</details>

5. Spot the bug:

```python
k = self.wk(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
k = k.repeat(1, self.H // self.Hkv, 1, 1)
```

<details><summary>Answer</summary>
`repeat` tiles the head axis as 0,1,...,Hkv-1,0,1,... so query head 1 is paired with kv head 1 instead of kv head 0. The model trains fine from scratch (the assignment is arbitrary if consistent) but loads any GQA checkpoint incorrectly. Use `repeat_interleave(self.H // self.Hkv, dim=1)`.
</details>

6. Mixtral has 46.7B parameters and 12.9B active. A dense 13B model and Mixtral cost about the same FLOPs per token. Why does Mixtral generally serve fewer tokens per second per GPU than the dense 13B in a memory-bound decode regime?

<details><summary>Answer</summary>
Decode at small batch is bound by weight memory traffic, not FLOPs. Every step must read the weights of every expert that any token in the batch selects; at batch 1 that is 2 of 8 experts per layer, but at batch 32 nearly all 8 experts are touched per layer, so the bytes read approach the full 47B parameters, about 3.6 times the dense model's. The FLOP parity argument only holds for throughput in the compute-bound regime.
</details>

7. In a looped model with random $T$ during training, why is truncated backprop through the last $k$ iterations not the same as training a $k$-block model?

<details><summary>Answer</summary>
The forward pass still runs all $T$ iterations, so the state entering the last $k$ iterations is the result of $T - k$ applications of the same block, and the block must learn a step that is useful when applied to its own outputs, not to embeddings. The gradient is biased (it ignores how earlier iterations would change) but the block is still being trained as a reusable step. A $k$-block model never sees its own output as input.
</details>

8. The load-balancing loss uses $f_e$, a non-differentiable count. Why not use $\sum_e P_e^2$ alone, which is differentiable and also minimized at uniform?

<details><summary>Answer</summary>
$\sum_e P_e^2$ balances the router's soft probabilities, but tokens are dispatched by the argmax (top-$k$), so the soft distribution can be uniform on average while the hard assignments are badly skewed (every token slightly prefers expert 3). The product $f_e P_e$ ties the differentiable term to the actual hard load: the gradient flows through $P_e$ but is weighted by what was really routed.
</details>

9. Give a reason the RoPE base was raised from 10000 to 500000 for Llama 3, in terms of the wavelengths of the slowest pairs.

<details><summary>Answer</summary>
The slowest pair has wavelength about $2\pi\beta$ positions. With $\beta = 10^4$ that is about 63k; with $\beta = 5 \times 10^5$ it is about 3.1M. At an 8k or 128k training context, a larger base means more pairs whose angle stays within a fraction of a full turn over the whole context, giving the model a set of nearly monotone position features to reason about long-range order, and leaving room to extend context without the low-frequency pairs wrapping around.
</details>

10. You tie the embedding and unembedding in a 128k-vocabulary model to save 525M parameters. What changes in the gradient the embedding table receives, and why do large models usually not tie?

<details><summary>Answer</summary>
A tied table gets gradient from two places: the input side (only the rows of tokens in the batch) and the output side (every row, every position, through the softmax). The output-side gradient dominates and forces the embedding geometry to serve the logit layer, whose ideal geometry (rows separable by the final hidden state) is not the same as a good input representation. At 8B parameters, 525M is 6.5 percent, cheap for the freedom; at 100M parameters it is a third of the model, and small models tie.
</details>

## What will change, what will not

The residual-stream picture is durable. Whatever replaces attention or the MLP, the architecture of "a per-token state that sublayers read and add to" has survived every change since 2017 because it is what makes depth trainable, and the analysis habits it supports (per-sublayer contributions, additive writes, a final norm that sets the logit scale) transfer to state-space and hybrid models as well.

Parameter and FLOP accounting is durable. The identities $6ND$, "cache = $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$ per token", and "active versus total" are arithmetic, not architecture, and they will be the first thing you compute about any new model. Practice until they are reflex.

The relative-position argument is durable even if RoPE is not. Any position scheme must answer the same question: what does the score depend on, and what happens outside the training range. Bases, scaling factors, and the specific rotation are choices that have already changed several times.

The specific configs will not last. Head counts, $H_{kv} = 8$, $d_{ff} = 3.5d$, eight experts with top-2, and the norm placement are all outcomes of ablations at a particular scale with particular kernels. Multiples of 64 and 256 appear because of tensor-core tile shapes, and will change when hardware does. Treat every number in a config as a decision to be re-derived, not a constant.

MoE routing and looped depth are the moving parts. Both trade parameters against compute, and both are active research areas: routing losses, capacity handling, and expert specialization on one side, and halting, cache handling, and training curricula for recurrent depth on the other. Expect the mechanisms in this chapter to be replaced; expect the trade itself (memory versus FLOPs per token) to remain the axis along which models are placed.

## Read next

1. "A Mathematical Framework for Transformer Circuits", Elhage, 2021. The residual-stream reading of the architecture used throughout this chapter.
2. "RoFormer: Enhanced Transformer with Rotary Position Embedding", Su, 2021. The rotation derivation and the long-range decay analysis.
3. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", Ainslie, 2023. The grouping, the mean-pool conversion, and the quality-versus-cache evidence.
4. "GLU Variants Improve Transformer", Shazeer, 2020. Where SwiGLU and the $8d/3$ parity convention come from.
5. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", Fedus, 2021. Top-1 routing, the auxiliary loss, capacity factor.
6. "The Llama 3 Herd of Models", Grattafiori, 2024. The 8B config counted above and the reasoning behind its choices.
7. "Universal Transformers", Dehghani, 2018. The weight-tied, looped block with adaptive halting.
8. "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", Geiping, 2025. Prelude, looped core, coda; random-depth training; the cache discussion at scale.
