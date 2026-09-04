---
title: "Lab 18: Mixture of experts"
kind: permanent
topics: [lab]
chapter: 18
station: moe
recipe: recipes/moe_nano.py
reading_time: 50 min
---

# Lab 18: Mixture of experts

## What you will be able to do

1. Write an MoE layer from its four parts (router logits, top-$k$ selection, renormalized gates, expert MLPs) in PyTorch with real dispatch, and say where the router's gradient comes from and why a top-1 layer with renormalized gates would receive none.
2. Count active and total parameters for any MoE config, reproduce Mixtral 8x7B's 46.7B and 12.9B, and turn the two counts into training FLOPs, weight memory, and decode bytes per token as a function of batch size.
3. Derive the Switch Transformer balance loss and its gradient on a router logit, explain the router z-loss, expert capacity and token dropping, and the bias-based loss-free balancing used in DeepSeek-V3.
4. Recognize routing collapse in a per-expert load histogram and a per-domain usage matrix, and know which knob (balance coefficient, router init, noise, bias) to turn.
5. Train a small MoE on the stories-and-arithmetic mixture on the 5090 with `recipes/moe_nano.py`, read its per-expert load and per-domain usage logs, and compare it fairly against a dense model with the same active parameter count.

## The idea in one paragraph

A transformer block spends most of its parameters in the MLP, and every token pays for all of them. A mixture of experts replaces that one MLP with several (the experts) and a small router that looks at each token and picks one or two of them; the token runs through only the experts it was sent to, and the results are blended by the router's weights. Parameters now grow with the number of experts while the compute per token does not, which is the whole attraction, and it comes with two costs. Every expert must still be in memory even though most sit idle for any given token, and the router has to be pushed to spread tokens out, because a router left alone finds the expert that trains fastest and sends everything there. Lab 11 counted Mixtral's parameters and wrote down the balance loss; this chapter builds the layer, breaks its routing, and shows what the load logs look like on a two-domain mixture. In the moe station in the browser a four-expert top-1 MoE head trains on the stories plus arithmetic mixture with a live expert-usage heatmap per domain and the balance loss drawn next to the task loss.

## The math

### The layer

Take one token's residual-stream vector $x \in \mathbb{R}^d$ (Lab 11). An MoE layer with $E$ experts and $k$ active experts computes, in order:

$$
s = x W_r \in \mathbb{R}^E, \qquad r = \mathrm{softmax}(s), \qquad \mathcal{T} = \mathrm{TopK}(r, k),
$$

$$
g_e = \begin{cases} \dfrac{r_e}{\sum_{e' \in \mathcal{T}} r_{e'}} & e \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases}, \qquad y = \sum_{e \in \mathcal{T}} g_e\, \mathrm{FFN}_e(x).
$$

$W_r \in \mathbb{R}^{d \times E}$ is the router, a single linear map with no bias; $s$ are the router logits; $r$ the router probabilities; $\mathcal{T}$ the index set of the $k$ largest; $g$ the gates, renormalized so they sum to one over the chosen experts; and each $\mathrm{FFN}_e$ is an ordinary MLP, SwiGLU in the Llama family, with its own weights $W_{g,e}, W_{u,e} \in \mathbb{R}^{d \times d_{ff}}$ and $W_{d,e} \in \mathbb{R}^{d_{ff} \times d}$. The layer output $y$ is added to the residual stream exactly as a dense MLP's output would be.

Where the gradient goes. $\mathrm{TopK}$ is a selection, not a differentiable function: the indices in $\mathcal{T}$ carry no gradient. The task loss reaches $W_r$ only through the gate values $g_e$ that multiply the expert outputs. That has a consequence you should check before writing any variant. With $k = 1$ and renormalized gates, $g_e = r_e / r_e = 1$ for the chosen expert, a constant, and the router gets no gradient from the task at all; it would never learn which expert suits which token. The Switch Transformer therefore uses the raw probability $r_e$ as the top-1 gate, so that the chosen expert's gate is a differentiable function of the logits. With $k = 2$ (Mixtral) renormalization is fine, because the ratio $r_{e_1} / (r_{e_1} + r_{e_2})$ still depends on the logits. DeepSeek-V3 replaces the softmax with per-expert sigmoid affinities $\sigma(s_e)$ and renormalizes over the selected set, which decouples the experts' scores from each other; the gradient path is the same.

Dispatch, not masking. The formula could be evaluated by running every expert on every token and multiplying by a mostly-zero gate matrix; that gives the right numbers and none of the savings. A real layer sorts tokens by chosen expert, runs each expert on only its tokens (a grouped or batched GEMM over ragged groups), and scatters the results back with the gate weights. The snippet below does the dispatch with an explicit loop over experts, which is what the grouped kernel does in one launch.

### Active versus total, with the Mixtral count

Per MoE layer, the expert parameters are $E \cdot 3 d\, d_{ff}$ (SwiGLU) and the router adds $dE$. Only $k$ experts run per token. Define

$$
P_{\text{total}} = P_{\text{attn}} + E \cdot 3 d\, d_{ff} + dE + P_{\text{norm}}, \qquad P_{\text{active}} = P_{\text{attn}} + k \cdot 3 d\, d_{ff} + dE + P_{\text{norm}}
$$

per layer, and add the embedding and unembedding to both totals. Lab 11 did the Mixtral 8x7B arithmetic ($d = 4096$, $d_{ff} = 14336$, $E = 8$, $k = 2$, 32 layers, $V = 32000$): one expert is $3 \times 4096 \times 14336 = 176{,}160{,}768$ parameters, so eight are $1{,}409{,}286{,}144$ per layer and two are $352{,}321{,}536$; with attention at $41{,}943{,}040$, the router's $32{,}768$, and the norms, the per-layer total is $1{,}451{,}270{,}144$ and the per-layer active is $394{,}305{,}536$; times 32 plus the $262{,}144{,}000$ for the two vocabulary matrices and the final norm gives $46{,}702{,}792{,}704$ total and $12{,}879{,}925{,}248$ active, the published 46.7B and 12.9B. Notice the ratio of experts to everything else: $32 \times 1.409$B $= 45.1$B of the 46.7B, 96.5 percent, live in experts. The MoE did not make attention or the vocabulary cheaper; it made the MLP budget six times larger for the same per-token cost.

Training compute follows the active count: $C \approx 6 P_{\text{active}} D$ for $D$ tokens (Lab 11's $6ND$ with $N$ the parameters that actually multiply each token), so Mixtral trains for about the FLOPs of a 13B dense model. Memory follows the total: in bf16, $2 \times 46.7 \times 10^9 = 93.4$ GB of weights, which is three times the 5090's memory before any cache, so the model runs on this card only at 4-bit or with offloading (below). Decode bytes per token depend on the batch. At batch 1 a step reads attention weights plus $k$ experts per layer, about $2 \times 12.9 = 25.8$ GB in bf16. At batch $B$ under uniform routing the number of distinct experts touched in a layer has expectation

$$
\mathbb{E}[\text{experts touched}] = E \left(1 - \left(1 - \tfrac{k}{E}\right)^{B}\right),
$$

which for $E = 8$, $k = 2$ is 2 at $B = 1$, 4.6 at $B = 3$, 7.2 at $B = 8$, and 7.9 at $B = 16$. By batch 8 the step reads almost all 45B expert parameters, and the byte count is that of a 47B dense model while the arithmetic is that of a 13B one. This is the inference-side version of the active-versus-total distinction: FLOPs scale with active, memory and, above a small batch, memory traffic scale with total.

### Load balancing

Let a batch hold $N$ tokens. Define for each expert the fraction of routing slots it received and the mean router probability it was assigned:

$$
f_e = \frac{1}{Nk} \sum_{i=1}^{N} \mathbb{1}[e \in \mathcal{T}_i], \qquad P_e = \frac{1}{N} \sum_{i=1}^{N} r_{i,e},
$$

so that $\sum_e f_e = \sum_e P_e = 1$ (the Switch paper's $f_e$ is for $k = 1$; dividing by $Nk$ keeps the normalization for any $k$). The Switch Transformer auxiliary loss is

$$
\mathcal{L}_{\text{bal}} = \alpha\, E \sum_{e=1}^{E} f_e P_e .
$$

Where it comes from. You would like to penalize $\sum_e f_e^2$, which is minimized at $f_e = 1/E$ (by Cauchy-Schwarz, $\sum_e f_e^2 \ge (\sum_e f_e)^2 / E = 1/E$ with equality at uniform), but $f_e$ counts hard assignments and has no gradient. $P_e$ is the differentiable shadow of $f_e$: the router assigns high probability to the experts it routes to, so $P_e$ tracks $f_e$, and replacing one factor gives $\sum_e f_e P_e$, which the same inequality bounds below by $1/E$ when $P = f$. The factor $E$ makes the uniform value 1 regardless of $E$, so one coefficient $\alpha$ works across expert counts. Now the gradient. Treat $f_e$ as a constant (it is), and differentiate through the softmax: for token $i$ and logit $s_{i,e'}$, $\partial r_{i,e} / \partial s_{i,e'} = r_{i,e} (\delta_{ee'} - r_{i,e'})$, so

$$
\frac{\partial \mathcal{L}_{\text{bal}}}{\partial s_{i,e'}} = \frac{\alpha E}{N} \sum_e f_e\, r_{i,e}(\delta_{ee'} - r_{i,e'}) = \frac{\alpha E}{N}\, r_{i,e'} \left( f_{e'} - \sum_e f_e\, r_{i,e} \right).
$$

Read the bracket: the logit of expert $e'$ is pushed down when its load $f_{e'}$ is above the router-weighted average load $\sum_e f_e r_{i,e}$ for that token, and up when it is below. Overloaded experts lose probability, underloaded ones gain it, and the push is proportional to how much the router already likes $e'$ for this token, so the loss never forces a token onto an expert the router considers irrelevant. Typical $\alpha$ is $10^{-2}$; at $10^{-1}$ the routing term competes with the task and the router stops discriminating, which the snippet shows.

Router z-loss (ST-MoE, Zoph et al., 2022) is a second regularizer,

$$
\mathcal{L}_z = \frac{1}{N} \sum_{i=1}^{N} \left( \log \sum_{e} e^{s_{i,e}} \right)^2 ,
$$

with a coefficient around $10^{-3}$. It penalizes large logits. The motivation is numerical: routing decisions are argmaxes over a softmax, and when the logits are large the softmax saturates, small bf16 rounding differences flip decisions between forward passes, and training becomes unstable; keeping the log-partition near zero keeps the logits in a range where the softmax is smooth. It is also why routers are computed in fp32 in every serious implementation.

Expert capacity and token dropping. Fixed-shape dispatch buffers (needed for expert parallelism, below) hold at most $C$ tokens per expert per batch,

$$
C = \left\lceil \text{capacity factor} \times \frac{N k}{E} \right\rceil ,
$$

so with capacity factor 1.0 each expert accepts exactly its uniform share and nothing more. A token that arrives at a full expert is dropped: it skips the layer, and its residual stream passes through unchanged (with top-2 it may still be served by its other expert). Factors of 1.0 to 1.25 in training and higher at evaluation are common; the drop fraction is a metric to log, because a router that is balanced on average can still overflow one expert in one batch, and dropped tokens are silent quality loss. Dropless MoE (MegaBlocks, Gale et al., 2022) removes the limit by expressing the expert computation as a block-sparse matmul over variable-size groups, which is what most single-node implementations do now; capacity survives where the all-to-all needs fixed buffers.

Loss-free balancing with a per-expert bias (Wang et al., 2024; used in DeepSeek-V3). Keep a bias $b_e$ per expert that enters only the selection, not the gate:

$$
\mathcal{T} = \mathrm{TopK}(\sigma(s) + b, k), \qquad g_e = \frac{\sigma(s_e)}{\sum_{e' \in \mathcal{T}} \sigma(s_{e'})} \text{ for } e \in \mathcal{T}.
$$

After each step, compare each expert's load with the mean: overloaded experts get $b_e \leftarrow b_e - \gamma$, underloaded ones $b_e \leftarrow b_e + \gamma$, with a small step such as $\gamma = 10^{-3}$. No gradient term touches the router from balancing, so the task loss alone decides the gate values and the bias only reshuffles marginal tokens; the DeepSeek-V3 report keeps a very small sequence-level balance loss alongside it as a guard. As far as this chapter can vouch, the mechanism is exactly this bias update on the selection scores; the report has the schedule details.

### Fine-grained and shared experts

DeepSeekMoE (Dai et al., 2024) makes two changes to the Mixtral shape. Fine-grained experts: split each expert's $d_{ff}$ by $m$ and multiply the count by $m$, activating $mk$ of $mE$. The active parameters and FLOPs are unchanged, but the number of distinct expert combinations a token can use grows from $\binom{E}{k}$ to $\binom{mE}{mk}$: for $E = 8$, $k = 2$ that is 28 combinations, and with $m = 4$ it is $\binom{32}{8} = 10{,}518{,}300$. The claim is that finer experts specialize more cleanly and combine more flexibly. Shared experts: reserve $k_s$ experts that every token uses, unrouted, alongside the $k$ routed ones, on the argument that common knowledge otherwise gets duplicated across many routed experts. DeepSeek-V3 has 256 routed experts with 8 active plus 1 shared expert, 671B total and 37B active parameters; the ratio total to active is about 18, against Mixtral's 3.6, which is what fine-grained routing buys.

### Routing collapse, and how to see it

The feedback loop. At initialization all experts are equally bad. A small asymmetry sends slightly more tokens to expert 3; expert 3 receives more gradient, improves faster, wins more tokens, and after a few hundred steps the router sends nearly everything to it while the others sit at their initialization. The model is then a dense model with one expert's capacity and $E$ experts' memory. The symptoms, in order of how early you can see them: the balance loss $E \sum_e f_e P_e$ climbs above 1 (uniform) toward $E$ (everything on one expert); the per-expert load histogram grows a spike; the entropy of the load, $-\sum_e f_e \log f_e$, falls from $\ln E$; the maximum expert share exceeds a few times $1/E$; some experts receive zero tokens over a window (dead experts); and if you have a per-domain usage matrix (rows domains, columns experts, entries the share of that domain's slots), a whole column takes every row. Causes: no balance term, a router initialized with large weights so early softmaxes are peaked, a learning rate that is too high for the router, and top-1 routing, which has the sharpest feedback. Fixes: the balance loss at $\alpha \approx 10^{-2}$, router weights initialized near zero (the snippet uses a standard deviation of 0.01), the bias mechanism, and, in the original sparsely-gated layer of Shazeer et al. (2017), Gaussian noise added to the logits before the top-$k$ so that early decisions are not locked in. The moe station is the cheapest place to watch it: at coefficient 0 one column usually takes everything within the first hundred steps, and at 0.01 the columns even out.

### Expert parallelism and all-to-all

When the experts do not fit on one device, place $E / P$ experts on each of $P$ devices. A layer then runs in three phases: an all-to-all in which every device sends each token's activation to the device holding its chosen expert; the expert MLPs, each on its own device; and a second all-to-all that returns the outputs to the devices that own the tokens. The communication volume per layer is about $N k d$ activations each way, so it grows with $k$ and $d$, and the layer finishes when the busiest device does: an imbalance of 2 to 1 in routing is a 2 to 1 imbalance in both compute and buffer size, which is why capacity factors exist and why balancing is a systems problem before it is a modeling one. On the 5090 there is no expert parallelism, all experts are local, and the relevant costs are memory and the grouped GEMM; keep the picture anyway, because it explains why every large MoE paper reports load statistics next to loss.

### Inference cost and offloading

A 3B-active model with 20B total parameters trains and generates with 3B parameters of arithmetic per token and 20B parameters of memory. In bf16 that is 40 GB, more than this card holds, even though each token touches 6 GB of weights. Two ways out. Quantize the experts: at 4 bits the 20B is about 10 to 12 GB with scales and fits with room for a cache; the quality cost is usually small because expert weights are large and well-conditioned matrices. Or offload: keep attention, norms, embeddings, and any shared experts on the GPU and hold the routed experts in CPU memory, copying each expert to the card when a token needs it. The bound is PCIe, not GDDR7. With a stated assumption of 50 GB/s achieved on a PCIe 5.0 x16 link, one bf16 Mixtral expert of 352 MB takes about 7 ms to move, and a token that needs two uncached experts in each of 32 layers waits about 450 ms just for transfers; at 4-bit the expert is about 90 MB and the figure is about 115 ms. What makes offloading tolerable in practice is that routing is not uniform in time: consecutive tokens reuse experts, so an LRU cache of a few experts per layer on the GPU hits often, and the router's decision for layer $\ell + 1$ can be predicted from layer $\ell$'s hidden state to prefetch before it is needed (Eliseev and Mazur, 2023, do both for Mixtral on consumer cards). Decode throughput on an offloaded MoE is set by the expert miss rate times the transfer time, and you should measure the miss rate on your traffic before promising a tokens-per-second number.

### Upcycling a dense model

Sparse upcycling (Komatsuzaki et al., 2022) starts an MoE from a trained dense checkpoint: copy the dense MLP of every layer into all $E$ experts, add a router initialized near zero, keep attention and embeddings, and continue training. At step zero every expert computes the same function, so the layer is exactly the dense model and nothing is lost; the router's early decisions are arbitrary, the balance loss (or noise) spreads tokens out, and the experts diverge as they see different tokens. The gain is that the MoE inherits the dense model's training rather than starting from scratch, and the loss of the upcycled model at a given additional compute is below both continuing the dense model and training an MoE fresh, in the paper's regime. The trap is symmetry: with identical experts and a zero router the gradient on the router is zero at step zero, so either the router needs a small random initialization, or the balance term or noise must break the tie. For your own experiments, upcycling the Lab 02 checkpoint into `recipes/moe_nano.py` is the quickest way to an MoE that is already fluent.

### MoE and LoRA

Lab 15 tells the story: a LoRA recipe with a hand-written `target_modules` list of dense-model names (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) was applied to a 35B-total, 3B-active MoE, matched only a fraction of the modules, and silently trained ten of forty layers. MoE implementations name and store expert weights differently from dense ones: as per-expert submodules with their own names, or as one fused three-dimensional parameter per layer that is not an `nn.Linear` at all. A name list written for a dense model matches attention and nothing in the experts, or nothing in some layers. The fix is `target_modules="all-linear"`, which attaches an adapter to every `nn.Linear` except the output head, followed by an assertion that counts adapters per layer before the first step. Two caveats that the fix does not cover. Fused expert tensors are still not `nn.Linear` and stay frozen under `all-linear`; if the experts are where you need capacity, you need an adapter that understands the fused layout, or an unfused checkpoint. And the router is a small `nn.Linear` that `all-linear` will adapt; a rank-8 update to a $d \times E$ matrix can move routing a great deal, so decide deliberately whether the router is in or out and log the load statistics during fine-tuning either way.

## Build it small

The snippet is a top-2 MoE MLP with four experts and real dispatch, trained by Adam on two synthetic domains that differ in both their inputs (shifted along a fixed direction $u$) and their target functions ($\tanh$ of one linear map against $\sin$ of another). Domain 0 supplies 80 percent of the tokens, as a real mixture would. It trains three times with the balance coefficient at 0, 0.01, and 0.1, and prints per-domain error, the balance loss (1.0 at uniform), the load per expert, and the per-domain usage matrix.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
D, H, E, K = 16, 64, 4, 2                        # input width, expert hidden, experts, top-k


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(D, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)              # near-uniform routing at the start
        self.W1 = nn.Parameter(torch.randn(E, D, H) / D ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(E, H))
        self.W2 = nn.Parameter(torch.randn(E, H, D) / H ** 0.5)

    def forward(self, x):
        probs = self.router(x).softmax(-1)                          # (N, E) router probabilities
        topv, topi = probs.topk(K, dim=-1)                          # (N, K) chosen experts
        gates = topv / topv.sum(-1, keepdim=True)                   # renormalize over the chosen K
        y = torch.zeros_like(x)
        for e in range(E):                                          # dispatch: each expert sees only its tokens
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel():
                h = F.gelu(x[rows] @ self.W1[e] + self.b1[e]) @ self.W2[e]
                y.index_add_(0, rows, gates[rows, slot, None] * h)
        f = torch.bincount(topi.flatten(), minlength=E).float() / topi.numel()   # fraction of slots per expert
        P = probs.mean(0)                                                          # mean router prob per expert
        return y, E * (f * P).sum(), topi                          # Switch balance loss: min 1 at uniform


u = F.normalize(torch.randn(D), dim=0)                             # domain direction
maps = [torch.randn(D, D) / D ** 0.5 for _ in range(2)]

def batch(n, frac=0.8):                                             # two domains, different inputs AND functions
    dom = (torch.arange(2 * n) >= int(2 * n * frac)).long()         # domain 0 is 80 percent of the tokens
    x = torch.randn(2 * n, D) + (2 * dom[:, None] - 1) * 1.5 * u    # domain 0 sits at -1.5u, domain 1 at +1.5u
    y = torch.where(dom[:, None] == 0, torch.tanh(x @ maps[0]), torch.sin(x @ maps[1]))
    return x, y, dom

for alpha in (0.0, 0.01, 0.1):
    torch.manual_seed(1)
    model = MoE(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1500):
        x, y, _ = batch(256)
        out, aux, _ = model(x)
        loss = F.mse_loss(out, y) + alpha * aux
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x, y, dom = batch(4096)
        out, aux, topi = model(x)
        usage = torch.stack([torch.bincount(topi[dom == d].flatten(), minlength=E).float() for d in range(2)])
        usage = usage / usage.sum(-1, keepdim=True)                 # rows: domain, cols: expert, share of slots
        load = torch.bincount(topi.flatten(), minlength=E).float() / topi.numel()
        print(f"alpha={alpha}: mse per domain = {[round(F.mse_loss(out[dom == d], y[dom == d]).item(), 4) for d in range(2)]}"
              f"  balance loss = {aux.item():.3f}  load per expert = {load.numpy().round(3)}")
        print("  per-domain expert usage (rows: domain 0, domain 1; cols: experts 0..3):")
        print("  " + str(usage.numpy().round(3)).replace("\n", "\n  "))
```

Output from a run on the CPU with PyTorch 2.10, about 10 seconds:

```
alpha=0.0: mse per domain = [0.0216, 0.1093]  balance loss = 1.180  load per expert = [0.12  0.371 0.368 0.141]
  per-domain expert usage (rows: domain 0, domain 1; cols: experts 0..3):
  [[0.034 0.451 0.444 0.072]
   [0.465 0.052 0.066 0.418]]
alpha=0.01: mse per domain = [0.0219, 0.114]  balance loss = 1.038  load per expert = [0.119 0.317 0.295 0.269]
  per-domain expert usage (rows: domain 0, domain 1; cols: experts 0..3):
  [[0.033 0.36  0.322 0.284]
   [0.463 0.145 0.187 0.206]]
alpha=0.1: mse per domain = [0.0277, 0.1447]  balance loss = 0.990  load per expert = [0.234 0.222 0.276 0.268]
  per-domain expert usage (rows: domain 0, domain 1; cols: experts 0..3):
  [[0.228 0.201 0.291 0.28 ]
   [0.257 0.308 0.218 0.218]]
```

What happened. With no balance term the experts specialised by domain on their own: domain 0 uses experts 1 and 2 for 90 percent of its slots, domain 1 uses experts 0 and 3 for 88 percent, and a top-2 layer with four experts and two domains has found the obvious partition. It did not collapse, which is worth understanding rather than assuming: the router started near uniform, Adam normalizes the per-parameter step so the rich-get-richer feedback is damped, and with top-2 both chosen experts receive gradient every step. Collapse is easy to provoke by moving away from that setting (top-1, a large router initialization, SGD), and the station's top-1 head shows it at coefficient 0. But the specialised solution is imbalanced, because the domains are: experts 1 and 2 carry 37 percent of the load each and experts 0 and 3 carry 12 to 14 percent, so the balance loss reads 1.18. At $\alpha = 0.01$ the loss falls to 1.04 and the load flattens toward uniform (0.12, 0.32, 0.30, 0.27), and the way it gets there is instructive: expert 3, which was serving the minority domain, now takes 28 percent of the majority domain's slots, so specialisation is partly traded for balance, at a small cost in error (domain 1's MSE from 0.109 to 0.114). At $\alpha = 0.1$ the load is nearly uniform, the balance loss is below 1 (the soft probabilities are flatter than the hard loads), the usage matrix is close to a constant, and the error is 28 percent worse on domain 0 and 32 percent worse on domain 1: the router has been told that balance matters more than the task and it now routes almost at random. With a 50/50 mixture (change `frac` to 0.5) the specialised partition is balanced by construction and all three coefficients give the same matrix and the same error; balance and specialisation only fight when the data are skewed, which real data always are.

The station shows the same matrix on real tokens: rows stories and arithmetic, columns experts, and the balance loss on the chart. Run it at coefficient 0 and then 0.01 and look at whether the two rows prefer different columns.

## Build it real

`recipes/moe_nano.py` is the Lab 02 trainer with the MLP of every block replaced by the MoE layer above, trained on the two-domain stories-and-arithmetic mixture that the midtrain station and `recipes/midtrain.py` use (Lab 03), so the per-domain numbers line up with what you have already seen. Arguments: `--n-expert` and `--top-k` set $E$ and $k$ (defaults 8 and 2); `--d-ff` is the width of one expert; `--balance aux|bias|none` selects the Switch loss with `--balance-coef` (default 0.01), the per-expert bias update with `--bias-step` (default 0.001), or nothing; `--z-loss` adds the router z-loss with its coefficient; `--capacity-factor` enables token dropping (default 0, dropless); `--router-init` sets the router's initial standard deviation; `--upcycle CKPT` starts from a Lab 02 dense checkpoint by copying its MLP into every expert; `--dense` trains a dense control with `--d-ff` scaled so that its parameter count matches the MoE's active count; and `--mix`, `--steps`, `--lr`, `--batch`, `--seq-len`, and the model-shape flags are as in Lab 02.

What it logs. `METRIC` lines with `loss`, `val_loss_stories`, `val_loss_arith`, `aux` (the balance loss), `load` (the vector $f_e$), `load_entropy`, `max_share`, `dropped_frac` when capacity is on, and every evaluation a `usage` matrix (rows domains, columns experts). A `RESULT` line at the end carries the final per-domain losses and the final usage matrix, and, when `--dense` was also run, the two are printed side by side.

Sizing. A six-layer, width-384, six-head model with $E = 8$ experts of $d_{ff} = 1024$ and $k = 2$ has about $8 \times 3 \times 384 \times 1024 \approx 9.4$M expert parameters per layer, 57M in total across six layers, of which a quarter is active per token; with attention and the character vocabulary the total is around 60M and the active count around 18M. It fits in a few GB. Time is set by the dispatch, not the arithmetic: at this width every expert GEMM is tiny and the per-expert loop launches $E$ small kernels per layer per step, so expect a few thousand steps in tens of minutes rather than the throughput Lab 02 reports for the dense model; `torch.compile` helps, and a grouped-GEMM kernel would help more, but at this scale the loop is fine. Use the formula $6 P_{\text{active}} D$ to compare compute with the dense control, and measure `tokens_per_s` for both so you can see the dispatch overhead as a ratio.

What to watch. `aux` should fall from its initial value toward 1 within the first few hundred steps and stay there; if it climbs, look at `max_share` and `load` in the same lines and expect a spike. `load_entropy` near $\ln E$ (2.08 for eight experts) is healthy; it will not be exactly there and should not be, since some imbalance is the price of specialisation. The `usage` matrix is the interesting object: rows are the two domains, and the question is whether they have different column profiles. With `--balance-coef 0.1` expect the matrix to flatten and both validation losses to rise, the toy's result at scale. With `--balance none --router-init 1.0` expect a collapse you can time-stamp from the `max_share` column. And compare the final validation losses with the `--dense` control at matched active parameters; the MoE should win on at least the minority domain, since it has four times the parameters to spend, and if it does not, the routing is the first suspect.

## How it goes wrong

Routing collapse. Symptom: `aux` climbs, `max_share` goes above 0.5, one column of `usage` takes both rows, and validation loss matches a dense model with one expert's capacity. Cause: no balance term, a large router initialization, or top-1 routing with a high learning rate. Fix: `--balance aux --balance-coef 0.01` or `--balance bias`, `--router-init 0.01`, and log load from step one so you see the spike form.

Over-balancing. Symptom: `aux` sits at or below 1, `usage` is a flat matrix, and both validation losses are worse than the dense control. Cause: the balance coefficient is large enough that the router ignores content, as at $\alpha = 0.1$ in the snippet. Fix: lower the coefficient by 10 and confirm the usage rows separate again; if you need hard balance for a systems reason, use the bias mechanism, which does not put a routing term in the task gradient.

Dead experts after token dropping. Symptom: with capacity on, an expert's load falls to zero and never recovers, and `dropped_frac` is high early. Cause: an expert that overflowed its capacity in the first steps received truncated gradient, fell behind, lost tokens, and now receives none; capacity turned a transient imbalance into a permanent one. Fix: train dropless or with a capacity factor of 1.25 or more, and drop only at evaluation; in a fixed-buffer system, raise the factor during warmup.

Router in bf16. Symptom: training is stable for thousands of steps and then the loss spikes without an obvious data cause; routing decisions for the same batch differ between two forward passes. Cause: large router logits, a saturated softmax, and top-$k$ ties resolved differently under bf16 rounding. Fix: compute the router in fp32 (cast $x$ before $W_r$) and add the z-loss.

A top-1 layer that never learns to route. Symptom: the usage matrix is whatever the initialization gave and never moves; the experts specialise by accident or not at all. Cause: renormalized gates with $k = 1$ are the constant 1 and pass no gradient to the router. Fix: use the raw probability as the gate for top-1, or use $k = 2$.

LoRA that trains a fraction of the model. Symptom: the trainable-parameter count is far below what the rank implies, the reward or loss does not move, and some layers report zero adapters. Cause: `target_modules` names from a dense model do not match the MoE's expert modules or its fused expert tensors. Fix: `target_modules="all-linear"`, an adapter count per layer asserted before training, and a deliberate decision about the router (Lab 15).

Out of memory at inference after sizing by active parameters. Symptom: a "3B" model fails to load on 32 GB. Cause: the 3B is the active count; the total is what must be resident. Fix: size by total parameters times bytes per weight, quantize the experts, or offload them and budget the PCIe transfer time with the miss rate.

The wrong baseline. Symptom: the MoE "wins" against a dense model with the same total parameters, or "loses" against a dense model trained with the same wall-clock. Cause: neither comparison holds compute fixed. Fix: compare at matched active parameters and matched tokens (same $6 P_{\text{active}} D$), report the total-parameter and memory cost next to it, and run at least two seeds; the fair claim is "better loss at the same FLOPs for more memory".

## Measure it

Task loss per domain against two dense controls: one with the same active parameters (the compute-fair comparison the MoE should win) and, when memory is the question, one with the same total parameters (which the MoE should lose, and the gap tells you what routing cost). Report both with the token count and two seeds.

Routing health, every evaluation: the balance loss (near 1 is healthy; the uniform value is 1 by construction, so it is comparable across $E$), load entropy against $\ln E$, maximum expert share against $1/E$, the count of experts below a small share threshold (dead experts), and the dropped fraction when capacity is on. None of these should be perfect; a load entropy at exactly $\ln E$ with a flat usage matrix is over-balancing, not health.

Specialisation: the per-domain usage matrix, and, as a single number, the mutual information between domain and expert computed from it, which is zero for a flat matrix and $\ln 2$ at most for two domains routed disjointly. Compare it across balance coefficients; the number should fall as the coefficient rises, and the loss should tell you where the trade stops paying.

Cost: tokens per second against the dense-active control (the dispatch overhead), weight bytes resident, and decode bytes per token at your serving batch from the experts-touched formula, which is the number that decides whether the model is fast on this card.

## Exercises

1. Compute total and active parameters for an MoE with $d = 2048$, $d_{ff} = 5632$, $E = 16$, $k = 2$, 24 layers, $H = 16$, $H_{kv} = 4$, $V = 32000$, untied. Check: per expert $3 \times 2048 \times 5632 = 34.6$M; attention with GQA $2048 \times 2048 \times 2 + 2 \times 2048 \times 512 = 10.5$M; per layer total about $16 \times 34.6 + 10.5 \approx 564$M and active about $2 \times 34.6 + 10.5 \approx 80$M; totals about 13.7B and 2.0B including the two vocabulary matrices ($2 \times 32000 \times 2048 = 131$M).

2. Show that $\sum_e f_e P_e \ge 1/E$ when $P = f$ and find the minimizer. Then construct $f \ne P$ with $\sum_e f_e P_e < 1/E$ and explain why the loss can read below 1 in the snippet's $\alpha = 0.1$ run. Check: Cauchy-Schwarz gives the bound with equality at uniform; with $f = (1, 0)$ and $P = (0.4, 0.6)$ for $E = 2$ the sum is 0.4 and $E$ times it is 0.8; soft probabilities flatter than hard loads give a value below 1.

3. Modify the snippet to top-1 with renormalized gates and confirm the router receives no gradient (`model.router.weight.grad` is zero after `backward`); then switch the gate to the raw probability and confirm it trains. Check: with renormalization the gate is identically 1 and the gradient is exactly zero; with the raw probability the usage matrix separates the domains within a few hundred steps.

4. Implement the per-expert bias update in the snippet (bias enters `topk` only, updated by $\pm \gamma$ after each step from the batch load) with the balance loss off. Check: on the 80/20 mixture the load flattens to within a few percent of uniform while the usage matrix keeps a visible domain structure and the error is closer to the $\alpha = 0$ run than to the $\alpha = 0.1$ run.

5. For Mixtral at 4-bit weights (assume 0.55 bytes per expert parameter including scales) on the 5090 with everything resident, compute the decode byte bound per token at batch 1, 4, and 16 using the experts-touched formula, and the resulting tokens per second at $\beta = 1.525$ TB/s. Check: expert bytes per layer are about $0.55 \times 176$M times 2, 5.5, and 7.9 experts touched; with attention and the vocabulary matrices left in bf16, the step reads about 9.4 GB at batch 1 (a bound near 160 tokens per second for the single stream), 20 GB at batch 4 (about 75 per stream, 300 aggregate), and 28 GB at batch 16 (about 55 per stream, 880 aggregate). Per-stream speed falls as the batch touches more experts, while aggregate throughput still rises.

6. On the 5090, run `recipes/moe_nano.py` with `--balance aux` at coefficients 0, 0.01, and 0.1, and `--dense`, all at matched tokens, two seeds each. Check: the coefficient-0 run either collapses or specialises depending on `--router-init`; the 0.01 run has the best minority-domain loss; the 0.1 run's usage matrix is flat and its losses approach the dense control's; and the seed-to-seed spread is reported next to every difference you claim.

## Test yourself

1. A colleague says "Mixtral is a 47B model, so it needs 47B parameters of compute per token." Correct this and give the two numbers that matter.

<details><summary>Answer</summary>
Compute per token follows the active count: attention plus two of eight experts per layer, 12.9B parameters, about $2 \times 12.9$B FLOPs per token in the forward pass. Memory follows the total: 46.7B parameters, 93.4 GB in bf16. The two are not interchangeable, and at decode with batch above about 8 the memory traffic per step also approaches the total because nearly every expert is touched by some token in the batch.
</details>

2. Derive the gradient of the Switch balance loss with respect to one router logit and state in words what it does to an overloaded expert.

<details><summary>Answer</summary>
With $f_e$ constant and $P_e = \frac{1}{N}\sum_i r_{i,e}$, the softmax Jacobian gives $\partial \mathcal{L}_{\text{bal}} / \partial s_{i,e'} = \frac{\alpha E}{N} r_{i,e'}\left(f_{e'} - \sum_e f_e r_{i,e}\right)$. For an expert whose load $f_{e'}$ is above the router-weighted mean load for that token, the gradient is positive and gradient descent lowers its logit, in proportion to how much probability the router already gives it on this token. Underloaded experts' logits rise.
</details>

3. Why does the top-1 Switch layer use the raw router probability as the gate rather than renormalizing over the selected set?

<details><summary>Answer</summary>
With one selected expert, renormalization makes the gate $r_e / r_e = 1$, a constant with no dependence on the logits; the task loss then reaches the router only through the non-differentiable selection, which is to say not at all. Using $r_e$ itself keeps the gate a differentiable function of the logits so the router learns which expert lowers the loss for which token.
</details>

4. Spot the bug:

```python
probs = self.router(x).softmax(-1)
topv, topi = probs.topk(1, dim=-1)
gates = topv / topv.sum(-1, keepdim=True)
```

<details><summary>Answer</summary>
Top-1 with renormalized gates: `gates` is identically 1, the router receives no gradient from the task, and routing never changes from its initialization. Use `gates = topv` for top-1, or top-2 with renormalization.
</details>

5. Under uniform routing with $E = 8$, $k = 2$, how many distinct experts does a layer touch in expectation at batch 4, and what does this imply for the bytes read per decode step compared with batch 1?

<details><summary>Answer</summary>
$8(1 - 0.75^4) = 8 \times 0.684 \approx 5.5$ experts, against 2 at batch 1. The step reads about 2.7 times the expert bytes for 4 times the tokens, so per-token bytes fall, but not by the factor of 4 a dense model would give; by batch 16 the layer touches essentially all 8 experts and the step reads the full model.
</details>

6. Why can the balance loss read below 1, its supposed minimum, at the end of the $\alpha = 0.1$ run in the snippet?

<details><summary>Answer</summary>
The bound $\sum_e f_e P_e \ge 1/E$ holds when $P = f$. The loss couples the hard load $f$ with the soft probability $P$, and a strongly balanced router has soft probabilities flatter than its hard loads (many tokens near ties), so $\sum_e f_e P_e$ can dip below $1/E$ and the scaled loss below 1. It is a sign of over-balancing, not of a bug.
</details>

7. In the snippet the $\alpha = 0$ run did not collapse. Name three things about the setup that damped the rich-get-richer feedback, and one change that would likely bring collapse back.

<details><summary>Answer</summary>
A router initialized at standard deviation 0.01, so early routing is near uniform and every expert gets gradient; Adam, whose per-parameter normalization stops the favoured expert's larger gradient from turning into a larger step; and top-2 with renormalized gates, so two experts train on every token. Switching to top-1, initializing the router at standard deviation 1, or using plain SGD with a high rate would each strengthen the feedback; the station's top-1 head at coefficient 0 shows the collapse.
</details>

8. A LoRA fine-tune on an MoE reports 0.4 percent trainable parameters and a loss that does not move. Give the diagnosis and the two-line fix, and say what `all-linear` still leaves frozen.

<details><summary>Answer</summary>
The `target_modules` list names dense-model projections that do not exist in the MoE's expert modules, so adapters attached only to attention (or to some layers). Fix: `target_modules="all-linear"` and an assertion counting adapters per layer before the first step. Fused expert tensors that are `nn.Parameter` rather than `nn.Linear` remain frozen even under `all-linear`, and the router, which is an `nn.Linear`, gets an adapter whether or not you wanted routing to move.
</details>

9. What does the capacity factor bound, and why does a router that is balanced on average still drop tokens at capacity factor 1.0?

<details><summary>Answer</summary>
It bounds the number of tokens any expert accepts in one batch, $\lceil \text{cf} \times Nk/E \rceil$, which fixes the dispatch buffer size for the all-to-all and the maximum per-expert compute. Balance on average is a statement about the mean of $f_e$ over batches; within a single batch the counts fluctuate (a batch of arithmetic tokens overloads the arithmetic experts), and at factor 1.0 there is no slack, so every fluctuation above the mean is a drop.
</details>

10. Upcycling copies the dense MLP into every expert. Why is the model unchanged at step zero, what is the gradient on a zero-initialized router at that moment, and how does training ever break the symmetry?

<details><summary>Answer</summary>
Identical experts give identical outputs, so $\sum_e g_e \mathrm{FFN}_e(x) = \mathrm{FFN}(x)$ for any gates that sum to one; the layer is the dense MLP. The task gradient on the router is proportional to differences between expert outputs, which are zero, so a zero router receives no task gradient. Symmetry is broken by a small random router initialization (different tokens go to different experts and the experts then diverge), by noise on the logits, or by the balance term, which spreads tokens even when the task is indifferent.
</details>

## What will change, what will not

The active-versus-total accounting is arithmetic and will not change: FLOPs per token follow the parameters a token multiplies, memory follows the parameters that exist, and decode traffic slides from one to the other as the batch grows. Every new sparse architecture (more experts, finer experts, shared experts, experts inside attention, mixtures of depths) will be placed on that same map, and the first thing to compute about any of them is the two counts and the experts-touched curve.

The routing problem will not go away, but its solution will keep changing. Any mechanism that assigns discrete work to discrete workers by a learned score has the rich-get-richer feedback, and every fix is a way of injecting a preference for balance without telling the router what to think: an auxiliary loss, noise, capacity limits, a bias on the selection scores. The trend from Switch to DeepSeek-V3 is toward mechanisms that keep the balancing pressure out of the task gradient; expect that to continue, and expect the specific coefficients and update rules to be replaced.

The shapes will change. Eight experts with two active was a 2023 configuration, 256 with 8 active and one shared is a 2024 one, and neither is a law. Fine-grained experts, shared experts, and expert counts in the hundreds are all bets that combinatorial capacity is worth the routing and communication cost; the systems side (grouped GEMMs, all-to-all, offloading, quantized experts) will decide how far that goes on given hardware, and a consumer card with 32 GB will always push toward fewer total parameters, quantized, with the shared and attention weights resident.

What will not change on your card: a model is fast on it when its resident bytes fit and its decode traffic at your batch size divides into 1.5 TB/s to give the tokens per second you want. The MoE moves both numbers at once, in opposite directions, and the usage matrix tells you whether the extra memory bought anything.

## Read next

1. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", Shazeer, 2017. The modern MoE layer, noisy top-$k$ gating, and the first load-balancing losses.
2. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", Lepikhin, 2020. Top-2 routing, expert capacity, and the all-to-all in a real distributed system.
3. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", Fedus, 2021. Top-1 routing, the $f_e P_e$ balance loss, capacity factor, and the case for many cheap experts.
4. "ST-MoE: Designing Stable and Transferable Sparse Expert Models", Zoph, 2022. The router z-loss, fp32 routing, and what goes wrong at scale.
5. "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", Gale, 2022. Dropless MoE as block-sparse matmul, and why capacity limits are a systems artifact.
6. "Mixtral of Experts", Jiang, 2024. The 8x7B configuration counted in Lab 11 and this chapter, with routing analyses by domain.
7. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", Dai, 2024. Fine-grained and shared experts, with the combinatorial argument.
8. "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", Wang, 2024, and the DeepSeek-V3 technical report, DeepSeek-AI, 2024. The per-expert bias on selection scores and its use at 671B total, 37B active.
9. "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", Komatsuzaki, 2022. Copying the dense MLP into every expert and what it saves.
