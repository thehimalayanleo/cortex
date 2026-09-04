---
title: "Lab 12: Optimizers and schedules"
kind: permanent
topics: [lab]
chapter: 12
station: pretrain
recipe: recipes/optim_bench.py
reading_time: 55 min
---

## What you will be able to do

1. Derive Adam and AdamW from the two moment estimates, including why the bias correction has the form it has and why decoupled decay is not the same as L2 regularization.
2. Choose $\beta_2$, $\epsilon$, warmup, clipping, and the decay set for a transformer pretraining run and explain each choice in terms of a failure it prevents.
3. Implement cosine and warmup-stable-decay schedules and say what the loss does during a cooldown and why.
4. Explain, from the size of an Adam update, why the hidden-layer learning rate should shrink with width, and apply the muP transfer rule.
5. Implement Muon (orthogonalized momentum through Newton-Schulz), say what it changes about the update, and run the AdamW-versus-Muon benchmark on the 5090.

## The idea in one paragraph

Every optimizer in this chapter takes a noisy gradient and turns it into a step. Plain gradient descent uses the gradient as is, so the step is as large as the gradient and as noisy. Momentum averages the gradient over time to cut the noise. Adam additionally divides each coordinate by a running estimate of its own scale, so that every parameter moves by about the learning rate per step no matter how big or small its gradients are. AdamW shrinks the weights toward zero on the side, outside the normalization. The schedule sets how big the step is over the course of training: small at the start while the estimates are poor, large in the middle, and decaying at the end to let the parameters settle out of the noise. Muon changes what "normalize" means for a weight matrix: instead of dividing each entry by its own scale, it replaces the whole update matrix by the nearest orthogonal matrix, so that every direction in the matrix moves by the same amount.

## The math

### Setup

Let $\theta \in \mathbb{R}^p$ be the parameters, $\mathcal{L}(\theta)$ the expected loss, and $g_t = \nabla \mathcal{L}_{B_t}(\theta_{t-1})$ the gradient on minibatch $B_t$ at step $t$, an unbiased but noisy estimate of $\nabla \mathcal{L}(\theta_{t-1})$. All updates below are per coordinate unless a matrix is named explicitly.

### SGD with momentum

$$v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta v_t.$$

Unrolling, $v_t = \sum_{i=1}^{t} \mu^{t-i} g_i$. If the gradient were constant at $g$, the buffer would converge to $g / (1 - \mu)$, so the effective step is $\eta / (1 - \mu)$ times the gradient: at $\mu = 0.9$ momentum multiplies the learning rate by 10 in the steady state, which is why learning rates quoted for SGD with momentum are not comparable with those quoted without it. Nesterov momentum evaluates the gradient after a provisional momentum step, which in the common implementation becomes $\theta_t = \theta_{t-1} - \eta (g_t + \mu v_t)$; it damps the overshoot slightly and is what Muon uses inside.

The weakness for transformers: SGD moves each coordinate by $\eta$ times its gradient, and gradient scales differ by orders of magnitude between the embedding table (sparse, large per-row), the attention projections, and the layer norm gains. One $\eta$ cannot serve them all.

### Adam, derived

The goal is a step whose size per coordinate does not depend on the gradient's scale. Take two exponential moving averages, of the gradient and of its square:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,$$

with $m_0 = v_0 = 0$. Unroll the first:

$$m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i.$$

Suppose for the moment that the gradients are stationary with mean $\mathbb{E}[g]$. Then

$$\mathbb{E}[m_t] = (1 - \beta_1)\, \mathbb{E}[g] \sum_{i=1}^{t} \beta_1^{t-i} = (1 - \beta_1)\, \mathbb{E}[g]\, \frac{1 - \beta_1^t}{1 - \beta_1} = (1 - \beta_1^t)\, \mathbb{E}[g].$$

The estimator is biased toward zero by the factor $1 - \beta_1^t$, which starts at $1 - \beta_1$ and goes to 1. Dividing by it gives the corrected estimate $\hat m_t = m_t / (1 - \beta_1^t)$, and the same argument gives $\hat v_t = v_t / (1 - \beta_2^t)$. The update is

$$\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Why the correction matters in practice, not just in principle: without it, the first step is $\eta\, (1 - \beta_1) g_1 / \sqrt{(1 - \beta_2) g_1^2} = \eta\, (1 - \beta_1) / \sqrt{1 - \beta_2}$ in magnitude. At $\beta_1 = 0.9, \beta_2 = 0.999$ that is $3.16 \eta$, a step three times larger than any step in the steady state, taken at initialization where the landscape is least forgiving. At $\beta_2 = 0.95$ it is $0.45 \eta$. The correction makes the first step exactly $\eta\, \mathrm{sign}(g_1)$.

The steady-state behaviour is what makes Adam usable for transformers. By Jensen's inequality $\mathbb{E}[g]^2 \le \mathbb{E}[g^2]$, so $|\hat m_t| / \sqrt{\hat v_t}$ is at most about 1, and every coordinate moves at most about $\eta$ per step regardless of its gradient scale. The ratio $\hat m / \sqrt{\hat v}$ is a signal-to-noise estimate: near 1 where the gradient is consistent, near 0 where it flips sign from batch to batch. So $\eta$ is a step size in parameter units, and a learning rate of $6 \times 10^{-4}$ means "no parameter moves more than about $6 \times 10^{-4}$ per step".

### AdamW: decoupled decay

Weight decay as L2 regularization adds $\frac{\lambda}{2} \|\theta\|^2$ to the loss, so the gradient becomes $g_t + \lambda \theta_{t-1}$, and that sum is what Adam normalizes. The decay a coordinate feels is then $\eta \lambda \theta / (\sqrt{\hat v} + \epsilon)$: coordinates with large gradient variance are decayed less. That is backwards; the coordinates with the noisiest gradients are the ones you most want regularized. Loshchilov and Hutter (2017) moved the decay out of the normalized term:

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \right) = (1 - \eta\lambda)\, \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Each step multiplies the weight by $(1 - \eta\lambda)$. Two consequences you can compute. First, the decay has a timescale: after $k$ steps at constant $\eta$ the weight has been shrunk by $(1 - \eta\lambda)^k \approx e^{-\eta\lambda k}$, so the memory of the initialization decays over $\tau = 1 / (\eta \lambda)$ steps. At $\eta = 6 \times 10^{-4}$ and $\lambda = 0.1$, $\tau \approx 16{,}700$ steps. A run shorter than $\tau$ never reaches the regime the decay was set for, and a schedule that sends $\eta \to 0$ freezes the decay along with the updates. Second, there is an equilibrium norm. Model the normalized update as a random vector with unit root-mean-square entries and direction uncorrelated with $\theta$. Then per coordinate $\mathbb{E}[\theta_t^2] = (1 - \eta\lambda)^2\, \mathbb{E}[\theta_{t-1}^2] + \eta^2$, and at the fixed point $\mathbb{E}[\theta^2] \approx \eta^2 / (2 \eta \lambda) = \eta / (2\lambda)$, so the root-mean-square weight settles near $\sqrt{\eta / (2\lambda)}$; with the numbers above, $0.055$. Weight decay in AdamW is therefore not really a regularizer of the loss; it is a controller for the weight norm, and through the norms' scale invariance, for the effective learning rate.

### $\epsilon$ and $\beta_2$ in practice

$\epsilon$ is documented as a numerical guard, and at $10^{-8}$ it looks harmless. It is not harmless when $\sqrt{\hat v}$ is comparable to it, and $\sqrt{\hat v}$ is the root-mean-square gradient per coordinate, which for a large model with a mean-reduced loss can be $10^{-7}$ or smaller in some layers. Where $\sqrt{\hat v} \ll \epsilon$, the update degrades to $\eta \hat m / \epsilon$, plain momentum SGD with learning rate $\eta / \epsilon$, and those coordinates move much more slowly than the rest. Wortsman et al. (2023) document this as a scale-dependent instability. The fix is to log the root-mean-square gradient per parameter group and confirm it is at least a hundred times $\epsilon$; if not, lower $\epsilon$ (values down to $10^{-15}$ are used) or make it relative to the parameter scale as Adafactor does.

$\beta_2$ sets the memory of the second-moment estimate: an EMA with coefficient $\beta_2$ averages over about $1 / (1 - \beta_2)$ steps, 1000 at $0.999$ and 20 at $0.95$. Suppose the gradient's scale jumps (a new data source enters the mixture, a bad batch, a sharp region). $\hat v$ lags, the ratio $\hat m / \sqrt{\hat v}$ exceeds 1 for as many steps as $\hat v$ takes to catch up, and the model takes steps larger than $\eta$ in a direction it has not verified. With $\beta_2 = 0.999$ that lasts hundreds of steps and is the classic loss spike. With $\beta_2 = 0.95$ it lasts a few dozen. The cost of the smaller $\beta_2$ is a noisier denominator, which is affordable because the batch is large. Large-model recipes (GPT-3, Llama) use $\beta_1 = 0.9$, $\beta_2 = 0.95$, and the Marin reference config's `AdamConfig(learning_rate=6e-4, weight_decay=0.1)` sits in the same family.

### Schedules

Warmup: $\eta_t = \eta_{\max}\, t / W$ for $t < W$. Two things it protects against. Adam's moment estimates are unreliable for the first tens of steps even with bias correction (a mean of five gradients is not a mean), and the landscape at initialization is sharp along a few directions; a full-size step early can put the run in a basin it never leaves. Post-norm models (see Lab 11) also need it for the gradient-imbalance reason. Warmup lengths from a few hundred to a few thousand steps are common, and the cost of a longer warmup is a few percent of the token budget.

Cosine decay from $W$ to $T$:

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - W}{T - W}\right)\right).$$

It requires you to know $T$ at the start. Stop early and you never decayed; continue past $T$ and you are training at $\eta_{\min}$.

Warmup-stable-decay (WSD): warm up, hold $\eta_{\max}$ for most of training, then decay to zero over the last fraction $D$ of steps, linearly or with $1 - \sqrt{\cdot}$ shape. The decay phase is the cooldown you watch in the midtrain station. Two facts from Hägele et al. (2024) drive its adoption. A WSD run with a decay of about 10 to 20 percent of the steps matches a cosine run of the same length, and because the stable phase has no fixed end, you can branch a cooldown off any checkpoint, which makes a single long run yield a scaling-law's worth of endpoints and makes the "how many tokens" decision reversible.

Why the loss drops sharply during decay: at constant $\eta$ the iterate does not sit at a minimum; it bounces in a region whose radius scales with $\eta$ times the gradient noise. Decaying $\eta$ shrinks that region, which is the same thing iterate averaging does. The cooldown dip is mostly noise being removed, not new structure being learned, which is why a very short decay recovers most of it and why a model taken from the stable phase without decay looks worse than it is.

### Gradient clipping

Compute the global norm over all parameters, $\|g\|_2 = \sqrt{\sum_{\text{all}} g_i^2}$, and rescale:

$$g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right).$$

With $c = 1$ this leaves ordinary steps alone and shrinks the rare batch whose gradient is an outlier. Clip after gradient accumulation, once, on the accumulated gradient; clipping each micro-batch separately changes the objective. Log the pre-clip norm every step: it rises before a loss spike, and if the clip fires on most steps then $c$ is acting as a learning-rate multiplier ($\eta c / \|g\|$) and you should say so or raise it.

### Which parameters get weight decay

Decay the two-dimensional weight matrices. Do not decay norm gains and biases: a gain that is followed by a residual add and another norm has a scale that barely affects the function, so decaying it only changes the effective learning rate of the layer, and decaying a bias pulls it toward a value with no meaning. The embedding table is two-dimensional and codebases disagree; nanoGPT decays it, many production configs do not. Whatever you choose, do it explicitly in parameter groups:

```python
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
```

### muP in outline

Under the standard parametrization (weights initialized with variance $1 / \text{fan\_in}$, one learning rate for everything), the best learning rate moves as you change the width $d$, so every width needs its own sweep. Yang et al. (2022) showed how to parametrize so that the optimum stays put, and the Adam version of the argument fits in a paragraph.

An Adam update to a hidden matrix $W \in \mathbb{R}^{d \times d}$ has entries of size about $\eta$, with sign given by the gradient. The gradient of a loss with respect to $W$ is a sum of outer products $\delta x^\top$ of backpropagated errors and inputs, so the update is aligned with the inputs it was computed on. Apply the updated matrix to a typical input $x$ with $O(1)$ entries:

$$(\Delta W x)_i = \sum_{j=1}^{d} \Delta W_{ij} x_j.$$

Because $\Delta W_{ij}$ is correlated with $x_j$ (it was built from inputs like $x$), the $d$ terms add coherently and the sum is $O(\eta d)$, not the $O(\eta \sqrt d)$ you would get from independent terms. So the change in every pre-activation grows linearly with width at fixed $\eta$. To keep the feature change $O(1)$ as $d$ grows you need $\eta_{\text{hidden}} \propto 1 / d$. That is the transfer rule: tune $\eta$ at a base width $d_0$, then use $\eta_{\text{hidden}}(d) = \eta_0\, d_0 / d$ for the hidden matrices. Two companion rules follow from the same coherence argument. The output layer's initialization or multiplier is scaled by $1 / d$ so that logits stay $O(1)$ as the hidden state's coherent components grow, and attention logits are divided by $d_h$ rather than $\sqrt{d_h}$, because after training $q$ and $k$ are correlated and their dot product is $O(d_h)$. Embedding learning rates stay constant with width because each row is updated by only its own token's gradients. The rule transfers the learning rate across width at fixed depth; transfer across depth needs a separate scaling of the residual branches and is a more recent result.

### Muon: orthogonalized momentum

For a weight matrix $W \in \mathbb{R}^{m \times n}$ with gradient $G$, Adam treats the $mn$ entries as unrelated scalars. Muon (Jordan, 2024) asks what the best update is if you measure its size by the spectral norm, the largest factor by which it can stretch any input:

$$\Delta W^\star = \arg\min_{\|\Delta W\|_2 \le 1} \langle G, \Delta W \rangle.$$

Write $G = U \Sigma V^\top$. By von Neumann's trace inequality, $|\langle G, \Delta W \rangle| \le \sum_i \sigma_i(G)\, \sigma_i(\Delta W) \le \sum_i \sigma_i(G)$ when all singular values of $\Delta W$ are at most 1, with equality at $\Delta W = -U V^\top$. So the steepest-descent direction under the spectral norm is the gradient with every singular value set to 1: same singular vectors, flat spectrum. Directions the gradient barely touches (small $\sigma_i$) are amplified to the same size as the dominant ones. The claim, borne out empirically, is that in a transformer these rare directions carry useful signal that Adam's per-coordinate scaling does not recover.

Muon applies this to the momentum buffer rather than the raw gradient:

$$M_t = \mu M_{t-1} + G_t, \qquad O_t = \mathrm{Orth}(M_t), \qquad W_t = W_{t-1} - \eta\, s\, O_t,$$

where $\mathrm{Orth}(M) = U V^\top$ is the polar factor and $s$ is a scale discussed below.

Computing $U V^\top$ by SVD every step is too slow. Newton-Schulz iteration computes it with matrix multiplies. The key fact: for any odd polynomial $p(X) = a X + b (X X^\top) X + c (X X^\top)^2 X$, writing $X = U \Sigma V^\top$ gives $X X^\top = U \Sigma^2 U^\top$ and therefore $p(X) = U\, p(\Sigma)\, V^\top$. The polynomial acts on the singular values individually and leaves the singular vectors alone. First normalize $X_0 = M / \|M\|_F$ so that every singular value lies in $(0, 1]$ (since $\sigma_{\max} \le \|M\|_F$). Then iterate $X_{k+1} = p(X_k)$ with a $p$ that pushes values in $(0, 1]$ toward 1. The textbook choice $p(\sigma) = \frac{3}{2}\sigma - \frac{1}{2}\sigma^3$ has a stable fixed point at 1 but multiplies a small $\sigma$ by only 1.5 per iteration, so a badly conditioned $M$ needs many iterations. Jordan's coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ have $p'(0) = 3.4445$, so small singular values grow 3.4 times per iteration, and five iterations bring every value into a band around 1 (roughly 0.7 to 1.2 in practice) without converging exactly. For an optimizer a flat-ish spectrum is all that was wanted, and five iterations in bf16 are cheap.

The scale $s$: $U V^\top$ has Frobenius norm $\sqrt{\min(m, n)}$, so its entries have root-mean-square $1 / \sqrt{\max(m, n)}$, which is width-dependent and much smaller than Adam's near-unit normalized update. Two conventions exist. The original multiplies by $\sqrt{\max(1, m/n)}$; the Moonshot variant (Liu et al., 2025) multiplies by $0.2 \sqrt{\max(m, n)}$ so that the update's root-mean-square entry is about 0.2, matching what AdamW's normalized update typically is, which lets you reuse an AdamW learning rate and weight decay for the Muon groups. Moonshot also added decoupled weight decay to Muon, because at scale the weight norms otherwise grow without bound and the effective learning rate of the norm-followed layers collapses.

What the cost is: per matrix per step, five iterations each with one $m \times m$ by $m \times n$ product, one $m \times m$ square, and one more $m \times m$ by $m \times n$ product (with $m \le n$ after transposing). For Llama-3 8B's $4096 \times 14336$ MLP matrices that is about $2 \cdot 4096^2 \cdot 14336 \cdot 2 + 2 \cdot 4096^3 \approx 1.1 \times 10^{12}$ FLOPs per iteration, $5.5 \times 10^{12}$ per step, against the $6 \times 4096 \times 14336 \times (\text{tokens per step})$ the matrix costs in forward and backward; at one million tokens per step that is $3.5 \times 10^{14}$, so the orthogonalization is under 2 percent overhead. At tiny batches the ratio flips.

What Muon does not do, and where it is known to struggle. It is defined only for two-dimensional hidden weights; embeddings, the unembedding, norm gains, and biases are trained with AdamW alongside, so a Muon run is always a two-optimizer run. Fused QKV weights should be orthogonalized per projection (or per head group), not as one wide matrix, because the spectral-norm argument is about the map each projection implements. Convolution kernels are flattened to two dimensions. The update ignores per-coordinate scale entirely, which is why it is wrong for embeddings, whose rows have wildly different gradient statistics. Under sharded data parallelism the full matrix must be gathered for the Newton-Schulz step, which costs communication that Adam does not need. And the evidence base is smaller than Adam's: the modded-nanoGPT speedrun results and the Moonlight report (a 16B-parameter MoE trained with Muon, reporting better compute efficiency than AdamW at matched tokens) are the main public data points, and momentum, learning rate, and the scale convention still need re-tuning when you move to a new model size.

## Build it small

A from-scratch AdamW that is checked against `torch.optim.AdamW`, a Newton-Schulz orthogonalizer checked by its singular values, and a Muon step, all run on the same toy regression.

```python
import torch, torch.nn as nn

def adamw_step(p, g, st, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    st["t"] += 1; t = st["t"]
    st["m"].mul_(b1).add_(g, alpha=1 - b1)               # first moment
    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)        # second moment
    m_hat = st["m"] / (1 - b1 ** t)                      # bias correction
    v_hat = st["v"] / (1 - b2 ** t)
    p.mul_(1 - lr * wd)                                  # decoupled decay
    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

def newton_schulz(M, steps=5):                           # M: (m, n) -> approx polar factor U V^T
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (M.norm() + 1e-7)                            # singular values now in (0, 1]
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X              # odd polynomial acts on singular values
    return X.T if tall else X

def muon_step(p, g, st, lr, mu=0.95):
    st["m"].mul_(mu).add_(g)
    O = newton_schulz(st["m"])
    p.add_(O, alpha=-lr * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)

def make_data(n=4096, d=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    W1, W2 = torch.randn(d, 64, generator=gen), torch.randn(64, 1, generator=gen)
    return X, torch.tanh(X @ W1) @ W2 / 8

def train(step_fn, lr, steps=300, seed=0):
    torch.manual_seed(seed)
    X, Y = make_data()
    net = nn.Sequential(nn.Linear(32, 128, bias=False), nn.Tanh(), nn.Linear(128, 1, bias=False))
    state = [dict(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p)) for p in net.parameters()]
    for i in range(steps):
        idx = torch.randint(0, len(X), (256,))
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
        grads = torch.autograd.grad(loss, list(net.parameters()))
        with torch.no_grad():
            for p, g, st in zip(net.parameters(), grads, state):
                step_fn(p, g, st, lr)
    with torch.no_grad():
        return ((net(X) - Y) ** 2).mean().item()

if __name__ == "__main__":
    # 1) my AdamW matches torch.optim.AdamW to float precision
    torch.manual_seed(0); w = torch.randn(16, 8); g = torch.randn(16, 8)
    ref = w.clone().requires_grad_(True); opt = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    st = dict(t=0, m=torch.zeros_like(w), v=torch.zeros_like(w))
    for _ in range(20):
        ref.grad = g.clone(); opt.step(); adamw_step(w, g, st, 1e-2)
    print("adamw matches torch:", torch.allclose(w, ref.detach(), atol=1e-6))
    # 2) Newton-Schulz returns a near-orthogonal matrix
    sv = torch.linalg.svdvals(newton_schulz(torch.randn(64, 32)))
    print(f"NS singular values in [{sv.min():.2f}, {sv.max():.2f}]")   # a band around 1, not exactly 1
    # 3) same toy regression, three optimizers, each at its best lr from a coarse sweep
    sgd = lambda p, g, st, lr: p.add_(g, alpha=-lr)
    for name, fn, lr in (("sgd", sgd, 0.3), ("adamw", adamw_step, 3e-2), ("muon", muon_step, 3e-2)):
        print(f"{name:6s} final mse {train(fn, lr):.5f}")
```

Expected output: `adamw matches torch: True`, a singular-value band such as `[0.68, 1.09]` (the exact numbers depend on the random matrix; the point is that they are not all 1.00), and three final losses, which with this seed are about `0.226` for SGD, `0.081` for AdamW, and `0.038` for Muon. The learning rates were each picked from a coarse sweep (SGD diverges at 1.0, AdamW is best at 0.03, Muon is best at 0.03 and degrades at 0.3). Do not read the ranking as a result about language models; it is a two-layer tanh regression. Read it as a check that the three mechanics are implemented correctly, and then vary the sweep yourself to feel how wide each optimizer's good region is.

## Build it real

The recipe `recipes/optim_bench.py` trains a nanoGPT-scale decoder (the Lab 11 block, 12 layers, $d = 768$, 12 heads, about 124M parameters with a GPT-2 tokenizer) on a tokenized text shard and compares AdamW against Muon at a matched token budget. In the browser, the pretrain station runs the same loop on the 2-layer, width-48 character model; you can watch its learning-rate curve and loss on the same axes the recipe logs.

Data: a tokenized `.bin` of `uint16` GPT-2 token ids, produced the nanoGPT way from any corpus you have (a FineWeb-Edu sample is the usual choice); the script takes `--data path/to/train.bin --val path/to/val.bin`. Model: `--n-layer 12 --d 768 --n-head 12 --seq-len 1024`. Optimizer: `--optimizer adamw|muon`, with `--lr`, `--wd`, `--beta1`, `--beta2`, `--eps` for the AdamW groups and `--muon-lr`, `--muon-momentum`, `--muon-scale keller|moonshot` for the Muon groups (hidden matrices only; embeddings, head, and norms always use AdamW). Schedule: `--schedule cosine|wsd --warmup 500 --decay-frac 0.2`. Batch: `--tokens-per-step 524288 --micro-batch 32`, which the script turns into gradient accumulation; `--clip 1.0`; `--steps`; `--seed`; `--out runs/<name>` for a JSONL log and checkpoint.

What to watch in the log, every step: training loss, learning rate, pre-clip gradient norm and whether the clip fired, root-mean-square gradient per parameter group (to check $\epsilon$), update-to-weight ratio per group (the root-mean-square of $\Delta\theta$ over the root-mean-square of $\theta$; a common rule of thumb is around $10^{-3}$ in the stable phase, and a group far outside that is under- or over-trained), weight norms per group (should plateau under decay, per the equilibrium above), tokens per second, and peak memory. Validation loss every 100 steps on a fixed 2M-token slice.

Time: use $6ND$. A 124M model on 200M tokens is $6 \times 1.24 \times 10^8 \times 2 \times 10^8 = 1.5 \times 10^{17}$ FLOPs. If you assume the card sustains $4 \times 10^{13}$ FLOP/s on this model (about 40 percent of a $10^{14}$ bf16 peak; measure it, the script prints it), that is about 3700 seconds, roughly an hour per run. A billion tokens is five hours. An A/B with two optimizers, three learning rates each, and two seeds at 200M tokens is a twelve-hour job; run it overnight with `--steps` chosen from the token budget divided by tokens per step (200M / 0.5M = 381 steps, so 400 steps with a 40-step warmup). Memory for the 124M model at micro-batch 32 and 1024 tokens is a few GB; the card is not the constraint here.

## How it goes wrong

Loss spike after a data-mixture change. Symptom: the loss climbs sharply within a few hundred steps of a new source entering the mix (the midtrain station shows exactly this event), sometimes recovers, sometimes does not. Cause: $\beta_2 = 0.999$ makes $\hat v$ lag the new gradient scale for hundreds of steps, so updates exceed $\eta$. Fix: $\beta_2 = 0.95$, global-norm clipping at 1, and a short re-warmup at the mixture boundary.

Divergence in the first hundred steps. Symptom: loss goes to NaN before the warmup ends, or a run that is stable at $\eta = 3 \times 10^{-4}$ dies at $6 \times 10^{-4}$. Cause: warmup too short for the model (post-norm needs more), or no warmup with Adam so the first steps use moment estimates from three gradients. Fix: warmup to at least a few hundred steps and check the pre-clip gradient norm in the first steps; it should fall, not rise.

Frozen weights in bf16. Symptom: the loss plateaus early and the update-to-weight ratio for some groups is zero. Cause: master weights kept in bf16. A weight near 1.0 has a bf16 spacing of $2^{-8} \approx 0.0039$, and an update of $\eta \times 1 = 6 \times 10^{-4}$ is below half the spacing, so it rounds away. Fix: fp32 master weights and fp32 optimizer state; bf16 for activations and (optionally) gradients only.

$\epsilon$ dominating. Symptom: some layers learn far slower than others; lowering $\epsilon$ by three orders of magnitude changes the loss curve. Cause: root-mean-square gradients below $10^{-7}$ in those layers. Fix: log per-group gradient scale, set $\epsilon$ at least a hundred times smaller than the smallest.

Cosine with the wrong horizon. Symptom: a run stopped at 60 percent of $T$ is worse than a WSD run with a 10 percent cooldown at the same tokens; or a run resumed past $T$ makes no progress. Cause: cosine never decayed in the first case and sits at $\eta_{\min}$ in the second. Fix: WSD when the token budget is not final; if you must use cosine, treat $T$ as fixed and never resume past it without a new schedule.

Clipping per micro-batch. Symptom: gradient accumulation gives a different loss curve than the same tokens in one batch, and the clip fires constantly. Cause: each micro-batch gradient is clipped before summation, so the sum is not the batch gradient. Fix: accumulate unclipped, clip once, step.

Muon on the wrong parameters. Symptom: training with Muon is unstable or the embedding never learns. Cause: the embedding or unembedding matrix was routed to Muon (they are two-dimensional and easy to sweep up), or fused QKV was orthogonalized as one matrix. Fix: build the Muon group by name (hidden attention and MLP matrices only), split fused projections, and print the group membership.

Learning rate not re-tuned with batch size. Symptom: a config that trained well at 0.5M tokens per step trains worse at 2M with the same $\eta$. Cause: the gradient is less noisy, so the optimal $\eta$ moves (McCandlish et al., 2018), and the number of steps for the same tokens falls fourfold, which also shortens the warmup and the decay's timescale $1 / (\eta\lambda)$ in steps. Fix: re-sweep $\eta$ and set warmup and decay in tokens, not steps.

## Measure it

The primary metric is validation loss at a matched token budget, and the primary rule is that no optimizer comparison is valid at a single learning rate. Sweep $\eta$ over at least a factor of ten in half-decade steps for each optimizer, take the best, and report the whole curve: the width of the region within 0.01 nats of the best is a robustness measure that matters as much as the minimum. Run two or three seeds at the best setting; a difference smaller than the seed-to-seed spread is not a result.

Report loss against wall-clock as well as against tokens, because Muon's orthogonalization is extra compute and a two-optimizer step has more launches; at the 124M scale the overhead should be small (the FLOP estimate above), and if it is not, the Newton-Schulz step is running in fp32 or on a fused matrix it should not be.

For schedules, compare cosine to WSD at the same total steps and the same peak $\eta$. Hägele's result is that a 10 to 20 percent decay matches cosine; if your WSD run is clearly worse, the decay is too short or the peak too high for the stable phase. Save the pre-decay checkpoint and run two decay lengths from it to see the dip's dependence on length.

For muP, the check is the transfer itself: sweep $\eta$ at $d = 256$ and at $d = 768$ under the muP scaling and plot loss against $\eta_0$ (the base-width learning rate). The minima should align; under standard parametrization they will not.

For the numbers themselves: the loss you get depends on data and tokens, so there is no absolute target; what is good is a gap between optimizers that is consistent across seeds and learning rates, and an update-to-weight ratio, gradient norm, and weight norm that behave as the derivations above say they should (norms plateau under decay, gradient norm falls during warmup and stays flat, clip fires rarely).

## Exercises

1. Show that Adam with $\beta_1 = 0$ and $\epsilon = 0$ reduces to sign descent when $\beta_2 = 0$. Check: $m_t = g_t$, $v_t = g_t^2$, update $= \eta\, g_t / |g_t| = \eta\, \mathrm{sign}(g_t)$. With $\beta_2 > 0$ the denominator averages past squares and the step becomes a signal-to-noise-weighted sign.

2. Compute the decay timescale $\tau = 1 / (\eta\lambda)$ and the equilibrium weight root-mean-square for $\eta = 3 \times 10^{-4}, \lambda = 0.1$ and for $\eta = 1 \times 10^{-3}, \lambda = 0.01$. Check: 33k steps and 0.039; 100k steps and 0.22. The second setting barely decays within a 50k-step run and lets weights grow four times larger.

3. Implement the WSD schedule as a function of step and plot it against cosine with the same $W$ and $T$. Then compute the average learning rate over training for both; the check is that WSD's average is higher (at $D = 0.2$ and linear decay, $\eta_{\max}(1 - 0.5 D)$ ignoring warmup, versus about $\eta_{\max} / 2$ for cosine), which is one reason its stable phase covers more ground per step and its cooldown drops further.

4. Verify the coherence argument numerically: train a 2-layer MLP at widths 128, 512, 2048 with Adam at a fixed $\eta$ and log the root-mean-square change in the first layer's pre-activations after one step. Check: it grows roughly in proportion to width; with $\eta \propto 1/d$ it stays flat.

5. Modify `newton_schulz` to use the classical coefficients $(1.5, -0.5, 0)$ and count how many iterations a random $64 \times 32$ matrix needs before its smallest singular value exceeds 0.5. Check: the classical iteration grows a small singular value by 1.5 per step, so from $\sigma = 0.02$ it needs about $\ln(25) / \ln(1.5) \approx 8$ iterations; the quintic gets there in about 3.

6. Add decoupled weight decay to `muon_step` and re-run the toy with the Moonshot scale $0.2 \sqrt{\max(m, n)}$ and the AdamW learning rate. Check: with the scale matched, the best Muon $\eta$ should land within a factor of about 2 of the best AdamW $\eta$ instead of the different range the unscaled version needs.

## Test yourself

1. Adam's steady-state update is at most about $\eta$ per coordinate. Under what circumstances is it much larger, and for how long?

<details><summary>Answer</summary>
When the gradient scale rises faster than $\hat v$ can follow. $\hat v$ is an EMA over about $1/(1-\beta_2)$ steps, so after a jump in gradient magnitude by a factor $r$, the ratio $\hat m / \sqrt{\hat v}$ can approach $r$ and decays back over that window. At $\beta_2 = 0.999$ that is hundreds of steps of oversized updates; at $0.95$ it is tens. The bias correction does not help here; it corrects the zero initialization, not a distribution shift.
</details>

2. Without bias correction, is the first Adam step too large or too small? Give the factor for $(\beta_1, \beta_2) = (0.9, 0.95)$ and $(0.9, 0.999)$.

<details><summary>Answer</summary>
Both moments are underestimated, but the square root on the denominator makes the denominator's underestimate the larger effect when $1 - \beta_2 < (1 - \beta_1)^2$. The factor is $(1 - \beta_1) / \sqrt{1 - \beta_2}$: $0.1 / \sqrt{0.05} = 0.45$ (too small) for $\beta_2 = 0.95$ and $0.1 / \sqrt{0.001} = 3.16$ (too large) for $0.999$. Most people guess "too small" for both.
</details>

3. Weight decay with a norm-followed matrix. Explain why decaying $W$ when the next operation is RMSNorm does not regularize the function, and what it does instead.

<details><summary>Answer</summary>
RMSNorm$(cWx) = $ RMSNorm$(Wx)$ for any $c > 0$, so the function is invariant to the scale of $W$. Decay shrinks $\|W\|$, and since Adam's update has a fixed size around $\eta$ per coordinate, a smaller $\|W\|$ means each update rotates $W$ by a larger angle: the effective learning rate of that layer rises. Decay is a controller of the angular update size, and the equilibrium $\sqrt{\eta / (2\lambda)}$ is where it settles.
</details>

4. Spot the bug:

```python
for micro in batches:
    loss = model(micro) / len(batches)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step(); opt.zero_grad()
```

<details><summary>Answer</summary>
The clip runs inside the accumulation loop, on the partially accumulated gradient, so later micro-batches are clipped relative to a norm that includes earlier ones and the final gradient is not a clipped version of the batch gradient. Move the clip after the loop, before `opt.step()`.
</details>

5. Estimate, with stated assumptions, the memory of AdamW state for a 1.5B-parameter model in mixed precision, and say whether it fits on 32 GB with activations.

<details><summary>Answer</summary>
Assume fp32 master weights (4 bytes), fp32 moments (8 bytes), bf16 working weights (2 bytes), bf16 gradients (2 bytes): 16 bytes per parameter, 24 GB for 1.5B. That leaves 8 GB for activations, CUDA context, and fragmentation, which at 1024-token sequences and a small micro-batch is feasible with activation checkpointing and not without. The practical answer is that 1.5B is the edge, and 8-bit optimizer states (2 bytes for both moments) or LoRA are how you get room.
</details>

6. Why does muP scale the attention logits by $1 / d_h$ instead of $1 / \sqrt{d_h}$, when Lab 11 derived $\sqrt{d_h}$ from the variance of a dot product?

<details><summary>Answer</summary>
The $\sqrt{d_h}$ argument assumes $q$ and $k$ are independent with unit-variance entries, which is true at initialization. After training, $q$ and $k$ are correlated by construction (the model learns to make relevant pairs align), and a dot product of correlated $d_h$-dimensional vectors is $O(d_h)$. To keep logits $O(1)$ across head widths in the trained regime, muP divides by $d_h$. At the base width the two conventions differ by a constant that the tuned learning rate absorbs; the point is how the scale transfers.
</details>

7. Muon sets every singular value of the update to 1. What happens to a direction in which the momentum buffer is pure noise?

<details><summary>Answer</summary>
It gets amplified to the same size as the signal directions. Orthogonalization removes magnitude information entirely, including the information that a direction is small because it is noise. This is why Muon needs momentum before orthogonalization (the buffer averages noise down first), why it is wrong for embeddings whose rows have very different signal-to-noise, and why the practical band of learning rates is narrower than the spectral-descent story suggests.
</details>

8. A run uses WSD with a stable phase at $\eta_{\max}$. You branch a 10 percent cooldown at 40 percent of the planned tokens and again at 100 percent. The two cooled-down losses differ by 0.15 nats, and the 40 percent stable-phase loss (before cooldown) differs from the 100 percent stable-phase loss by only 0.05 nats. What does the difference between those two gaps tell you?

<details><summary>Answer</summary>
The stable-phase loss is dominated by the noise-ball term, which is the same at both points because $\eta$ is the same, so it understates the progress made between 40 and 100 percent. The cooldown removes the noise term and reveals the 0.15-nat gap that is real learning. Comparing runs by their stable-phase loss underestimates the value of more tokens; always compare after cooldown.
</details>

9. Under sharded data parallelism (ZeRO stage 2 or 3), what extra communication does Muon need per step that AdamW does not, and roughly how much for a $4096 \times 14336$ matrix in bf16 across 8 GPUs?

<details><summary>Answer</summary>
AdamW's update is elementwise, so each GPU updates its own shard. Newton-Schulz needs the full momentum matrix, so each GPU must gather the shards it does not own before the iteration (or one GPU does the iteration and scatters the result). The matrix is $4096 \times 14336 \times 2$ bytes $= 117$ MB; an all-gather moves about $7/8$ of that into each GPU, about 103 MB per matrix per step, and there are two or three such matrices per layer.
</details>

10. Spot the flaw in the argument: "AdamW's decay is $\theta \leftarrow (1 - \eta\lambda)\theta$, so during a cosine decay to zero the weight decay also goes to zero, and the final weights are under-regularized; we should keep $\lambda$ constant in absolute terms by using $\lambda / \eta_t$."

<details><summary>Answer</summary>
The decay's job is to hold the weight norm at its equilibrium against the updates, and the updates shrink with $\eta$ too, so the ratio is what the equilibrium depends on. Making the decay independent of $\eta$ would shrink the weights as the updates vanish, driving the norm down (and the effective learning rate of norm-followed layers up) exactly when the run is supposed to settle. Some implementations do scale decay with the schedule and some do not; the argument that one is "correct" fails because the equilibrium, not the per-step decay, is the object of interest.
</details>

## What will change, what will not

The derivations are durable. The bias-correction factor, the equilibrium norm under decoupled decay, the $1/(1-\beta)$ window of an EMA, the coherence argument for $\eta \propto 1/d$, the trace-inequality argument for the polar factor, and the $6ND$ time estimate are results about the update rules, not about any framework. They will let you read the next optimizer's paper and know within a page what it changes.

The habit of instrumenting the run is durable. Pre-clip gradient norm, per-group gradient scale, update-to-weight ratio, weight norm, and loss after cooldown are the quantities every failure mode above shows up in first. Whatever the optimizer, log them.

The hyperparameters will not last. $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, $\lambda = 0.1$, a 20 percent cooldown, five Newton-Schulz iterations with these coefficients, and the $0.2$ scale factor are what worked at particular scales with particular data. Each has a derivation that tells you what to re-check when the scale changes, and you should re-check rather than copy.

Adam's dominance is the thing most likely to change. It has lasted a decade because its failure modes are understood and cheap to avoid, not because it is optimal; matrix-aware methods (Shampoo, Muon, and their descendants) are the current challengers, and the durable content of that line is the idea that a weight matrix's update should be judged as a linear map, by its spectrum, rather than as a bag of scalars. Expect the specific orthogonalization and the scale conventions to be replaced; expect the spectral view to stay.

The scheduling picture will shift toward "never finish": WSD-style runs with branched cooldowns, continued pretraining, and midtraining mixtures (the station's cooldown is a small version of this) make the notion of a run with a fixed $T$ less central. The noise-ball explanation of the cooldown dip is the part that survives.

## Read next

1. "Adam: A Method for Stochastic Optimization", Kingma, 2014. The original derivation, including the bias correction and the bound on the update size.
2. "Decoupled Weight Decay Regularization", Loshchilov, 2017. Why L2 in the loss and decoupled decay differ under Adam, and the AdamW rule.
3. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", Hägele, 2024. The evidence that a short cooldown matches cosine and the branching methodology.
4. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", Yang, 2022. muP; read the intuition sections and the transfer tables before the theory.
5. "Small-scale proxies for large-scale Transformer training instabilities", Wortsman, 2023. Loss spikes, the $\epsilon$ problem, and what to log to see them coming.
6. "An Empirical Model of Large-Batch Training", McCandlish, 2018. The gradient-noise scale and why the best learning rate and batch size move together.
7. "Muon is Scalable for LLM Training", Liu, 2025. The Moonshot scaling and weight-decay additions, and the largest public Muon run.
8. "Shampoo: Preconditioned Stochastic Tensor Optimization", Gupta, 2018. The earlier matrix-aware optimizer that Muon can be read as a cheap special case of.
