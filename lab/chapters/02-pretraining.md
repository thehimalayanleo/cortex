---
title: "Lab 02: Pretraining a decoder"
kind: permanent
topics: [lab]
chapter: 2
station: pretrain
recipe: recipes/pretrain_nano.py
reading_time: 55 min
---

## What you will be able to do

- Derive the next-token cross-entropy from maximum likelihood, and say exactly which quantity the loss cannot go below and why.
- Budget a run in FLOPs from parameter count and token count, convert the budget to hours on one RTX 5090 under stated assumptions, and place the run on a Chinchilla-style scaling curve without over-reading the fitted constants.
- Set warmup, peak learning rate, cosine decay, and batch size with a reason for each number, and measure whether the batch is below or above the critical size.
- Read a loss curve: identify the uniform baseline, the unigram plateau, the power-law tail, the train-validation gap, and the signature of a spike, and know the fix for each spike cause.
- Run a nanoGPT-style model on TinyStories on the 5090 and read Marin's `train_lm` call as a specification of the same run.

## The idea in one paragraph

Pretraining is one loss, applied to every position of every window: predict the next token, and pay the negative log-probability you assigned to what actually came next. Because the loss is a log-likelihood, minimizing it is fitting a probability model to the data, and the best possible value is the entropy of the data itself, not zero. Everything else in this chapter is about spending compute efficiently on that one objective: how many parameters against how many tokens, how large a batch before extra examples stop buying fewer steps, how to raise and lower the learning rate so the optimizer neither diverges at the start nor rattles around at the end. The pretrain station in the browser trains a 2-layer, width-48, 3-head character model with exactly this loss; watch it fall from $\ln V$ toward the corpus entropy and the attention map organize into a diagonal and word-boundary stripes.

## The math

### Cross-entropy from maximum likelihood

Let $p^*$ be the unknown distribution over token sequences from which the corpus was drawn, and let $p_\theta$ be the model. A decoder factorizes a sequence by the chain rule,

$$p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}),$$

where each factor is a softmax over $V$ tokens computed from the positions before $t$. Given a corpus of $N_{\text{tok}}$ tokens, maximum likelihood chooses $\theta$ to maximize $\sum \ln p_\theta(x)$, equivalently to minimize the average negative log-likelihood per token,

$$L(\theta) = -\frac{1}{N_{\text{tok}}} \sum_{t} \ln p_\theta(x_t \mid x_{<t}).$$

This is the cross-entropy between the empirical next-token distribution and the model's. In the limit of a large corpus it converges to $\mathbb{E}_{p^*}[-\ln p_\theta(x_t \mid x_{<t})]$, and one line of algebra shows what that expectation contains. Write $p^*$ and $p_\theta$ for the conditional next-token distributions at some context; then

$$-\mathbb{E}_{p^*}[\ln p_\theta] = -\mathbb{E}_{p^*}[\ln p^*] + \mathbb{E}_{p^*}\left[\ln \frac{p^*}{p_\theta}\right] = H(p^*) + \mathrm{KL}(p^* \,\|\, p_\theta).$$

The KL divergence is non-negative and zero only when $p_\theta = p^*$, so the loss is bounded below by $H(p^*)$, the conditional entropy of the next token given its context, averaged over contexts. That is the loss floor. Natural language has a positive entropy rate; a model that reached zero loss on real text would be a bug (see the failure modes). The floor is not knowable for real corpora, but for synthetic data you can compute it: the snippet below generates text by choosing uniformly among 12 words whose average length with the trailing space is $3.75$ characters, so the entropy rate is $\ln 12 / 3.75 = 0.663$ nats per character, and you can watch the model approach that number and stop. Chinchilla-style fits (below) estimate the floor for a real corpus as the constant $E$ in the loss formula.

The gradient of the loss with respect to the logits $z$ at one position is $\partial L / \partial z_j = p_\theta(j) - \mathbb{1}[j = x_t]$: push down every token in proportion to the probability you gave it, push up the true one. Every position in every window contributes such a term, which is why a $B \times T$ batch is $BT$ training examples.

### Compute in FLOPs

A linear layer with weight $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ applied to one token costs $2 d_{\text{in}} d_{\text{out}}$ floating-point operations forward (one multiply and one add per weight), which is 2 FLOPs per parameter. The backward pass computes two matmuls of the same shape, one for the gradient with respect to the input and one for the gradient with respect to the weight, so 4 more FLOPs per parameter. Summing over all weight matrices, the cost of one token is about $6N$ FLOPs, where $N$ is the number of parameters that participate in matmuls (the unembedding counts; the input embedding lookup is a gather and costs nothing). Over $D$ tokens,

$$C \approx 6 N D.$$

Attention adds a term that does not scale with parameters. At each layer, each token forms dot products with the keys before it ($2 \cdot t \cdot d_{\text{model}}$ FLOPs at position $t$, summed over heads) and a weighted sum over values of the same size. Averaging $t$ over a causal window of length $T$ gives $T/2$, so the forward cost is about $2 T d_{\text{model}}$ per layer per token and the total with backward is about $6\, n_{\text{layer}} T d_{\text{model}}$. The per-token cost is therefore

$$c \approx 6N + 6\, n_{\text{layer}} \, T \, d_{\text{model}}.$$

Worked example, GPT-2 small: $N = 124 \times 10^6$, $n_{\text{layer}} = 12$, $d_{\text{model}} = 768$, $T = 1024$. $6N = 7.44 \times 10^8$; the attention term is $6 \times 12 \times 1024 \times 768 = 5.66 \times 10^7$, about 8 percent of $6N$. At $T = 8192$ it would be 60 percent, which is why long-context training is done as a short late stage (Lab 03). Model FLOPs utilization is the ratio of the rate you actually sustain, $c \times (\text{tokens per second})$, to the hardware's peak; it is the single number that tells you whether the training loop is healthy.

### Scaling laws, stated carefully

Hoffmann et al. (2022) fit the final loss of many runs to

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},$$

and report, for their third fitting approach, $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$. Read the form before the numbers. $E$ is the estimated floor: the loss an infinitely large model trained on infinite data would reach, which is the entropy of their data under their tokenizer plus whatever the model class cannot capture. The second term is the penalty for finite parameters; the third for finite data. Both penalties fall as power laws, which is the empirical content of the fit.

Now minimize $L$ subject to a fixed budget $C = 6ND$. Substitute $D = C / (6N)$:

$$L(N) = E + A N^{-\alpha} + B \left(\frac{6N}{C}\right)^{\beta}.$$

Set the derivative to zero: $-\alpha A N^{-\alpha - 1} + \beta B \, 6^\beta C^{-\beta} N^{\beta - 1} = 0$, so $N^{\alpha + \beta} = \dfrac{\alpha A}{\beta B \, 6^\beta} C^\beta$, giving

$$N_{\text{opt}} \propto C^{\beta / (\alpha + \beta)}, \qquad D_{\text{opt}} \propto C^{\alpha / (\alpha + \beta)}.$$

With the fitted exponents, $\beta / (\alpha + \beta) = 0.45$ and $\alpha / (\alpha + \beta) = 0.55$; their other two fitting methods give values close to $0.5$ for both. The practical summary is that parameters and tokens should grow at about the same rate, and that at the scales they studied the compute-optimal ratio came out near 20 tokens per parameter.

Three cautions. The constants are properties of one corpus, one tokenizer, and one architecture family, and they do not transfer; $E = 1.69$ says nothing about TinyStories. The exponents are what generalizes, approximately, and even they move with data quality. And "compute-optimal" means optimal for training cost alone; if the model will be served, the cost of inference favors a smaller model trained on far more tokens than 20 per parameter, which is why Llama-family models are trained well past that ratio (Sardana and Frankle, 2023, work this out).

Worked example for the 5090. Take $N = 124 \times 10^6$ and 20 tokens per parameter, $D = 2.5 \times 10^9$. Then $C = 6 \times 1.24 \times 10^8 \times 2.5 \times 10^9 = 1.86 \times 10^{18}$ FLOPs. Assume the card sustains $4 \times 10^{13}$ FLOP/s on this workload, which is 40 percent utilization of a $10^{14}$ FLOP/s dense BF16 rate (an assumption; the recipe prints the measured value). Then the run takes $1.86 \times 10^{18} / (4 \times 10^{13}) = 4.65 \times 10^{4}$ seconds, about 13 hours. A 30M-parameter model at the same ratio needs $D = 6 \times 10^8$, $C = 1.08 \times 10^{17}$, and about 45 minutes at the same rate. Plugging the 124M point into the fitted formula gives $1.69 + 406.4 / (1.24 \times 10^8)^{0.34} + 410.7 / (2.5 \times 10^9)^{0.28} = 1.69 + 0.72 + 0.96 = 3.37$ nats; the arithmetic is the exercise, the value belongs to their corpus.

### Learning-rate schedule

The optimizer is AdamW. With gradient $g_t$, first and second moment estimates

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,$$

bias-corrected as $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$, the update is

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right),$$

with the weight decay $\lambda$ applied directly to the parameters rather than added to the gradient (the "decoupled" in AdamW). Typical values for decoders are $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$, gradient norm clipped to $1$. The choice $\beta_2 = 0.95$ rather than the Adam default $0.999$ matters: the second moment is a running average over roughly $1 / (1 - \beta_2)$ steps, and with a 1,000-step memory a sudden increase in gradient magnitude is divided by a stale, small $\sqrt{\hat v}$ and produces a huge step. A 20-step memory tracks the change.

The learning rate $\eta_t$ follows warmup then cosine decay. For $t < T_w$,

$$\eta_t = \eta_{\max} \frac{t}{T_w},$$

and for $T_w \le t \le T_{\text{total}}$,

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{t - T_w}{T_{\text{total}} - T_w}\right)\right),$$

with $\eta_{\min}$ commonly $0.1 \eta_{\max}$. Warmup exists because at initialization the moment estimates are built from a handful of gradients and the attention logits are far from their eventual scale, so full-size steps in the first hundred iterations move the weights into regions the optimizer cannot recover from. Decay exists because of the noise ball: with a constant learning rate the iterate does not converge to a minimum but fluctuates around it with a spread proportional to $\eta$, and the loss carries an excess term proportional to $\eta$ as well. Lab 03 derives that term and shows it as the drop you get from a cooldown. The cosine shape is a convention; a constant rate followed by a short linear cooldown ("warmup-stable-decay") reaches similar final losses and has the advantage that any intermediate checkpoint can be cooled down and evaluated, which is the arrangement Lab 03 uses.

The peak rate is the one hyperparameter that must be swept. It scales down with width; for a 124M model with this optimizer, values around $6 \times 10^{-4}$ are standard, and the Marin call below uses exactly that.

### Batch size and the critical batch

Let $G$ be the true gradient of the expected loss and let a batch of $B$ examples produce an estimate $\hat G$ with $\mathbb{E}[\hat G] = G$ and covariance $\Sigma / B$, where $\Sigma$ is the per-example gradient covariance. On a locally quadratic loss with Hessian $H$, a step of size $\eta$ along $-\hat G$ decreases the expected loss by

$$\Delta L(\eta) = \eta |G|^2 - \frac{\eta^2}{2} \left( G^\top H G + \frac{\mathrm{tr}(H \Sigma)}{B} \right).$$

Maximizing over $\eta$ gives

$$\Delta L_{\text{opt}} = \frac{\Delta L_{\max}}{1 + B_{\text{noise}} / B}, \qquad B_{\text{noise}} = \frac{\mathrm{tr}(H \Sigma)}{G^\top H G},$$

where $\Delta L_{\max} = |G|^4 / (2 G^\top H G)$ is the decrease a noiseless step would achieve. This is the gradient noise scale of McCandlish et al. (2018). A batch of size $B$ achieves a fraction $1 / (1 + B_{\text{noise}} / B)$ of the ideal step. So the number of steps to reach a target loss is $S = S_{\min}(1 + B_{\text{noise}} / B)$, the number of examples is $E = SB = S_{\min}(B + B_{\text{noise}})$, and with $E_{\min} = S_{\min} B_{\text{noise}}$ the two obey

$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1.$$

Below $B_{\text{noise}}$, doubling the batch nearly halves the steps and costs almost no extra examples. Above it, doubling the batch barely reduces steps and doubles the examples burned. The critical batch is the knee, and it is not a constant: as training proceeds $|G|$ shrinks faster than $\Sigma$, so $B_{\text{noise}}$ grows, which is the argument for ramping the batch up during a run. You can estimate it without the Hessian by measuring $|\hat G|^2$ at two batch sizes and solving $\mathbb{E}|\hat G_B|^2 = |G|^2 + \mathrm{tr}(\Sigma) / B$ for $|G|^2$ and $\mathrm{tr}(\Sigma)$, then taking $B_{\text{simple}} = \mathrm{tr}(\Sigma) / |G|^2$. For 124M-class decoders on web text a batch of about half a million tokens is the customary setting; on one GPU you reach it with gradient accumulation, and the recipe's `--batch_tokens` argument is that number.

### Reading the loss curve

At step zero a model with small output weights predicts the uniform distribution and the loss is $\ln V$; the pretrain station draws this as a horizontal reference line. In the first few hundred steps the loss drops to the unigram entropy, because learning the token frequencies needs only the output bias. Then it falls through the bigram entropy as the model learns to use the previous token. After that it follows a slow power law in steps, which looks like a straight line on log-log axes and like a curve that is always flattening on linear axes. There is no clean plateau, so "it has converged" is never true of a pretraining run; it has run out of budget.

The validation loss should track the training loss closely for the first epoch. A gap that opens is memorization, and on a small corpus it opens as soon as the second epoch begins. A spike is a sudden jump of the training loss by a large fraction of its value over a few steps, usually followed by a slow recovery, sometimes by divergence. Its causes and fixes are in the failure modes. Reading the curve with the learning rate overlaid, as the station does, is the habit to build: most spikes occur just after warmup ends, when the rate is at its peak and the second-moment estimate has not yet caught up with the loss landscape.

## Build it small

A 2-layer causal transformer on synthetic text whose true entropy rate you can compute, with the uniform and unigram reference lines printed alongside. CPU, about half a minute.

```python
# Lab 02, build it small: a 2-layer causal transformer on synthetic text, with the
# two reference lines every loss curve should be read against: ln(V) and the unigram entropy.
import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "park", "big", "red", "a"]
gen = torch.Generator().manual_seed(1)
text = " ".join(words[i] for i in torch.randint(len(words), (4000,), generator=gen).tolist())
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data)); train, val = data[:n], data[n:]

counts = torch.bincount(train, minlength=V).float(); p = counts / counts.sum()
H1 = -(p[p > 0] * p[p > 0].log()).sum().item()          # unigram entropy: a context-free model cannot beat this
floor = math.log(len(words)) / (sum(map(len, words)) / len(words) + 1)   # true entropy: ln(12) nats per word, spread over chars
print(f"V={V}  ln V={math.log(V):.3f}  unigram {H1:.3f}  true floor {floor:.3f} nats/char")

T, B, d, heads, layers = 32, 32, 48, 3, 2

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, mask):
        h = s.ln1(x)
        x = x + s.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        s.blocks = nn.ModuleList(Block() for _ in range(layers))
        s.ln, s.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
    def forward(s, idx):
        t = idx.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)   # True = blocked
        x = s.tok(idx) + s.pos(torch.arange(t))
        for b in s.blocks: x = b(x, mask)
        return s.head(s.ln(x))

def batch(src):
    i = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[j:j + T] for j in i]); y = torch.stack([src[j + 1:j + T + 1] for j in i])
    return x, y

model = GPT(); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
steps, warmup = 600, 50
for step in range(1, steps + 1):
    lr = 3e-3 * step / warmup if step < warmup else 3e-3 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
    for g in opt.param_groups: g["lr"] = lr
    x, y = batch(train)
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))    # mean over B*T next-char predictions
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 100 == 0 or step == 1:
        with torch.no_grad():
            xv, yv = batch(val); vl = F.cross_entropy(model(xv).view(-1, V), yv.view(-1)).item()
        print(f"step {step:4d}  lr {lr:.2e}  train {loss.item():.3f}  val {vl:.3f}")
```

Expected output: the header line `V=17  ln V=2.833  unigram 2.455  true floor 0.663 nats/char`, a first-step loss just above $\ln V$ (about $3.07$, because the random output weights are not exactly zero), a loss near $0.81$ at step 100, and single-batch values between $0.67$ and $0.71$ from step 300 on; each printed validation value is one batch of 1,024 characters and carries a noise of a few hundredths. Averaged over 100 validation batches the final loss is about $0.69$. The model does not reach $0.663$: the first prediction in each window sees one character of context and pays about $1.1$ nats, the next few positions are also short of context, and positions past the eighth average about $0.68$, which is finite capacity and finite steps. Increase $T$ and the context-starvation part of the gap shrinks. Two things to try: set `warmup = 1` and raise the peak rate to `1e-2` to see a spike; change `diagonal=1` to `diagonal=2` in the mask so that each position can also see the next token, which is its own target, and watch the loss fall far below the floor, which is the signature of leakage.

## Build it real

`recipes/pretrain_nano.py` is a single-file trainer in the nanoGPT style that reads the `train.bin` and `val.bin` shards written by Lab 01's `curate.py`. The model is a pre-norm decoder: token embedding tied to the unembedding, learned positions by default with `--rope` to switch to rotary (which Lab 03 needs for context extension), `F.scaled_dot_product_attention` for the flash kernel, GELU MLP at four times the width, and `--n_layer --n_head --n_embd --seq_len` to size it. Training uses fused AdamW with $\beta_2 = 0.95$, weight decay $0.1$ on matrices only, clipping at $1$, bf16 autocast, `torch.compile`, and gradient accumulation to reach `--batch_tokens`. The schedule is `--warmup_steps`, `--lr`, and either `--schedule cosine --min_lr_ratio 0.1` or `--schedule wsd --cooldown_frac 0.2`, the latter being what Lab 03 continues from. Evaluation runs every `--eval_every` steps on a fixed set of validation windows chosen once, so the number is comparable across the run. Checkpoints hold weights, optimizer state, the step, and the RNG state, so a resumed run reproduces the un-interrupted one.

Two presets bracket what one 5090 can do. `--preset char-tiny` trains a character-level model (vocabulary from the shard metadata, 4 layers, width 256, $T = 256$) on TinyStories at $2^{16}$ tokens per batch; it reaches recognizable stories within minutes and is the right first run. `--preset gpt2-small` trains the 124M configuration with the GPT-2 tokenizer at $T = 1024$ and $2^{19}$ tokens per batch, and its wall time is the worked example above: the number of steps you ask for times $2^{19}$ tokens times $c$ FLOPs, divided by the rate the log reports. Memory is not the constraint at this size: weights, gradients, and two Adam moments in fp32 cost 16 bytes per parameter, about 2 GB for 124M, and activations at a micro-batch of 16 windows fit comfortably in 32 GB; if you see an out-of-memory error, halve `--micro_batch` and the accumulation doubles automatically, leaving the optimization unchanged.

The log line is `step, train loss, val loss, lr, grad norm, tokens/s, MFU`. Watch the first 200 steps for the drop from $\ln V$; watch the gradient norm, which should settle to a roughly constant value after warmup and whose sudden growth precedes a spike by a few steps; and watch MFU, which should be stable once compilation finishes and which tells you, before any loss number does, whether the run is worth its hours. Time per run: `char-tiny` in the tens of minutes; `gpt2-small` at the 20-tokens-per-parameter point in the low tens of hours under the assumption stated above, and you will want to run it for fewer steps first to see the curve.

### Marin's `train_lm`, line by line

The lab's overview station quotes the beginner tutorial from `experiments/tutorials/train_tiny_model.py`:

```python
model = train_lm(
    name="checkpoints/tiny-tinystories",
    model=llama_nano,
    datasets={tinystories_tokenized: 1.0},
    optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
    batch_size=32, seq_len=model.max_seq_len,
    num_train_steps=100,
)
```

`train_lm` returns a step in Marin's dependency graph, not a trained model; the graph executor runs it when something downstream asks for it, after the tokenization step it depends on. `name` is the step's identity and its checkpoint path, so re-running with the same name reuses the result. `model=llama_nano` is a configuration object for a Llama-shaped decoder (RMSNorm, rotary positions, SwiGLU MLP) at a size chosen to run on a CPU; the recipe's `--rope` flag is the closest the nano trainer comes to this shape. `datasets` maps tokenized caches to sampling weights; a single cache at weight $1.0$ is the pure-pretraining case, and Lab 03 changes this line and nothing else. `optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1)` sets the two values you sweep; the values you do not see (betas, warmup fraction, the decay shape, the minimum rate, the clip threshold) are fields of that config class with defaults, and the first thing to do with a new pipeline is read that class rather than assume the defaults match another framework's. `batch_size=32` is sequences per step, `seq_len` is the window, so tokens per step is their product and total tokens $D$ is that product times `num_train_steps`. With 100 steps this run is a smoke test; put your own $N$ and $D$ through $C = 6ND$ before launching anything longer.

## How it goes wrong

1. The loss sits at $\ln V$ and does not move. Either the learning rate is effectively zero (a scheduler that starts at zero and is never stepped, or a parameter group with the wrong key), or the targets are constant. Check that `lr` in the log is nonzero after warmup, and print one batch to verify that `y` is `x` shifted by one.

2. The loss falls far below any plausible entropy within a few hundred steps. The causal mask is missing or off by one, so position $t$ can see $x_{t+1}$; or the validation windows overlap the training windows; or the target was included in the input. In the toy this shows up as a loss below $0.663$. Fix the mask (`torch.triu(..., diagonal=1)` blocks strictly future positions) and split at the document level.

3. A spike after warmup, then a slow recovery. The peak rate is too high for the width, or $\beta_2 = 0.999$ let the second moment go stale, or the attention logits grew until the softmax saturated. Lower the peak by a factor of two, set $\beta_2 = 0.95$, and if spikes persist add QK-layernorm (normalize queries and keys before the dot product) or a z-loss term $10^{-4} (\ln Z)^2$ that keeps the softmax normalizer near one. Wortsman et al. (2023) show these instabilities appear at small scale if you raise the rate, so you can reproduce and fix them on the 5090.

4. `NaN` after a few thousand steps in fp16. Activations overflowed the 16-bit exponent range. Use bf16, which has the fp32 exponent range, and never fp16 without loss scaling.

5. Validation loss rises while training loss keeps falling. The corpus is small and the run has entered its second or third epoch, or the data has duplicates (Lab 01). Compute the epoch count from tokens seen divided by corpus tokens; stop or add data.

6. MFU is a fraction of what a matmul benchmark on the same card achieves. The micro-batch is too small to fill the GPU, `torch.compile` is off, attention is falling back to the math kernel because of an unsupported mask or dtype, or the data loader is on the critical path. Raise the micro-batch until memory is used, confirm the flash kernel is selected, and load shards with memory mapping.

7. The loss curve is smooth but worse than a reference at equal tokens. The batch is far above the critical size, so the run is spending examples without reducing steps; or far below it, so the noise ball is large and the cooldown has not yet removed it. Estimate $B_{\text{simple}}$ from two batch sizes and compare with `--batch_tokens`.

8. A resumed run does not reproduce the interrupted one. The RNG state or the data-loader position was not checkpointed, so the resumed run sees different batches. Checkpoint both; a correct resume gives a loss curve indistinguishable from the original.

## Measure it

The primary number is held-out loss in nats per token on a fixed validation set, reported with the token count at which it was measured, because the loss without its $D$ is not a result. Report bits per byte alongside it (Lab 01) so the number survives a tokenizer change. For reference on what a healthy small run looks like, the nanoGPT README reports its 124M reproduction on OpenWebText reaching a validation loss of about $2.85$ nats, and the released OpenAI GPT-2 checkpoint of the same size evaluating at about $3.11$ on that data; your TinyStories numbers will be much lower because the corpus is simpler, and comparing across corpora is meaningless.

The second number is the train-validation gap at the end of the run, which should be small (a few hundredths of a nat) for a single-epoch run on deduplicated data.

The third is MFU, which for a well-tuned single-GPU trainer at this size sits between 30 and 50 percent of peak; below 20 percent means something in failure mode 6 is wrong.

For downstream evaluation, `lm-eval-harness` runs on the saved checkpoint; at 124M only a few tasks are above chance (LAMBADA-style next-word accuracy is the one that moves first), and the honest use of the harness at this scale is to compare two runs, not to report absolute numbers. The sample quality check is cheap and telling: generate from a fixed prompt with temperature $1$ at several checkpoints and read them; the point where the samples become coherent stories on TinyStories is visible in the loss curve as the end of the fast phase.

## Exercises

1. Compute $c$ for a 6-layer, width-384 model at $T = 512$ with a 50k vocabulary (count the unembedding in $N$), and the hours to train it on $10^9$ tokens at an assumed $3 \times 10^{13}$ FLOP/s. Check: $N \approx 6 \times 12 \times 384^2 + 384 \times 50{,}000 \approx 2.98 \times 10^7$, $c \approx 1.86 \times 10^8$, so about $1.7$ hours.

2. Modify the toy so that $T = 128$ and report the final validation loss. Check: it moves closer to $0.663$, because fewer positions are context-starved; the remaining gap is capacity and finite training.

3. Implement the two-batch-size estimator of $B_{\text{simple}}$ in the toy and print it every 100 steps. Check: it grows over training as $|G|^2$ falls.

4. Replace the cosine schedule in the toy with a constant rate and report the final loss over 3 seeds against cosine. Check: cosine is lower by an amount comparable to the noise-ball prediction, and the gap shrinks if you lower the constant rate.

5. Derive the loss at step zero when the output weights are drawn from $\mathcal N(0, \sigma^2)$ instead of being zero, to second order in $\sigma$. Check: the expected loss is $\ln V + \frac{1}{2}\sigma^2 d \,(1 - 1/V)$ plus higher-order terms, which is why step-one losses sit slightly above $\ln V$ in the toy.

6. Run `char-tiny` on the 5090 with peak rates $\{1, 3, 10\} \times 10^{-3}$ and identify the highest rate that does not spike. Check: the gradient-norm column rises for several steps before the spike; use it as the early warning in later runs.

## Test yourself

1. A colleague argues that with enough parameters and data the pretraining loss will approach zero. Give the one-line refutation and say what number it approaches instead.

<details><summary>Answer</summary>
$L = H(p^*) + \mathrm{KL}(p^* \| p_\theta) \ge H(p^*)$, and the conditional entropy of natural language given its context is strictly positive because the next token is not determined by the past. The loss approaches $H(p^*)$, which Chinchilla-style fits estimate as the constant $E$.
</details>

2. Where does the factor of 3 between forward (2 FLOPs per parameter) and total (6) come from, and which of the two backward matmuls could you skip for the first layer?

<details><summary>Answer</summary>
Backward computes the gradient with respect to the weight ($x^\top \delta$) and with respect to the input ($\delta W^\top$), each the size of the forward matmul, giving 2 + 2 + 2 = 6. For the first matmul after the embedding lookup, the gradient with respect to the input is only needed to update the embedding table, which is a gather, so if embeddings were frozen that matmul could be skipped; in practice it is tiny relative to the rest and nobody bothers.
</details>

3. You have $C = 10^{19}$ FLOPs. Using $N_{\text{opt}} \propto C^{0.5}$ and the anchor that at $C = 1.86 \times 10^{18}$ the optimal point was $N = 1.24 \times 10^8$, $D = 2.5 \times 10^9$, estimate $N_{\text{opt}}$ and $D_{\text{opt}}$.

<details><summary>Answer</summary>
The ratio of budgets is $5.38$, so both scale by $\sqrt{5.38} = 2.32$: $N \approx 2.9 \times 10^8$, $D \approx 5.8 \times 10^9$. Check: $6ND = 6 \times 2.9 \times 10^8 \times 5.8 \times 10^9 = 1.0 \times 10^{19}$. The anchor is itself an assumption (the 20-tokens-per-parameter rule), which is the point: without a fit on your own data the law gives you exponents, not positions.
</details>

4. Spot the bug: `lr = max_lr * step / warmup if step < warmup else max_lr * 0.5 * (1 + math.cos(math.pi * step / total))`.

<details><summary>Answer</summary>
The cosine phase uses `step / total` rather than `(step - warmup) / (total - warmup)`, so the rate at the end of warmup is not `max_lr` but `max_lr * 0.5 * (1 + cos(pi * warmup / total))`, a discontinuous drop at the warmup boundary, and the rate reaches its minimum at `total` only by coincidence of the same endpoint. The schedule is not what was intended and the drop at `step == warmup` will show in the loss curve as a small kink.
</details>

5. The noise scale estimator $\mathbb{E}|\hat G_B|^2 = |G|^2 + \mathrm{tr}(\Sigma)/B$ is applied with $B_{\text{small}} = 32$ and $B_{\text{big}} = 512$ sequences, giving $|\hat G|^2 = 4.0$ and $1.3$. Compute $B_{\text{simple}}$.

<details><summary>Answer</summary>
Two equations: $4.0 = g + s/32$ and $1.3 = g + s/512$. Subtracting, $2.7 = s(1/32 - 1/512) = s \times 0.02930$, so $s = 92.2$ and $g = 1.3 - 92.2/512 = 1.12$. $B_{\text{simple}} = s/g \approx 82$ sequences. The estimate is noisy; average $|\hat G|^2$ over several batches at each size before solving.
</details>

6. Why does $\beta_2 = 0.999$ cause spikes when the gradient norm suddenly rises, but not when it suddenly falls?

<details><summary>Answer</summary>
The update is $\hat m / \sqrt{\hat v}$. When gradients rise, $\hat m$ (memory about 10 steps) tracks the rise but $\hat v$ (memory about 1,000 steps) stays small, so the ratio and the step blow up. When gradients fall, $\hat v$ stays large and the step becomes too small, which slows training but does not destabilize it. The asymmetry is why a shorter second-moment memory is the safer error.
</details>

7. A run at $T = 8192$ shows MFU of 25 percent while the same model at $T = 1024$ showed 45 percent. Is the run necessarily unhealthy?

<details><summary>Answer</summary>
Not necessarily. MFU as usually computed counts only the $6N$ matmul FLOPs; at $T = 8192$ the attention term is around 60 percent of $6N$ for a 124M-shaped model, so a large share of the real work is not counted, and the apparent MFU falls. Recompute with $c = 6N + 6\, n_{\text{layer}} T d$ before concluding the loop is slow. If the corrected number is still low, the attention kernel is the suspect.
</details>

8. In the toy at $T = 32$, a colleague estimates the context-starvation penalty as one position in 32 paying the unigram entropy instead of the floor, $(2.455 - 0.663)/32 = 0.056$ nats, and is puzzled that the averaged validation loss is $0.69$, a gap of only $0.03$. Find the error.

<details><summary>Answer</summary>
The first prediction in a window is not made with zero context. Position $t$ predicts $x_{t+1}$ from $x_{\le t}$, so the first target $x_2$ is predicted from $x_1$, one character of context, and its loss is $H(x_2 \mid x_1)$, not the unigram entropy. Measured, that position pays about $1.12$ nats, contributing $(1.12 - 0.663)/32 \approx 0.014$; the next few positions contribute smaller amounts, and positions past the eighth average about $0.68$. So roughly half the $0.03$ gap is context starvation and half is finite capacity and steps. The naive estimate exceeds the entire gap because it forgets the shift by one.
</details>

9. Under a WSD schedule, a checkpoint at 60 percent of the run is cooled down and evaluated. A colleague compares it with a cosine run of the same total length at its 60 percent point and declares WSD superior. What is wrong with the comparison?

<details><summary>Answer</summary>
The cosine run at 60 percent still carries its noise-ball excess because its rate is still well above the minimum; the cooled WSD checkpoint has had its excess removed. The fair comparison is against a cosine run whose total length is 60 percent of the original, so both have decayed at the point of evaluation. WSD's practical advantage is that it does not require deciding the length in advance, not that its loss at a given token count is lower than a properly matched cosine run.
</details>

10. Give a concrete mechanism by which increasing the vocabulary from 32k to 128k at fixed $d_{\text{model}} = 768$ can make a 124M-class model worse at equal FLOPs, even though it improves compression.

<details><summary>Answer</summary>
The embedding and unembedding matrices grow from $32{,}000 \times 768 = 24.6$M to $98$M parameters, so at fixed total $N$ the transformer body shrinks, and the unembedding matmul's cost per token grows fourfold. At the same time the number of tokens per byte falls, so each byte gets fewer forward passes of a smaller body. For small models the body is where the loss is won, and the trade goes against the larger vocabulary; it reverses at larger widths, which is why vocabulary size scales with model size.
</details>

## What will change, what will not

The decomposition $L = H(p^*) + \mathrm{KL}$ is a theorem and the loss floor is a property of the data. Any future objective that is a proper scoring rule for next-token prediction shares it; objectives that are not (multi-token prediction, diffusion over tokens, latent-variable models) change the floor's definition but not the fact of one. The FLOP accounting $6N$ per token is a property of dense matmuls with backpropagation and will survive as long as those are the primitive; mixture-of-experts changes $N$ to active parameters, and hardware that fuses forward and backward would change the constant, not the structure.

The scaling-law form $E + A N^{-\alpha} + B D^{-\beta}$ is an empirical fit. The exponents near $0.3$ and the 20-tokens-per-parameter ratio are the least durable numbers in this chapter; they have already been revised by better data, and they do not apply to models trained for inference efficiency. The procedure (fit on small runs, extrapolate one order of magnitude, check) is what to keep.

The noise-scale argument and the trade-off $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$ are derived from a quadratic model and hold for any first-order optimizer with minibatch noise. AdamW's specific constants ($\beta_2 = 0.95$, decay $0.1$, clip at $1$) are conventions of this era, and optimizers that shape the update differently (second-order approximations, orthogonalized updates, parameterizations that transfer the learning rate across widths) are already displacing some of them. Warmup and decay respond to two mechanisms, unreliable early statistics and the noise ball, that any such optimizer still faces, so some form of both will remain.

The nanoGPT trainer, `torch.compile`, bf16, and the flash attention kernel are tooling and will be replaced. The habit of overlaying the learning rate on the loss curve, watching the gradient norm, and reporting loss with its token count will not.

## Read next

- "Scaling Laws for Neural Language Models", Kaplan, 2020. The first systematic power-law study and the source of the $6ND$ accounting.
- "Training Compute-Optimal Large Language Models", Hoffmann, 2022. The Chinchilla fit, its three estimation methods, and the equal-scaling conclusion.
- "An Empirical Model of Large-Batch Training", McCandlish, 2018. The gradient noise scale and the steps-versus-examples trade-off derived above.
- "Decoupled Weight Decay Regularization", Loshchilov, 2019. Why weight decay is applied to the parameters rather than the gradient in AdamW.
- "Small-scale Proxies for Large-scale Transformer Training Instabilities", Wortsman, 2023. Reproduces attention-logit growth and output-logit divergence at small scale and evaluates QK-layernorm and z-loss as fixes.
- "PaLM: Scaling Language Modeling with Pathways", Chowdhery, 2022. Introduces the z-loss and documents spike handling by rewinding and skipping batches.
- "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", Eldan, 2023. The dataset the recipes use and evidence that small models learn grammar and narrative on it.
- "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws", Sardana, 2023. Why deployed models are trained past 20 tokens per parameter.
