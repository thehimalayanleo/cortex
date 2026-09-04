---
title: "Lab 03: Mid-training, cooldown, and data mixtures"
kind: permanent
topics: [lab]
chapter: 3
station: midtrain
recipe: recipes/midtrain.py
reading_time: 55 min
---

## What you will be able to do

- Derive why a run's loss drops when the learning rate decays, predict the size of the drop for a simple model, and choose a cooldown length with a reason.
- Change the data mixture late in a run, separate the forgetting that a cooldown will cure from the forgetting that only replay can cure, and measure both with held-out per-domain losses.
- Continue pretraining from a checkpoint without a loss spike, by re-warming the rate and deciding what to do with the optimizer state.
- Extend a rotary-position model's context as a mid-training stage, and say which dimensions of the position encoding the extension changes.
- Run a two-domain mixture-and-cooldown grid on the 5090 and read its results as a table of per-domain losses rather than a single number.

## The idea in one paragraph

A pretraining run does not end when the token budget runs out; it ends with a stretch in which two things change at once, the learning rate goes to zero and the data shifts toward what you most want the model to be good at. The learning-rate part is mechanical: with a constant rate the weights never settle, they orbit the minimum at a distance set by the rate, and letting the rate decay lets them fall in, which shows up as a loss drop that costs nothing. The data part is a trade: whatever you upweight gets better, whatever you downweight gets worse, and the only way to know the exchange rate is to hold out a slice of each domain and watch both losses. Continued pretraining and long-context extension are the same stage with different data. The midtrain station in the browser continues the pretrain station's model on a stories-and-arithmetic mixture while the rate cools, and draws both domain losses with the rate overlaid; the two effects, the trade and the drop, are both visible in under a minute.

## The math

### The noise ball

Take the simplest loss with curvature, $f(\theta) = \frac{h}{2}\theta^2$ for a scalar $\theta$ and $h > 0$, and stochastic gradient descent whose gradient estimate carries additive noise $\xi_t$ with mean zero and variance $\sigma^2$:

$$\theta_{t+1} = \theta_t - \eta (h \theta_t + \xi_t) = (1 - \eta h)\theta_t - \eta \xi_t.$$

Because the noise has mean zero, $\mathbb{E}[\theta_t] \to 0$ geometrically at rate $(1 - \eta h)$ whenever $0 < \eta h < 2$. The second moment does not go to zero. Write $v_t = \mathbb{E}[\theta_t^2]$; squaring the update and using independence of $\xi_t$ from $\theta_t$,

$$v_{t+1} = (1 - \eta h)^2 v_t + \eta^2 \sigma^2.$$

This is a linear recurrence with fixed point

$$v^* = \frac{\eta^2 \sigma^2}{1 - (1 - \eta h)^2} = \frac{\eta \sigma^2}{2h - \eta h^2}.$$

The expected loss at stationarity is $\frac{h}{2} v^*$, so the excess loss above the minimum is

$$\mathbb{E}[f] - f_{\min} = \frac{\eta \sigma^2}{4 - 2\eta h} \approx \frac{\eta \sigma^2}{4} \quad (\eta h \ll 1).$$

Two things to read off. The excess is proportional to the learning rate, so halving the rate halves the noise ball, and letting the rate go to zero collapses it: that is the cooldown drop. And to first order the excess does not depend on the curvature $h$: high-curvature and low-curvature directions each contribute $\eta \sigma_i^2 / 4$. In $d$ dimensions with per-direction noise variances $\sigma_i^2$ the excess is $\sum_i \eta \sigma_i^2 / (4 - 2\eta h_i)$, and in a model where the noise is spread over many directions the drop from a cooldown is large because it is a sum over all of them.

What does depend on curvature is time. The mean relaxes at rate $\eta h$, so a direction with curvature $h_i$ takes about $1 / (\eta h_i)$ steps to forget where it was. When the rate decays, the noise ball in each direction shrinks only as fast as that direction can relax, and directions with $\eta_t h_i \ll 1/(\text{steps remaining})$ freeze with their variance intact. This is the argument for a cooldown that is a fraction of the run rather than a few hundred steps: Hägele et al. (2024) report that a cooldown of roughly 10 to 20 percent of the total steps, from a constant rate, matches or beats a cosine schedule of the same length, and that a $1 - \sqrt{\cdot}$ shape does slightly better than linear because it spends more of the cooldown at low rates where the slow directions drain.

For the linear-regression toy below, everything is computable. With inputs $x \sim \mathcal N(0, I_d)$ and labels $y = x^\top w + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma^2)$, the loss $\frac{1}{2}\mathbb{E}(x^\top \theta - y)^2 = \frac{\sigma^2}{2} + \frac{1}{2}\|\theta - w\|^2$ has curvature $h = 1$ in every direction and floor $\sigma^2 / 2$, the analogue of the data entropy in Lab 02. A minibatch of $b$ examples gives a gradient whose noise at the optimum has variance $\sigma^2 / b$ per coordinate. So the predicted excess is

$$d \cdot \frac{\eta \sigma^2 / b}{4 - 2\eta},$$

and the snippet prints this next to the measured value.

### Why annealing data matters more than its token count

During the cooldown the model sees a small fraction of its total tokens, yet reports of large runs consistently anneal on curated high-quality data in that window (MiniCPM, Llama 3, and the domain-upsampling study by Blakeney et al., 2024, all describe this). The quadratic picture explains why the window is not proportionally unimportant. Along a direction with curvature $h$, the iterate after a sequence of steps with rates $\eta_t$ is a weighted average of the per-step targets, with weight $\eta_t h \prod_{s > t}(1 - \eta_s h)$ on step $t$. The total weight carried by a final window $W$ is

$$1 - \prod_{t \in W}(1 - \eta_t h) \approx 1 - \exp\left(-h \sum_{t \in W} \eta_t\right).$$

For directions whose curvature times the cooldown's learning-rate mass is large, this saturates at one: the final window entirely determines where those directions end up, no matter what came before. For directions where it is small, the cooldown barely moves them. So the annealing data rewrites the fast directions (surface statistics, formatting, the distribution of the last few tokens' features) and leaves the slow directions (whatever took the whole run to learn) to the bulk of pretraining. That is exactly the division you want if the annealing data is high quality and in the target format, and exactly the danger if it is contaminated (see the failure modes). The Llama 3 report also uses this sensitivity in reverse: a short anneal from a mid-run checkpoint onto a candidate dataset is a cheap probe of that dataset's value.

### Changing the mixture and the two kinds of forgetting

Recall from Lab 01 that a mixture with weights $w$ trains on $L_{\text{mix}}(\theta) = \sum_i w_i L_i(\theta)$. Its minimizer $\theta^*(w)$ is a compromise; changing $w$ to $w'$ moves the target to $\theta^*(w')$, and the model follows at the rate set by $\eta$ and the curvatures. Define the forgetting of domain $A$ over a mid-training stage as

$$F_A = L_A(\theta_{\text{end}}) - L_A(\theta_{\text{start}}),$$

measured on held-out $A$ data. It has two parts. The intrinsic part, $L_A(\theta^*(w')) - L_A(\theta^*(w))$, is the price of the new compromise; no schedule removes it, only putting $A$ back into the mixture (replay) does. The transient part is the noise ball plus the distance the model has not yet traveled, and a cooldown removes it. The toy separates them: with a lopsided mixture the intrinsic part dominates and the cooldown changes $L_A$ little; with a mild mixture the transient part is visible as the difference between the constant-rate and cooled-down runs.

The toy also shows a noise source that pure pretraining does not have. When each step samples a domain and then a minibatch, the gradient's variance includes a term from the domain choice, $w_A w_B \|\nabla L_A - \nabla L_B\|^2$, which is large exactly when the domains disagree. The constant-rate mixture runs therefore sit further above their optimum than the single-domain run did, and the cooldown removes that too. In the linear toy the intrinsic forgetting has a closed form: with $\Delta = w_B - w_A$ the difference between the two domains' true weight vectors, the mixture optimum is $\theta^*(w) = w_A + w_B \Delta$ (writing $w_B$ for the weight on domain $B$), so $L_A(\theta^*) = \sigma^2/2 + \frac{1}{2} w_B^2 \|\Delta\|^2$ and $L_B(\theta^*) = \sigma^2/2 + \frac{1}{2}(1 - w_B)^2 \|\Delta\|^2$. The snippet prints these optima next to the measured losses.

For replay, Ibrahim et al. (2024) measured continued pretraining of decoders onto a new corpus with a fraction of the old one mixed back in, and found that a few percent of replay recovers most of the old-domain loss under a mild distribution shift, with more needed under a strong shift, at a cost on the new domain that is small relative to the forgetting avoided. The fraction is a hyperparameter of the two-domain grid below, not a constant.

### Continued pretraining from a checkpoint

A checkpoint at the end of a cooled-down run sits in a small noise ball at a low rate. Continuing it on new data requires raising the rate again, and the re-warm reinflates the noise ball while the new distribution pulls the target away: the loss on both old and new data rises for a while before the new-data loss starts falling. Gupta et al. (2023) document this bump and show that re-warming to a peak below the original (a fraction of it, swept per run) and re-decaying gives the best final loss on the new data, at the cost of some old-data loss that replay addresses. Two decisions about state. If you restore the optimizer's Adam moments from the checkpoint, the first steps are well scaled and a short re-warm suffices; if you reset them (a different framework, a changed parameter set), the second-moment estimate starts at zero and warmup is mandatory, for the reason in Lab 02. Either way, treat the continued run as a new run with its own warmup, peak, and cooldown, all shorter than the original.

### Long-context extension as a mid-training stage

Rotary position embedding rotates each pair of query and key dimensions $(2i, 2i+1)$ at position $m$ by the angle $m \theta_i$, with

$$\theta_i = \text{base}^{-2i / d_{\text{head}}}, \qquad i = 0, \dots, d_{\text{head}}/2 - 1,$$

so that the attention score between positions $m$ and $n$ depends on the relative angle $(m - n)\theta_i$. Dimension pair $i$ therefore has a wavelength $\lambda_i = 2\pi / \theta_i = 2\pi \cdot \text{base}^{2i / d_{\text{head}}}$: the first pairs turn once every few positions and encode local order; the last pairs turn once every tens of thousands of positions. With base $10{,}000$ and $d_{\text{head}} = 64$, the longest wavelength is $2\pi \cdot 10^{4 \cdot 62/64} \approx 47{,}000$ positions. A model trained at context $L$ has never seen relative angles beyond $L\theta_i$ in the slow dimensions, so at positions past $L$ those dimensions produce values the attention heads have never been calibrated on, and the loss rises sharply just past $L$.

Position interpolation (Chen et al., 2023) scales positions down, $m \to m L / L'$, so that a context of $L'$ spans the same angles as the trained $L$, and fine-tunes briefly on long documents. It works because interpolating between seen angles is far easier than extrapolating past them. Its cost is resolution: the fast dimensions are compressed too, and neighboring tokens become harder to tell apart. The fix used by Llama-family long-context models is to raise the base instead ("adjusted base frequency"), which stretches the slow wavelengths while leaving the fast ones nearly unchanged; the YaRN paper gives the base change that makes the slowest dimension interpolate by exactly a factor $s = L'/L$ as

$$\text{base}' = \text{base} \cdot s^{\,d_{\text{head}} / (d_{\text{head}} - 2)},$$

and adds per-dimension blending between interpolation and no change according to each wavelength's ratio to $L$. Whatever the scheme, the model then needs a short training stage on documents that actually are long, which is where the attention cost from Lab 02 comes back: at $T = 32{,}768$ the attention term $6\, n_{\text{layer}} T d$ for a 124M-shaped model is $6 \times 12 \times 32{,}768 \times 768 = 1.8 \times 10^9$ FLOPs per token, more than twice the $6N$ term. The stage is therefore short in tokens and expensive per token, and it is done last, after the mixture change and within or just before the cooldown, on a mixture of long documents and replayed short ones so that short-context loss does not regress.

### Measuring it with held-out per-domain losses

Everything above is measured with one instrument: for each domain $i$, a held-out set split at the document level (Lab 01), and the loss $L_i(\theta_t)$ evaluated on a fixed set of windows every few hundred steps. Plot every $L_i$ on one axis with $\eta_t$ overlaid. The cooldown drop is the fall in every curve as $\eta_t \to 0$; forgetting is a curve that rises; the trade is the pair of curves moving in opposite directions after the mixture changes. For long-context runs, add the loss by position bucket (tokens $0$ to $L$, $L$ to $2L$, and so on), which is the number that tells you the extension worked.

## Build it small

Two-domain linear regression with minibatch noise. Phase one trains on domain $A$ at a constant rate and compares the measured excess loss with the noise-ball prediction. Phase two continues from that point on a mixture, at a constant rate and with a linear cooldown, and prints held-out losses on both domains next to the losses of the mixture-optimal weights.

```python
# Lab 03, build it small: the noise ball and forgetting, on two-domain linear regression.
# One weight vector, two domains that agree on half the coordinates and disagree on the other half.
import torch
torch.manual_seed(0)
d, b, sigma, eta = 20, 8, 0.5, 0.05             # dims, minibatch, label noise std, peak learning rate
wA = torch.randn(d)
wB = wA.clone(); wB[d // 2:] += 0.5 * torch.randn(d // 2)   # domain B disagrees with A on the second half
delta2 = ((wB - wA) ** 2).sum().item()

def sample(w, n):                                # y = x.w + noise, x standard normal
    x = torch.randn(n, d); return x, x @ w + sigma * torch.randn(n)

XA, YA = sample(wA, 20000); XB, YB = sample(wB, 20000)   # held-out sets, one per domain
def held_out(w):
    return tuple(0.5 * ((X @ w - Y) ** 2).mean().item() for X, Y in ((XA, YA), (XB, YB)))

def train(w, steps, mix_b, cooldown):
    w = w.clone()
    for t in range(steps):
        lr = eta * (1 - t / steps) if cooldown else eta        # linear-to-zero or constant
        src = wB if torch.rand(()) < mix_b else wA             # sample the domain, then a minibatch
        x, y = sample(src, b)
        grad = x.T @ (x @ w - y) / b                            # gradient of 0.5*mean squared error
        w -= lr * grad
    return w

floor = 0.5 * sigma ** 2                                       # irreducible loss: half the label variance
print(f"floor (the data-entropy analogue) = {floor:.4f}")

w0 = train(torch.zeros(d), 4000, mix_b=0.0, cooldown=False)    # "pretraining" on A only, constant lr
lA0, lB0 = held_out(w0)
pred = d * eta * sigma ** 2 / (b * (4 - 2 * eta))               # noise-ball excess from the derivation
print(f"pretrain on A, constant lr: L_A {lA0:.4f} (excess {lA0 - floor:.4f}, predicted {pred:.4f})  L_B {lB0:.4f}")

for mix_b in (0.3, 0.9):
    optA = floor + 0.5 * mix_b ** 2 * delta2                   # loss of the mixture-optimal weights
    optB = floor + 0.5 * (1 - mix_b) ** 2 * delta2
    for cooldown in (False, True):
        lA, lB = held_out(train(w0, 4000, mix_b, cooldown))
        print(f"midtrain mix_b={mix_b} cooldown={str(cooldown):5s}: L_A {lA:.4f} (opt {optA:.4f})"
              f"  L_B {lB:.4f} (opt {optB:.4f})  forgetting of A {lA - lA0:+.4f}")
```

Expected output, with this seed: the floor is $0.125$; after pretraining on $A$ the measured excess is about $0.009$ against a predicted $0.008$, and $L_B$ is around $1.4$ because the model has never seen $B$. At `mix_b=0.3` the constant-rate run lands at $L_A \approx 0.31$ against an optimum of $0.246$, and the cooled run at about $0.24$, on the optimum: the transient part of forgetting was about $0.06$ and the cooldown removed it, while the intrinsic part (the optimum itself, $0.12$ above the floor) remains. At `mix_b=0.9` both runs sit near $L_A \approx 1.2$ and $L_B \approx 0.14$: the mixture has moved the target most of the way to $B$, forgetting of $A$ is about $1.1$ and almost entirely intrinsic, and the cooldown makes little difference to $A$ because there is nothing transient left to remove. Change `mix_b` to `0.0` in the second phase with cooldown and you get the pure cooldown drop on $A$ alone, from $0.134$ to within a thousandth of the floor.

## Build it real

`recipes/midtrain.py` continues a checkpoint written by Lab 02's `pretrain_nano.py` and is built around one output: a CSV of held-out per-domain losses against step and learning rate, which the browser station draws for the toy and which you will plot for the real run. Its arguments are `--init` for the checkpoint, `--data_a` and `--data_b` for two shard directories from Lab 01 (each with its own document-level `val.bin`), `--mix_b` for the sampling weight of domain $B$, `--steps`, and the schedule: `--lr` as the re-warmed peak (default half the original), `--rewarmup_steps`, and `--schedule` chosen from `constant`, `linear`, `cosine`, and `one_minus_sqrt`, with `--cooldown_frac` for how much of the stage decays. `--restore_optimizer` loads the Adam moments from the checkpoint; without it the moments start at zero and the recipe refuses to run with `--rewarmup_steps 0`. `--anneal_data` and `--anneal_frac` swap in a third shard directory for the last fraction of the stage, which is how you test annealing on high-quality data. For context extension, `--seq_len` and `--rope_base` override the checkpoint's values (the checkpoint must have been trained with `--rope`), and the evaluation adds loss by position bucket to the CSV.

The two-domain experiment that pairs with the station is a grid, and the recipe runs it with `--grid mix_b=0.3,0.6,0.9 schedule=constant,linear`. Domain $A$ is TinyStories, the corpus of the Lab 02 checkpoint. Domain $B$ should be something the checkpoint is bad at and that is cheap to make: the recipe's `--make_arithmetic` writes shards of lines like `37 + 58 = 95` through Lab 01's tokenizer, matching the station's second corpus, and a small Python-code subset is the natural second choice. Start from a `--preset char-tiny` checkpoint saved at the end of its stable phase (a WSD run at `--cooldown_frac 0`, so that the mid-training stage owns the cooldown). Each cell continues for a few thousand steps; at the `char-tiny` size the whole grid runs in well under an hour on the 5090, and the compute per cell is the Lab 02 formula, steps times tokens per step times $c$, divided by the measured rate. Add `--replay 0.05` as a fourth axis once the first grid makes sense. At `gpt2-small` size the same grid is a day, and you run the two most informative cells.

What to watch in the logs: the first few hundred steps, where a re-warm bump is normal and a spike is not; the two domain losses after the mixture change, which should move in opposite directions and then both fall when the cooldown begins; and, in the extension mode, the loss in the highest position bucket, which should come down to within a few hundredths of the lowest bucket by the end of the stage. If it does not, the base or the scale is wrong, not the training length.

## How it goes wrong

1. The loss spikes in the first hundred steps of the continued run. The rate was re-warmed to the original peak in too few steps, or the optimizer state was reset and there was no warmup at all. Re-warm over one to two percent of the stage's steps to a peak at or below half the original, or restore the moments.

2. Domain $A$ loss climbs for the whole stage and the cooldown does not bring it back. Intrinsic forgetting from a lopsided mixture; the compromise moved. Add replay of $A$ at a few percent and re-run; if $A$ genuinely does not matter, say so in the report rather than hiding the curve.

3. The cooldown drop is smaller than expected. The cooldown is too short for the slow directions to drain, or the rate at the start of the cooldown was already low (a cosine run near its end has nothing left to cool). Check the rate at the start of the cooldown and lengthen it to a tenth or a fifth of the stage.

4. Annealing on a new dataset produces a large benchmark gain. Before celebrating, run the 13-gram decontamination from Lab 01 between that dataset and the benchmark. The annealing window is the most sensitive part of the run to its data, which is precisely why contamination there is the most damaging.

5. In the extension run, the loss is flat up to the original context and then rises steeply. The positions were extended without changing the rotary base or interpolating; the slow dimensions are extrapolating. Set `--rope_base` per the formula above or use interpolation, then train.

6. Long-position loss is fine but short-context loss and the old domain both regressed. Interpolation compressed the fast dimensions, or the extension stage had no short documents. Use the base adjustment rather than plain interpolation, and mix in replayed short-context data.

7. A cooled-down mid-training checkpoint beats the un-cooled pretraining checkpoint at the same token count and the difference is attributed to the new data. The comparison is confounded by the cooldown. Compare cooled against cooled: cool the pretraining checkpoint with the original mixture for the same number of steps and use that as the baseline.

8. Per-domain held-out losses look implausibly good and stop moving. The held-out windows overlap the training shards, because the split was done on windows rather than documents, or because domain $B$ was generated with the same seed for train and validation. Rebuild the split at the document level with distinct seeds.

## Measure it

The primary artifact is a table with one row per experimental cell and one column per domain, each entry the held-out loss in nats per token at the end of the stage, plus a column for the loss of the pretraining checkpoint cooled on its original mixture. From it, report three derived numbers.

The cooldown gain per domain, the constant-rate loss minus the cooled loss at equal tokens. It should be positive on every domain, and in the toy it is predicted by $d \eta \sigma^2 / (4b)$; in a real model it is typically a few hundredths of a nat and larger when the constant rate was higher.

Forgetting $F_A$ for the old domain, the loss after the stage minus the loss before, on held-out $A$, at matched schedules. Its dependence on `mix_b` is the trade curve, and its reduction with `--replay` is the price of replay expressed in domain $B$ loss.

For extension, loss by position bucket. A successful extension has the last bucket within a few hundredths of the first on long held-out documents, and a short-context loss that has not regressed by more than the seed noise; a needle-retrieval probe (a fact placed at a controlled depth, a question at the end) is a cheap complement that `lm-eval-harness` does not provide and the recipe includes as `--needle`.

For a downstream check at `gpt2-small` size, `lm-eval-harness` on a task from each domain, used to compare cells, not to report absolute numbers.

## Exercises

1. Re-derive the stationary variance for SGD with momentum $\mu$ on the scalar quadratic and show that the excess loss becomes $\eta \sigma^2 / (4(1 - \mu))$ to first order. Check: momentum $0.9$ multiplies the noise ball by ten at the same rate, which is why the effective rate $\eta / (1 - \mu)$ is what matters.

2. In the toy, set `cooldown` to a $1 - \sqrt{t/\text{steps}}$ shape and compare final losses with linear over 5 seeds at `mix_b = 0.3`. Check: the difference is small in this convex toy; the shape matters when slow directions exist, which the isotropic toy lacks. Then make it anisotropic by scaling half the input coordinates by $0.2$ and repeat.

3. Compute the intrinsic forgetting of $A$ at `mix_b` of $0.3$, $0.6$, $0.9$ from the closed form, then measure it with cooled runs. Check: $\frac{1}{2} w_B^2 \|\Delta\|^2$ with the printed `delta2`; measured values agree to within the noise of the held-out estimate.

4. Add replay to the toy: after the mixture stage, continue for 1,000 cooled steps at `mix_b = 0.9` versus `mix_b = 0.8` and compare the pair of losses. Check: $L_A$ falls by about $\frac{1}{2}(0.81 - 0.64)\|\Delta\|^2$ and $L_B$ rises by about $\frac{1}{2}(0.04 - 0.01)\|\Delta\|^2$; write the exchange rate as a ratio.

5. For a checkpoint trained at $L = 1024$ with base $10{,}000$ and $d_{\text{head}} = 64$, compute the base needed for $L' = 8192$ from the formula, and count how many dimension pairs have wavelength longer than $1024$ before and after. Check: $s = 8$, exponent $64/62 = 1.032$, base$' \approx 10{,}000 \times 8.55 \approx 85{,}000$; before, pairs with $2\pi \cdot 10^{4 \cdot 2i/64} > 1024$, i.e. $2i/64 > \log_{10}(163) / 4 = 0.553$, so $i \ge 18$, fourteen pairs; after, the threshold moves to $2i/64 > \log_{85000}(163) = 0.449$, so $i \ge 15$, seventeen pairs.

6. Run the `char-tiny` grid on the 5090 and produce the table from the Measure it section. Check: the `constant` column is above the `linear` column on both domains in every row, and forgetting of $A$ increases monotonically with `mix_b`.

## Test yourself

1. The noise-ball excess is $\eta\sigma^2/4$ per direction independent of curvature. A colleague concludes that the cooldown length also does not depend on curvature. What is wrong?

<details><summary>Answer</summary>
The stationary excess is curvature-independent, but the rate at which a direction approaches stationarity is $\eta h_i$ per step. During a cooldown the rate is falling, and a direction whose $\eta_t h_i$ is small relative to the inverse of the remaining steps cannot drain before the rate reaches zero; it is left with the variance it had. Low-curvature directions therefore need a long cooldown, which is the reason cooldown length is a fraction of the run and not a fixed number of steps.
</details>

2. Give a one-sentence reason why the noise-ball derivation predicts that a run with twice the batch size at the same rate has half the cooldown drop, and say what that implies about comparing schedules across batch sizes.

<details><summary>Answer</summary>
The gradient noise variance scales as $1/b$, so the stationary excess $\eta\sigma^2/(4b)$ halves when $b$ doubles, and so does the drop that removing it produces. A schedule comparison made at one batch size does not transfer: a large-batch run gains less from cooling and more from a higher peak rate, so the optimal schedule shape moves with the batch.
</details>

3. Spot the bug. A mid-training script restores the model weights from a checkpoint, constructs a fresh `AdamW`, and starts at the original peak rate with `warmup_steps=0`, arguing that the weights are already well trained so warmup is unnecessary.

<details><summary>Answer</summary>
Warmup protects the optimizer state, not the weights. A fresh AdamW has $v = 0$, and after bias correction the first steps divide the gradient by its own magnitude, producing updates of size $\eta$ in every coordinate regardless of the gradient's scale; at the original peak that is a large, poorly scaled move from a good point, and the loss spikes. Either restore the moments or re-warm.
</details>

4. During annealing, a high-quality dataset is upweighted to 50 percent of the last 5 percent of the run. Estimate the fraction of the run's tokens it accounts for and explain why its effect can nonetheless exceed that of a source worth 10 percent of the full run.

<details><summary>Answer</summary>
It is $0.5 \times 0.05 = 2.5$ percent of the tokens. But its influence on the fast directions is the saturating quantity $1 - \exp(-h \sum_{W} \eta_t)$, which for those directions is near one during any window at the end, so it sets their final values regardless of everything earlier; the 10 percent source, spread over the whole run, had its contribution to those directions overwritten by every later window. For slow directions the ordering is reversed, and the 10 percent source wins. Which matters more depends on what the evaluation measures.
</details>

5. Estimate the intrinsic forgetting of domain $A$ if a decoder is continued on domain $B$ alone (no replay) for long enough to converge, in the linear-toy model, and say what it means that the answer does not depend on the schedule.

<details><summary>Answer</summary>
With `mix_b=1` the optimum is $w_B$ itself and $L_A(\theta^*) = \sigma^2/2 + \frac{1}{2}\|\Delta\|^2$, so forgetting is $\frac{1}{2}\|\Delta\|^2$, the full disagreement between the domains. No schedule changes a stationary point; schedules change only how far along the path you get and how much noise you carry. Anything that preserves $A$ must change the objective, which is what replay does.
</details>

6. A position-interpolated model at scale $s = 4$ answers needle-retrieval probes at 16k tokens perfectly but its loss on ordinary 1k-token documents rose by $0.05$ nats. Explain both facts with the wavelength picture.

<details><summary>Answer</summary>
Interpolation divides every angle by four, so the slow dimensions now span the trained range across 16k tokens and long-range attention works, which is what the probe tests. The fast dimensions are also divided by four: a pair that once distinguished neighbors by a quarter turn now moves a sixteenth of a turn per token, and local order is blurrier, which costs loss on every document at every length. The base adjustment stretches the slow dimensions without compressing the fast ones and avoids the second effect.
</details>

7. The two-domain grid shows that at `mix_b=0.6` with cooldown, $L_A$ is lower than the pretraining checkpoint's $L_A$, even though $A$ was downweighted. A colleague calls this positive transfer from $B$. Give the more likely explanation and the experiment that distinguishes them.

<details><summary>Answer</summary>
The pretraining checkpoint was saved at the end of a stable phase at a constant rate, so it carries its full noise ball; the mid-training cell was cooled. The comparison is cooldown against no cooldown, not $B$ against no $B$. Cool the pretraining checkpoint on the original mixture for the same number of steps and compare that value; positive transfer is what remains, if anything does.
</details>

8. Compute the per-token FLOP cost at $T = 32{,}768$ for a 124M model with 12 layers and width 768, and the fraction that is attention.

<details><summary>Answer</summary>
$6N = 7.44 \times 10^8$; attention $6 \times 12 \times 32{,}768 \times 768 = 1.81 \times 10^9$; total $2.55 \times 10^9$ FLOPs per token, of which 71 percent is attention. A billion tokens of extension therefore cost about $3.2$ times what a billion tokens at $T = 1024$ cost ($8.0 \times 10^8$ per token there), which is why the stage is short.
</details>

9. Domain $B$ is generated on the fly (arithmetic lines) with a fixed seed for both training and held-out shards. The per-domain held-out loss on $B$ reaches a value below any plausible entropy. What happened, and what is the correct floor for that domain?

<details><summary>Answer</summary>
The held-out lines are the training lines; the model memorized them. Generate the two splits with distinct seeds and, better, from disjoint operand ranges. The floor for arithmetic with uniformly random operands is the entropy of the operands spread over the characters of the line (the answer digits are deterministic given the operands), which you can compute exactly the way Lab 02's toy computes its floor.
</details>

10. Why does the argument that "the cooldown removes noise" not imply that a very low constant rate throughout the run is just as good?

<details><summary>Answer</summary>
Progress along a direction goes as $\eta h_i$ per step, so a low rate reaches the same point in proportionally more steps; the noise ball is small but the model never gets near the minimum within the budget. A high rate makes progress and a cooldown at the end removes the noise it accumulated. The schedule buys both. The toy shows this if you run phase one at `eta = 0.0005` for the same 4,000 steps: that is two time constants of $1/(\eta h)$, the weights are still far from $w_A$, and $L_A$ ends near $0.35$ instead of $0.134$, even though the noise ball at that rate is a hundredth of the size.
</details>

## What will change, what will not

The noise-ball derivation is elementary and exact for a quadratic with additive noise, and the qualitative consequences (excess loss proportional to the rate, drain time inversely proportional to rate times curvature) survive in any first-order stochastic optimizer. What will change is the schedule shape considered standard: cosine, WSD, and $1 - \sqrt{\cdot}$ cooldowns are conventions, and whichever one is current in five years will still be doing the same thing, spending learning-rate mass early and removing variance late.

The two-kinds-of-forgetting split is a statement about stationary points and holds for any objective that is a weighted sum over domains; it does not depend on the architecture. The empirical replay fractions and the "few percent" guidance are tied to the models and shifts studied and will be revised. Methods that change the objective (regularizers toward the old weights, parameter isolation, model merging) attack the intrinsic part and may become standard in continued pretraining; the measurement, held-out loss per domain on document-level splits, is what to keep regardless.

The rotary-position analysis in terms of wavelengths is specific to RoPE, which is the dominant position encoding today and may not remain so. The general point, that any position encoding has a trained range and that extension means either staying inside it or teaching the model the outside of it briefly on real long documents, transfers to whatever replaces it. The specific base numbers and the YaRN formula are current tooling.

The practice of annealing on curated data is likely to grow, because the argument for it is structural, and the boundary between "mid-training" and "post-training" (Lab 04) is already blurring as instruction-formatted data moves into the cooldown. The habit this chapter tries to install, comparing cooled checkpoints only against cooled checkpoints and reading a table of per-domain losses rather than one number, is the durable part.

## Read next

- "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", Hägele, 2024. The systematic study of warmup-stable-decay cooldowns, their length, and their shape.
- "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", Hu, 2024. An early detailed account of annealing on high-quality and instruction data during the decay phase.
- "Does your data spark joy? Performance gains from domain upsampling at the end of training", Blakeney, 2024. Direct measurements of upweighting high-quality domains in the final stretch of pretraining.
- "Simple and Scalable Strategies to Continually Pre-train Large Language Models", Ibrahim, 2024. Re-warming, re-decaying, and replay fractions for continued pretraining under weak and strong distribution shift.
- "Continual Pre-Training of Large Language Models: How to (re)warm your model?", Gupta, 2023. The re-warming bump and the choice of a lower second peak.
- "Extending Context Window of Large Language Models via Positional Interpolation", Chen, 2023. Position interpolation and why interpolating beats extrapolating for RoPE.
- "YaRN: Efficient Context Window Extension of Large Language Models", Peng, 2023. The wavelength-based view of RoPE dimensions and the base-adjustment formula.
- "The Llama 3 Herd of Models", Dubey, 2024. Annealing data, using short anneals to value datasets, and staged context extension as a late pretraining stage.
