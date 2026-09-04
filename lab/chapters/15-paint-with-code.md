---
title: "Lab 15: Train a model to paint with code"
kind: permanent
topics: [lab]
chapter: 15
station: paint
recipe: recipes/paint_grpo.py
reading_time: 70 min
---

## What you will be able to do

1. Explain why a policy that writes a program which is then rendered is a better RL target than a policy that emits pixels: the program is inspectable and editable, and the reward is computed on the render.
2. Write down the render-in-the-loop step (generate, render, score, update), cost it term by term, and say which term dominates in the Hugging Face reproduction and why.
3. Build a composite reward from a gate, a length term, a pairwise judge and an aesthetic scorer; derive the variance of a four-comparison judge score from the binomial formula; and show with the GRPO advantage formula why one gate rejection in a group of eight can flatten the signal between the other seven.
4. Name the reward hacks specific to image rewards (blank canvases, degenerate repetition, judge exploitation), and explain how the gate and the exclusion of failed renders interact with group normalization.
5. Avoid the LoRA-on-MoE trap, and scale the whole idea down to one RTX 5090 with a tiny drawing DSL, a 0.5B policy and an image-similarity reward, first in 60 lines of PyTorch, then with `recipes/paint_grpo.py`.

## The idea in one paragraph

Instead of training a model to output pixels, you train a language model to write a short program that paints, run the program, and score the picture. The model never sees the picture; it only sees a number. The reward is a mix of cheap checks (did the program compile and put paint on the canvas), a learned aesthetic scorer, and a vision model that compares the render against a small pool of paintings a person rated by hand. Because the reward is computed on a render of a program, everything upstream stays legible: you can read the program, change one brush call and re-run it, and you can point the same training loop at a different pool of references and get a different taste without touching a line of code. The price is that every rollout needs a render before it can be scored, and in the reproduction this chapter follows, that render, not the 35B model, is what the step time is made of. The optimizer is GRPO from Lab 05, unchanged; what is new is the environment around it and the ways an image reward can be gamed.

The source for every number in this chapter is the Hugging Face blog post "Training a coding model to paint watercolours with TRL and OpenEnv" by Sergio Paniego (September 2026), which reproduces Surya Narreddi's original project. Where this chapter says "the post", it means that.

## The math

### The objects

Fix a prompt $x$ (in the post, a subject such as "a peach hibiscus" inside a restrictive system prompt). The policy $\pi_\theta(z \mid x)$ emits a program $z$, a token sequence. A renderer $\rho$ maps a program to an image, $\rho(z) = I \in [0, 1]^{H \times W \times 3}$, or to a failure symbol $\bot$ if the program does not compile, times out, or the rendering service is unavailable. A reward $R(I, x) \in [0, 1]$ is computed on the image. The objective is the one from Lab 05 with the reward composed through the renderer:

$$
J(\theta) = \mathbb{E}_{z \sim \pi_\theta(\cdot \mid x)} \big[ R(\rho(z), x) \big].
$$

Nothing about $\rho$ is differentiable (it is a browser running JavaScript), and nothing about $R$ needs to be (two of its terms are other neural networks called as black boxes). That is why this is a policy-gradient problem and not a differentiable-rendering problem: the gradient of $J$ is $\mathbb{E}[R \, \nabla_\theta \log \pi_\theta(z \mid x)]$, and it only ever touches the policy's log-probabilities of its own tokens. The renderer could be anything, which is the point.

Why programs and not pixels. Three reasons, in decreasing order of how much they matter. First, the output is code: about 150 lines of JavaScript per painting in the post, which you can read, edit and re-run, so a bad painting can be diagnosed to a brush call. Second, the medium can be restricted at the language level. The post's system prompt allows ten of p5.brush's 47 methods: `scaleBrushes`, `noStroke`, `fill`, `noFill`, `fillBleed`, `fillTexture`, `beginShape`, `vertex`, `endShape` and `circle`. Everything the model can do is filled shapes with bleed and texture, and the watercolour look comes from the library's pigment simulation, not from the model. The post reports that widening this allowlist crashed more sketches and broke the look. Third, the reward is on the render, so you can swap the aesthetic target (a different reference pool, a different scorer) without changing the policy, the renderer or the trainer.

### The step and its cost

One GRPO step with a render in the loop is four phases. For a group of $G$ rollouts per prompt and $P$ prompts,

$$
T_{\text{step}} = T_{\text{gen}}(P G, L) + T_{\text{render}}(P G) + T_{\text{score}}(P G) + T_{\text{update}},
$$

where $L$ is the completion length. Generation scales with $P G L$ divided by whatever throughput your sampler sustains. Rendering scales with the number of rollouts divided by how many render in parallel, times the per-render latency. Scoring is one aesthetic-model forward per image plus, for the pairwise judge, $2n$ vision-language calls per image ($n$ references, both presentation orders). The update is a forward and backward over $P G L$ tokens and is the smallest term.

The post's numbers, all from the post: a single render takes 69 to 96 seconds against a 90 second deadline, a step of eight rollouts takes fifteen to eighteen minutes, and rendering is 70 to 80 percent of that. Read that back through the formula: 70 to 80 percent of 15 to 18 minutes is roughly 10.5 to 14.4 minutes of rendering for eight rollouts, which at 69 to 96 seconds per render is consistent with the renders running close to one after another rather than eight at once. Whether that is what happened is not stated; what is stated is that the environment ran on a CPU-only Space, so headless Chromium rasterized the WebGL canvas in software, and p5.brush's bleeds and textures are heavy pixel work. The lesson does not depend on the exact parallelism: with a 35B mixture-of-experts policy on an H200, the model was not the bottleneck. A render farm was.

Two consequences for your own design. The render deadline is a reward-shaping decision in disguise: a 90 second timeout with renders that take 69 to 96 seconds means some fraction of honest programs will be cut off by the clock, and how you score a timeout decides whether the policy learns to write cheaper programs or learns nothing. And the cheapest lever on step time is not the model. It is the renderer, which is why the scaled-down version in this chapter renders in numpy in milliseconds and moves the cost back to generation.

### The composite reward

The post's reward has four terms with fixed weights:

$$
R = 0.05 \cdot \text{gate} + 0.05 \cdot \text{len} + w_J \cdot J + w_H \cdot H,
$$

where the gate is one if the sketch compiles, paints something and does not cheat (for example by writing text on the canvas), the length term is a soft push toward longer code, $J \in [0, 1]$ is the pairwise judge's score, and $H$ is the HPSv3 aesthetic score. The three runs differ only in $(w_J, w_H)$: judge-led $(0.60, 0.30)$, hps-led $(0.30, 0.60)$, hps-only $(0.00, 0.90)$.

Why is there a gate at all, when the judge and the scorer would presumably rate a blank canvas poorly on their own? Because you do not know that they would. An aesthetic scorer trained on photographs and generated images has never been asked about a near-blank canvas, and its output there is whatever the network extrapolates; a vision-language judge asked "which of these two is the better watercolour" may pick the blank one for its "restraint". The gate is a term whose value on the degenerate cases you have verified by construction. It is small (0.05) because its job is not to drive learning but to guarantee the sign of the reward on the cases where the learned terms are unreliable. The length term is the same idea from the other side: a soft prior against the trivial short program, weighted low enough that it cannot be the thing the policy optimizes.

There is a second, less obvious job for the gate, and it comes from the advantage formula rather than the reward. That is the next section but one.

### Pairwise judging versus absolute scoring

The judge in the post is Qwen3-VL-30B-A3B-Instruct. It sees the candidate next to four references drawn at random from a pool of 178 hand-rated paintings (half from a "love" tier and half from an "okay" tier), is told what to weigh (bleeds, translucent washes, soft edges), sees each comparison in both presentation orders, and returns the share of comparisons the candidate wins.

Why pairwise rather than asking the judge for a number out of ten? Recall the Bradley-Terry model from Lab 05: if every image has a latent quality $s$, the probability the judge prefers the candidate $c$ over a reference $r$ is

$$
P(c \succ r) = \sigma(s_c - s_r).
$$

An absolute score asks the judge to report $s_c$ on a scale it has to invent, and language-model judges are known to compress toward a few favourite values and to shift with the wording of the prompt (Lab 09 covers this in detail). A comparison only asks for the sign of $s_c - s_r$, which is the quantity the Bradley-Terry model says the judge can actually report, and the references pin the scale: a score of 0.5 means "as good as the pool", whatever the pool is. The expected judge score against a random reference is

$$
\mathbb{E}[J] = \mathbb{E}_{r \sim \text{pool}} \big[ \sigma(s_c - s_r) \big],
$$

which is monotone increasing in $s_c$, so the judge's expectation is a faithful (if nonlinear) reading of quality relative to the pool, and the pool is the reward. Change the pool and you change what is being learned without touching any code, which is the post's central design claim.

Now the variance, because it is large. With $n$ comparisons each won with probability $p$, the number of wins is $W \sim \text{Binomial}(n, p)$ and the score $J = W / n$ has

$$
\mathbb{E}[J] = p, \qquad \mathrm{Var}[J] = \frac{p (1 - p)}{n}.
$$

At $n = 4$ and $p = 0.5$ the standard deviation is $\sqrt{0.25 / 4} = 0.25$, and $J$ can only take the values $0, 0.25, 0.5, 0.75, 1$. Multiplied by the judge-led weight of 0.60, one term of the reward carries a noise standard deviation of 0.15 on a reward whose entire rise over a run was 0.27 in the post. The post's "both presentation orders" doubles the count to $n = 8$ comparisons, which would bring the standard deviation to about 0.18 if the two orders were independent; they are not (the same pair, swapped), so the true improvement is somewhere between none and that. When the references have different qualities the wins are Bernoulli with different $p_r$ and $W$ is Poisson-binomial with variance $\sum_r p_r (1 - p_r) \le n \bar p (1 - \bar p)$, so the binomial bound is the worst case. The practical reading: with four references, a single candidate's judge score is a coarse, noisy estimate, and GRPO's group of eight is what averages it. Raising $n$ costs $2n$ vision-model calls per rollout, which is a real bill at 240 episodes per step. And note the ceiling: once a candidate beats every reference, $J$ saturates at one and the judge has nothing left to say, which is why the post keeps half the references in the easier "okay" tier and lists "move the mix from easy to hard as the run advances" as a next step.

### What the gate does to the group advantage

GRPO's advantage for rollout $i$ in a group of $G$ (Lab 05) is

$$
\hat A_i = \frac{r_i - \mu}{\sigma}, \qquad \mu = \frac{1}{G} \sum_j r_j, \qquad \sigma = \sqrt{\frac{1}{G} \sum_j (r_j - \mu)^2}.
$$

Take a group of eight in which seven rollouts are honest paintings with rewards $0.70 \pm 0.03$ and one fails the gate and lands near zero. Then $\mu \approx 0.61$ and $\sigma \approx \sqrt{(7 \times 0.0875^2 + 0.61^2) / 8} \approx 0.23$. The seven good rollouts get advantages of about $+0.38$ each, and the differences between them, which is where the information about what makes a better painting lives, are $0.03 / 0.23 \approx 0.13$ apart. Drop the failed rollout and the same seven have $\sigma \approx 0.03$, so the same differences are about one standard deviation apart. One gate rejection has divided the useful signal in the group by roughly eight, and spent most of the step's gradient on the one lesson the policy already knows (do not paint a blank canvas). This is exactly what the post reports as the reason it switched TRL's `scale_rewards` from group normalization to `none`: one gate rejection was shrinking every other advantage in the group. With `scale_rewards none` the advantage is $r_i - \mu$, the failed rollout gets a large negative advantage, and the honest rollouts keep their $\pm 0.03$ spread intact. Lab 05 gives the same recommendation from Dr. GRPO for a different reason (difficulty bias across prompts); here the reason is outliers within a prompt, and the two arguments point the same way.

### Excluding failures instead of punishing them

A rollout whose render times out or whose scorer was unreachable is not a bad painting; it is a missing measurement. About 1.5 percent of rollouts in the post failed this way, and 5.2 percent in the worst run. Scoring them as zero teaches the policy to avoid whatever those programs had in common, which is nothing, so it trains on noise. The post's fix is to return `None` and drop the rollout from the group. In the formula, the group statistics are computed over the valid subset $V \subseteq \{1, \dots, G\}$,

$$
\mu_V = \frac{1}{|V|} \sum_{j \in V} r_j, \qquad \hat A_i = \begin{cases} r_i - \mu_V & i \in V \\ 0 & i \notin V, \end{cases}
$$

and a rollout with zero advantage contributes zero to the policy gradient, so its tokens are neither pushed up nor down. If $|V| \le 1$ the group has no baseline and is dropped entirely. Two things to notice. The effective group size shrinks, so the baseline is noisier for that prompt, which is a small price. And exclusion is only correct when the failure is independent of the policy's behaviour. If long programs are what time out, then excluding timeouts removes the one signal that would teach the policy to write shorter ones, and the honest choice is a soft length penalty before the deadline (Lab 05's overlong shaping) rather than silence. In the post the failures were infrastructure; in your own environment, check the correlation between failure and program length before you choose.

### The LoRA-on-MoE trap

The base model in the post is Qwen/Qwen3.5-35B-A3B, a mixture of experts with 35B total and about 3B active parameters, fine-tuned with LoRA. The standard PEFT recipe names the modules to adapt: a list such as `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, written for dense models. This architecture names most of its projections differently and stores the routed experts as fused tensors, so the hand list matched only a fraction of the modules, and the post reports the adapter was training ten layers out of forty: thirty of forty layers were frozen and nobody had said so. Training still ran; the reward just did not move. The fix was `target_modules="all-linear"`, which attaches an adapter to every `nn.Linear` except the output head. The post notes that even `all-linear` leaves the fused expert tensors frozen (they are not `nn.Linear` modules), but everything else received an adapter and that was enough to learn.

The check you should run before any PEFT job on an unfamiliar architecture takes four lines: iterate over `model.named_modules()`, collect the layer index of every module that has a `lora_A` child, and assert that the set equals `range(num_layers)`. Print the trainable-parameter fraction too, but do not trust it alone: a hand list that catches attention in every layer can look healthy while every MLP is frozen. The post's other three trainer changes, also from the post, were the learning rate from 2e-5 to 5e-5 (the ceiling that the "LoRA Without Regret" guidance uses for GRPO), the scheduler from linear decay to `constant_with_warmup` with 5 warmup steps (linear decay had spent most of the learning rate by mid-run), and `scale_rewards` from group to `none` for the reason derived above.

## Build it small

The snippet below is the whole loop with every piece replaced by the smallest thing that has the same shape. The program is six strokes from a categorical vocabulary of 144 tokens (shape, position on a 6 by 6 grid, radius); the renderer is a numpy union of filled circles and squares on a 16 by 16 binary canvas; the reference is an ellipse that the DSL cannot draw exactly; the reward is a 0.05 gate for a non-blank canvas plus 0.95 times intersection over union; 2 percent of renders "time out" at random and return `None`; and the update is GRPO with the group mean and standard deviation computed over valid rollouts only, one optimizer pass per batch (so the importance ratio is one and clipping is inert, as in Lab 05's discussion of `num_iterations`). The brush field is twice the size of the canvas, so most random strokes miss it: that is how the toy gets its near-blank canvases.

```python
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); rng = np.random.default_rng(0)
S, K, G, P, D = 16, 6, 8, 16, 32          # canvas side, strokes per program, group size, groups per step, width
GRID, RAD = 6, (2, 4)                     # brush field is 6x6 cells over [-S/2, 3S/2): most cells miss the canvas
V = 2 * GRID * GRID * len(RAD)            # 144 stroke tokens: (shape, cell, radius); token V is "start"

def decode(tok):                          # token id -> (shape, cx, cy, r); shape 0 = circle, 1 = square
    shape, rest = divmod(tok, GRID * GRID * len(RAD))
    cell, ri = divmod(rest, len(RAD))
    gx, gy = divmod(cell, GRID)
    return shape, (gx + 0.5) * 2 * S / GRID - S / 2, (gy + 0.5) * 2 * S / GRID - S / 2, RAD[ri]

yy, xx = np.mgrid[0:S, 0:S]
def render(prog):                         # numpy renderer: union of filled circles and squares on a binary canvas
    canvas = np.zeros((S, S), bool)
    for tok in prog:
        shape, cx, cy, r = decode(int(tok))
        canvas |= ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) if shape == 0 else (abs(xx - cx) <= r) & (abs(yy - cy) <= r)
    return canvas

ref = ((xx - 8) / 6.5) ** 2 + ((yy - 8) / 3.5) ** 2 <= 1      # procedural reference: an ellipse, not in the DSL

def reward(prog):                         # gate 0.05 for a non-blank canvas + 0.95 * IoU; None for a failed render
    if rng.random() < 0.02: return None   # simulated infrastructure failure (timeout), not the policy's fault
    img = render(prog)
    iou = (img & ref).sum() / max((img | ref).sum(), 1)
    return 0.05 * float(img.any()) + 0.95 * iou

class Policy(nn.Module):                  # categorical over stroke tokens, conditioned on slot and previous token
    def __init__(s):
        super().__init__()
        s.slot, s.prev = nn.Embedding(K, D), nn.Embedding(V + 1, D)
        s.head = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, V))
    def rollout(s, n):                    # sample n programs; keep per-token log-probs with grad (single-pass GRPO)
        prev, toks, lps = torch.full((n,), V), [], []
        for t in range(K):
            dist = torch.distributions.Categorical(logits=s.head(s.slot(torch.full_like(prev, t)) + s.prev(prev)))
            tok = dist.sample(); toks.append(tok); lps.append(dist.log_prob(tok)); prev = tok
        return torch.stack(toks, 1), torch.stack(lps, 1)

pi = Policy(); opt = torch.optim.Adam(pi.parameters(), 1e-3)
for step in range(401):
    toks, lp = pi.rollout(P * G)
    raw = [reward(p) for p in toks.tolist()]
    r = torch.tensor([np.nan if x is None else x for x in raw]).view(P, G)
    ok = ~r.isnan(); n = ok.sum(1, keepdim=True).clamp(min=1)          # group statistics over valid rollouts only
    mean = torch.where(ok, r, 0.).sum(1, keepdim=True) / n
    std = (torch.where(ok, (r - mean) ** 2, 0.).sum(1, keepdim=True) / n).sqrt()
    adv = torch.where(ok, (r - mean) / (std + 1e-4), 0.)               # failed rollouts get zero advantage
    loss = -(adv.view(-1) * lp.mean(1)).mean()                          # REINFORCE with the group baseline
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        imgs = [render(p) for p in toks.tolist()]
        blank = np.mean([not im.any() for im in imgs]); cov = np.mean([im.mean() for im in imgs])
        best = max((im & ref).sum() / (im | ref).sum() for im in imgs)
        print(f"step {step:3d} reward {r[ok].mean():.3f} blank {blank:.2f} coverage {cov:.2f} "
              f"best_iou {best:.2f} failed {(~ok).float().mean():.3f}")
print("\n".join("".join("#" if v else "." for v in row) for row in render(toks[r.nan_to_num(-1).argmax()].tolist())))
```

What to expect qualitatively: the mean reward rises quickly, the blank fraction goes to zero first, paint coverage grows toward the ellipse's area (about 0.28 of the canvas is the target, and the DSL overshoots it), and the reward then plateaus because the grid is too coarse to trace an ellipse. Output from one CPU run with this seed (about eight seconds on a laptop, PyTorch 2.13):

```
step   0 reward 0.170 blank 0.03 coverage 0.19 best_iou 0.45 failed 0.031
step  50 reward 0.351 blank 0.01 coverage 0.35 best_iou 0.50 failed 0.039
step 100 reward 0.490 blank 0.00 coverage 0.52 best_iou 0.54 failed 0.023
step 150 reward 0.542 blank 0.00 coverage 0.52 best_iou 0.53 failed 0.023
step 200 reward 0.547 blank 0.00 coverage 0.53 best_iou 0.53 failed 0.039
step 250 reward 0.549 blank 0.00 coverage 0.54 best_iou 0.53 failed 0.016
step 300 reward 0.549 blank 0.00 coverage 0.54 best_iou 0.53 failed 0.000
step 350 reward 0.549 blank 0.00 coverage 0.54 best_iou 0.53 failed 0.016
step 400 reward 0.549 blank 0.00 coverage 0.54 best_iou 0.53 failed 0.008
```

Three things to read off, and one to be honest about. The reward rose from 0.17 to 0.55 with no labels and a non-differentiable renderer in the loop. The blank fraction was the first thing to go, which is the toy version of the post's finding that in every run the first thing the model learned was to stop producing bad paintings. The failed fraction hovers around the 2 percent I injected and never affects the curve, because those rollouts contribute nothing. And the honest part: by step 150 every rollout in the batch is the same program (the best IoU equals the mean, and coverage stops moving), which is entropy collapse. At learning rate 3e-3 the same run collapsed by step 50 to the same 0.53 plateau; at 3e-4 it took 400 steps to reach 0.53. The final canvas printed at the end is one wide blob that covers the ellipse and spills past it: the policy found the DSL's best approximation and then stopped exploring. The fix is the one from Lab 05 (temperature above one during rollouts, an entropy bonus, or clip-higher with several optimizer passes), and exercise 3 asks you to add it.

## Build it real

Two versions: the post's, which you should read as the reference configuration, and the scaled-down one that runs on the 5090.

The post's configuration, all values from the post. Base model Qwen/Qwen3.5-35B-A3B with a LoRA on `all-linear`, bf16, gradient checkpointing. Pure RL with TRL's `GRPOTrainer`, no SFT stage. Reward as in the math section, with the judge Qwen3-VL-30B-A3B-Instruct comparing against 4 random references from the 178-painting pool and HPSv3 as the aesthetic scorer; the pool itself was generated by four open models writing p5.brush sketches from openly licensed hibiscus photos, refined over three rounds of vision-model feedback, and rated one painting at a time by the author. Learning rate 5e-5, `constant_with_warmup` with 5 warmup steps, `scale_rewards none`, 8 rollouts per step, 240 episodes, per-device batch 1 with gradient accumulation 8, max completion length 8192, sampling with top-p 0.95 and top-k 20. Renders in headless Chromium with a 90 second timeout; failures excluded as `None`. Judge-led and hps-led ran 110 steps and hps-only 60, on one H200. The launch command from the post:

```bash
hf jobs uv run train/watercolour_grpo.py --flavor h200 --timeout 48h --secrets HF_TOKEN -- \
  --env-url https://<you>-watercolour-env.hf.space \
  --model Qwen/Qwen3.5-35B-A3B --lora --all-linear --bf16 --gradient-checkpointing \
  --subject 'a peach hibiscus' --references 4 \
  --top-p 0.95 --top-k 20 \
  --lr 5e-5 --lr-scheduler constant_with_warmup --warmup-steps 5 \
  --scale-rewards none \
  --steps 110 --n-episodes 240 --num-generations 8 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --max-completion-length 8192 \
  --run-tag my-run --out <you>/watercolour-grpo --push-to-hub
```

What it produced, from the post: mean group reward from the first third to the final third of each run went from 0.58 to 0.71 for hps-only, 0.45 to 0.72 for judge-led, and 0.57 to 0.82 for hps-led. The more weight the judge carried, the lower the start and the noisier the climb; judge-led spent its first thirty steps nearly flat. In hps-only, three quarters of the rise came from bad paintings becoming rare, the best painting of each step moved only +0.034 while the median moved +0.155, and the run had a ceiling of 0.771 even if every rollout matched its good ones. With the judge present the top of the distribution moved too, and paint coverage doubled (0.11 to 0.23 in judge-led, 0.13 to 0.30 in hps-led) where hps-only barely moved it. One more finding worth keeping: the system prompt asked for fifteen to thirty filled shapes, the policy produced seven to nine on average, and the count barely correlated with reward; the policy is not rewarded for obeying that sentence, so it does not.

This does not fit on one 32 GB card as written: a 35B MoE in bf16 is 70 GB of weights before any adapter, and the environment needs a browser farm and two hosted scorers. `recipes/paint_grpo.py` keeps the loop and shrinks every piece. The policy is a 0.5B instruct model in bf16 with a LoRA (or full fine-tuning; both fit with room for vLLM colocated at a modest KV budget, using the same `GRPOTrainer` settings as `recipes/grpo.py` from Lab 05). The medium is a tiny drawing DSL, one command per line (`circle x y r`, `rect x y w h`, `fill gray`, `bleed amount`), that the recipe parses with a strict grammar and renders in numpy to a small grayscale canvas in milliseconds, so the step time is generation again and the render farm is gone. The reward is the same shape as the post's: a 0.05 gate (parses, at least one command, non-blank canvas, no unknown commands), a 0.05 soft length term, and 0.90 of an image-similarity score against a reference image (intersection over union on thresholded ink, or one minus a normalized L1 distance on the blurred canvas; the recipe exposes both). There is no learned scorer and no judge, deliberately: the point of the first run is to see the loop work with a reward you can read, and exercise 5 adds a pairwise judge on top.

Arguments. `--smoke` runs three steps with a group of four and a 64-token completion cap, prints one rendered canvas per step as ASCII, and exits; run it first, every time you touch the reward. `--steps` is the number of optimizer steps (start at 100). `--group` is $G$, the rollouts per prompt (default 8; the post used 8). `--reference` is either a path to a small grayscale image or the name of a procedural reference the recipe ships (`ellipse`, `ring`, `two-blobs`); the ellipse is the one from the snippet above. `--out` is the output directory for the adapter, a CSV of per-step reward, blank fraction, coverage and failed fraction, and a contact sheet of the best canvas per step so you can watch the paintings change. Logs to watch are the ones from Lab 05 (`reward`, `reward_std`, `completions/mean_length`, `kl`, `frac_zero_var`) plus the recipe's `blank_frac` and `coverage`; `blank_frac` should reach zero in the first few dozen steps and `coverage` should move toward the reference's ink area without overshooting it by more than a factor of two.

Time, as a formula, because it depends on your completion length. A step of $P$ prompts, $G$ rollouts and $L$ tokens generates $P G L$ tokens; with 16 prompts, a group of 8 and 256-token programs that is about 33k tokens, and the step time is that divided by the tokens per second vLLM sustains for a 0.5B model on the card (measure it once with `--bench-gen`), plus a training forward and backward over the same tokens, which for 0.5B parameters is small. The numpy render adds milliseconds per rollout and does not appear. A hundred steps is therefore minutes, not hours, and you can afford five seeds per setting, which is the budget you need to say anything about a reward change with the judge-noise numbers from the math section in mind.

The in-browser station "paint" shows the same loop live: each step it samples a group of candidate programs, renders them side by side in the browser, prints the reward under each canvas and the group-normalized advantage next to it, and you can watch the blank canvases disappear from the group in the first steps and the spread inside the group shrink later.

## How it goes wrong

Blank canvases win. Symptom: the reward is flat or rising while every render is white or a single faint wash. Cause: the learned terms of the reward were never validated on degenerate inputs and happen to score emptiness above a mediocre painting, or the gate is missing. Fix: a gate whose value on blank, single-colour and text-covered canvases you have verified by hand, and a coverage metric logged every step so you see it before the judge does.

The reward rises, the paintings stop changing. Symptom: the group's rewards converge to one value, every rollout in a group renders the same picture, `reward_std` and entropy fall together. Cause: entropy collapse under group normalization, which the toy above shows by step 150 at learning rate 1e-3 and by step 50 at 3e-3; the post also notes that within each run the paintings look alike and the rewards inside each group get closer as training advances. Fix: rollout temperature at or above one, an entropy bonus, clip-higher with more than one optimizer pass, or a smaller learning rate and more steps.

One outlier flattens the group. Symptom: with `scale_rewards` on group, the reward moves only when a group contains a gate rejection, and the advantages among honest rollouts are tiny. Cause: the standard deviation is dominated by the outlier, as derived above. Fix: `scale_rewards none`, which the post adopted, and a lower gate weight so a rejection is a clear negative without being the only thing the step learns.

Failures trained as negatives. Symptom: about one to five percent of rollouts score exactly zero with no visible defect in the program; the reward curve is noisy and the policy drifts toward shorter or simpler programs without the pictures improving. Cause: timeouts and scorer outages counted as reward zero. Fix: return `None`, exclude from the group, drop groups with fewer than two valid rollouts, and log the failed fraction so a rise in it shows up as an infrastructure problem and not as a policy one.

The adapter trains a fraction of the model. Symptom: the reward does not move over dozens of steps, gradient norms are small, and the trainable-parameter count is lower than you expected. Cause: the LoRA `target_modules` list does not match the architecture's module names; in the post's MoE it left thirty of forty layers frozen. Fix: `target_modules="all-linear"`, and the per-layer adapter assertion from the math section before the first step.

The learning rate is gone by mid-run. Symptom: the reward rises for the first fraction of the run and then flattens while everything else looks healthy. Cause: a linear-decay scheduler on a short run has spent most of the learning rate by the time the policy escapes the initial degenerate outputs. Fix: `constant_with_warmup`, as the post did, with a handful of warmup steps.

The judge is being played. Symptom: the judge term rises faster than the aesthetic term, the paintings acquire a feature (a border, a signature-like mark, text, a colour cast) that the references do not have, and a human rater disagrees with the judge's ordering. Cause: the vision-language judge has a preference the pool does not encode, and the policy found it. Fix: the gate's cheat detector (the post's rejects text on the canvas), position-swapped comparisons, a second judge held out from training, and a monthly human re-rating of the best canvases (Lab 09).

Degenerate repetition. Symptom: programs become a single shape call repeated with tiny offsets, or the same colour laid down dozens of times; coverage rises, length rises, the picture is a blob. Cause: the length term and a coverage-sensitive scorer reward both, and repetition is the cheapest way to raise both. Fix: cap the length term (it is 0.05 in the post for this reason), penalize duplicate calls in the gate, and read fifty programs at every checkpoint.

## Measure it

For the loop itself: mean group reward by thirds of the run (the post's table), the fraction of rollouts under 0.3 (the post's definition of a bad painting; it fell from 99 to 16 across judge-led's thirds), the blank fraction and paint coverage per step, and the best and median reward per step reported separately. The post's argument that hps-only got more reliable without getting better rests entirely on the best moving +0.034 while the median moved +0.155; if you only log the mean you cannot make that distinction. For the reward's components: log each term separately, because a composite reward that rises tells you nothing about which term rose, and a judge term rising while the aesthetic term falls is the signature of judge exploitation. For the judge: score the same image twice and compute agreement (the post lists this as untested), and score each candidate against a held-out reference set the policy never saw. For the policy: KL to the reference at the end (Lab 05's few hundredths of a nat is the target), mean completion length against the reward, and the count of distinct programs within a group as an entropy proxy. What is good: blank fraction at zero within the first tenth of the run, a median that keeps rising after the blank fraction hits zero, a best that moves at all, and a human rating of the final best canvases that agrees with the judge's ordering on at least a clear majority of pairs. There is no number for taste; there is a pool, and the pool is a set of choices you can defend.

## Exercises

1. In the group-of-eight example from the math section, compute the advantages of the seven honest rollouts with and without the gate rejection under both `scale_rewards` settings, and state the ratio by which their spread shrinks. Check: with group scaling the spread among the seven drops from about one standard deviation to about 0.13; with `none` it is unchanged at $\pm 0.03$.

2. Run the snippet with the failure probability at 0.10 and compare the curves with `None` exclusion (as written) against scoring failures as zero. Check: with exclusion the reward curve is within noise of the 0.02 run; with zeros, the reward the policy sees is lower and noisier and the plateau arrives later.

3. Add entropy to the snippet: sample with temperature 1.5 during rollouts and compute the log-probabilities under temperature 1 for the update, or add $0.01 \times$ the mean entropy to the objective. Check: the collapse is delayed past step 150 and the best IoU rises above 0.53 at least once; then report whether the mean reward at step 400 is higher or lower, because exploration is not free.

4. Replace the IoU reward with a binomial judge: for each candidate, draw four references from a fixed pool of DSL renders with known IoU to the ellipse, declare a win when the candidate's IoU exceeds the reference's, and use the share of wins. Check: with four references the reward takes five values, the group's `reward_std` is larger than with IoU, and the curve at learning rate 1e-3 is noticeably noisier; raising to sixteen references smooths it.

5. Run `recipes/paint_grpo.py --smoke`, then a 100-step run on `--reference ellipse` at group 8 with five seeds. Check: `blank_frac` reaches zero before step 20 in every seed, and the step-100 reward's seed-to-seed standard deviation is what you must beat before claiming a reward change helped.

6. Write a deliberately exploitable gate (accept any program with at least one command) and a length term with weight 0.3, run 200 steps, and describe the degenerate program the policy finds. Then restore the weights and rerun from the same checkpoint. Check: the second run's coverage falls back toward the reference's area and the number of distinct commands per program rises.

## Test yourself

1. Why does the policy gradient not need the renderer to be differentiable, and what would you gain and lose by making it so?

<details><summary>Answer</summary>
The gradient of $\mathbb{E}_{z \sim \pi_\theta}[R(\rho(z))]$ is $\mathbb{E}[R(\rho(z)) \nabla_\theta \log \pi_\theta(z)]$; the reward enters as a scalar weight and its dependence on $z$ is never differentiated, so $\rho$ and $R$ can be a browser and two remote models. A differentiable renderer would give you $\nabla_z R(\rho(z))$, a per-stroke direction that says how to move a coordinate to raise the reward, which is far lower variance than a scalar. You would lose the discrete program (tokens are not differentiable, so you would be optimizing continuous stroke parameters instead), the freedom to use any library, and the legibility that was the reason for painting with code in the first place.
</details>

2. The post's judge compares against four references in both orders. Under the Bradley-Terry model, what is the expected judge score of a candidate whose quality equals the pool's median, and why does that not equal 0.5 in general?

<details><summary>Answer</summary>
The expectation is $\mathbb{E}_{r}[\sigma(s_c - s_r)]$ over the pool. With $s_c$ at the median, half the references have $s_r$ above $s_c$ and half below, but $\sigma$ is nonlinear, so the average of $\sigma$ over an asymmetric spread of differences is not $\sigma(0)$. If the "okay" tier sits far below the candidate and the "love" tier only slightly above, the wins against the okay tier are near certain and the losses against the love tier are near coin flips, and the expectation is well above 0.5. This is also why the pool composition (half love, half okay in the post) is a reward-shaping choice, not a detail.
</details>

3. Derive the variance of the judge score for $n$ references and say how many references would be needed to make the judge's noise standard deviation, at weight 0.6, smaller than 0.05 on the total reward.

<details><summary>Answer</summary>
$J = W/n$ with $W \sim \text{Binomial}(n, p)$, so $\mathrm{Var}[J] = p(1-p)/n \le 1/(4n)$. The judge's contribution has standard deviation $0.6 \sqrt{p(1-p)/n} \le 0.3 / \sqrt{n}$. Setting $0.3 / \sqrt{n} \le 0.05$ gives $n \ge 36$ references, or 72 vision-model calls per rollout with both orders, which at 8 rollouts per step and 240 episodes is why nobody does it and why the group average has to carry the load instead.
</details>

4. A colleague proposes making the gate multiplicative, $R = \text{gate} \times (\text{len} + w_J J + w_H H)$, so a rejected sketch gets exactly zero. What changes in the group advantage, and when would you prefer it?

<details><summary>Answer</summary>
Nothing changes for honest rollouts, and a rejected rollout's reward is now exactly zero rather than near zero, so the outlier is slightly larger and the standard-deviation shrinkage under group scaling slightly worse. The real difference is in what is exposed to hacking: with an additive gate, a rejected sketch can still collect the learned terms (a blank canvas that the aesthetic scorer happens to like), so the additive form relies on the learned terms being low there. The multiplicative form guarantees zero. You would prefer it when you do not trust the learned scorers on degenerate inputs, which is most of the time, and pair it with `scale_rewards none`.
</details>

5. Explain, using the advantage formula, why excluding a failed rollout is not the same as giving it the group's mean reward, even though both produce zero advantage for that rollout.

<details><summary>Answer</summary>
Both give the failed rollout $\hat A = 0$. But a rollout scored at the mean still enters the mean and the standard deviation: it does not move $\mu$ (by construction) yet it lowers $\sigma$ by adding a zero-deviation term to the average of squared deviations, so every other advantage in the group is inflated by a factor of $\sqrt{G / |V|}$. With exclusion the statistics are computed over the valid rollouts only and the others are unchanged. At $G = 8$ with one failure the inflation is about 7 percent, small but systematic, and it grows when failures cluster.
</details>

6. The post's policy ignored the instruction to paint fifteen to thirty shapes and produced seven to nine. Is this reward hacking?

<details><summary>Answer</summary>
No. Reward hacking is raising the reward through a channel the designer did not intend. Here the reward does not pay for shape count, the count barely correlates with reward in any run, and the policy allocated its effort to what the reward measures. It is a demonstration that in pure RL the system prompt is not a constraint, only an initialization: anything you want enforced must be in the reward or in the gate. The hacking version of this story would be the policy discovering that a particular count raised the judge's score without improving the picture.
</details>

7. Spot the bug in this group-statistics code for a batch with `None` rewards:

```python
r = torch.tensor([0.0 if x is None else x for x in raw]).view(P, G)
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
```

<details><summary>Answer</summary>
Failed rollouts are turned into zeros and then included in the mean and standard deviation, so they receive a large negative advantage and are trained as bad paintings, and they inflate $\sigma$ for everyone else. The fix is a validity mask: compute the mean and standard deviation over valid entries only, set the advantage of invalid entries to zero, and drop groups with fewer than two valid rollouts. Note also that `torch.std` uses the unbiased estimator by default, which is fine but differs from the population version used in the snippet, so do not compare the two runs' `reward_std` directly.
</details>

8. Your render farm can run 8 renders in parallel at 80 seconds each and generation takes 60 seconds per group of 8. You are offered a policy twice as fast to sample. How much does the step time improve?

<details><summary>Answer</summary>
The step is 60 seconds of generation plus 80 seconds of rendering (all eight in parallel), 140 seconds; halving generation gives 30 plus 80, 110 seconds, a 21 percent improvement. Doubling the render parallelism instead does nothing here (the eight already fit), but halving the render latency gives 60 plus 40, 100 seconds, a 29 percent improvement. In the post's regime, where rendering is 70 to 80 percent of the step, the model is the wrong thing to optimize; the renderer is.
</details>

9. Why does the post keep half the references in the "okay" tier, and what would happen to the judge's signal late in a run if the pool were "love" only?

<details><summary>Answer</summary>
A weak early policy compared only against the best references loses every comparison, so $J = 0$ for every rollout in the group and the judge term contributes zero advantage; the run would have to be carried by the aesthetic scorer until the policy is good enough to win sometimes. The okay tier gives the early policy something it can beat. Late in the run the opposite happens: a strong policy beats the okay tier every time, those comparisons carry no information, and the judge's effective $n$ halves. That is why the post lists moving the mix from easy to hard as a next step: a curriculum on the reference pool.
</details>

10. The toy collapses to a single program at IoU 0.53. Is that a failure of GRPO or of the DSL?

<details><summary>Answer</summary>
Both, and you should separate them. The DSL's grid spacing is 5.33 pixels on a 16-pixel canvas with radii of 2 and 4, so the best achievable IoU against the ellipse is well below one; the plateau's height is the DSL's. The fact that the whole batch became one program is GRPO's: once every rollout in a group is identical the advantage is zero everywhere and learning stops, whether or not a better program exists. Raising the plateau needs a finer DSL; escaping it needs exploration. The post's version of the same distinction is that hps-only had a ceiling of 0.771 (the reward's) and the paintings within each run looked alike (the optimizer's).
</details>

## What will change, what will not

The durable core is the decomposition: a policy over programs, a renderer you do not differentiate through, a reward computed on the render, and a policy gradient with a group baseline. That structure will outlive p5.brush, Chromium, HPSv3 and the particular judge, because it is what lets you swap any of them. The Bradley-Terry reading of a pairwise judge and the binomial variance of a share-of-wins score are theorems about comparisons, not about vision models, and they will hold for whatever judge you use next. The interaction between a gate outlier and the group standard deviation is arithmetic, and it will keep biting anyone who leaves reward scaling on by default with a bimodal reward.

Reward hacking is permanent here as everywhere (Lab 05), and image rewards give the optimizer more surface than a math checker does: blank canvases, repetition, text on the canvas, and whatever the next judge happens to like. The detectors (a gate you verified by hand, per-term logging, a held-out judge, and looking at the pictures) are the durable defense; the specific exploits will be replaced.

What will change: the cost structure. The post's step is 70 to 80 percent rendering because a browser rasterized watercolour in software on a CPU; a GPU renderer or a cheaper medium moves the bottleneck back to generation, and a 4B policy (which the post reports could already write valid sketches) moves the bill down by an order of magnitude. Expect single-turn painting with eyes closed to give way to loops where the policy sees its own render and revises, because the post's reference pool was itself made with three rounds of vision feedback and the later rounds were better. Expect the pool, not the hyperparameters, to be where the work goes: the post says the pool is the reward function, and that moves the job from tuning to curation, which Lab 01 covers for text.

What is open: whether a judge that only ever sees the pool can teach anything the pool does not contain, how consistent a vision-language judge is with itself (the post lists scoring the same image twice as untested), and whether the legibility of programs survives the policy getting good, when 150 lines of shapes may become as opaque as pixels.

## Read next

1. Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons, Bradley and Terry, 1952. The model behind every pairwise judge and the reason a share of wins is the right statistic.
2. Deep Reinforcement Learning from Human Preferences, Christiano, 2017. Learning a reward from comparisons and optimizing a policy against it; the template the reference pool follows.
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, Shao, 2024. GRPO as used by TRL, with the group-normalized advantage this chapter takes apart.
4. Understanding R1-Zero-Like Training: A Critical Perspective, Liu, 2025. The case against dividing by the group standard deviation, which the post reached from the other direction.
5. LoRA: Low-Rank Adaptation of Large Language Models, Hu, 2021. What `target_modules` selects and why a wrong list fails silently.
6. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, Zheng, 2023. Position bias and the reason the post presents every comparison in both orders; read with Lab 09.
7. Scaling Laws for Reward Model Overoptimization, Gao, 2022. What happens when a policy is optimized against a proxy longer than the proxy deserves; the aesthetic scorer here is such a proxy.
8. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis, Wu, 2023. The lineage of the HPS aesthetic scorers, of which the post uses the v3 successor.
