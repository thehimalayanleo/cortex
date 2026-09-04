---
title: "Lab 19: Post-training multimodal models: understanding and generation"
kind: permanent
topics: [lab]
chapter: 19
station: none
recipe: recipes/vlm_grpo.py
reading_time: 75 min
---

## What you will be able to do

1. Draw the sample, score, learn, sync loop with its three roles (rollout engine, trainer, weight sync) and say exactly what changes in each role when the prompt contains an image and when the policy is a diffusion model rather than a token decoder.
2. Trace an image from pixels to positions in the decoder's sequence (processor, vision encoder, projector, interleaving), write the loss mask for a vision-language rollout, and cost the rollout in tokens.
3. Explain why a flow-matching sampler has no policy gradient, derive the marginal-preserving SDE that gives every denoising step a Gaussian transition probability, and write the Flow-GRPO update from that transition; place DiffusionNFT, ReFL and Flow-DPPO relative to it.
4. Design a verifiable reward for a geometry task and a learned reward (PickScore) for images, name what each can be hacked on, and specify the rollout record you log so that hacking is visible in a table before it is visible in the pictures.
5. Scale both lanes to one RTX 5090: a 2B-class VLM with LoRA and GRPO on a small verifiable image task via `recipes/vlm_grpo.py`, and a small flow model with a cheap reward via the snippet in this chapter.

## The idea in one paragraph

Everything in Lab 05 assumed the model reads text and writes text. Two changes cover most of what people now call multimodal post-training. In the first, the model still writes text but its prompt contains an image: a vision encoder turns the image into a few hundred vectors, a small projector maps them into the language model's embedding space, and they are dropped into the sequence where the image placeholder was, so the decoder reads them like any other tokens and the RL loop (sample a response, score it with a checker, compute a group advantage, update, sync weights to the sampler) is unchanged. In the second, the model writes an image, and here the loop breaks at the first step: a flow-matching sampler is a deterministic ODE, so the same prompt and the same starting noise always produce the same picture, there is no distribution over actions, and there is nothing to take a policy gradient of. Flow-GRPO's fix is to add noise to the sampler in a way that keeps the marginal distributions of the original model, which turns every denoising step into a Gaussian transition with a log-probability, and after that the same GRPO update applies with the denoising steps playing the role of tokens. The framework this chapter follows, Miles from RadixArk, keeps one infrastructure for both lanes and swaps only the rollout engine, the reward and the loss.

The sources for this chapter are the RadixArk post "Post-training with Miles to Understand and Generate the Multimodal World" (Miles Team, September 3, 2026), its Geo3K VLM documentation page, its Cosmos3 diffusion recipe page, and the Miles documentation index. Where this chapter says "the post" or "the docs" it means those pages, and where they do not state something it says so. The derivations are mine and are labeled as such.

## The math

### One loop, two lanes

The post's loop is four words: sample, score, learn, sync. In its words, "feed the model an image and a prompt, generate a response, score it, and update the model from the reward." Three roles run it. A rollout engine generates: SGLang for language and vision-language models, SGLang-diffusion for generative models. A trainer updates the weights: Megatron-LM or FSDP2 for LLMs and VLMs, FSDP2 for the diffusion transformer (the post's phrase is "FSDP2 trains the DiT"). And a weight-sync path carries the new parameters back to the rollout engines, which the post's final figure labels "sync: new weights, or LoRA adapters". The docs index adds that new weights "reach the engines in-loop in seconds, even on a trillion-parameter model", with peer-to-peer RDMA as the fast path when rollout and training live on different machines, and that in Miles-diffusion LoRA weights can be shipped to colocated rollout engines over CUDA IPC instead of replicating the whole model.

The post's figure 7 is the whole design: the stages are shared and only what flows between them changes. For a VLM the rollout engine emits a response and the trainer trains on it. For a diffusion model the rollout engine emits a trajectory plus an output (the denoising path and the decoded image), the reward scores the output, and the trainer trains on the denoising steps. The reward stage accepts verifiers or model-based scorers in both lanes.

Write the loop as an objective so the two lanes can be compared term by term. With a prompt $x$ (text, possibly with an image), a policy $\pi_\theta$, a reward $R$, and a frozen reference $\pi_{\text{ref}}$,

$$
J(\theta) = \mathbb{E}_{x} \, \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)} \big[ R(\tau, x) \big] - \beta \, \mathbb{E}_{x} \, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big),
$$

where $\tau$ is whatever the policy emits: a token sequence $y_1, \dots, y_T$ for a VLM, a sequence of latent states $x_T \to x_{T-1} \to \dots \to x_0$ for a diffusion model (the post writes both chains exactly this way). Lab 05 derived the policy gradient $\mathbb{E}[R \, \nabla_\theta \log \pi_\theta(\tau \mid x)]$ and the GRPO estimator with a group baseline. Both need one thing from the policy: a log-probability of the thing it sampled, decomposed into per-step terms. For a VLM that is $\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$, available for free. For a diffusion sampler it is not available at all until you change the sampler, which is the subject of the third subsection.

### How an image becomes tokens, and what that does to the loss mask

The post's description of the VLM pipeline is four sentences and I will keep its order. "The processor first prepares the image, choosing a resolution and dividing it into patches. A vision encoder turns those patches into visual representations. A projector then maps them into the embedding space used by the language model." Then: "When the prompt contains an image placeholder, those visual representations are inserted at that position. From the decoder's perspective, visual tokens are simply part of the same sequence as text tokens." Video, the post adds, is the same idea with a time dimension: visual information spanning multiple frames.

Write it down. Let the processor resize the image to $H \times W$ pixels and cut it into patches of side $p$, giving $n_p = (H/p)(W/p)$ patches. The vision encoder $E$ (a ViT in the post's figure) maps patches to vectors, $E: \mathbb{R}^{n_p \times 3p^2} \to \mathbb{R}^{n_v \times d_v}$, where $n_v$ may be smaller than $n_p$ if the encoder or the processor merges neighbouring patches. The projector $W_{\text{proj}}: \mathbb{R}^{d_v} \to \mathbb{R}^{d}$ maps each vector into the decoder's embedding width $d$. The prompt is a token sequence with one placeholder position $j$; the decoder's input embeddings are

$$
h = \big( e(x_1), \dots, e(x_{j-1}), \; W_{\text{proj}} E(I)_1, \dots, W_{\text{proj}} E(I)_{n_v}, \; e(x_{j+1}), \dots, e(x_{m}) \big),
$$

a sequence of length $m - 1 + n_v$ in which $n_v$ positions were never produced by the token embedding table. Everything downstream is the transformer of Lab 11. The response tokens $y_1, \dots, y_T$ are appended and sampled one at a time as usual.

Two consequences follow, and the pages are silent on both, so this is my reading of the arithmetic. The loss mask: the policy's log-probability is a sum over response tokens only, as in Lab 04's answer-only loss. Text prompt positions have mask zero because they are given; the $n_v$ image positions have mask zero for a stronger reason, there is no token there to have a probability, and a loop that computes the loss over "all non-padding positions" will try to score a label that does not exist. It is the first thing to assert. The trainable parameters: nothing in the objective says whether $E$ and $W_{\text{proj}}$ receive gradient, and the pages do not state what Miles trains on Geo3K. A LoRA on the language model with the encoder and projector frozen is the cheap default and the one `recipes/vlm_grpo.py` uses; unfreezing the projector is exercise 4.

For rollouts the consequence is cost. Every rollout carries $n_v$ extra prefix tokens, and $n_v$ depends on the resolution the processor chose. With patch side $p = 14$, a 2 by 2 merge of neighbouring patches, and a 448 by 448 image (assumptions typical of current open VLMs; check your model's processor), $n_p = 32 \times 32 = 1024$ and $n_v = 256$. A 1,024 by 1,024 image at the same settings is about $1{,}336$ tokens, five times the prefix for the same question. Whether the $G$ samples of one prompt share the encoder and prefill through a prefix cache is an engine detail the pages do not discuss, and it is the difference between paying for the encoder once or $G$ times per prompt. One hardware fact from the docs page applies to this card: on Blackwell, which "currently does not support fa3", the recipe passes `--sglang-mm-attention-backend sdpa` and `--attn-implementation flash_attention_2`.

### RL for understanding: Geo3K with a verifiable reward

The post's example trains Qwen3.5-9B on Geo3K, a set of geometry problems with a diagram, a question, and a numeric answer. The held-out problem in its figure reads "Find the value of the variable $y$ in the figure", with angles marked $(3x - 15)^\circ$, $(y + 25)^\circ$ and $105^\circ$, and the correct answer is 50. The docs page uses the processed dataset `chenhegu/geo3k_imgurl`, whose rows have three fields, `problem` (text), `answer` (a string such as "270"), and `images` (a list), and its default model is Qwen3-VL-8B-Instruct, with a supported list running from Qwen3-VL-2B-Instruct up to Qwen3-VL-235B-A22B-Instruct plus the Thinking variants of each size. The post and the docs page use different models for the same task; I report both.

The reward. The docs say three configurations were tried, a task-specific checker with a tolerance of 0.05 "to handle rounding in ground truth labels", the same with tolerance zero, and "the default math RM", that "all three performed similarly", and that the default math reward is used "for simplicity". The SFT preparation script on the same page formats the target as `Answer: \boxed{<answer>}`, which tells you the checker's contract: extract the boxed answer, compare with the gold string. This is Lab 06's verifiable reward with an image in the prompt, and it is binary. Write $R(y, x) = \mathbb{1}[\text{extract}(y) = \text{gold}(x)]$. The GRPO objective is then Lab 05's, unchanged:

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}, \qquad
J = \mathbb{E}\Big[ \frac{1}{G} \sum_i \frac{1}{|o_i|} \sum_t \min\big(\rho_{i,t} \hat A_i, \, \mathrm{clip}(\rho_{i,t}, 1 \pm \epsilon) \hat A_i\big) - \beta \hat D_{i,t} \Big],
$$

with the ratio and the $k_3$ KL estimator as before, and $|o_i|$ counting response tokens only.

What an eval reward of 0.429 at baseline means. The post's figure labels the base checkpoint "eval 0.429". With a binary reward, the mean over a held-out set is the fraction of problems the checker accepts, so the base model solves 42.9 percent of held-out Geo3K under this checker. That number is the most useful thing to know before an RL run, because it predicts how much of the sampling budget GRPO will throw away: a prompt with per-sample pass rate $p$ gives an all-correct group with probability $p^G$ and an all-wrong one with $(1 - p)^G$, and either way the advantage is zero everywhere. If every prompt sat at $p = 0.429$ with $G = 8$, $0.429^8 \approx 0.0011$ and $0.571^8 \approx 0.011$, about 1.2 percent of groups wasted, close to GRPO's best case. The real waste is larger because pass rates are spread across prompts (some diagrams the model always reads correctly, some never); `frac_zero_var` from Lab 05 measures it. The post's interactive figure also plots response length and follows one problem across checkpoints; the text states only the baseline number, so I cannot report the curve's endpoint.

One numerical fact from the docs page is arithmetic, so it belongs here. Their first checker returned "format scores" of 0 and 0.9, and under fp32 "fractional values like 0.9 can't be exactly represented, so when all samples in a group have the same reward, reward - mean doesn't equal zero, creating spurious gradient signal." The size of the effect: 0.9 is stored as the nearest representable value, a sum of $G$ copies divided by $G$ need not return it exactly, and when it does not, $r_i - \mu \approx 6 \times 10^{-8}$ for every sample with a rounding-dependent sign, and $\sigma$ is the same $6 \times 10^{-8}$, so $(r_i - \mu) / (\sigma + \varepsilon)$ with $\varepsilon = 10^{-6}$ is about $\pm 0.056$ for a group that carries no information. I checked in PyTorch: for $G \in \{3, 6, 7, 12, 24\}$ identical 0.9 rewards give an advantage magnitude of 0.056; for $G \in \{5, 8, 10, 16\}$ the mean happened to be exact and the advantage was zero, which is why the artifact looks random when you meet it. The docs' fix is binary rewards, with a float16 cast of the reward tensor as the fallback; the cleaner fix is an explicit all-equal test that zeroes the group's advantages, which `recipes/vlm_grpo.py` does.

### RL for generation: why an ODE has no policy gradient

The post contrasts the two chains. Tokens are "a sequence of discrete decisions" $y_1 \to y_2 \to \dots \to y_T$; a diffusion model "passes through a sequence of increasingly clean latent states" $x_T \to x_{T-1} \to \dots \to x_0$. Then: "A standard flow-matching sampler typically follows a deterministic ordinary differential equation (ODE). Given the same prompt and starting noise, it follows the same path every time."

Set up the flow-matching model so that the statement can be turned into an equation. I use the convention where $s \in [0, 1]$ runs from noise to data, because it makes the derivation cleaner; the Flow-GRPO paper and most samplers use $t = 1 - s$, and I will translate at the end. Data $x_1 \sim p_{\text{data}}$, noise $\varepsilon \sim \mathcal{N}(0, I)$, the interpolant

$$
x_s = s \, x_1 + (1 - s) \, \varepsilon,
$$

and the model $u_\theta(x, s, c)$ trained by flow matching to regress the velocity, $u_\theta(x_s, s, c) \approx \mathbb{E}[x_1 - \varepsilon \mid x_s, c]$, with $c$ the conditioning. Write $p_s$ for the marginal density of $x_s$. The sampler integrates the ODE

$$
\frac{dx}{ds} = u_\theta(x, s, c), \qquad x(0) = \varepsilon,
$$

with an Euler step $x_{k+1} = x_k + u_\theta(x_k, s_k, c) \, \Delta$ on a grid $s_k = k / N$. This is the post's figure 3 in symbols: a text encoder produces $c$, the DiT "predicts velocity $v_\theta$", the update repeats over denoising steps, and a VAE decoder turns $x_1$ into pixels.

Now the problem. For a fixed $(c, \varepsilon)$ the endpoint $x_1 = \Phi_\theta(\varepsilon, c)$ is a deterministic function, and the only randomness is $\varepsilon$, which the model did not choose. The per-step "policy" $p(x_{k+1} \mid x_k)$ is a Dirac delta at $x_k + u_\theta \Delta$; its log-probability is $+\infty$ on the path and $-\infty$ off it, and $\nabla_\theta \log p$ is undefined. In code, `torch.distributions.Normal(mean, 0.0)` raises a `ValueError`, which is the right behaviour. A policy-gradient method needs a distribution over actions that the parameters shape, and the ODE provides none. The post's figure 5 caption: "The deterministic sampler traces the same path on every run."

### The SDE that keeps the marginals, and the policy it defines

The post says Flow-GRPO works "by injecting controlled noise into the ODE and turning the sampling process into an equivalent stochastic differential equation (SDE). Each denoising transition can then be treated as a stochastic action with a tractable transition probability." The word doing the work is "equivalent": the SDE must have the same marginals $p_s$ as the ODE, otherwise the noise would change what the model generates before any training happened. Here is why such an SDE exists and what it is. This derivation is mine, following the standard Fokker-Planck argument.

The ODE transports $p_s$ according to the continuity equation $\partial_s p_s = -\nabla \cdot (u \, p_s)$. Consider instead the SDE $dx = b(x, s) \, ds + g_s \, dW$ with a scalar noise scale $g_s$. Its density obeys the Fokker-Planck equation $\partial_s p = -\nabla \cdot (b \, p) + \tfrac{1}{2} g_s^2 \Delta p$. Choose $b = u + \tfrac{1}{2} g_s^2 \nabla \log p_s$. Then

$$
-\nabla \cdot (b \, p_s) + \tfrac{1}{2} g_s^2 \Delta p_s
= -\nabla \cdot (u \, p_s) - \tfrac{1}{2} g_s^2 \nabla \cdot (p_s \nabla \log p_s) + \tfrac{1}{2} g_s^2 \Delta p_s
= -\nabla \cdot (u \, p_s),
$$

because $p_s \nabla \log p_s = \nabla p_s$ and $\nabla \cdot \nabla p_s = \Delta p_s$. The extra drift and the diffusion cancel exactly, and the SDE has the same marginals as the ODE for any $g_s \ge 0$. At $g_s = 0$ it is the ODE. That is the whole family of "equivalent" samplers, and the score $\nabla \log p_s$ is the price of admission.

You do not have a score model, but you do not need one, because for this interpolant the score is a function of the velocity. Condition on $x_s = x$: since $x = s \, x_1 + (1 - s) \varepsilon$ and $u = \mathbb{E}[x_1 - \varepsilon \mid x]$, you have $\mathbb{E}[x_1 \mid x] = (x - (1 - s) \mathbb{E}[\varepsilon \mid x]) / s$ and therefore $u = x / s - \mathbb{E}[\varepsilon \mid x] / s$, that is, $\mathbb{E}[\varepsilon \mid x] = x - s \, u$. For a Gaussian perturbation the score is $\nabla \log p_s(x) = -\mathbb{E}[\varepsilon \mid x] / (1 - s)$ (Tweedie's identity applied to the noise component with standard deviation $1 - s$). So

$$
\nabla \log p_s(x) = -\frac{x - s \, u_\theta(x, s, c)}{1 - s},
\qquad
dx = \Big[ u_\theta - \frac{g_s^2}{2 (1 - s)} \big( x - s \, u_\theta \big) \Big] ds + g_s \, dW.
$$

Translate to the paper's convention with $t = 1 - s$, $v = -u$ (the velocity pointing toward noise), $dt = -ds$, and $\sigma_t = g_{1-t}$: the drift becomes $v_\theta + \frac{\sigma_t^2}{2t}(x_t + (1 - t) v_\theta)$, which is the form Flow-GRPO writes. The paper's noise schedule, as I recall it, is $\sigma_t = a \sqrt{t / (1 - t)}$ with a scalar $a$ that sets how much exploration you inject; check the paper for the exact discretization before copying it. The snippet below uses $g_s = a \sqrt{1 - s}$ instead, which makes the correction coefficient $g_s^2 / (2(1 - s)) = a^2 / 2$ a constant and sends the noise to zero at the data end; it is a legitimate member of the family and easier to read.

Discretize with Euler-Maruyama on the grid $s_k = k / N$, $\Delta = 1 / N$:

$$
x_{k+1} = \mu_\theta(x_k, k, c) + g_{s_k} \sqrt{\Delta} \, \xi_k, \qquad \mu_\theta(x_k, k, c) = x_k + \Big[ u_\theta - \frac{g_{s_k}^2}{2 (1 - s_k)} (x_k - s_k u_\theta) \Big] \Delta, \qquad \xi_k \sim \mathcal{N}(0, I).
$$

Now each step is a Gaussian policy, $\pi_\theta(x_{k+1} \mid x_k, c) = \mathcal{N}\big(\mu_\theta(x_k, k, c), \, g_{s_k}^2 \Delta \, I\big)$, with a log-density

$$
\log \pi_\theta(x_{k+1} \mid x_k, c) = -\frac{\| x_{k+1} - \mu_\theta(x_k, k, c) \|^2}{2 g_{s_k}^2 \Delta} - \frac{D}{2} \log \big( 2 \pi g_{s_k}^2 \Delta \big),
$$

which depends on $\theta$ through $\mu_\theta$ and is differentiable. The trajectory's log-probability is the sum over $k$, and this is the quantity the post means by "tractable transition probability". Note that these log-densities are routinely positive (the variance is small and $D$ is the latent dimension), which surprises people who are used to token log-probabilities; only differences and ratios matter.

The Flow-GRPO update is now GRPO from Lab 05 with steps in place of tokens. For a prompt $c$, sample a group of $G$ trajectories from the SDE, decode each endpoint, score it with $r_i = R(\text{decode}(x^{(i)}_N), c)$, form the group-normalized advantage $\hat A_i$, and maximize

$$
J = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{N} \sum_{k=0}^{N-1} \min\big( \rho_{i,k} \hat A_i, \; \mathrm{clip}(\rho_{i,k}, 1 \pm \epsilon) \hat A_i \big) - \beta \, \mathrm{KL}\big( \pi_\theta(\cdot \mid x_k^{(i)}) \,\|\, \pi_{\text{ref}}(\cdot \mid x_k^{(i)}) \big),
\qquad
\rho_{i,k} = \frac{\pi_\theta(x^{(i)}_{k+1} \mid x^{(i)}_k)}{\pi_{\text{old}}(x^{(i)}_{k+1} \mid x^{(i)}_k)}.
$$

The KL here is exact and closed-form, because both policies are Gaussians with the same variance: $\mathrm{KL} = \| \mu_\theta - \mu_{\text{ref}} \|^2 / (2 g_{s_k}^2 \Delta)$. No $k_3$ estimator is needed. Read the post's figure 5 caption with this in hand: "The stochastic sampler injects noise, so its trajectories diverge and each one scores differently; they are colored by group-relative advantage, blue where it is negative and red where it is positive."

Which steps get noise. The Cosmos3 recipe page adds a detail the derivation does not force: only some steps are stochastic. It passes `--diffusion-sde-candidate-steps 8,9,10,11`, and a selector called `epoch_global_random_choice` "draws two steps per epoch" from that list, because "the Cosmos3 checkpoint's Karras flow-sigma grid puts head steps 1 to 7 at sigma > 0.96 with |dt| < 0.02; steps 8 to 11 are the useful high-noise segment", and "step numbers are not transferable across sigma-grid families: re-derive candidates from |dt| when changing model or grid." In the formula, a step with $g_{s_k} = 0$ has no log-probability term and no ratio, so the sum over $k$ runs over the stochastic steps only; the trainer recomputes only those, and the exploration sits where a step actually moves the latent. The pages do not say why two steps rather than four; treat the count as a knob.

### The alternatives, as the post places them

DiffusionNFT "takes a different route. Rather than treating the denoising trajectory itself as a sequence of policy actions, it scores completed generations and turns those rewards into positive and negative training signals inside the flow-matching objective." So the reward enters the regression loss that trained the model in the first place, with positively and negatively scored generations pushing the velocity field in opposite directions, and the sampler stays as it was. The post gives no formula and neither will I; the paper is in the reading list. ReFL and Flow-DPPO are named by the post as methods that "make different choices about where and how the reward influences training", with the remark that "the field is still actively exploring the best formulations", and that is all the post says about them. Miles-diffusion treats all of these as swappable pieces: the post lists "loss, rollout dynamics, training-batch preparation, reward, and denoising-step strategy" as the things an algorithm changes while the distributed system stays fixed.

### Rewards for images, and where they break

The Cosmos3 recipe uses PickScore as the reward, colocated with training and rollout, and reports that it raises PickScore (`rollout/reward/raw_mean`) "from ~0.77 to ~0.85 over 250 rollouts" on text-to-image at 832 by 480, one frame; the post's figure 6 plots "the average PickScore over a batch of held-out eval prompts". That is all the pages say about the reward model. PickScore, from its paper, is a CLIP-style scorer trained on human choices between pairs of generated images for the same prompt: a learned preference model in Lab 05's sense, fit on a finite set of comparisons, trustworthy near the images it was fit on and extrapolating freely elsewhere. An optimizer will move toward whatever the scorer likes that the raters never had a chance to disagree with: a palette, a contrast level, a centred subject, a "look". This is the aesthetics hack, the image analogue of the length hack. Its symptoms are a reward that keeps rising after the pictures stop improving, images for different prompts converging on one style, and a second scorer, a judge model run the way Lab 09 prescribes, or a person disagreeing with the training scorer's ordering of late checkpoints. The pages do not report any of this on Cosmos3 and I am not claiming it happened there; Lab 15's gate and pairwise judge are the same argument for a different reward. A verifiable image reward (an OCR check that requested text appears, which the Miles diffusion docs use in their Stable Diffusion 3.5 recipe) has holes too, but they are the checker's holes, and you can read the checker.

### The rollout record

The rollout table is where hacking shows up first, and it is the same table in both lanes. Log one row per sample with these fields:

| field | VLM understanding | diffusion generation |
|---|---|---|
| group id | prompt id (question plus image hash) | prompt id |
| input | the question text and a reference to the image | the prompt text and the seed noise |
| completion or sample | the full response text | a thumbnail of the decoded image and the trajectory id |
| reward | the checker's 0 or 1 and the extracted answer | the scorer's value (and each term if composite) |
| advantage | group-normalized, with the group mean and std | the same |
| KL | mean per-token $k_3$ against the reference | mean per-step closed-form Gaussian KL |
| length or steps | response tokens | which steps were stochastic |
| log-probability | sum over response tokens under the sampler | sum over stochastic steps under the sampler |

Reading it. Sort by group and look at the spread of rewards inside each group; an all-equal group has zero advantage and should be counted, not trained on. Plot reward against KL over the run: a reward that keeps rising while the KL accelerates is a policy leaving the reference's region, which is where a learned scorer stops being trustworthy. In the VLM table, read the highest-advantage completion in each group and ask whether the extracted answer is the model's answer or one of several numbers in the text; several boxed answers, or a boxed answer before the reasoning, are the standard exploits, and length growing while eval reward is flat is their signature. In the diffusion table, look at the top-advantage thumbnails across prompts; a shared palette, composition or texture the prompts did not ask for means the scorer is being optimized rather than the prompt. And compare the sampler's log-probability with the trainer's recomputation at the first optimizer pass; they must agree to numerical precision or the ratio is meaningless, which is Lab 05's synchronization check in a new costume. The snippet below prints four such rows.

### The cost of a step

Write the step time as the sum of its phases,

$$
T_{\text{step}} = T_{\text{rollout}} + T_{\text{score}} + T_{\text{train}} + T_{\text{sync}},
$$

and cost the rollout in both lanes. A VLM rollout for $P$ prompts, $G$ samples and $L$ response tokens generates $P G L$ tokens with a KV cache, one decoder forward over one position each, plus a vision-encoder forward per image (or per sample, if the prefix is not shared) and a prefill over $n_v$ image tokens and the question. A diffusion rollout for the same $P G$ samples runs $N$ denoising steps, each one DiT forward over all latent tokens with no cache, because the whole latent changes every step. With a VAE spatial factor $f$, a DiT patch size $q$ and an $H \times W$ image, $n_\ell = (H / (f q)) (W / (f q))$; for 832 by 480 with $f = 8$ and $q = 2$ (assumptions; the Cosmos3 page states the VAE's 4x temporal compression, not its spatial factor), $n_\ell = 52 \times 30 = 1{,}560$, so $N = 30$ steps is 30 full forwards over 1,560 tokens per sample, doubled if classifier-free guidance runs two passes, plus a VAE decode and the reward model's forward. A 1,000-token text response is 1,000 cached forwards over one token each. Per sample the diffusion rollout does more work by a large factor, and video multiplies $n_\ell$ by the number of latent frames. That is why the post's design records trajectories in the rollout engine rather than regenerating them in the trainer, why "reward workers" are a separate role, and why Miles-diffusion ships LoRA weights over IPC rather than full weights: in this lane generation is the step, and everything else is arranged to keep the generation engine fed.

Two more cost facts from the Cosmos3 page, both about precision rather than speed. Timesteps are kept in fp32 with no scaling, because "the Karras grid is non-integer and sgl-d conditions on exact fp32 values; bf16 rounds 993.25 to 992", and conditioning tensors are passed through without a dtype cast because "mRoPE position ids sit at about 15000 where bf16 spacing is 128; a boundary cast scrambles rotary phases". These are mismatches between the rollout engine and the trainer of exactly the kind that make $\rho_{i,k} \ne 1$ at the first pass, and they are the diffusion analogue of Lab 05's "the reference is not the model you think".

## Build it small

The snippet is Flow-GRPO on a 2-D flow model, every piece present and none of them hidden. Stage one trains a conditional flow model by flow matching on a ring of eight Gaussian modes, where the prompt $c \in \{0, 1, 2, 3\}$ owns the pair of adjacent modes $(2c, 2c + 1)$, so the "pretrained generator" spreads its mass over both modes of its pair. Stage two is RL: the reward is 1 if the endpoint lands nearer the counterclockwise mode of the pair than the clockwise one, the sampler is the SDE derived above with $g_s = a \sqrt{1 - s}$, the group is $G = 16$ samples per prompt, the update is the clipped surrogate with two optimizer passes over the same trajectories and an exact Gaussian KL to a frozen copy of the pretrained flow. A second metric the reward cannot see, the fraction of endpoints within 0.5 of any true mode, tells you whether the policy is choosing between modes or leaving the data manifold. Pass the KL weight as the first argument.

```python
import torch, torch.nn as nn, math, copy, sys
torch.manual_seed(0)
N, G, P, A, EPS = 10, 16, 4, 0.8, 0.2          # sampler steps, group size, prompts, noise scale a, clip band
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # KL weight to the frozen pretrained flow
ANG = torch.arange(8) * (math.pi / 4)           # 8 modes on a ring; prompt c in 0..3 owns the pair (2c, 2c+1)
CENTERS = 2.0 * torch.stack([ANG.cos(), ANG.sin()], 1)

def sample_data(c):                              # x_1 ~ p_data(. | c): either mode of the pair, plus jitter
    return CENTERS[2 * c + torch.randint(0, 2, c.shape)] + 0.15 * torch.randn(len(c), 2)

class Flow(nn.Module):                           # velocity field u_theta(x, s, c); s = 0 is noise, s = 1 is data
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(4, 16)
        s.net = nn.Sequential(nn.Linear(19, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(s, x, t, c): return s.net(torch.cat([x, t[:, None], s.emb(c)], 1))

def step_dist(model, x, k, c):                   # one Euler-Maruyama transition as a Gaussian policy
    s, dt = k / N, 1.0 / N
    u = model(x, torch.full((len(x),), s), c)
    mean = x + (u - 0.5 * A * A * (x - s * u)) * dt     # drift = u + (g^2/2) grad log p_s, with g_s = A sqrt(1-s)
    std = A * math.sqrt(1 - s) * math.sqrt(dt)           # ODE sampler is the same with A = 0: a Dirac, no log-prob
    return torch.distributions.Normal(mean, std), x + u * dt

@torch.no_grad()
def rollout(model, c, stochastic=True):          # returns the trajectory (N+1 points) and per-step old log-probs
    x, xs, lps = torch.randn(len(c), 2), [], []
    for k in range(N):
        dist, x_ode = step_dist(model, x, k, c); xs.append(x)
        x = dist.sample() if stochastic else x_ode
        lps.append(dist.log_prob(x).sum(-1))
    xs.append(x); return torch.stack(xs), torch.stack(lps)   # shapes (N+1, B, 2) and (N, B)

def reward(x, c):                                # 1 if the endpoint is nearer the counterclockwise mode of the pair
    return ((x - CENTERS[2 * c + 1]).norm(dim=-1) < (x - CENTERS[2 * c]).norm(dim=-1)).float()
def on_manifold(x):                              # quality check the reward does not see: near any true mode
    return ((x[:, None] - CENTERS[None]).norm(dim=-1).min(1).values < 0.5).float().mean().item()

flow = Flow(); opt = torch.optim.Adam(flow.parameters(), 2e-3)
for it in range(3000):                           # stage 1: flow matching, the "pretrained" generator
    c = torch.randint(0, 4, (256,)); x1, e, s = sample_data(c), torch.randn(256, 2), torch.rand(256)
    loss = ((flow(s[:, None] * x1 + (1 - s[:, None]) * e, s, c) - (x1 - e)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

ref = copy.deepcopy(flow).requires_grad_(False); opt = torch.optim.Adam(flow.parameters(), 3e-4)
for step in range(41):                           # stage 2: Flow-GRPO on the SDE sampler
    c = torch.arange(P).repeat_interleave(G)     # P prompts, G samples each: the group shares the prompt
    xs, lp_old = rollout(flow, c)
    r = reward(xs[-1], c).view(P, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    if step == 0:                                # the rollout record you should log: group, sample, reward, advantage, log-prob
        for i in range(4): print("record", int(c[i]), [round(v, 2) for v in xs[-1][i].tolist()], float(r.view(-1)[i]),
                                 round(float(adv[i]), 2), round(float(lp_old[:, i].sum()), 2))
    for _ in range(2):                           # two passes on the same trajectories, so the ratio does work
        d = [(step_dist(flow, xs[k], k, c)[0], step_dist(ref, xs[k], k, c)[0]) for k in range(N)]
        lp = torch.stack([p.log_prob(xs[k + 1]).sum(-1) for k, (p, _) in enumerate(d)])
        kl = torch.stack([torch.distributions.kl_divergence(p, q).sum(-1) for p, q in d])   # exact, same std
        ratio = (lp - lp_old).exp()              # per-step importance ratio, shape (N, B)
        surr = torch.min(ratio * adv, ratio.clamp(1 - EPS, 1 + EPS) * adv)
        opt.zero_grad(); (-(surr - BETA * kl).mean()).backward(); opt.step()
    if step % 10 == 0:
        c_ev = torch.arange(P).repeat_interleave(64)
        x_ode = rollout(flow, c_ev, stochastic=False)[0][-1]
        print(f"step {step:2d} sde reward {r.mean():.2f} on-manifold {on_manifold(xs[-1]):.2f} | "
              f"ode reward {reward(x_ode, c_ev).mean():.2f} on-manifold {on_manifold(x_ode):.2f} | "
              f"kl/step {kl.mean():.3f} zero-var groups {(r.std(1) == 0).float().mean():.2f}")
```

Output from one CPU run with this seed and no KL (`python flow_grpo_toy.py 0`, about ten seconds including the flow-matching stage):

```
record 0 [1.21, 1.48] 1.0 0.65 10.37
record 0 [1.41, 1.59] 1.0 0.65 4.72
record 0 [2.11, -0.05] 0.0 -1.44 7.83
record 0 [1.54, 1.1] 1.0 0.65 10.06
step  0 sde reward 0.56 on-manifold 0.92 | ode reward 0.53 on-manifold 0.94 | kl/step 0.001 zero-var groups 0.00
step 10 sde reward 0.55 on-manifold 0.81 | ode reward 0.63 on-manifold 0.91 | kl/step 0.127 zero-var groups 0.00
step 20 sde reward 0.75 on-manifold 0.92 | ode reward 0.74 on-manifold 0.85 | kl/step 0.215 zero-var groups 0.00
step 30 sde reward 0.83 on-manifold 0.69 | ode reward 0.80 on-manifold 0.62 | kl/step 0.451 zero-var groups 0.00
step 40 sde reward 0.97 on-manifold 0.61 | ode reward 0.95 on-manifold 0.66 | kl/step 0.561 zero-var groups 0.50
```

And with the KL at $\beta = 0.03$ (`python flow_grpo_toy.py 0.03`), same seed, same records at step 0:

```
step  0 sde reward 0.56 on-manifold 0.92 | ode reward 0.53 on-manifold 0.94 | kl/step 0.001 zero-var groups 0.00
step 10 sde reward 0.55 on-manifold 0.86 | ode reward 0.63 on-manifold 0.93 | kl/step 0.102 zero-var groups 0.00
step 20 sde reward 0.75 on-manifold 0.94 | ode reward 0.74 on-manifold 0.90 | kl/step 0.140 zero-var groups 0.00
step 30 sde reward 0.81 on-manifold 0.77 | ode reward 0.79 on-manifold 0.88 | kl/step 0.255 zero-var groups 0.00
step 40 sde reward 0.94 on-manifold 0.83 | ode reward 0.92 on-manifold 0.88 | kl/step 0.298 zero-var groups 0.50
```

What I observed. The pretrained model starts at reward 0.56 under the SDE and 0.53 under the ODE, the coin flip between the two modes of each pair that flow matching learned, with samples on the manifold 92 to 94 percent of the time. The four records are a group in miniature: three samples landed near mode 1 (centred at $(1.41, 1.41)$) with reward 1 and advantage $+0.65$, one landed near mode 0 at $(2.11, -0.05)$ with reward 0 and advantage $-1.44$; the log-probabilities are positive, as the math section warned, and vary by a factor of two within the group because the noise draws differ. Over forty steps the SDE reward climbs to 0.97 and the ODE reward, measured with the deterministic sampler the model will actually be used with, climbs with it to 0.95: training with injected noise improves the noiseless sampler, which is the practical claim of Flow-GRPO and the reason the SDE is a training device rather than a change to the product. By step 40 half the groups have zero variance and the run is done in Lab 05's sense.

Now the honest part, which is why the on-manifold column exists. Without a KL term the on-manifold fraction falls from 0.92 to 0.61 while the reward rises. Print the endpoints after training and they have drifted past mode 1, counterclockwise, to points like $(1.5, 1.9)$, half a unit from any true mode. The reward only asks "nearer to mode 1 than mode 0", which stays true beyond mode 1, so the policy is rewarded for leaving the data distribution and nothing in the objective objects. That is reward hacking on a two-line reward, and it is what the aesthetics hack on PickScore looks like when the picture is two numbers. With $\beta = 0.03$ the reward reaches 0.94 instead of 0.97 and the on-manifold fraction holds at 0.83 to 0.88, with the final per-step KL about half the unregularized run's; at $\beta = 0.1$ (not shown) reward 0.89 and on-manifold 0.86 to 0.92. The trade is Lab 05's, and the reward alone would never have told you it was being made.

Set `A = 0` and the snippet raises a `ValueError` from `Normal` at the first step: there is no policy without the noise. Set `A = 2.0` and the samples are off the manifold at step 0, before any training, because ten Euler-Maruyama steps integrate a strongly stochastic SDE badly even though the continuous SDE has the right marginals; the error grows with $g_s^2 \Delta$, and moderate noise and a few stochastic steps (the Cosmos3 recipe's choice) are both responses to that.

## Build it real

The post's runs are not one-card runs. The Geo3K docs default to eight GPUs (`MILES_SCRIPT_NUM_GPUS=8`), a Qwen3-VL-8B-Instruct policy, the Megatron backend (`MILES_SCRIPT_TRAIN_BACKEND=megatron`, or `fsdp`), the `chenhegu/geo3k_imgurl` dataset, and a launch of `./examples/geo3k_vlm/run_geo3k_vlm.sh` with `MILES_SCRIPT_MODEL_NAME` to swap the model (the page's example is `Qwen3-VL-4B-Instruct`). The Cosmos3 recipe is `scripts/run_diffusion_grpo_cosmos3_pickscore_t2i_4gpu.py`, four GPUs with training, rollout and PickScore colocated, a 16B model (8B understanding tower, frozen, plus 8B generation tower, with LoRA on the generation tower's attention projections only: `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`), a packed single-sample forward that forces `--rollout-microgroup-size 1`, and an environment variable `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` because "RL scores raw samples". Batch sizes, learning rates and group sizes for either recipe are in the scripts, not on the pages, so I do not report them.

`recipes/vlm_grpo.py` is the one-card version of the understanding lane. The policy is `Qwen/Qwen3-VL-2B-Instruct` (the smallest entry in the docs' supported list) in bf16, 4 GB of weights, with a LoRA on every linear layer of the language model and the vision encoder and projector frozen, trained with TRL's `GRPOTrainer` and the same settings as Lab 05's recipe: group-normalized advantages with `scale_rewards` exposed, the clipped ratio, the $k_3$ KL, and `num_iterations` for passes per batch. The loss mask is asserted, not assumed: before the first step the recipe builds one batch, checks that the number of supervised positions equals the number of response tokens, and refuses to start otherwise. Generation is colocated on the same card, through vLLM when the installed vLLM supports the model and its image inputs and through `transformers` generation otherwise; the recipe reports which at startup and prints tokens per second either way.

Arguments. `--model` is the Hugging Face id (default `Qwen/Qwen3-VL-2B-Instruct`; any model the processor and the LoRA target list can handle). `--dataset` is either `chenhegu/geo3k_imgurl` (the docs' Geo3K, with the boxed-answer checker as the reward) or `synthetic`, a task the recipe generates in-process: a rendered image of a few coloured shapes with a question such as "how many red circles are there" and an integer answer, so the loop runs offline and the checker is exact. `--steps` is the number of optimizer steps (start at 100 on Geo3K). `--group` is $G$ (default 8). `--lora` takes a rank (default 16; `0` means full fine-tuning of the language model, which fits for 2B but leaves less room for the KV cache). `--smoke` runs three steps on the synthetic task with a group of four and a 64-token cap, prints one rollout record per step with the fields from the table in the math section, and exits; run it every time you touch the reward or the mask. The logs are Lab 05's (`reward`, `reward_std`, `completions/mean_length`, `kl`, `clip_ratio`, `frac_zero_var`) plus `image_tokens` (the mean $n_v$ per prompt, so you notice when a resolution change doubles your prefill) and `extract_fail` (the fraction of completions with no boxed answer, which should fall to near zero in the first steps and, if it rises later, is the model discovering that the checker tolerates something it should not).

Time, as a formula. A step of $P$ prompts, $G$ samples and $L$ response tokens generates $P G L$ tokens plus $P$ (or $P G$) vision-encoder forwards and $P G$ prefills of $n_v + |q|$ tokens. With 8 prompts, $G = 8$, $L = 512$ and $n_v = 256$, that is 32k generated tokens and 64 prefills of about 300 tokens; at whatever tokens per second the sampler sustains for a 2B model on the card (the recipe measures it with `--bench-gen`), the generated tokens set the step time, and the training forward and backward over the same 32k tokens plus 64 image prefixes is a fraction of it. A hundred steps is under an hour if the sampler is doing a few thousand tokens per second, which a 2B model on this card should; the base pass rate on Geo3K for a 2B model is the number to measure first, with the `frac_zero_var` arithmetic from the math section, before committing to the run.

For the generation lane there is no shipped recipe; this is the plan for this card. Replace the snippet's 2-D flow with a small open flow-matching text-to-image model in bf16 with a LoRA on the DiT's attention at a small resolution (Stable Diffusion 3.5 Medium is the model the Miles diffusion docs use for their two-GPU quick start; with the text encoders run once and offloaded and a 512 by 512 image, it fits in 32 GB). Make two to four middle steps stochastic, as the Cosmos3 recipe does, so the trainer recomputes only those. Use a reward you can read before a learned one: an OCR check that a requested word appears (the Miles docs' SD3.5 reward) or a compressibility score of the kind the DDPO paper used, then PickScore with a second scorer held out. Cost it first: at $N = 20$, $G = 8$, four prompts and a 1,024-token latent, one step is 640 DiT forwards over 1,024 tokens plus 32 VAE decodes and 32 scorer forwards, minutes per step on this card, which is the post's point about where the time goes.

## How it goes wrong

Spurious gradient from a non-binary reward. Symptom: groups whose samples all received the same reward still produce advantages of a few hundredths with random sign, and the run drifts without the reward moving. Cause: the docs' fp32 story; $(r - \mu) / (\sigma + \varepsilon)$ turns a $6 \times 10^{-8}$ rounding error into an advantage of 0.056. Fix: binary rewards where the task allows, an explicit all-equal test that zeroes the group, and the docs' float16 cast as a blunt instrument.

Image positions in the loss. Symptom: the trainer reports a token count per sample far above the response length, the loss is dominated by positions that never change, and the KL to the reference is large at step zero. Cause: the mask treats every non-padding position as a target, including the $n_v$ image positions and the question. Fix: the assertion in the recipe (supervised positions equal response tokens), run before the first step.

Rollout and trainer disagree on the image. Symptom: the importance ratios at the first optimizer pass are not one and the clip fraction is high from the start, Lab 05's synchronization symptom, but the weights are synchronized. Cause: the rollout engine's processor and the trainer's chose different resolutions or merge settings, so the two sides see different image token counts, or different attention backends produced different logits (the docs' Blackwell note is one instance). Fix: compare the two sides' log-probabilities on one batch before training, and pin the processor configuration on both.

The ODE was used for rollouts. Symptom: every sample in a group for the same seed is identical, `reward_std` is zero for every group, and the log-probabilities are infinite or the code errors on a zero scale. Cause: the sampler was not switched to the SDE, or the noise scale was set to zero on every step. Fix: at least one stochastic step with $g > 0$, and a check that `reward_std` is nonzero for most groups at step zero.

Too much noise. Symptom: at step zero, before any update, the SDE samples score worse than the ODE samples, and the on-manifold or quality metric is already down. Cause: the discretized SDE with a large $g_s^2 \Delta$ does not preserve the marginals the continuous SDE does; the toy shows this at $A = 2$. Fix: a smaller noise scale, stochastic steps only where the grid moves the latent (the Cosmos3 page's "re-derive candidates from $|dt|$"), and the step-zero comparison of SDE against ODE as a standing check.

The scorer is being optimized instead of the prompt. Symptom: PickScore rises past the point where the images improve, samples for different prompts converge on a style, and a held-out scorer or a person disagrees with the training scorer on late checkpoints. Cause: a learned reward extrapolated beyond its training pairs; the toy's on-manifold collapse is the same mechanism with a two-line reward. Fix: a KL weight tuned against a quality metric, a second scorer never used for training, a gate for degenerate images (Lab 15), and thumbnails in the rollout table.

Timestep precision. Symptom: on a model with a non-integer sigma grid, the trainer's log-probabilities drift from the sampler's on some steps only. Cause: the Cosmos3 page's bf16 rounding of timesteps (993.25 to 992) or of position ids, so the trainer conditions on a slightly different step than the sampler did. Fix: fp32 timesteps and pass-through conditioning dtypes, as the recipe's family config does.

Generation is the run. Symptom: GPU utilization is high, the reward curve is fine, and a hundred steps takes a day. Cause: the arithmetic in the cost section, $N$ full forwards per sample with no cache, doubled by guidance. Fix: fewer stochastic steps does not help (the sampler still runs all $N$), so reduce resolution, $N$ or $G$, cache the text-encoder outputs, and put the reward model on the same card only if it fits beside the sampler; then accept the bill, because in this lane the sampler is the cost and the post's infrastructure is built around that fact.

## Measure it

For the understanding lane: held-out pass rate under the same checker (the post's eval reward; 0.429 is the Qwen3.5-9B base on Geo3K), pass@k for $k$ up to 16 so you can tell sharpening from learning (Lab 05), response length against reward, `extract_fail`, the KL to the reference at the end (a few hundredths of a nat per token for a run that stayed close), and a second checker on the same completions as the hacking detector (Lab 06). What is good is a held-out pass rate that rises while pass@16 rises with it and length stays flat; a pass@1 that rises to meet a flat pass@16 is selection, not learning, and is still worth having.

For the generation lane: the training scorer's mean on held-out prompts (the post's figure 6; the Cosmos3 page's 0.77 to 0.85 over 250 rollouts is the scale of movement one recipe reports), the same prompts scored by a held-out scorer, a prompt-adherence check that the training reward does not contain (an OCR check, an object-count check from a VLM judge with the position-swapping and calibration cautions of Lab 09, or a person), a diversity measure across seeds for a fixed prompt (pairwise distance in a feature space, or simply the number of visually distinct images in a group of eight), the mean per-step KL, and the step-zero comparison of SDE and ODE samples under the scorer. What is good is a training score that rises with the held-out score and the adherence check, at a diversity that has not collapsed, and a KL that is still small when you stop. A training score that keeps rising alone is the aesthetics hack, and the earliest place you will see it is the rollout table's thumbnails, not any curve.

## Exercises

1. Verify the marginal-preserving property of the SDE numerically. Sample 10,000 endpoints from the snippet's pretrained flow with the ODE and with the SDE at $A = 0.8$ and compare the two histograms of the angle around the ring. Check: the mode weights agree within sampling noise; then raise $A$ to 2.0 and watch the SDE histogram smear, which is discretization error, not a property of the continuous SDE.

2. Make only two of the ten steps stochastic (say $k = 3$ and $k = 4$), as the Cosmos3 recipe does with its candidate steps, and rerun. Check: the reward still rises, the KL is computed on two steps only, and the run is faster per step by about the ratio of trainer forwards; report whether the on-manifold fraction is better or worse than the all-steps run at the same final reward.

3. Compute $n_v$ for your chosen VLM's processor at three resolutions (the processor's default, half, and double) by running one image through it and counting placeholder expansions. Check: the count scales roughly with pixel area, and the doubling costs you more prefill than the response length in a typical Geo3K rollout.

4. Run `recipes/vlm_grpo.py --dataset synthetic --steps 100` twice, once with the projector frozen and once with it trainable (a one-line change in the LoRA target list). Check: on the synthetic counting task, report held-out accuracy and the KL for both; then say which one you would trust on real images, given that only the language model saw the reward.

5. Reproduce the fp32 artifact: build reward tensors of $G$ copies of 0.9 for $G$ from 2 to 32, compute the group-normalized advantage with $\varepsilon = 10^{-6}$, and tabulate its magnitude. Check: the artifact appears for some $G$ and not others (I saw it at 3, 6, 7, 12 and 24 and not at 5, 8, 10, 16), and an explicit all-equal test removes it for every $G$.

6. Write the hackable reward on purpose: in the snippet, reward the endpoint's angle directly (larger angle, higher reward) instead of the nearer-mode test, run without KL, and describe what the policy does to the ring. Then add the KL and find the smallest $\beta$ at which the on-manifold fraction stays above 0.85 at step 40. Check: without KL the endpoints leave the ring entirely; the $\beta$ you find is the number you would have to defend for a real scorer.

## Test yourself

1. Show that the SDE $dx = [u + \tfrac{1}{2} g^2 \nabla \log p_s] \, ds + g \, dW$ has the same marginals as the ODE $dx = u \, ds$, and say what breaks when you discretize it.

<details><summary>Answer</summary>
Fokker-Planck for the SDE gives $\partial_s p = -\nabla \cdot (u p) - \tfrac{1}{2} g^2 \nabla \cdot (p \nabla \log p) + \tfrac{1}{2} g^2 \Delta p$; since $p \nabla \log p = \nabla p$, the last two terms cancel and what remains is the continuity equation of the ODE, so the marginals coincide for any $g \ge 0$. Discretization breaks it because Euler-Maruyama with step $\Delta$ is only first-order accurate and its error grows with $g^2 \Delta$; the score also blows up near $s = 1$ (the $1/(1-s)$ factor), so the noise scale must vanish there or the step must be tiny, which is why $g_s = a\sqrt{1 - s}$ or a schedule like Flow-GRPO's is used and why recipes make only some steps stochastic.
</details>

2. Why is the score expressible through the velocity for the linear interpolant, and what would you need if the interpolant were not linear?

<details><summary>Answer</summary>
For $x_s = s x_1 + (1 - s) \varepsilon$ the velocity is $u = \mathbb{E}[x_1 - \varepsilon \mid x_s]$ and $x_s = s \mathbb{E}[x_1 \mid x_s] + (1 - s) \mathbb{E}[\varepsilon \mid x_s]$, two linear equations in the two conditional means, so $\mathbb{E}[\varepsilon \mid x_s] = x_s - s u$; Tweedie then gives $\nabla \log p_s = -\mathbb{E}[\varepsilon \mid x_s] / (1 - s)$. The identity uses that the noise enters with a known scalar coefficient $(1-s)$ and that $x_s$ is linear in $(x_1, \varepsilon)$. For a general interpolant $\alpha_s x_1 + \sigma_s \varepsilon$ the same two-equation trick works with $\alpha_s, \sigma_s$ and their derivatives; for a nonlinear coupling you would need a separately trained score model or the denoiser's own estimate of $\varepsilon$.
</details>

3. In the snippet the log-probabilities are around $+5$ to $+10$ per trajectory. Explain the sign, and why it is harmless for GRPO but would matter for a maximum-likelihood evaluation.

<details><summary>Answer</summary>
Each step is a Gaussian with variance $g_{s_k}^2 \Delta$, which is about $0.064 (1 - s_k)$ here, so the density at a typical sample exceeds one and its log is positive; summing ten of them gives a positive number. GRPO uses only the ratio $\pi_\theta / \pi_{\text{old}}$ and the KL, both of which are differences of log-densities at the same point, so the normalizing constant and its sign cancel. For a likelihood evaluation you would be comparing densities across different discretizations or noise scales, where the constants do not cancel, and a positive log-likelihood in one grid means nothing against another.
</details>

4. The base VLM scores 0.429 on held-out Geo3K. A colleague proposes $G = 4$ to halve the rollout cost. Estimate the fraction of wasted groups under a uniform pass-rate model at $G = 4$ and $G = 8$, and then explain why the uniform model understates the waste.

<details><summary>Answer</summary>
At $p = 0.429$: $p^4 + (1-p)^4 = 0.034 + 0.106 = 0.14$ versus $p^8 + (1-p)^8 = 0.0011 + 0.011 = 0.012$, so $G = 4$ wastes about twelve times as many groups as $G = 8$ for half the samples, and the useful samples per rollout token are lower. The uniform model understates the waste because real pass rates are spread over prompts; by Jensen's inequality on the convex functions $p^G$ and $(1-p)^G$, the average of the wasted fraction over a spread of pass rates exceeds the wasted fraction at the average pass rate, and the spread on a diagram task is large (some diagrams are always read correctly).
</details>

5. Why does a group of identical fractional rewards produce a nonzero advantage in fp32, and why does the same group with binary rewards not?

<details><summary>Answer</summary>
Integers and $0$ and $1$ are exact in fp32, so the sum and the division by $G$ are exact, $r_i - \mu = 0$ exactly, and the advantage is exactly zero. A value like 0.9 is stored with a rounding error, and the reduction (sum then divide) can round differently from the stored value, giving $r_i - \mu$ of order $10^{-8}$ with the same sign for every sample; the standard deviation is of the same order, so $(r_i - \mu)/(\sigma + \varepsilon)$ is order $10^{-8} / (10^{-8} + 10^{-6}) \approx 0.01$ to $0.06$ rather than zero. Whether it happens depends on $G$ and the reduction order, which is why it looks intermittent, and why an explicit all-equal test is the right fix rather than a bigger $\varepsilon$.
</details>

6. Spot the bug in this Flow-GRPO recomputation of the log-probabilities:

```python
xs, lp_old = rollout(flow, c)                 # trajectory from the sampler
for _ in range(2):
    xs_new, lp = rollout(flow, c)             # recompute under the current policy
    ratio = (lp - lp_old).exp()
```

<details><summary>Answer</summary>
The second call draws a fresh trajectory, so `lp` is the log-probability of different states under the current policy, not of the sampled states `xs` under the current policy; the ratio compares densities at two different points and has no importance-sampling meaning, and the gradient does not flow because `rollout` is under `no_grad`. The correct recomputation evaluates $\log \pi_\theta(x_{k+1} \mid x_k)$ at the stored $x_k, x_{k+1}$ with gradients enabled, which is what `step_dist(flow, xs[k], k, c)[0].log_prob(xs[k + 1])` does in the snippet. This is the diffusion version of Lab 05's rule that the ratio is computed on the old samples.
</details>

7. Explain why a LoRA on the generation tower's attention only, with the understanding tower frozen "inside the training graph" as the Cosmos3 page puts it, still requires the understanding tower to run in the trainer's forward pass, and what that costs.

<details><summary>Answer</summary>
The page says the two towers process one packed text plus vision sequence, so the generation tower's activations depend on the understanding tower's outputs at every layer; freezing the understanding tower's parameters (by name fragments, per the page) stops their updates but not their forward computation, which must run to produce the activations the trainable parts consume. The cost is the full 16B forward per training step rather than 8B, plus the activation memory of the frozen half if it sits on the backward path, though no gradient is accumulated into its weights. Detaching the tower would change the model; freezing keeps it.
</details>

8. A run's PickScore rises from 0.77 to 0.85 while a held-out OCR check on the same prompts falls. Give two mechanisms that produce this pattern and the log field that separates them.

<details><summary>Answer</summary>
First mechanism: the scorer rewards a style (contrast, palette, composition) that the policy can produce without rendering the requested text, so the images get prettier and less faithful; the signature is thumbnails across prompts converging in look and a KL that grows steadily. Second mechanism: the noise scale is high enough that the SDE samples used for training are off the model's manifold, the policy learns to score well on those, and the ODE samples the OCR check sees are not the ones being scored; the signature is a gap between SDE and ODE rewards at the same step. The separating fields are the per-step KL and the paired SDE versus ODE evaluation; the first mechanism shows both samplers agreeing and drifting, the second shows them disagreeing.
</details>

9. Compare the per-sample rollout cost of a 1,000-token VLM response and a 20-step, 1,024-latent-token image with guidance, counting forwards times tokens, and say which of the two is easier to speed up by an engine improvement.

<details><summary>Answer</summary>
The VLM response is 1,000 decoder forwards over one new token each with a KV cache, plus the prefill of the question and $n_v$ image tokens once: roughly $1{,}000 + n_v + |q|$ token-forwards, memory-bound per step. The image is $20 \times 2 \times 1{,}024 = 40{,}960$ token-forwards through the DiT, compute-bound, plus a VAE decode. The diffusion sample is around forty times the token-forwards, and each is a full-sequence forward. Text generation benefits from batching, paged KV caches and speculative decoding (Lab 17); diffusion sampling benefits from fewer steps (distilled samplers), lower resolution, and caching the guidance branch, but not from a cache over steps, since the whole latent changes each step. The text side has more engine levers; the image side's lever is the sampler itself.
</details>

10. The Cosmos3 page says step numbers "are not transferable across sigma-grid families". Explain what a candidate step list means in the SDE formulation and how you would derive one for a new model.

<details><summary>Answer</summary>
The candidate list names the grid indices $k$ at which $g_{s_k} > 0$; every other step is an ODE step with no transition density and no term in the surrogate. Which indices are useful depends on how much the latent moves at each index, which is $|dt|$ times the drift on that model's grid; the page reports that on Cosmos3's Karras grid the first seven steps have sigma above 0.96 and $|dt| < 0.02$, so noise injected there is spent on steps that barely move the state. For a new model, print the grid's sigmas and $|dt|$ per step, pick the contiguous segment where $|dt|$ is largest at high noise, and confirm at step zero that SDE samples on those steps match ODE samples under the scorer before training.
</details>

## What will change, what will not

The durable part of the understanding lane is the sequence: an image becomes vectors in the decoder's embedding space, those positions carry no loss, and the RL loop is Lab 05's with a longer prefix. Encoders, projectors, merge factors and resolutions will keep changing, and so will the token counts; the loss mask rule and the cost formula will not. The Geo3K base rate and the zero-variance arithmetic are the same computation for any verifiable task at any base rate.

The durable part of the generation lane is the theorem: a deterministic sampler has no policy, and any SDE of the form drift plus $\tfrac{1}{2} g^2 \nabla \log p$ keeps the marginals and gives every step a Gaussian transition. That is a fact about Fokker-Planck, not about Flow-GRPO, and whatever replaces Flow-GRPO will either use the same family with a different schedule and choice of stochastic steps, or take DiffusionNFT's route of putting the reward into the regression objective and leaving the sampler alone. The post's own wording, that "the field is still actively exploring the best formulations", is the honest state of it; the noise schedule, the number of stochastic steps, and the clip and KL settings are the parts I expect to look different in a year.

The infrastructure claim will hold longer than any algorithm: one loop, three roles, and modality-specific pieces that plug into it. The post's figure 7 is a design you can copy at any scale, and this lab's recipes are that figure with the numbers shrunk. What will change is the cost structure: today the diffusion sampler is the step, and few-step distilled samplers, cheaper VAEs and better choices of stochastic steps will move the bill toward the reward model and the trainer, where the text lane already is.

Reward hacking is permanent, and image rewards give the optimizer more surface than a checker does: the toy left the data manifold on a two-line reward in forty steps. The detectors (a KL you tuned against a quality metric, a held-out scorer, a step-zero SDE versus ODE comparison, and thumbnails in the rollout table) are the durable defense. What is open: whether learned image rewards can be made robust enough to optimize for long without a human in the loop, whether the same weights can be post-trained to understand and generate in one loop (the post's closing sentence is that they will), and what the right unit of credit assignment is for a denoising trajectory, where the steps are not tokens and the last one is not where the decision was made.

## Read next

1. Flow-GRPO: Training Flow Matching Models via Online RL, Liu, 2025. The ODE-to-SDE conversion, the per-step Gaussian policy and the GRPO update this chapter derives; the post's reference for the method.
2. DiffusionNFT: Online Diffusion Reinforcement with Forward Process, Zheng, 2025. The alternative the post names, with the reward inside the flow-matching objective and no change to the sampler.
3. Training Diffusion Models with Reinforcement Learning, Black, 2023. DDPO; the first treatment of denoising steps as a Markov decision process with a policy gradient, and the source of the cheap compressibility and aesthetic rewards.
4. Flow Matching for Generative Modeling, Lipman, 2022. The interpolant, the velocity regression, and the ODE sampler that the SDE is built on.
5. Score-Based Generative Modeling through Stochastic Differential Equations, Song, 2021. The family of SDEs sharing marginals with a probability-flow ODE; the Fokker-Planck argument in section 3 is theirs.
6. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation, Kirstain, 2023. PickScore, the reward in the Cosmos3 recipe, and what it was trained on.
7. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, Xu, 2023. A learned image reward and ReFL, the reward-feedback fine-tuning method the post names.
8. Visual Instruction Tuning, Liu, 2023. The encoder-projector-decoder pipeline (LLaVA) that the post's figure 1 describes; read with Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning, Lu, 2021, which introduced the Geometry3K problems behind Geo3K.
