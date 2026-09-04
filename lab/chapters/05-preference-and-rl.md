---
title: "Lab 05: Preference optimization and RL for language models"
kind: permanent
topics: [lab]
chapter: 5
station: posttrain
recipe: recipes/dpo.py and recipes/grpo.py
reading_time: 80 min
---

## What you will be able to do

1. Train a reward model from pairwise comparisons, state exactly what it can and cannot identify, and explain why a per-prompt constant never matters.
2. Write the KL-regularized RLHF objective, derive its closed-form optimum, and turn that closed form into the DPO loss with every algebra step shown.
3. Derive PPO for a language model from the policy gradient up: importance ratio, clipped surrogate, value function, GAE recursion, token-level KL penalty, and a memory budget for the four models on one 32 GB card.
4. Write the GRPO objective from DeepSeekMath exactly, explain each normalization, know what Dr. GRPO and DAPO change and why, and implement one GRPO step in under 60 lines.
5. Diagnose likelihood displacement, over-optimization and reward hacking from the logs of a real run, and run DPO then GRPO on the 5090 with TRL.

## The idea in one paragraph

After SFT the model can answer, but it cannot tell a good answer from a slightly better one, because SFT only ever showed it one answer per question. Preference training shows it two and says which one a person preferred. The cleanest way to use that information is to fit a reward model that scores answers, then ask the policy to produce high-scoring answers while staying close to the SFT model, because a reward model fit on a few thousand comparisons is only trustworthy near the answers it was trained on. That objective, reward minus a KL penalty, has a closed-form solution, and if you substitute the solution back into the preference model the reward disappears and you get a loss you can apply directly to the policy: that is DPO. When the reward is not a learned model but a checker (the answer is right or it is not, the tests pass or they do not), you no longer need a reward model at all; you sample several answers per question, score them, and push the policy toward the ones that scored above the group's average: that is GRPO. Both are ways of spending a small amount of comparison signal without letting the model wander away from what SFT taught it.

## The math

### Preferences and the Bradley-Terry model

Fix a prompt $x$ and two responses $y_1, y_2$. A human says which is better. The Bradley-Terry model assumes there is a latent score $r(x, y)$ such that

$$
P(y_1 \succ y_2 \mid x) = \frac{e^{r(x, y_1)}}{e^{r(x, y_1)} + e^{r(x, y_2)}} = \sigma\big(r(x, y_1) - r(x, y_2)\big),
$$

where $\sigma(z) = 1 / (1 + e^{-z})$. It is the model you get if each response's perceived quality is its score plus independent Gumbel noise and the annotator picks the larger. Two consequences follow immediately. Only differences of scores appear, so $r$ is identified only up to an additive function of $x$: $r(x, y) + c(x)$ gives identical predictions for every $c$. And the model says nothing about ties or about intensity of preference; a pair preferred 51 to 49 and a pair preferred 99 to 1 are both a single bit.

A reward model $r_\phi$ is a language model with its output head replaced by a scalar head read at the final token. Given a dataset $\mathcal{D}$ of triples $(x, y_w, y_l)$ with $y_w$ preferred, the maximum-likelihood objective is

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) \Big].
$$

Its gradient is $-\mathbb{E}\big[\big(1 - \sigma(\Delta)\big)\big(\nabla_\phi r_\phi(x, y_w) - \nabla_\phi r_\phi(x, y_l)\big)\big]$ with $\Delta = r_\phi(x, y_w) - r_\phi(x, y_l)$: pairs the model already gets right with a large margin contribute almost nothing, pairs it gets wrong contribute fully. The reward model's accuracy on held-out pairs is its only honest number; on human data, agreement between two annotators is itself far from perfect, so accuracy in the high sixties to mid seventies is what a reward model that matches the annotators looks like, and higher numbers on the training pairs mean memorization.

### The RLHF objective and its closed-form optimum

Let $\pi_\theta(y \mid x)$ be the policy being trained and $\pi_{\text{ref}}(y \mid x)$ a frozen copy of the SFT model. The objective is

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ r(x, y) \big] - \beta \, \mathbb{E}_{x \sim \mathcal{D}} \, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big),
$$

with $\beta > 0$. The KL term is what keeps the policy inside the region where the reward model was trained. To find the optimum, fix one $x$ and write the per-prompt objective as a functional of the distribution $\pi$:

$$
\mathbb{E}_{y \sim \pi}[r(x, y)] - \beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)}
= -\beta \sum_y \pi(y) \left[ \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} - \frac{r(x, y)}{\beta} \right].
$$

Define the Gibbs distribution $\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x) \exp\big(r(x, y) / \beta\big) / Z(x)$ with $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)$ so that it sums to one. Then $\log \pi_{\text{ref}}(y) + r(x, y)/\beta = \log \pi^*(y) + \log Z(x)$, and the bracket above equals $\log \frac{\pi(y)}{\pi^*(y)} - \log Z(x)$. Substituting,

$$
\text{per-prompt objective} = -\beta \, \mathrm{KL}\big(\pi \,\|\, \pi^*\big) + \beta \log Z(x).
$$

The second term does not depend on $\pi$ and the first is maximized (at zero) when $\pi = \pi^*$. So

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\Big(\frac{r(x, y)}{\beta}\Big).
$$

Read it as a reweighting of the reference: every response keeps its reference probability times an exponential bonus for reward, with $\beta$ as temperature. Small $\beta$ lets the reward dominate; large $\beta$ pins the policy to the reference. You cannot sample from $\pi^*$ directly because $Z(x)$ sums over every possible response, which is why PPO exists (it climbs toward $\pi^*$ with gradients) and why DPO exists (it uses $\pi^*$ without ever computing $Z$).

### The policy gradient

Let $J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$ for some reward $R$ (fold the prompt into the notation). Since $\nabla_\theta \pi_\theta(y) = \pi_\theta(y) \nabla_\theta \log \pi_\theta(y)$,

$$
\nabla_\theta J = \sum_y R(y) \nabla_\theta \pi_\theta(y) = \mathbb{E}_{y \sim \pi_\theta} \big[ R(y) \nabla_\theta \log \pi_\theta(y) \big].
$$

For any baseline $b$ that does not depend on $y$, $\mathbb{E}_{y \sim \pi_\theta}[b \nabla_\theta \log \pi_\theta(y)] = b \nabla_\theta \sum_y \pi_\theta(y) = b \nabla_\theta 1 = 0$, so you may replace $R(y)$ by $R(y) - b$ without bias, and a good $b$ (close to $\mathbb{E}[R]$) reduces variance. A response is a sequence of tokens $y = (y_1, \dots, y_T)$, and $\log \pi_\theta(y) = \sum_t \log \pi_\theta(y_t \mid y_{<t})$, so the gradient decomposes into per-token terms. If you have a per-token advantage $A_t$ (an estimate of how much better the token $y_t$ was than the policy's average at that prefix), the estimator is

$$
\nabla_\theta J \approx \frac{1}{T} \sum_{t=1}^{T} A_t \, \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}).
$$

This is REINFORCE with a baseline. It is unbiased but needs fresh samples from $\pi_\theta$ for every gradient step, and sampling from a language model is the expensive part.

### Importance sampling and the clipped surrogate

To take several gradient steps on one batch of samples, the samples are from an older policy $\pi_{\text{old}}$, and you correct with importance weights. For the per-token ratio

$$
\rho_t(\theta) = \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\text{old}}(y_t \mid y_{<t})},
$$

the surrogate $L^{\text{IS}}(\theta) = \mathbb{E}_{\pi_{\text{old}}}\big[\rho_t(\theta) A_t\big]$ has the same gradient as $J$ at $\theta = \theta_{\text{old}}$ (because $\nabla \rho_t = \rho_t \nabla \log \pi_\theta$ and $\rho_t = 1$ there), so maximizing it is a first-order-correct way to improve $J$. The problem is that it is only correct near $\theta_{\text{old}}$: the ratio can be driven to large values by a few tokens, and the surrogate would happily do that. PPO's clipped surrogate is

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[ \min\big( \rho_t A_t, \; \mathrm{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \, A_t \big) \Big].
$$

Why this bounds the update: take a token with $A_t > 0$. The objective wants $\rho_t$ to grow. Once $\rho_t > 1 + \epsilon$, the clipped term is the constant $(1+\epsilon)A_t$, and the unclipped term $\rho_t A_t$ is larger, so the minimum is the constant and its gradient with respect to $\theta$ is zero. The token stops contributing. For $A_t < 0$ the objective wants $\rho_t$ to shrink; once $\rho_t < 1 - \epsilon$ the clipped term $(1-\epsilon) A_t$ is larger than $\rho_t A_t$ (both negative, the clipped one less negative), again the minimum is the constant and the gradient is zero. In both cases the surrogate is a pessimistic lower bound on the unclipped one, and the gradient is only nonzero for tokens whose ratio is still within the band in the direction the advantage wants. Note the asymmetry: a token with $A_t > 0$ and $\rho_t < 1 - \epsilon$ is not clipped (the min picks $\rho_t A_t$), because moving it back toward the band is allowed. Clipping is therefore not a hard constraint on the ratio; it removes the incentive to leave the band, and after a few epochs on the same batch you check the fraction of clipped tokens, which should be a few percent, not half.

### The value function and GAE

The advantage $A_t$ needs a baseline that depends on the prefix, not just on the prompt. In the language-model MDP the state at step $t$ is the prefix $s_t = (x, y_{<t})$, the action is the token $y_t$, and the value $V(s_t) = \mathbb{E}_{\pi}[\text{return from } t \mid s_t]$. With a reward $r_t$ at each step and a discount $\gamma$, the one-step temporal-difference residual is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
$$

which is an unbiased estimate of the advantage of $y_t$ if $V$ is exact, and biased otherwise. The Monte Carlo alternative, $\sum_{l \ge 0} \gamma^l r_{t+l} - V(s_t)$, is unbiased regardless of $V$ but has variance that grows with the horizon. Generalized advantage estimation interpolates with a parameter $\lambda \in [0, 1]$:

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \, \delta_{t+l}.
$$

At $\lambda = 0$ it is $\delta_t$; at $\lambda = 1$ the telescoping sum $\sum_l \gamma^l \delta_{t+l} = \sum_l \gamma^l r_{t+l} - V(s_t)$ (with $V(s_T) = 0$) is Monte Carlo minus baseline. In practice you compute it backward with the recursion

$$
\hat{A}_T = 0, \qquad \hat{A}_t = \delta_t + \gamma \lambda \, \hat{A}_{t+1},
$$

and train the value head to regress $V(s_t)$ onto the return target $\hat{A}_t + V(s_t)$. For language models $\gamma = 1$ almost always (there is no reason to discount within one response), $\lambda$ around 0.95, and the reward is sparse: $r_t = 0$ for every token except the last, which receives the reward model's score, plus the per-token KL penalty described next.

### Token-level and sequence-level KL

The objective's KL is between sequence distributions, but the policy is a product of per-token conditionals, and so is the reference, so

$$
\mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big) = \mathbb{E}_{y \sim \pi_\theta} \Big[ \sum_{t} \log \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\text{ref}}(y_t \mid y_{<t})} \Big].
$$

The sequence-level KL is the expected sum of per-token log ratios along sampled trajectories. This gives two ways to use it. The InstructGPT and PPO way puts the per-token log ratio into the reward: $r_t \leftarrow r_t - \beta \log \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\text{ref}}(y_t \mid y_{<t})}$, so the penalty flows through the advantage and the critic like any other reward, and credit is assigned per token. The GRPO way keeps the KL out of the reward and adds it as a separate loss term, estimated per token. For the estimator, let $u_t = \pi_{\text{ref}}(y_t \mid y_{<t}) / \pi_\theta(y_t \mid y_{<t})$ with $y$ sampled from $\pi_\theta$. Three unbiased estimators of the per-token $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$:

$$
k_1 = -\log u_t, \qquad k_2 = \tfrac{1}{2}(\log u_t)^2, \qquad k_3 = u_t - \log u_t - 1.
$$

$k_1$ is the plain log ratio, unbiased and high-variance, and can be negative on a given sample. $k_2$ is a low-variance but biased estimate of the KL (it is exact only to second order). $k_3$ is unbiased because $\mathbb{E}_{\pi_\theta}[u_t] = \sum_y \pi_\theta(y) \pi_{\text{ref}}(y) / \pi_\theta(y) = 1$, so $\mathbb{E}[k_3] = 1 - \mathbb{E}[\log u_t] - 1 = \mathbb{E}[-\log u_t] = \mathrm{KL}$; and it is nonnegative for every sample because $u - \log u - 1 \ge 0$ with equality at $u = 1$. That is why GRPO uses it. One caveat you should know: when $k_3$ is used as a loss term and differentiated, its gradient is not the gradient of the true KL; it is a control-variate-like estimator whose expectation matches only when $\pi_\theta$ is close to $\pi_{\text{ref}}$. Near the reference it behaves well, which is the regime the penalty is meant to keep you in.

### The four models and a 32 GB budget

PPO for a language model holds four networks: the policy (trained), the reference (frozen, for the KL), the value model (trained), and the reward model (frozen). The memory in bytes is

$$
M \approx 2 N_{\pi} + 2 N_{\text{ref}} + 2 N_{V} + 2 N_{\text{RM}} + 12 N_{\text{train}} + M_{\text{act}} + M_{\text{gen}},
$$

with weights in bf16 (2 bytes), $12$ bytes per trainable parameter for the fp32 master copy and two Adam moments, $M_{\text{act}}$ the activation memory of the training forward and backward, and $M_{\text{gen}}$ the KV cache and working memory of the generation engine. A worked example with stated assumptions: a 3B policy (6 GB) with a LoRA adapter, so the reference is the same weights with the adapter disabled ($N_{\text{ref}}$ costs nothing extra); a value head as a single linear layer on top of the policy trunk (negligible, shared weights, one extra forward output); a separate 0.5B reward model (1 GB); trainable parameters 50M in the adapter plus the head, $0.6$ GB; activations for a batch of 8 sequences of 1,024 tokens with gradient checkpointing on a 3B model with hidden 2,048 and 36 layers about $8 \times 1024 \times 2048 \times 2 \times 36 \approx 1.2$ GB plus the logits, which chunked cross-entropy keeps small; and 8 to 12 GB reserved for vLLM's KV cache if generation is colocated on the same card. Total around 20 GB with headroom. Full fine-tuning of the same 3B policy would add $12 \times 3\text{B} = 36$ GB in optimizer state alone, which is why on one card the policy is LoRA or the model is around 1B. Doubling the policy to 7B roughly doubles every term except the reward model and does not fit with colocated generation unless you shrink the batch and the KV cache, at which point generation dominates wall-clock. This budget is also the argument for DPO and GRPO: DPO removes the value and reward models and needs only policy and reference; GRPO removes the value model and, with a verifier, the reward model too.

### DPO: the derivation

Start from the closed form and solve for the reward:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\Big(\frac{r(x, y)}{\beta}\Big)
\;\Longrightarrow\;
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

Now put this reward into the Bradley-Terry probability for a pair $(y_w, y_l)$ with the same prompt:

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{= 0}.
$$

The intractable normalizer cancels because both responses share the prompt. Replace the unknown optimal policy $\pi^*$ by the parameterized policy $\pi_\theta$ and maximize the likelihood of the observed preferences, exactly as for the reward model:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \Big[ \log \sigma\Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \Big) \Big].
$$

This is the formula the posttrain station shows next to its DPO tab. The quantity $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward: DPO is reward-model training where the reward model is defined by the policy. The two log-probabilities are sums over response tokens of per-token log-probabilities, which is why a DPO step is two forward passes of the policy (chosen and rejected) plus two of the reference, and why the reference log-probabilities can be precomputed once.

The gradient tells you what a step does. With $\Delta = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$,

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E}\Big[ \sigma(-\Delta) \Big( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \Big) \Big].
$$

Each pair pushes up the log-probability of the chosen and pushes down the rejected, weighted by $\sigma(-\Delta)$, how wrong the implicit reward currently is on that pair. Pairs the policy already ranks correctly with a wide margin stop contributing, exactly as in the reward model loss. What the gradient does not contain is any term that anchors $\log \pi_\theta(y_w \mid x)$ on its own: only the difference is constrained.

### DPO failure modes, in the math

Likelihood displacement. Because the loss depends only on $\Delta$, the optimizer is free to decrease both $\log \pi_\theta(y_w)$ and $\log \pi_\theta(y_l)$ as long as the rejected falls faster. It routinely does. The mechanism: $y_w$ and $y_l$ usually share most of their tokens (same prompt, similar wording, same first sentence), so the two gradient terms point in nearly the same direction and largely cancel in the shared part, while the net gradient on the shared tokens has the sign of the rejected term when the rejected response is slightly more probable or longer. Probability mass leaves both responses and lands on sequences the pairs never mention. Watch `logps/chosen` in the logs: if it trends down over training, the model is learning to prefer $y_w$ over $y_l$ by becoming less likely to say either. The standard mitigations add an anchor: an SFT term on the chosen response with weight $\alpha$ (RPO does this; the recipe exposes it as `--rpo-alpha`), or a margin term. The proper diagnosis is on the data: pairs with high embedding similarity between chosen and rejected displace the most, and filtering or rewriting them helps more than any loss change.

Over-optimization. The implicit reward is fit on a finite set of pairs. As training continues, the margin on training pairs grows without bound (nothing stops it, the loss just goes to zero slowly) while the held-out win rate against the SFT model rises, plateaus and then falls, because the policy is now exploiting regularities of the training pairs (length, formatting, certain phrases) rather than whatever the annotators meant. This is the same phenomenon as reward-model over-optimization in PPO, with the policy playing both roles. The lever is $\beta$ and the number of epochs, and the only detector is a held-out evaluation that is not the training loss.

The role of $\beta$. Small $\beta$ (0.01 to 0.05) means the implicit reward is a large multiple of a small log-ratio, so the policy can drift far from the reference before the margin saturates; you get bigger wins on the preference data and more displacement and forgetting. Large $\beta$ (0.5 and up) pins the policy: the loss saturates at small log-ratio changes, wins are modest, and the model stays close to SFT. The browser station lets you slide $\beta$ from 0.1 to 2 and watch the reward margin and the answers; the collapse at low $\beta$ is visible in a few hundred steps on the toy model. Values around 0.1 are the usual starting point at scale, and the right value is found by measuring held-out win rate and retention at three settings.

Length. If the chosen responses in your pairs are on average longer, DPO learns that longer is better, because length is a feature that separates the classes. This is a data artifact, not an algorithmic one, and the check is to compare mean lengths of chosen and rejected before you train. Length-normalized variants (SimPO divides the log-probability by the token count and drops the reference) reduce the incentive at the cost of changing what is being optimized.

### GRPO: the exact objective

For a prompt $q$, sample a group of $G$ outputs $\{o_1, \dots, o_G\}$ from the old policy, score each with a reward $r_i$ (a verifier, a reward model, or a sum of both), and define the group-normalized advantage

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)},
$$

assigned to every token of output $i$ (outcome supervision). The DeepSeekMath objective is

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\} \sim \pi_{\text{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big( \min\big( \rho_{i,t} \hat{A}_i, \; \mathrm{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \hat{A}_i \big) - \beta \, \hat{D}_{i,t} \Big) \right],
$$

with the per-token ratio $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})$ and the per-token KL estimator

$$
\hat{D}_{i,t} = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1,
$$

which is $k_3$ from above. Compare with PPO term by term. The clipped surrogate is the same. The advantage is no longer from a critic; the group mean is the baseline and the group standard deviation is a per-prompt scale. The KL is a loss term, not a reward. And there is a $1/|o_i|$ normalizer per output, so each output contributes equally regardless of length. Because the group mean includes the sample itself, the baseline is very slightly correlated with the sample, which introduces a bias of order $1/G$; the leave-one-out variant (RLOO) removes it by using the mean of the other $G-1$ rewards. At $G = 8$ or more the difference is small.

What Dr. GRPO changes and why. Two of the normalizers introduce biases the authors of Dr. GRPO named. The $1/|o_i|$ term makes a negative-advantage token in a long output cost less than the same token in a short one, so among wrong answers the objective prefers longer ones, and among right answers shorter ones; the empirical signature is response length growing during training without accuracy improving. The standard-deviation division up-weights prompts whose group has low reward variance (a question the model almost always gets right, or almost always wrong) relative to prompts of medium difficulty, which is the opposite of what you want. Dr. GRPO drops both: it divides by a fixed constant instead of $|o_i|$ and uses $r_i - \mathrm{mean}$ without the standard deviation. TRL exposes the second choice as `scale_rewards` and the first as the loss normalization type.

What DAPO changes. Clip-higher: use a larger upper clip bound $\epsilon_{\text{high}}$ than lower $\epsilon_{\text{low}}$ (0.28 and 0.2 in the paper), because for a low-probability token with positive advantage the symmetric band allows only a tiny absolute increase in probability, and low-probability tokens are where exploration lives; the symmetric band leads to entropy collapse. Dynamic sampling: groups where every reward is identical have zero advantage everywhere and contribute nothing but noise to the gradient; DAPO keeps sampling until the batch is full of groups with nonzero variance. Token-level loss: normalize by the total number of tokens across the batch instead of per output, so long outputs are not down-weighted token by token (the same fix as Dr. GRPO's, motivated from the other side). Overlong reward shaping: rather than a hard zero for truncated outputs, a soft penalty that grows with the overrun, so the length signal is not a cliff. The parts of these variants I am confident about are the ones listed here; treat other details as things to check against the papers.

When GRPO beats PPO. When the reward is a verifiable outcome on a single long response (math with a checkable final answer, code with unit tests), the group baseline is cheap and accurate enough, a value model of the same size would be hard to train on sparse terminal reward, and you were going to sample several outputs per prompt anyway for pass@k reasons. It also halves the memory. When it does not: with dense per-step rewards (a critic can exploit them, the group baseline cannot), in multi-turn environments where sampling $G$ full trajectories per prompt is expensive and the states within a trajectory differ, and when within-group reward variance is often zero (easy or impossible prompts), where GRPO throws samples away and a critic would still learn from them. The group baseline also has higher per-sample variance than a good learned baseline, which shows up as noisier training at small $G$.

## Build it small

Two snippets, each complete. The first is tabular DPO: a policy over eight actions, a hidden reward, a reference policy, Bradley-Terry labels. It checks that DPO recovers the reward up to a constant and lands on the closed-form $\pi^*$.

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
K, beta, N = 8, 0.5, 20000
r_true = torch.tensor([0.0, 0.5, 1.0, 2.0, -0.5, 0.3, 1.5, -1.0])   # hidden reward of each action
ref_logits = 0.5 * torch.randn(K)                                    # a non-uniform reference policy
logp_ref = F.log_softmax(ref_logits, -1)
pi_star = F.softmax(ref_logits + r_true / beta, -1)                  # closed-form optimum of the objective

i = torch.randint(0, K, (N,)); j = torch.randint(0, K, (N,)); i, j = i[i != j], j[i != j]
i_wins = torch.rand(len(i)) < torch.sigmoid(r_true[i] - r_true[j])   # Bradley-Terry labeller
w, l = torch.where(i_wins, i, j), torch.where(i_wins, j, i)

theta = ref_logits.clone().requires_grad_()                          # policy starts at the reference
opt = torch.optim.Adam([theta], 0.05)
for step in range(400):
    logp = F.log_softmax(theta, -1)
    margin = beta * ((logp[w] - logp_ref[w]) - (logp[l] - logp_ref[l]))
    loss = -F.logsigmoid(margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()

r_hat = beta * (F.log_softmax(theta, -1) - logp_ref).detach()       # implicit reward
r_hat, r_c = r_hat - r_hat.mean(), r_true - r_true.mean()           # rewards are identified up to a constant
print("max |r_hat - r_true| after centering:", (r_hat - r_c).abs().max().item())
print("total variation to pi_star:", 0.5 * (F.softmax(theta, -1) - pi_star).abs().sum().item())
```

Expected output from one run with this seed: the centered reward error is about 0.06 and the total variation distance to $\pi^*$ is about 0.015. Both shrink as $N$ grows, because in the tabular case DPO's minimizer is exactly the Bradley-Terry maximum-likelihood reward, and the policy is $\pi^*$ for that reward. Try $N = 2000$ and watch the error grow: that is the finite-data over-optimization in miniature.

The second snippet is one GRPO step, repeated, on a verifiable toy: a two-token policy that must emit the digits of $a + b$. The reward is checkable per digit, so groups have varying rewards from the start.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
L, G, EPS, BETA, D = 2, 8, 0.2, 0.02, 64

class Policy(nn.Module):           # emits L digits one at a time, conditioned on the prompt and its history
    def __init__(s):
        super().__init__()
        s.emb_a, s.emb_b, s.emb_tok, s.emb_pos = (nn.Embedding(n, D) for n in (10, 10, 11, L))
        s.net = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, 10))
    def logits(s, a, b, hist, t):                # hist = previous output token, 10 = "start"
        return s.net(s.emb_a(a) + s.emb_b(b) + s.emb_tok(hist) + s.emb_pos(torch.full_like(a, t)))
    def logprobs(s, a, b, out):                  # log pi(out_t | a, b, out_<t) for every position t
        hist, lps = torch.full_like(a, 10), []
        for t in range(L):
            lp = F.log_softmax(s.logits(a, b, hist, t), -1)
            lps.append(lp.gather(1, out[:, t:t + 1]).squeeze(1)); hist = out[:, t]
        return torch.stack(lps, 1)               # shape (N, L)
    @torch.no_grad()
    def sample(s, a, b):
        hist, out = torch.full_like(a, 10), []
        for t in range(L):
            hist = torch.distributions.Categorical(logits=s.logits(a, b, hist, t)).sample(); out.append(hist)
        return torch.stack(out, 1)

def reward(a, b, out):                            # verifiable: the two digits of a + b, leading zero allowed
    tgt = torch.stack([(a + b) // 10, (a + b) % 10], 1)
    return (out == tgt).float().mean(1)           # 0, 0.5 or 1; exact-match version is .all(1).float()

pi, ref = Policy(), Policy(); ref.load_state_dict(pi.state_dict())
opt = torch.optim.Adam(pi.parameters(), 1e-3)
for step in range(801):
    a = torch.randint(0, 10, (16,)).repeat_interleave(G)      # 16 prompts, G samples each
    b = torch.randint(0, 10, (16,)).repeat_interleave(G)
    out = pi.sample(a, b)                                       # rollouts from the current policy
    r = reward(a, b, out).view(-1, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1, 1)  # group-relative
    with torch.no_grad(): lp_old, lp_ref = pi.logprobs(a, b, out), ref.logprobs(a, b, out)
    for _ in range(2):                                          # two optimizer passes over the same rollouts
        lp = pi.logprobs(a, b, out)
        ratio = (lp - lp_old).exp()                             # per-token importance ratio
        surr = torch.min(ratio * adv, ratio.clamp(1 - EPS, 1 + EPS) * adv)
        kl = (lp_ref - lp).exp() - (lp_ref - lp) - 1            # k3 estimator of KL(pi || ref), per token
        loss = -(surr - BETA * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        zero = (r.std(1) == 0).float().mean()
        print(f"step {step:3d} reward {r.mean():.2f} exact {(r == 1).float().mean():.2f} "
              f"kl {kl.mean():.3f} zero-variance groups {zero:.2f}")
```

Expected output from one CPU run with this seed (a minute or so):

```
step   0 reward 0.07 exact 0.01 kl 0.008 zero-variance groups 0.38
step 200 reward 0.80 exact 0.59 kl 1.419 zero-variance groups 0.81
step 400 reward 0.84 exact 0.68 kl 2.102 zero-variance groups 0.94
step 600 reward 0.93 exact 0.87 kl 1.510 zero-variance groups 0.94
step 800 reward 0.93 exact 0.85 kl 1.763 zero-variance groups 0.88
```

Three things to read off. The policy learns the task from reward alone, no labels, no critic. The KL to the random reference grows to well above one nat per token, because the reference is useless here and $\beta = 0.02$ is small; in a real run the reference is the SFT model and you want this number to stay under a few hundredths. And by step 400 more than ninety percent of groups have identical rewards, so most of each batch contributes zero gradient: that is the waste DAPO's dynamic sampling removes, and it is the reason the curve flattens. Change the reward to the exact-match version and you will see the opposite problem at the start: almost every group is all zeros and learning is slow until the first successes appear.

## Build it real

Two recipes, meant to be run in order on the SFT adapter from Lab 04.

`recipes/dpo.py` wraps TRL's `DPOTrainer`. It loads the SFT model (base plus merged adapter, or base plus adapter path via `--sft-adapter`), attaches a fresh LoRA for the preference stage, and uses the adapter-disabled model as the reference, which TRL does automatically when you pass a PEFT model and no explicit `ref_model`. That keeps one copy of the weights in memory. Arguments: `--data` is a JSONL of `{"prompt": [...messages...], "chosen": [assistant message], "rejected": [assistant message]}` in the conversational format so the chat template is applied consistently with Lab 04; `--beta` (default 0.1); `--lr` (default $5 \times 10^{-6}$ for a LoRA on top of an SFT model, noticeably lower than SFT because the loss is a ranking loss and the gradients on shared tokens are large); `--epochs` (default 1, rarely more than 2); `--max-len` and `--max-prompt-len`; `--loss-type` chooses sigmoid (DPO as derived), ipo, or hinge; `--rpo-alpha` adds the SFT anchor on the chosen response; `--precompute-ref` runs the reference forward once over the dataset and caches the log-probabilities, halving the per-step compute. The logs to watch are `rewards/chosen`, `rewards/rejected` (the implicit rewards, which start at zero because policy equals reference), `rewards/margins` (their difference, should rise), `rewards/accuracies` (fraction of pairs with positive margin, should rise toward but not reach one on training data, and the held-out value is the one that matters), and `logps/chosen`. If `logps/chosen` falls steadily, you are watching likelihood displacement.

Time, as a formula. A DPO step is four forward passes of a full sequence (two policy, two reference) and one backward through the two policy passes, so about $2 \times 2N + 2 \times 2N + 2 \times 2N = 12N$ FLOPs per token-pair with precomputed reference log-probabilities reducing it to $8N$. For an 8B model on 10,000 pairs of 1,000 tokens each, one epoch is $10^7$ token-pairs, about $8 \times 10^{17}$ FLOPs; at an assumed 30 percent of a 200 TFLOP/s peak that is a little under four hours. Halve it with a 4B model; iterate with a 1B.

`recipes/grpo.py` wraps TRL's `GRPOTrainer` with vLLM colocated on the same card for generation (`use_vllm=True`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization` around 0.3 so training and generation share the 32 GB). The reward is a list of Python functions the recipe imports from `--reward-module`; the default module implements a math checker that extracts the final boxed answer and compares it with the gold string after normalization, plus a format reward that checks the answer is present exactly once. Arguments: `--data` with a `prompt` column and a `solution` column; `--num-generations` ($G$, default 8); `--max-completion-length` (default 1,024; longer helps reasoning tasks and costs memory linearly); `--beta` (default 0.04; DAPO-style runs set it to zero and rely on clipping, which is a legitimate choice for verifiable rewards and a risky one for learned rewards); `--epsilon` and `--epsilon-high` for the clip band; `--scale-rewards` to switch the standard-deviation division on or off; `--loss-type` to choose per-sequence or token-level normalization; `--lr` (default $1 \times 10^{-6}$ for full fine-tuning of a small model, $1 \times 10^{-5}$ for a LoRA); `--num-iterations` for optimizer passes per batch of rollouts (1 makes the ratio exactly one and clipping inert, 2 to 4 is where clipping does work). The logs to watch are `reward` and `reward_std` per step, `completions/mean_length`, `kl`, `clip_ratio` (fraction of tokens clipped; a few percent is healthy), and the fraction of groups with zero reward variance, which the recipe computes and logs as `frac_zero_var`. Time is dominated by generation: a step of 16 prompts with $G = 8$ and 1,024-token completions is 131k generated tokens; at whatever tokens per second vLLM sustains on the card for your model size (measure it once with `--bench-gen`), that is the step time, and the training forward and backward are a small addition.

On model size for GRPO on this card: a 1.5B to 4B model in bf16 with a LoRA or full fine-tuning is comfortable; an 8B model fits for DPO but for GRPO forces small batches and a small KV cache, and the run becomes generation-bound. Start with a 1.5B math-capable base and a few hundred training prompts; the point of the first run is to see the reward curve, the length curve and the KL curve move together, not to set a number.

## How it goes wrong

Both log-probabilities fall during DPO. Symptom: `logps/chosen` and `logps/rejected` both decrease, margins still grow, and the model's outputs drift toward generic text that appears in neither response. Cause: likelihood displacement, worst on pairs where chosen and rejected are near-duplicates. Fix: filter pairs by similarity, add an SFT anchor with `--rpo-alpha`, or raise $\beta$.

Training accuracy reaches one, held-out win rate falls. Symptom: exactly that. Cause: over-optimization to features of the training pairs. Fix: stop earlier (the recipe saves checkpoints at each eval), raise $\beta$, more distinct pairs, and check for a length or formatting confound before blaming the algorithm.

Responses get longer every epoch. Symptom: mean completion length climbs while the held-out score does not. Cause in DPO: chosen responses are longer than rejected on average. Cause in GRPO: the per-sequence normalizer's length bias, or a reward that pays for something length correlates with (more steps of work, more chances to state the answer). Fix: audit the pair lengths; switch to token-level normalization; add a soft length penalty; check the verifier is not rewarding multiple answers.

The reference is not the model you think. Symptom: DPO margins are large at step zero, or the implicit reward is nonzero before training. Cause: the reference was loaded with a different template, a different adapter state, or a different precision from the policy. Fix: assert at step zero that policy and reference log-probabilities agree to numerical precision on a batch; the recipe does this and refuses to start otherwise.

Most groups have zero variance. Symptom: `frac_zero_var` above 0.7 and a flat reward curve. Cause: prompts are too easy or too hard for the current policy, so all $G$ samples score the same. Fix: curate prompts by the current policy's pass rate (keep those between about 0.2 and 0.8), raise $G$, or enable dynamic sampling.

Reward rises, answers get worse. Symptom: the verifier's reward climbs while a human reading the samples sees repeated phrases, answers stated several times, or outputs that pattern-match the checker. Cause: reward hacking; the checker has a hole. Common holes: the extractor takes the last number, so the model writes many numbers; the extractor is case- or whitespace-sensitive in a way the model discovers; a format reward is achievable without solving anything. Fix: read fifty samples at every checkpoint, make the extractor strict, and hold out a second checker the policy never sees.

Entropy collapses. Symptom: all $G$ samples become nearly identical early, KL stays low, reward plateaus below where it should. Cause: symmetric clipping starves low-probability tokens of gradient, or the sampling temperature is too low. Fix: clip-higher, temperature 1.0 during rollouts, and a small entropy bonus if the framework supports it.

Generation and training weights diverge. Symptom: the importance ratios at the first optimizer pass are not all one, or the clip fraction is high from the start. Cause: vLLM's copy of the weights was not synchronized after the last step, or the LoRA was merged for generation with a different precision. Fix: the recipe checks that the mean absolute log-ratio between the sampler's log-probabilities and the trainer's at the first pass is near zero; if it is not, fix synchronization before anything else.

## Measure it

For a reward model: held-out pairwise accuracy, and its value relative to inter-annotator agreement on the same data; a model that beats the annotators' agreement with each other has memorized. For DPO: implicit-reward accuracy on held-out pairs (the number the station shows as accuracy), a win rate against the SFT model on a held-out prompt set judged either by people or by a judge model with position swapping to remove order bias, the mean length ratio of policy to SFT outputs (a ratio well above one with a flat win rate is a length exploit), and the Lab 04 retention suite, because preference training forgets faster per step than SFT does. A held-out win rate in the high fifties to sixties against a good SFT model is what a working DPO run looks like on realistic data; a win rate near ninety with a large length ratio is a red flag, not a triumph. For GRPO with a verifier: pass@1 on a held-out set with the same checker, pass@k for $k$ up to 16 or so on the same set to see whether RL improved the model or only sharpened it (if pass@1 rises but pass@16 does not, the model is selecting among answers it could already produce; if both rise, it learned something), KL to the reference at the end (a few hundredths of a nat per token is a run that stayed close, above 0.5 is a run that left), and the accuracy of a second checker on the same outputs as the hacking detector. For everything: read the samples. No number replaces fifty read outputs per checkpoint.

## Exercises

1. Show that adding any function $c(x)$ to the reward leaves both the Bradley-Terry likelihood and the closed-form $\pi^*$ unchanged. Check: in $\pi^*$ the constant is absorbed by $Z(x)$; in Bradley-Terry it cancels in the difference.

2. In the tabular toy, replace the DPO loss with the IPO loss $\big(\Delta - \tfrac{1}{2\beta}\big)^2$ where $\Delta$ is the log-ratio difference without the $\beta$ factor. Check: the learned policy no longer matches $\pi^*$ for the Bradley-Terry reward, but the training loss is bounded and the margin stops growing; explain which of the two is the property you want.

3. Implement the leave-one-out baseline in the GRPO toy (mean over the other $G - 1$ rewards) and run both versions across five seeds. Check: the reward curves are within noise at $G = 8$; then set $G = 2$ and see the difference.

4. Switch the GRPO toy to exact-match reward and add dynamic sampling: keep drawing prompts until you have 16 groups with nonzero variance. Check: the number of prompts drawn per step starts high and falls as the policy improves; time-to-reward-0.5 improves against the version without it.

5. Run `recipes/dpo.py` at $\beta \in \{0.03, 0.1, 0.5\}$ on the same pairs with a 1B SFT model. Plot held-out accuracy, `logps/chosen`, and mean output length against training step. Check: the lowest $\beta$ has the fastest-falling chosen log-probability and the largest length change.

6. Write a deliberately hackable verifier (accept the answer if the gold string appears anywhere in the output), run GRPO for a few hundred steps, and describe the exploit the policy finds. Then fix the verifier and rerun from the same checkpoint. Check: the second run's reward is lower than the first at the same step, and the second checker agrees with it.

## Test yourself

1. Derive the DPO gradient from the loss and identify the factor that makes a training pair stop contributing. Then explain why the same factor is a problem for a pair whose label is wrong.

<details><summary>Answer</summary>
With $\Delta = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$ and $\frac{d}{d\Delta}\log \sigma(\Delta) = \sigma(-\Delta)$, the gradient is $-\beta \, \sigma(-\Delta) \big(\nabla \log \pi_\theta(y_w) - \nabla \log \pi_\theta(y_l)\big)$. The weight $\sigma(-\Delta)$ goes to zero as the pair is ranked correctly with a growing margin. For a mislabeled pair, $\Delta$ is driven negative by the correct examples, so $\sigma(-\Delta)$ approaches one: the wrong pair receives the largest weight in the batch and keeps pushing against the majority. Label noise is amplified rather than averaged out, and the cure is data cleaning or a robust loss, not a smaller learning rate.
</details>

2. A colleague claims the KL term in the RLHF objective is unnecessary if you just stop early. What is the strongest version of that claim, and where does it fail?

<details><summary>Answer</summary>
Strongest version: early stopping bounds how far the policy travels, and with a small learning rate that is a form of implicit regularization toward the initialization, so the penalty is redundant. Failure: the KL term regularizes in distribution space, per prompt, while early stopping bounds a parameter-space path. The policy can move a small distance in parameter space and a very large distance in output distribution on a subset of prompts (the ones the reward model overvalues), which is precisely the failure the KL is there to stop. Also, the closed form $\pi^*$ that DPO relies on exists only because of the KL term; without it, the optimum is a point mass on the highest-reward response, and DPO has no derivation.
</details>

3. In PPO, why is the value function usually not trained with the clipped surrogate, and what happens to GAE if the value head is badly wrong?

<details><summary>Answer</summary>
The value head is a regression target problem, not a policy improvement problem: it minimizes $(V(s_t) - \hat{R}_t)^2$ where $\hat{R}_t = \hat{A}_t + V_{\text{old}}(s_t)$ is the return estimate, sometimes with its own clipping to limit how far $V$ moves per update. If $V$ is badly wrong, the residuals $\delta_t$ are dominated by the value error rather than by the reward, and at $\lambda < 1$ GAE inherits that error as bias; at $\lambda = 1$ the value error cancels in the telescoping sum except at the start, so the advantage is unbiased but high variance. A wrong critic with $\lambda = 0.95$ can push every token in the wrong direction consistently, which is worse than noise.
</details>

4. Estimate the memory of a PPO run on a 7B policy with full fine-tuning, a separate 7B value model, a 7B reference and a 7B reward model, all bf16, before activations. Then say what you would change first to fit 32 GB.

<details><summary>Answer</summary>
Weights: four models at 14 GB each, 56 GB. Optimizer state for two trained 7B models at 12 bytes per parameter, 168 GB. Total above 220 GB before activations or generation. First change: LoRA on the policy (removes 84 GB of optimizer state and lets the reference share the base weights, removing 14 GB); second: a value head on the policy trunk instead of a separate model (removes 14 GB and its optimizer state); third: a small or rule-based reward. That lands near the budget in section 3 for a 3B policy; for 7B on one card you also need a smaller batch and a smaller KV cache, or a different algorithm.
</details>

5. Show that $k_3$ is an unbiased estimator of $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ when samples come from $\pi_\theta$, and say what it estimates if the samples come from $\pi_{\text{old}}$ instead.

<details><summary>Answer</summary>
With $u = \pi_{\text{ref}}(y) / \pi_\theta(y)$ and $y \sim \pi_\theta$: $\mathbb{E}[u] = \sum_y \pi_{\text{ref}}(y) = 1$ and $\mathbb{E}[-\log u] = \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$, so $\mathbb{E}[u - \log u - 1] = \mathrm{KL}$. If the samples come from $\pi_{\text{old}}$ (which they do after the first optimizer pass on a batch), $\mathbb{E}_{\pi_{\text{old}}}[u] = \sum_y \pi_{\text{old}}(y) \pi_{\text{ref}}(y) / \pi_\theta(y) \ne 1$ in general, and the estimator is biased by the mismatch between $\pi_{\text{old}}$ and $\pi_\theta$. Within the clip band the mismatch is small, which is one more reason the clip band matters.
</details>

6. Spot the bug in this GRPO advantage computation for a batch laid out as prompts times generations:

```python
r = rewards.view(G, -1)                  # rewards is length num_prompts * G
adv = (r - r.mean(0)) / (r.std(0) + 1e-4)
adv = adv.view(-1)
```

<details><summary>Answer</summary>
The rollouts are typically laid out with the $G$ generations of one prompt contiguous, so the correct reshape is `view(-1, G)` with statistics over dimension 1. Reshaping to `(G, -1)` and normalizing over dimension 0 mixes rewards across different prompts: each column holds one generation from each of $G$ different prompts, and the baseline becomes the mean over unrelated questions. The gradient is still a valid policy gradient (any baseline that does not depend on the sample is unbiased), but the variance reduction is gone, and prompts that are easy get uniformly positive advantages, which is the difficulty bias in its purest form.
</details>

7. DPO is derived assuming the pairs come from the reference policy's distribution (they are sampled from $\pi_{\text{ref}}$ and labeled). In practice the pairs come from other models or from people. What does that change, and what does it not?

<details><summary>Answer</summary>
Nothing in the algebra needs the pairs to come from $\pi_{\text{ref}}$; the cancellation of $Z(x)$ only needs both responses to share a prompt. What changes is the statistical meaning: the implicit reward is fit on responses that may have near-zero probability under the reference, so the log-ratios are large and poorly estimated there, and the policy is being asked to reorder mass in regions it never visits. This is the off-policy gap that on-policy variants (sample pairs from the current or SFT policy, label them, train) close, and it is the reason on-policy DPO usually beats DPO on pairs collected from a different model.
</details>

8. Your GRPO run has $G = 8$ and 60 percent of groups are all-correct. You are considering raising $G$ to 32 to reduce the zero-variance fraction. Estimate what fraction of groups will still be zero-variance under a per-prompt pass rate model, and say whether the compute is better spent elsewhere.

<details><summary>Answer</summary>
If a prompt's per-sample pass rate is $p$, a group of size $G$ is all-correct with probability $p^G$ and all-wrong with $(1-p)^G$. A prompt with $p = 0.95$ gives $p^8 \approx 0.66$ and $p^{32} \approx 0.19$; a prompt with $p = 0.99$ gives $0.92$ and $0.72$. Raising $G$ helps for moderately easy prompts and barely at all for near-saturated ones, at four times the generation cost. Filtering prompts by the current pass rate (dropping those above about 0.9) costs one estimate per prompt and recovers more useful groups per generated token. Dynamic sampling does the same thing online.
</details>

9. A run reports the implicit reward accuracy at 0.93 on the held-out pairs and a win rate of 0.52 against SFT. Reconcile the two numbers.

<details><summary>Answer</summary>
Held-out pair accuracy measures the ordering of two specific responses that already exist; it says the policy assigns higher log-ratio to the chosen one. The win rate measures the policy's own samples against the SFT model's samples. The two can diverge when the log-ratio ordering was achieved by lowering both responses' probabilities (displacement) while the policy's mode moved somewhere the pairs do not cover, or when the held-out pairs share the confound (length, format) that the policy learned and the judge does not reward. High accuracy with flat win rate is the signature of over-optimization to the pair distribution, and the win rate is the number to trust.
</details>

10. Why does GRPO assign the same advantage to every token of an output, and what is lost compared with a per-token advantage from a critic?

<details><summary>Answer</summary>
With only a terminal reward and no critic, there is no information to distinguish tokens within one output; the group-relative score is the only signal and it is per output. What is lost is credit assignment: a correct answer that contains a wrong intermediate step reinforces the wrong step, and a wrong answer that contains a good first half penalizes the good half. A critic, if it were accurate, would give the good half a positive advantage and the wrong step a negative one. Process reward models try to recover this signal without a critic; when they are available and accurate, per-step advantages are better, and when they are not, the outcome-only signal is at least unbiased with respect to the thing you can verify.
</details>

## What will change, what will not

The Bradley-Terry model, the KL-regularized objective, and its Gibbs-distribution optimum are the durable core. The observation that a preference is a single bit identifying reward only up to a per-prompt constant will hold for any pairwise data, and the closed form $\pi^* \propto \pi_{\text{ref}} \exp(r / \beta)$ is a theorem, not a design. DPO is that theorem substituted into that model. Whatever replaces DPO will still have to say what it does about the normalizer and what its implicit reward is.

The policy gradient and its variance-reduction machinery (baselines, importance ratios, trust regions) will also stay. The specific choice of clipped surrogate, the GAE parameters, the $k_3$ estimator, and which normalizers to divide by are engineering decisions that are actively being revised, and the Dr. GRPO and DAPO changes are examples of that revision happening within a single year. Expect the loss normalization, the clip band, and the sampling strategy to keep changing; expect the questions to stay the same: what is the baseline, what bounds the update, and how is length being priced.

Reward hacking is permanent. Any optimizer aimed at a proxy will find the proxy's holes, and the better the optimizer the faster it finds them. The detectors (a second checker, a held-out judge, a KL budget, and reading samples) are the durable defense; the particular exploits (last-number extraction, boxed answers repeated, length) will be replaced by new ones as the checkers improve.

What will change: the libraries and their argument names, the default $\beta$ values, the model sizes that fit on one card, the generation engine, and the idea that a 32 GB budget is tight. The memory formula in section 3 is how you recompute the budget for whatever hardware and model you have.

What is open: whether outcome-only rewards can teach reasoning that transfers beyond the checker's domain, whether process rewards can be made reliable without labeling every step, and whether on-policy preference data closes the gap between DPO and RL in general or only on the benchmarks where it has been measured.

## Read next

1. Deep Reinforcement Learning from Human Preferences, Christiano, 2017. The origin of learning a reward from pairwise comparisons and optimizing a policy against it.
2. Learning to summarize from human feedback, Stiennon, 2020. The first full RLHF pipeline on a language model, including the KL penalty and the observation that the reward model over-optimizes.
3. Training language models to follow instructions with human feedback, Ouyang, 2022. InstructGPT; PPO with the per-token KL in the reward and the pretraining-mix term.
4. Proximal Policy Optimization Algorithms, Schulman, 2017. The clipped surrogate and why it works; read with High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman, 2016, for GAE.
5. Direct Preference Optimization: Your Language Model is Secretly a Reward Model, Rafailov, 2023. The derivation in section 3, with the gradient analysis.
6. A General Theoretical Paradigm to Understand Learning from Human Preferences, Azar, 2023. IPO; the argument that the Bradley-Terry assumption plus deterministic preferences leads to unbounded margins, and a bounded alternative.
7. Scaling Laws for Reward Model Overoptimization, Gao, 2022. The gold-versus-proxy reward curves that define what over-optimization looks like and how it scales with reward model size and KL.
8. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, Shao, 2024. GRPO as written in section 3, including the group-normalized advantage and the $k_3$ KL term; followed by DeepSeek-R1, Guo, 2025, for GRPO with verifiable rewards at scale, and Understanding R1-Zero-Like Training, Liu, 2025, for the Dr. GRPO analysis of the normalizers.
