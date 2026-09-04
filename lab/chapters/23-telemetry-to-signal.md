---
title: "Lab 23: From product telemetry to training signal: the continual post-training loop"
kind: permanent
topics: [lab]
chapter: 23
station: none
recipe: recipes/data_prep.py
reading_time: 55 min
---

## What you will be able to do

- Name the five kinds of signal a product emits (accept, edit, reject, correction, outcome), write the estimator each one gives you, and say which of SFT, preference optimisation, or reward modelling each can feed.
- Convert a log of sessions into three files (SFT pairs, preference pairs, rewarded episodes) with a script that prints how much of the log was usable and why the rest was not.
- Explain the three ways this data lies (selection, position and length bias, label noise from hurried users) and the audit that catches each one.
- Run one round of the loop on your own traces (the Traces tab) and report the change in a held-out metric with an interval.
- State the deploy-retrain gap precisely: what changes between the policy that produced the data and the policy you train, and why that bounds how much of the log you can use.

## The idea in one paragraph

A model in a product is graded thousands of times a day by people who never fill in a form: they accept a suggestion, they edit it before sending, they ask again with a sharper prompt, they abandon the session. Each of these is a weak label. The whole craft is turning weak labels into a training signal that is *unbiased enough* and *dense enough* to move the model, and then closing the loop: train, deploy, collect again. The three products that have done this by hand (a code editor's completion model, a coding agent's tool policy, a customer-support agent) share one design: an event log with the model's output, the user's action, and the outcome, joined by session id; a miner that emits pairs; a trainer that consumes them on a schedule; an evaluation that decides whether the new checkpoint ships. Cortex's Traces tab is a small version of the first two boxes, built for one user, and Lab 21's pipeline is the third and fourth.

## The math

### Five events, three estimators

Let the model produce $y \sim \pi_\theta(\cdot \mid x)$ and the user act. Write the events as

- **accept** $a = 1$: the user used $y$ unchanged;
- **edit** $y \to y'$: the user changed it and used $y'$;
- **reject** $a = 0$: the user discarded it and moved on;
- **correction** $x \to x'$: the user rephrased the request after seeing $y$;
- **outcome** $o \in \{0,1\}$ or $o \in \mathbb{R}$: something later in the workflow succeeded or failed (a test passed, a ticket closed, a reply was sent).

Each gives a different estimator. An accept is a sample from the set $\{y : \text{user finds } y \text{ good enough}\}$, so accepted pairs $(x, y)$ are SFT data for a policy that imitates *what passed*, with the caveat that acceptance is a threshold, not a maximum. An edit gives the strongest signal: $(x, y')$ is an SFT pair for the corrected behaviour and $(x, y' \succ y)$ is a preference pair with a known direction, and the edit distance $d(y, y')$ is a graded magnitude. A reject alone is a negative without a positive; it becomes useful as the rejected side of a pair when the same $x$ later gets an accept. A correction is a signal about $x$: the model's answer to $x$ was wrong in a way that made the user say $x'$, so $(x, y_{x'})$ is a pair where the answer to the sharper prompt is what the vaguer prompt wanted. An outcome is a reward for the whole episode, sparse and delayed, and it is the only one of the five that is not a judgement by the person who asked.

The three targets:

$$\mathcal{L}_{\text{SFT}} = -\sum_{(x,y^+)} \log \pi_\theta(y^+ \mid x), \qquad
\mathcal{L}_{\text{DPO}} = -\sum_{(x, y^+, y^-)} \log \sigma\!\Big(\beta \log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)} - \beta \log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}\Big),$$

and, for outcomes, a reward model $r_\phi(x, y) \approx \mathbb{E}[o \mid x, y]$ trained by regression or a verifier if the outcome is computable, then GRPO or PPO against it (Lab 05). The rule of thumb: edits feed DPO best because both sides come from the same $x$ and the same session; accepts feed SFT; outcomes feed reward modelling only when there are enough of them to beat the noise, which usually means thousands of episodes per task family.

### Selection: the log is not a sample of the task

The user only acts on what the model showed. If the model is good at easy requests and bad at hard ones, accepts concentrate on the easy ones and SFT on accepts teaches the model to be good at what it was already good at. Write the probability that a request reaches the log as $q(x)$ and the task distribution as $p(x)$; the SFT objective on accepts estimates $\mathbb{E}_{q}[\cdot]$, not $\mathbb{E}_p[\cdot]$. The correction is importance weighting, $w(x) = p(x)/q(x)$, which you cannot compute exactly but can approximate by the inverse of the acceptance rate in a bucket of similar requests (cluster them with Lab 08). A cheaper fix that works in practice: stratify, cap the number of pairs per bucket, and keep every edit (edits are concentrated where the model was wrong, which is where you want the data).

### Position and length bias

A suggestion shown first is accepted more; a longer suggestion is edited more because there is more to edit, not because it is worse. Preference pairs built from edits therefore carry a length prior. Measure it: regress $\mathbb{1}[y^+ \text{ is longer}]$ on nothing and see how far from $0.5$ it is; then either balance the pairs by length or add the length-normalised DPO variant. Position bias is handled at logging time by randomising the order in which alternatives are shown, which is a product decision you must make before the data exists.

### Label noise

Users are hurried. An accept can be "I will fix it later"; an edit can be a stylistic preference of one person. Estimate the noise rate with a small hand-labelled set: sample 200 accepts, label whether each was actually correct, and the fraction that was not is $\eta$. For DPO, a symmetric flip rate $\eta$ shrinks the effective margin by $(1 - 2\eta)$, so at $\eta = 0.3$ the signal is at 40 percent strength and $\beta$ must drop to compensate. The audit is cheap and it is the one thing that separates a loop that improves the model from one that slowly teaches it the users' mistakes.

### The deploy-retrain gap

The data was produced by $\pi_{\theta_0}$; you train $\pi_{\theta_1}$ and deploy it; the next log comes from $\pi_{\theta_1}$. Every batch is off-policy with respect to the model you are training now, and the gap grows with how much the policy moved. For SFT this is harmless (it is imitation). For DPO the reference model should be the policy that produced the pairs, so keep $\pi_{\text{ref}} = \pi_{\theta_0}$ for pairs from round 0 and rotate it with the rounds. For RL with an outcome reward, off-policy samples need importance weights $\pi_{\theta}(y\mid x)/\pi_{\theta_0}(y \mid x)$ clipped as in PPO, and the clip is what bounds how stale a batch can be before it is useless: with $\epsilon = 0.2$, tokens whose probability moved by more than 20 percent contribute nothing. This is why the continual loop is a *schedule*: small rounds, frequent, with the reference rotated, rather than one big retrain on a year of logs.

### Forgetting and the tenant problem

Round after round of fine-tuning on a narrow product log erodes what the base model knew. Measure retention on a fixed general suite after every round (Lab 09); the fix is replay (mix a fraction of the original post-training data into every round), a low learning rate with L2-to-init, or adapters: a LoRA per customer on a shared base, which also solves the privacy problem of one tenant's edits teaching another tenant's model. Your own results on sparse versus dense updates apply here: on a real base model, a small dense update with replay retained more than a sparse one at matched quality.

## Build it small

The Traces tab has been recording your own sessions: pairs you accepted, edits you made, preferences you recorded. Run `data_prep.py --source traces --report`; it prints, per event kind, how many there were, how many became pairs, and why the rest were dropped (too short, duplicate, no positive side). Then `dpo.py` on the preference pairs against the SFT checkpoint from Lab 04 for 200 steps, and `eval_suite.py` before and after on the held-out split. Report: pairs used, the length-bias rate, the held-out win rate with a Wilson interval.

## Build it real

Design the log first. One JSON line per event with `session`, `turn`, `x`, `y`, `event` in {accept, edit, reject, correct, outcome}, `y_prime`, `outcome`, `model_version`, `shown_position`, `latency_ms`. Store the model version on every line: without it the deploy-retrain accounting is impossible. Mine daily; train weekly on the 5090 with replay at 30 percent; evaluate on the fixed suite and a fresh sample of last week's requests; ship only on a paired win. Lab 21's pipeline runner is the scheduler; add a `mine` stage before `sft` and a `retain` check after `eval`.

## How it goes wrong

- The model gets better at the log and worse at the product: selection bias; stratify and weight.
- Win rate on edits goes up, users complain about verbosity: length bias in the pairs; balance or normalise.
- Each round gains, the tenth round is worse than the first: forgetting; measure retention, add replay.
- DPO diverges: the reference model is stale relative to the pairs; rotate it with the rounds.
- The outcome reward looks great and the product does not: the outcome is gamed (tests that always pass, tickets closed by giving up); audit 100 by hand.

## Measure it

- Usable fraction of the log per event kind, and the drop reasons.
- Noise rate $\eta$ from 200 hand labels, per event kind.
- Length-bias rate of preference pairs.
- Held-out win rate with a Wilson interval and retention on the general suite, per round.
- Staleness: fraction of tokens outside the PPO clip when a batch is trained.

## Exercises

1. Write the miner for the event log above and run it on your traces; print the report.
2. Estimate $q(x)$ by clustering requests (Lab 08) and computing acceptance per cluster; reweight and compare the SFT result.
3. Build a length-balanced pair set and measure the difference in mean response length after DPO.
4. Simulate label noise by flipping 30 percent of pairs; find the $\beta$ that recovers the clean result.
5. Run two rounds with the reference rotated and two without; compare divergence.

## Test yourself

1. Which of the five events gives a preference pair with a known direction, and why is it the best one?
2. Why does SFT on accepts estimate the wrong expectation, and what is the weight that fixes it?
3. What does a symmetric label-flip rate $\eta$ do to the DPO margin?
4. Why must the DPO reference model be rotated in a continual loop?
5. What bounds the usefulness of a stale batch in PPO-style RL, numerically?
6. Name two fixes for forgetting across rounds and the measurement that tells you it is happening.
7. Why must the model version be logged on every event line?
8. What is length bias in edit-derived pairs, and how do you measure it in one line?
9. Why is an outcome reward the only event not judged by the asker, and what does that imply about gaming?
10. What is the difference between a correction and a reject as a training signal?

<details><summary>Answers</summary>

1. The edit: $(x, y' \succ y)$ comes from the same request, the same session, and the user's own hand; both sides are real and the direction is certain.
2. Accepts are drawn from $q(x)$, the requests that reached the log and got accepted, not from the task distribution $p(x)$; the weight is $p(x)/q(x)$, approximated by inverse acceptance rate per bucket.
3. Shrinks the effective margin by $(1-2\eta)$; at $\eta = 0.3$ the signal is at 40 percent strength.
4. Pairs are produced by the deployed policy; DPO's regularisation is toward the reference, so the reference must be the policy that generated the pairs or the objective drifts.
5. The clip $\epsilon$: tokens whose probability ratio moved beyond $1 \pm \epsilon$ get zero gradient, so a batch whose tokens have mostly moved contributes nothing.
6. Replay of original post-training data, L2-to-init or low learning rate, adapters per tenant; the measurement is retention on a fixed general suite per round.
7. To attribute each event to the policy that produced it, which is what the off-policy correction and the reference rotation need.
8. Edits favour the shorter side (or the longer, depending on the product); measure $P(\text{chosen is longer})$ against $0.5$.
9. It comes from the world (a test, a closed ticket), not from the person; anything computable can be gamed, so audit a sample by hand.
10. A reject says $y$ was bad with no replacement; a correction says $x$ was answered wrongly and supplies a sharper $x'$ whose answer is what $x$ wanted, giving a positive pair.

</details>

## What will change, what will not

Products will log more (screen recordings, tool traces, multi-turn outcomes) and the miners will get smarter, possibly models themselves. The estimators will not: an accept is a threshold sample, an edit is a directed pair, an outcome is a delayed reward, and each carries its selection and noise. The deploy-retrain gap is a property of any loop with a deployed policy, and small frequent rounds with a rotated reference is the answer in 2026 and will be the answer when the models are ten times bigger.

## Read next

- Rafailov et al., "Direct Preference Optimization" (2023); Meng et al., "SimPO" (2024) for the length-normalised variant.
- Ouyang et al., "Training language models to follow instructions with human feedback" (2022), for the original loop.
- Ziegler et al., "Fine-tuning language models from human preferences" (2019), for off-policy caveats.
- Cursor's and Windsurf's engineering posts on training completion models from accept/edit telemetry.
- Your own notes: sparse continual update, CAMP, plasticity audit (retention under repeated fine-tunes).
