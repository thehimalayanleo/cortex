"""GRPO with a verifiable-answer reward on the synthetic reasoning set (Lab 21: the training pie's RL stage).

What this teaches
  * the reward is a program, not a model: the completion is scored on the span
    after its last "answer:" against the gold answer from data_prep.py
        r = 0.2 * [an "answer:" marker is present] + 0.8 * [extracted answer == gold]
    so a policy that learned the format in SFT starts around 0.2 and climbs as
    sampling noise is trained away; there is nothing to reward-hack except the
    format bit, and that bit is worth less than being right
  * the same GRPO machinery as grpo_tool.py (group-normalized advantages, the
    clipped ratio, the k3 KL to a frozen reference) on a different environment,
    which is the point: the loop does not change, only score() and the tasks
  * why the RL stage sits after SFT: with no warm start the policy never emits
    the format, every group has zero variance, and the advantage is zero everywhere

How to run
  nano path (CPU, offline; --ckpt from sft_lora.py selects it, --smoke without a
  checkpoint runs a short warm-up first):
    python lab/recipes/grpo_reason.py --smoke --steps 30
    python lab/recipes/grpo_reason.py --ckpt out/sft/ckpt.pt --tasks-jsonl out/data_prep/reason_train.jsonl \
        --eval-jsonl out/data_prep/reason_heldout.jsonl --steps 30 --group 4 --batch 4
  real (RTX 5090): TRL GRPOTrainer with a small instruct model and this file's reward
    python lab/recipes/grpo_reason.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks-jsonl out/data_prep/reason_train.jsonl --steps 200 --group 8
  needs (real): pip install transformers trl peft datasets

Outputs: METRIC lines (reward_mean, kl, clip_frac, frac_format, frac_correct, reward_greedy on the held-out set),
ROLLOUT lines for the first group every --log-rollouts-every steps, a checkpoint (nano path) or adapter (real), RESULT.
"""
from __future__ import annotations

import inspect
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402
import data_prep as D  # noqa: E402

SYSTEM_PROMPT = ("Answer the question. Think in one or two short sentences, then finish with "
                 "'answer: <the answer>' and nothing after it.")


def build_parser():
    p = C.base_parser("grpo_reason", __doc__.split("\n")[0])
    p.add_argument("--ckpt", default=None, help="nano path: the SFT checkpoint to start from")
    p.add_argument("--tasks-jsonl", default=None, help="data_prep.py's reason_train.jsonl; generated in-file if omitted")
    p.add_argument("--eval-jsonl", default=None, help="held-out items for the greedy eval; generated in-file if omitted")
    p.add_argument("--n-tasks", type=int, default=300, help="generated tasks when --tasks-jsonl is absent")
    p.add_argument("--n-eval", type=int, default=48)
    p.add_argument("--group", type=int, default=None, help="G, completions per prompt")
    p.add_argument("--batch", type=int, default=None, help="prompts per step")
    p.add_argument("--mu", type=int, default=2, help="optimizer passes per batch of rollouts")
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--warm-steps", type=int, default=200, help="nano path without --ckpt: SFT steps before RL")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-prompt-len", type=int, default=256)
    p.add_argument("--log-rollouts-every", type=int, default=5, help="ROLLOUT lines for the first group every N steps (0 disables)")
    return p


# --------------------------------------------------------------------------- the environment


def normalize(s: str) -> str:
    return " ".join(s.lower().replace(".", " ").split())


def extract_answer(text: str) -> str | None:
    if "answer:" not in text:
        return None
    tail = text.rsplit("answer:", 1)[1].strip()
    return tail.splitlines()[0].strip() if tail else ""


def score(text: str, gold: str) -> tuple[float, dict]:
    """(reward, {format, correct}); format is 1 when an 'answer:' marker exists, correct when the span after it matches gold."""
    ans = extract_answer(text)
    parts = {"format": int(ans is not None), "correct": int(ans is not None and normalize(ans) == normalize(gold))}
    return 0.2 * parts["format"] + 0.8 * parts["correct"], parts


def load_tasks(path: str | None, n: int, seed: int) -> list[dict]:
    rows = C.read_jsonl(path) if path else D.make_reasoning(n, seed)
    out = []
    for r in rows:
        if r.get("question") and r.get("answer") is not None:
            out.append({"question": r["question"], "answer": str(r["answer"]), "chain": r.get("chain", ""), "prompt": r.get("prompt") or D.prompt_of(r["question"])})
    return out


# --------------------------------------------------------------------------- nano path: GRPO by hand


def token_logps(model, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids[:, :-1]).float()
    return F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]


def batched_generate(policy, tok, tasks, repeats, max_new, temperature, greedy, device):
    prompts = [tok.encode(t["prompt"]) for t in tasks for _ in range(repeats)]
    ids, mask = C.left_pad(prompts, tok.pad_id)
    out = C.generate(policy, ids.to(device), max_new, temperature=temperature, eos_id=tok.eos_id, greedy=greedy, attn_mask=mask.to(device))
    P = ids.shape[1]
    res = []
    for i, row in enumerate(out):
        comp = row[P:].tolist()
        if tok.eos_id in comp:
            comp = comp[: comp.index(tok.eos_id) + 1]
        res.append((tasks[i // repeats], prompts[i], comp, tok.decode(comp)))
    return res


def rollout(policy, tok, tasks, G, max_new, temperature, device):
    seqs, flags, rewards, parts_all, texts = [], [], [], [], []
    for task, prompt, comp, text in batched_generate(policy, tok, tasks, G, max_new, temperature, False, device):
        r, parts = score(text, task["answer"])
        seqs.append(prompt + comp)
        flags.append([0] * len(prompt) + [1] * len(comp))
        rewards.append(r)
        parts_all.append(parts)
        texts.append(text)
    ids, _ = C.pad_batch(seqs, tok.pad_id)
    cmask, _ = C.pad_batch(flags, 0)
    return ids.to(device), cmask.bool().to(device), torch.tensor(rewards, device=device), parts_all, texts


def warm_start(policy, tok, tasks, steps, device):
    C.status("warmup", f"{steps} SFT steps on {len(tasks)} demonstrations (no --ckpt given)")
    seqs, labels = [], []
    for t in tasks:
        p = tok.encode(t["prompt"])
        c = tok.encode(f"{t['chain']} answer: {t['answer']}", add_eos=True)
        seqs.append(p + c)
        labels.append([-100] * len(p) + c)
    max_len = policy.cfg.seq_len + 1
    ids, mask = C.pad_batch(seqs, tok.pad_id, max_len)
    lab, _ = C.pad_batch(labels, -100, max_len)
    lab[~mask] = -100
    opt = C.make_adamw(policy, 3e-3, 0.0)
    gen = torch.Generator().manual_seed(0)
    policy.train()
    for _ in range(steps):
        ix = torch.randint(0, len(seqs), (32,), generator=gen)
        loss = C.lm_loss(policy(ids[ix, :-1].to(device)), lab[ix, 1:].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    C.log(f"warm-up loss {loss.item():.3f}")


@torch.no_grad()
def greedy_eval(policy, tok, tasks, max_new, device):
    total, parts_sum = 0.0, {"format": 0, "correct": 0}
    for task, _, _, text in batched_generate(policy, tok, tasks, 1, max_new, 1.0, True, device):
        r, parts = score(text, task["answer"])
        total += r
        for k in parts_sum:
            parts_sum[k] += parts[k]
    n = max(1, len(tasks))
    return total / n, {k: v / n for k, v in parts_sum.items()}


def nano(args):
    device = C.pick_device(args.device)
    tasks = load_tasks(args.tasks_jsonl, args.n_tasks, args.seed)
    eval_tasks = load_tasks(args.eval_jsonl, args.n_eval, args.seed + 999)[: args.n_eval]
    if not tasks or not eval_tasks:
        raise SystemExit("no tasks: check --tasks-jsonl / --eval-jsonl")
    if args.ckpt:
        policy, tok, _ = C.load_checkpoint(args.ckpt, device)
        C.log(f"policy from {args.ckpt}, params={policy.num_params():,}")
    else:
        tok = C.CharTokenizer()
        policy = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=96, n_head=4, seq_len=192)).to(device)
        warm_start(policy, tok, tasks[:150], args.warm_steps, device)
    ref = __import__("copy").deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    r0, parts0 = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
    C.log(f"greedy reward before RL: {r0:.3f} parts={parts0}")
    C.metric(0, reward_greedy=r0, **{f"greedy_{k}": v for k, v in parts0.items()})

    opt = C.make_adamw(policy, args.lr, 0.0)
    rng = random.Random(args.seed)
    C.status("train", f"GRPO: {args.steps} steps, {args.batch} prompts x G={args.group}, mu={args.mu}, beta={args.beta}")
    trace = []
    for step in range(1, args.steps + 1):
        batch = rng.sample(tasks, min(args.batch, len(tasks)))
        ids, cmask, rewards, parts, texts = rollout(policy, tok, batch, args.group, args.max_new, args.temperature, device)
        r = rewards.view(len(batch), args.group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        with torch.no_grad():
            policy.eval()
            old_lp = token_logps(policy, ids)
            ref_lp = token_logps(ref, ids)
        policy.train()
        m = cmask[:, 1:].float()
        for _ in range(args.mu):
            lp = token_logps(policy, ids)
            ratio = torch.exp(lp - old_lp)
            pg = -torch.min(ratio * adv[:, None], ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip) * adv[:, None])
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1
            per_seq = ((pg + args.beta * kl) * m).sum(1) / m.sum(1).clamp(min=1)
            loss = per_seq.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            clip_frac = (((ratio - 1).abs() > args.eps_clip).float() * m).sum() / m.sum().clamp(min=1)
            kl_mean = (kl * m).sum() / m.sum().clamp(min=1)
            kl_seq = (kl * m).sum(1) / m.sum(1).clamp(min=1)
        if args.log_rollouts_every and (step % args.log_rollouts_every == 0 or step == 1):
            for i in range(args.group):
                C.rollout(step=step, group=0, idx=i, prompt=C.clip_text(batch[0]["prompt"], 300), completion=C.clip_text(texts[i], 600),
                          reward=float(rewards[i]), advantage=float(adv[i]), format=parts[i]["format"], correct=parts[i]["correct"],
                          kl=float(kl_seq[i]), expected=f"{batch[0]['chain']} answer: {batch[0]['answer']}")
        trace.append(rewards.mean().item())
        fields = dict(loss=loss.item(), reward_mean=rewards.mean(), reward_std=rewards.std(), kl=kl_mean, clip_frac=clip_frac,
                      completion_len=m.sum(1).mean(), zero_var_groups=(r.std(1) < 1e-6).float().mean(),
                      frac_format=sum(p["format"] for p in parts) / len(parts), frac_correct=sum(p["correct"] for p in parts) / len(parts))
        if step % 10 == 0 or step == args.steps:
            rg, pg_ = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
            fields.update(reward_greedy=rg, **{f"greedy_{k}": v for k, v in pg_.items()})
            C.log(f"step {step}: sample={texts[0]!r} reward={rewards[0].item():.2f}; greedy eval {rg:.3f}")
            policy.train()
        C.metric(step, **fields)
    r1, parts1 = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), policy, tok, args.steps)
    C.status("done", f"saved {path}")
    k = max(1, len(trace) // 3)
    C.result(reward_greedy_before=r0, reward_greedy_after=r1, sampled_reward_first_third=sum(trace[:k]) / k,
             sampled_reward_last_third=sum(trace[-k:]) / k, **{f"before_{k_}": v for k_, v in parts0.items()},
             **{f"after_{k_}": v for k_, v in parts1.items()}, n_eval=len(eval_tasks), steps=args.steps, checkpoint=path)


# --------------------------------------------------------------------------- real: TRL GRPOTrainer


def real(args):
    C.require("transformers")
    trl = C.require("trl")
    peft = C.require("peft")
    datasets = C.require("datasets")
    tasks = load_tasks(args.tasks_jsonl, args.n_tasks, args.seed)
    rows = [{"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["question"]}], "answer": t["answer"]} for t in tasks]
    ds = datasets.Dataset.from_list(rows)
    calls = {"n": 0}

    def reason_reward(completions, answer, prompts=None, **kw):
        out, parts_all, texts = [], [], []
        for c, gold in zip(completions, answer):
            text = c[-1]["content"] if isinstance(c, list) else c
            r, parts = score(text, gold)
            out.append(r)
            parts_all.append(parts)
            texts.append(text)
        calls["n"] += 1
        if args.log_rollouts_every and calls["n"] % args.log_rollouts_every == 0:
            G = min(args.group, len(out))
            grp = torch.tensor(out[:G])
            adv = (grp - grp.mean()) / (grp.std() + 1e-4) if G > 1 else grp * 0
            for i in range(G):
                pr = prompts[i] if prompts else ""
                q = pr[-1]["content"] if isinstance(pr, list) and pr else pr
                C.rollout(step=calls["n"], group=0, idx=i, prompt=C.clip_text(q, 300), completion=C.clip_text(texts[i], 600),
                          reward=out[i], advantage=float(adv[i]), **parts_all[i], expected=answer[i])
        return out

    sig = inspect.signature(trl.GRPOConfig.__init__).parameters
    cfg_kw = dict(output_dir=os.path.join(args.out, "trainer"), max_steps=args.steps, learning_rate=args.lr,
                  per_device_train_batch_size=args.group, gradient_accumulation_steps=1, num_generations=args.group,
                  max_completion_length=args.max_new, beta=args.beta, logging_steps=1, save_strategy="no", report_to="none",
                  bf16=torch.cuda.is_available(), seed=args.seed, temperature=args.temperature)
    for k, v in dict(epsilon=args.eps_clip, num_iterations=args.mu, max_prompt_length=args.max_prompt_len).items():
        if k in sig:
            cfg_kw[k] = v
    peft_config = peft.LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = trl.GRPOTrainer(model=args.model, reward_funcs=reason_reward, args=trl.GRPOConfig(**cfg_kw), train_dataset=ds,
                              peft_config=peft_config, callbacks=[C.make_metric_callback()])
    out = trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter)
    last = {k.replace("/", "_"): v for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items() if isinstance(v, (int, float))}
    C.status("done", f"adapter saved to {adapter}")
    C.result(train_loss=out.training_loss, steps=out.global_step, adapter=adapter, model=args.model, **last)


def main():
    args = build_parser().parse_args()
    use_nano = args.smoke or bool(args.ckpt)
    d = dict(steps=30, group=4, batch=4, lr=2e-4, max_new=64) if use_nano else dict(steps=200, group=8, batch=8, lr=1e-5, max_new=128)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (nano if use_nano else real)(args)


if __name__ == "__main__":
    main()
