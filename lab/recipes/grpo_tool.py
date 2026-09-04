"""GRPO for tool use with a verifiable reward (Lab 05/06: RL with verifiable rewards).

What this teaches
  * a tiny tool environment (three Python functions: add, lookup_capital,
    days_between) and a strict call format the policy must emit:
        <call>{"name": "add", "args": {"a": 3, "b": 5}}</call>
  * a shaped, verifiable reward: +0.3 if the call parses, +0.3 if the tool
    name is right, +0.4 if executing the call gives the expected answer
  * GRPO from scratch: for each prompt sample a group of G completions,
    turn rewards into advantages by normalizing inside the group,
        A_i = (r_i - mean(r)) / (std(r) + eps)
    then take PPO-style clipped policy-gradient steps on the completion
    tokens,
        L_pg = -min(rho * A, clip(rho, 1 - eps_clip, 1 + eps_clip) * A),   rho = pi_theta / pi_old
    plus a KL penalty to a frozen reference computed with the k3 estimator
        k3 = exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1
    which is unbiased for KL(pi || pi_ref) and never negative per sample.
  * why a warm start matters: a policy that never emits a parseable call gets
    reward 0 on every sample, every group has zero variance, and the
    advantage is zero everywhere. The smoke path runs a short SFT on a few
    demonstrations first (the real path starts from an instruct model).

How to run
  smoke (CPU, offline): the minimal GPT at character level; reward rises
  from the warm-started level as sampling noise is trained away:
    python lab/recipes/grpo_tool.py --smoke --steps 60 --group 4 --batch 4
  real (RTX 5090): TRL GRPOTrainer with a small instruct model and this
  file's reward function (the tasks are still generated in-file):
    python lab/recipes/grpo_tool.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200 --group 8
  needs: pip install transformers trl peft datasets

Swapping in NVIDIA's RL datasets: nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1
and nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 are
prompt-plus-verifier style datasets for exactly this loop. To use one, replace
make_tasks() with rows from the dataset: put the conversation into the "prompt"
column, keep whatever columns the dataset's verifier needs, and rewrite
score() to call that verifier. The GRPO machinery does not change.
"""
from __future__ import annotations

import inspect
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402

# --------------------------------------------------------------------------- the environment

CAPITALS = {"france": "paris", "japan": "tokyo", "peru": "lima", "kenya": "nairobi", "chile": "santiago",
            "egypt": "cairo", "norway": "oslo", "india": "new delhi"}


def add(a, b):
    return int(a) + int(b)


def lookup_capital(country):
    return CAPITALS[str(country).lower()]


def days_between(start, end):
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


TOOLS = {"add": add, "lookup_capital": lookup_capital, "days_between": days_between}
CALL_RE = re.compile(r"<call>(.*?)</call>", re.S)

SYSTEM_PROMPT = (
    "You can call exactly one tool. Reply with only the call, in this format and nothing else:\n"
    '<call>{"name": TOOL, "args": {...}}</call>\n'
    "Tools: add(a: int, b: int); lookup_capital(country: str); days_between(start: YYYY-MM-DD, end: YYYY-MM-DD)."
)


@dataclass
class Task:
    question: str
    tool: str
    args: dict
    answer: str

    @property
    def demo(self) -> str:
        return "<call>" + json.dumps({"name": self.tool, "args": self.args}, separators=(",", ":")) + "</call>"


def make_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    countries = list(CAPITALS)
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            tasks.append(Task(f"what is {a} + {b}", "add", {"a": a, "b": b}, str(a + b)))
        elif kind == 1:
            c = rng.choice(countries)
            tasks.append(Task(f"what is the capital of {c}", "lookup_capital", {"country": c}, CAPITALS[c]))
        else:
            d1 = date(2024, 1, 1) + timedelta(days=rng.randint(0, 40))
            d2 = d1 + timedelta(days=rng.randint(1, 30))
            tasks.append(Task(f"how many days from {d1.isoformat()} to {d2.isoformat()}", "days_between",
                              {"start": d1.isoformat(), "end": d2.isoformat()}, str((d2 - d1).days)))
    return tasks


def score(text: str, task: Task) -> tuple[float, dict]:
    """The reward. Returns (reward, {parse, tool, answer}) with each part 0 or 1."""
    parts = {"parse": 0, "tool": 0, "answer": 0}
    m = CALL_RE.search(text)
    if not m:
        return 0.0, parts
    try:
        call = json.loads(m.group(1))
        assert isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("args"), dict)
    except Exception:
        return 0.0, parts
    parts["parse"] = 1
    if call["name"] == task.tool:
        parts["tool"] = 1
    if call["name"] in TOOLS:
        try:
            if str(TOOLS[call["name"]](**call["args"])) == task.answer:
                parts["answer"] = 1
        except Exception:
            pass
    return 0.3 * parts["parse"] + 0.3 * parts["tool"] + 0.4 * parts["answer"], parts


# --------------------------------------------------------------------------- smoke: GRPO by hand


def build_parser():
    p = C.base_parser("grpo_tool", __doc__.split("\n")[0])
    p.add_argument("--ckpt", default=None, help="smoke: start from an SFT checkpoint instead of the built-in warm-up")
    p.add_argument("--group", type=int, default=None, help="G, completions per prompt")
    p.add_argument("--batch", type=int, default=None, help="prompts per step")
    p.add_argument("--mu", type=int, default=2, help="optimizer passes per batch of rollouts (ratio is 1 when mu=1)")
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--warm-steps", type=int, default=250, help="smoke: SFT steps on demonstrations before RL (keep it short: a memorized policy has no sampling variance and GRPO gets zero advantage)")
    p.add_argument("--warm-tasks", type=int, default=150, help="smoke: number of demonstrations for the warm-up")
    p.add_argument("--n-tasks", type=int, default=300)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-prompt-len", type=int, default=256)
    return p


def token_logps(model, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[:, t] | ids[:, :t]) for t >= 1; shape (N, L-1)."""
    logits = model(ids[:, :-1]).float()
    return F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]


def batched_generate(policy, tok, tasks, repeats, max_new, temperature, greedy, device):
    """One generate call for all tasks x repeats using left-padded prompts. Returns list of (task, completion_ids, text)."""
    prompts = [tok.encode(f"q: {t.question}\n") for t in tasks for _ in range(repeats)]
    ids, mask = C.left_pad(prompts, tok.pad_id)
    out = C.generate(policy, ids.to(device), max_new, temperature=temperature, eos_id=tok.eos_id, greedy=greedy,
                     attn_mask=mask.to(device))
    P = ids.shape[1]
    res = []
    for i, row in enumerate(out):
        comp = row[P:].tolist()
        if tok.eos_id in comp:
            comp = comp[: comp.index(tok.eos_id) + 1]
        res.append((tasks[i // repeats], prompts[i], comp, tok.decode(comp)))
    return res


def rollout(policy, tok, tasks, G, max_new, temperature, device):
    """Sample G completions per task. Returns padded ids, completion mask (N, L), rewards, parts, texts."""
    seqs, comp_flags, rewards, parts_all, texts = [], [], [], [], []
    for task, prompt, comp, text in batched_generate(policy, tok, tasks, G, max_new, temperature, False, device):
        r, parts = score(text, task)
        seqs.append(prompt + comp)
        comp_flags.append([0] * len(prompt) + [1] * len(comp))
        rewards.append(r)
        parts_all.append(parts)
        texts.append(text)
    ids, _ = C.pad_batch(seqs, tok.pad_id)
    cmask, _ = C.pad_batch(comp_flags, 0)
    return ids.to(device), cmask.bool().to(device), torch.tensor(rewards, device=device), parts_all, texts


def warm_start(policy, tok, tasks, steps, device):
    C.status("warmup", f"{steps} SFT steps on {len(tasks)} demonstrations")
    seqs, labels = [], []
    for t in tasks:
        p = tok.encode(f"q: {t.question}\n")
        c = tok.encode(t.demo, add_eos=True)
        seqs.append(p + c)
        labels.append([-100] * len(p) + c)
    ids, mask = C.pad_batch(seqs, tok.pad_id)
    lab, _ = C.pad_batch(labels, -100)
    lab[~mask] = -100
    opt = C.make_adamw(policy, 3e-3, 0.0)
    gen = torch.Generator().manual_seed(0)
    policy.train()
    for step in range(steps):
        ix = torch.randint(0, len(seqs), (32,), generator=gen)
        x, y = ids[ix, :-1].to(device), lab[ix, 1:].to(device)
        loss = C.lm_loss(policy(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    C.log(f"warm-up loss {loss.item():.3f}")


@torch.no_grad()
def greedy_eval(policy, tok, tasks, max_new, device):
    total, parts_sum = 0.0, {"parse": 0, "tool": 0, "answer": 0}
    for task, _, _, text in batched_generate(policy, tok, tasks, 1, max_new, 1.0, True, device):
        r, parts = score(text, task)
        total += r
        for k in parts_sum:
            parts_sum[k] += parts[k]
    n = len(tasks)
    return total / n, {k: v / n for k, v in parts_sum.items()}


def smoke(args):
    device = C.pick_device(args.device)
    tasks = make_tasks(args.n_tasks, args.seed)
    eval_tasks = make_tasks(24, args.seed + 999)
    if args.ckpt:
        policy, tok, _ = C.load_checkpoint(args.ckpt, device)
    else:
        tok = C.CharTokenizer()
        policy = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=96, n_head=4, seq_len=160)).to(device)
        t = C.Timer()
        warm_start(policy, tok, tasks[: args.warm_tasks], args.warm_steps, device)
        C.log(f"warm-up took {t.lap():.1f}s")
    ref = __import__("copy").deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    r0, parts0 = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
    C.log(f"greedy reward before RL: {r0:.3f} parts={parts0}")
    C.metric(0, reward_greedy=r0, **{f"greedy_{k}": v for k, v in parts0.items()})

    opt = C.make_adamw(policy, args.lr, 0.0)
    rng = random.Random(args.seed)
    C.status("train", f"GRPO: {args.steps} steps, {args.batch} prompts x G={args.group}, mu={args.mu}, beta={args.beta}")
    t = C.Timer()
    for step in range(1, args.steps + 1):
        batch = rng.sample(tasks, args.batch)
        ids, cmask, rewards, parts, texts = rollout(policy, tok, batch, args.group, args.max_new, args.temperature, device)
        # group-normalized advantages
        r = rewards.view(args.batch, args.group)
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
            kl = torch.exp(d) - d - 1                                  # k3 estimator
            per_seq = ((pg + args.beta * kl) * m).sum(1) / m.sum(1).clamp(min=1)
            loss = per_seq.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            clip_frac = (((ratio - 1).abs() > args.eps_clip).float() * m).sum() / m.sum()
            kl_mean = (kl * m).sum() / m.sum()
        fields = dict(loss=loss.item(), reward_mean=rewards.mean(), reward_std=rewards.std(), kl=kl_mean, clip_frac=clip_frac,
                      completion_len=m.sum(1).mean(), zero_var_groups=(r.std(1) < 1e-6).float().mean(),
                      frac_parse=sum(p["parse"] for p in parts) / len(parts), frac_tool=sum(p["tool"] for p in parts) / len(parts),
                      frac_answer=sum(p["answer"] for p in parts) / len(parts))
        if step % 10 == 0 or step == args.steps:
            rg, pg_ = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
            fields.update(reward_greedy=rg, **{f"greedy_{k}": v for k, v in pg_.items()})
            C.log(f"step {step}: sample={texts[0]!r} reward={rewards[0].item():.2f}; greedy eval {rg:.3f}")
            policy.train()
        C.metric(step, **fields)
    C.log(f"RL took {t.lap():.1f}s for {args.steps} steps")
    r1, parts1 = greedy_eval(policy, tok, eval_tasks, args.max_new, device)
    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), policy, tok, args.steps)
    C.status("done", f"saved {path}")
    C.result(reward_greedy_before=r0, reward_greedy_after=r1, **{f"before_{k}": v for k, v in parts0.items()},
             **{f"after_{k}": v for k, v in parts1.items()}, steps=args.steps, checkpoint=path)


# --------------------------------------------------------------------------- real: TRL GRPOTrainer


def real(args):
    transformers = C.require("transformers")  # noqa: F841
    trl = C.require("trl")
    peft = C.require("peft")
    datasets = C.require("datasets")
    tasks = make_tasks(args.n_tasks, args.seed)
    rows = [{"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t.question}],
             "tool": t.tool, "args_json": json.dumps(t.args), "answer": t.answer} for t in tasks]
    ds = datasets.Dataset.from_list(rows)

    def tool_reward(completions, tool, args_json, answer, **kw):
        out = []
        for c, tl, aj, an in zip(completions, tool, args_json, answer):
            text = c[-1]["content"] if isinstance(c, list) else c
            out.append(score(text, Task("", tl, json.loads(aj), an))[0])
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
    trainer = trl.GRPOTrainer(model=args.model, reward_funcs=tool_reward, args=trl.GRPOConfig(**cfg_kw), train_dataset=ds,
                              peft_config=peft_config, callbacks=[C.make_metric_callback()])
    out = trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter)
    last = {k.replace("/", "_"): v for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items()
            if isinstance(v, (int, float))}
    C.status("done", f"adapter saved to {adapter}")
    C.result(train_loss=out.training_loss, steps=out.global_step, adapter=adapter, model=args.model, **last)


def main():
    args = build_parser().parse_args()
    d = dict(steps=60, group=4, batch=8, lr=2e-4, max_new=80) if args.smoke else dict(steps=200, group=8, batch=8, lr=1e-5, max_new=128)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if args.smoke else real)(args)


if __name__ == "__main__":
    main()
