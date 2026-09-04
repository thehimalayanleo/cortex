"""GRPO where the reward is an image: a policy writes a drawing program, the program is rendered,
and the rendered picture is scored against a reference (a small, faithful version of the
Hugging Face post "Train to paint with code").

What this teaches
  * RL with a reward that is not a label: the policy never sees the reference
    image, only a scalar that says how close its rendered program came. Every
    part of the loop is inspectable here: a tiny drawing DSL, a numpy
    renderer, a reward with a gate, a length term and a similarity term, and
    the same GRPO update as grpo_tool.py (group-normalized advantages, a
    clipped ratio, and a k3 KL penalty to a frozen reference policy).
  * the DSL (one command per line, integer coordinates on a 64 x 64 canvas,
    colour is one of black dark gray light white):
        blob x y r colour          soft disc, gaussian falloff with sigma = r / 2
        circle x y r colour        hard-edged disc
        rect x y w h colour        axis-aligned rectangle
        stroke x1 y1 x2 y2 w colour   line segment of width w
  * the reward, in [0, 1]:
        gate       0.05 if at least one command parses and paints something
        length     0.05 * min(1, commands / 4)
        similarity 0.90 * (1 - RMSE(rendered, reference))      (pixels in [0, 1])
    An empty program scores 0 on all three; a blank canvas already gets a
    similarity credit, which is why the gate and the group-normalized
    advantage matter: the policy is rewarded for beating its own group, not
    for beating zero.

How to run
  smoke (CPU, offline): the minimal GPT at character level is the policy; a
  short SFT on random valid programs teaches the syntax (the real version
  starts from an instruct model that already knows the language), then GRPO
  raises the reward. The best canvas is saved as PGM (and PNG when Pillow is
  installed) under --out:
    python lab/recipes/paint_grpo.py --smoke --steps 60
    python lab/recipes/paint_grpo.py --smoke --steps 60 --reference my_photo.png
  real (RTX 5090): TRL GRPOTrainer, a small instruct model emitting the same
  DSL, the same reward function:
    python lab/recipes/paint_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200 --group 8
  needs: pip install transformers trl peft datasets   (Pillow optional)

How the Hugging Face post scales this (as described there): the policy is
Qwen3.5-35B-A3B with a LoRA on all linear layers; programs are p5.brush
sketches rendered in headless Chromium; the reward mixes a gate (0.05), a
length term (0.05), a Qwen3-VL pairwise judge (0.60) and the HPSv3 aesthetic
scorer (0.30); learning rate 5e-5 with a constant_with_warmup schedule, 8
rollouts per step, and scale_rewards set to none (advantages are centred but
not divided by the group standard deviation). This file keeps the structure
and swaps the judge for pixel similarity so the whole thing runs on a CPU.
"""
from __future__ import annotations

import copy
import inspect
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402

SIZE = 64
COLOURS = {"black": 0.0, "dark": 0.25, "gray": 0.5, "light": 0.75, "white": 1.0}
PROMPT = "draw:\n"
SYSTEM_PROMPT = (
    "Write a drawing program, one command per line, nothing else. Canvas is 64x64, integer coordinates.\n"
    "Commands: blob x y r colour | circle x y r colour | rect x y w h colour | stroke x1 y1 x2 y2 w colour\n"
    "colour is one of: black dark gray light white. Use 1 to 6 commands."
)

# --------------------------------------------------------------------------- DSL: parse and render

_YY, _XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def parse(program: str) -> list[tuple]:
    """Return the list of valid commands; invalid lines are skipped."""
    cmds = []
    for line in program.strip().splitlines():
        t = line.strip().split()
        if not t:
            continue
        try:
            if t[0] == "blob" and len(t) == 5:
                cmds.append(("blob", int(t[1]), int(t[2]), int(t[3]), COLOURS[t[4]]))
            elif t[0] == "circle" and len(t) == 5:
                cmds.append(("circle", int(t[1]), int(t[2]), int(t[3]), COLOURS[t[4]]))
            elif t[0] == "rect" and len(t) == 6:
                cmds.append(("rect", int(t[1]), int(t[2]), int(t[3]), int(t[4]), COLOURS[t[5]]))
            elif t[0] == "stroke" and len(t) == 7:
                cmds.append(("stroke", int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]), COLOURS[t[6]]))
        except (ValueError, KeyError):
            continue
    return cmds


def render(cmds: list[tuple]) -> np.ndarray:
    """Paint commands in order onto a black canvas. Returns float32 (SIZE, SIZE) in [0, 1]."""
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    for c in cmds:
        if c[0] in ("blob", "circle"):
            _, x, y, r, col = c
            if r <= 0:
                continue
            d = np.sqrt((_XX - x) ** 2 + (_YY - y) ** 2)
            a = np.exp(-(d ** 2) / (2 * (r / 2) ** 2)) if c[0] == "blob" else np.clip(r + 0.5 - d, 0, 1)
        elif c[0] == "rect":
            _, x, y, w, h, col = c
            a = ((_XX >= x) & (_XX < x + w) & (_YY >= y) & (_YY < y + h)).astype(np.float32)
        else:
            _, x1, y1, x2, y2, w, col = c
            px, py = _XX - x1, _YY - y1
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            t = np.clip((px * vx + py * vy) / L2, 0, 1) if L2 > 0 else np.zeros_like(px)
            d = np.sqrt((px - t * vx) ** 2 + (py - t * vy) ** 2)
            a = np.clip(w / 2 + 0.5 - d, 0, 1)
        canvas = canvas * (1 - a) + col * a
    return canvas


def procedural_reference() -> np.ndarray:
    """A few overlapping soft circles. Expressible in the DSL, so a perfect score is reachable."""
    return render([("blob", 22, 24, 12, 1.0), ("blob", 42, 30, 14, 0.75), ("blob", 30, 46, 10, 0.5)])


def load_reference(path: str) -> np.ndarray:
    Image = C.require("PIL.Image", "Pillow").Image
    im = Image.open(path).convert("L").resize((SIZE, SIZE))
    return np.asarray(im, dtype=np.float32) / 255.0


def reward(program: str, ref: np.ndarray) -> tuple[float, dict, np.ndarray]:
    cmds = parse(program)
    img = render(cmds)
    gate = 1.0 if cmds and img.max() > 0 else 0.0
    if not gate:
        return 0.0, {"gate": 0.0, "length": 0.0, "similarity": 0.0}, img
    length = min(1.0, len(cmds) / 4)
    sim = 1.0 - float(np.sqrt(np.mean((img - ref) ** 2)))
    return 0.05 * gate + 0.05 * length + 0.9 * sim, {"gate": gate, "length": length, "similarity": sim}, img


def random_program(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(1, 4)):
        k = rng.choice(["blob", "blob", "circle", "rect", "stroke"])
        col = rng.choice(list(COLOURS))
        if k in ("blob", "circle"):
            lines.append(f"{k} {rng.randint(4, 59)} {rng.randint(4, 59)} {rng.randint(3, 16)} {col}")
        elif k == "rect":
            lines.append(f"rect {rng.randint(0, 50)} {rng.randint(0, 50)} {rng.randint(4, 24)} {rng.randint(4, 24)} {col}")
        else:
            lines.append(f"stroke {rng.randint(0, 63)} {rng.randint(0, 63)} {rng.randint(0, 63)} {rng.randint(0, 63)} {rng.randint(1, 6)} {col}")
    return "\n".join(lines)


def save_canvas(img: np.ndarray, path_no_ext: str) -> list[str]:
    paths = []
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    with open(path_no_ext + ".pgm", "wb") as f:
        f.write(f"P5\n{SIZE} {SIZE}\n255\n".encode() + u8.tobytes())
    paths.append(path_no_ext + ".pgm")
    try:
        from PIL import Image

        Image.fromarray(u8).resize((256, 256), Image.NEAREST).save(path_no_ext + ".png")
        paths.append(path_no_ext + ".png")
    except ImportError:
        pass
    return paths


# --------------------------------------------------------------------------- smoke: GRPO by hand


def token_logps(model, ids):
    logits = model(ids[:, :-1]).float()
    return F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]


def warm_start(policy, tok, steps, n_programs, seed, device):
    rng = random.Random(seed)
    C.status("warmup", f"{steps} SFT steps on {n_programs} random valid programs (syntax only)")
    p = tok.encode(PROMPT)
    seqs, labels = [], []
    for _ in range(n_programs):
        c = tok.encode(random_program(rng), add_eos=True)
        seqs.append(p + c)
        labels.append([-100] * len(p) + c)
    ids, mask = C.pad_batch(seqs, tok.pad_id)
    lab, _ = C.pad_batch(labels, -100)
    lab[~mask] = -100
    opt = C.make_adamw(policy, 3e-3, 0.0)
    gen = torch.Generator().manual_seed(seed)
    policy.train()
    for _ in range(steps):
        ix = torch.randint(0, len(seqs), (32,), generator=gen)
        x, y = ids[ix, :-1].to(device), lab[ix, 1:].to(device)
        loss = C.lm_loss(policy(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    C.log(f"warm-up loss {loss.item():.3f}")


def log_group(step, prompt, texts, rewards, adv, parts, kl_seq, G):
    """One ROLLOUT line per sample of the first group of this step (rows 0..G-1)."""
    for i in range(G):
        C.rollout(step=step, group=0, idx=i, prompt=C.clip_text(prompt, 300), completion=C.clip_text(texts[i], 600),
                  reward=float(rewards[i]), advantage=float(adv[i]), gate=parts[i]["gate"], length=parts[i]["length"],
                  similarity=parts[i]["similarity"], kl=float(kl_seq[i]), n_commands=len(parse(texts[i])))


def smoke(args, ref):
    device = C.pick_device(args.device)
    tok = C.CharTokenizer()
    if args.ckpt:
        policy, tok, _ = C.load_checkpoint(args.ckpt, device)
    else:
        policy = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=96, n_head=4, seq_len=160)).to(device)
        warm_start(policy, tok, args.warm_steps, 256, args.seed, device)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = C.make_adamw(policy, args.lr, 0.0)
    prompt = torch.tensor([tok.encode(PROMPT)], device=device).repeat(args.batch * args.group, 1)
    P = prompt.shape[1]
    best = (-1.0, "", None)
    C.status("train", f"GRPO: {args.steps} steps, {args.batch} groups x G={args.group}, beta={args.beta}")
    for step in range(1, args.steps + 1):
        policy.eval()
        out = C.generate(policy, prompt, args.max_new, temperature=args.temperature, eos_id=tok.eos_id)
        seqs, flags, rewards, parts, texts = [], [], [], [], []
        for row in out:
            comp = row[P:].tolist()
            if tok.eos_id in comp:
                comp = comp[: comp.index(tok.eos_id) + 1]
            text = tok.decode(comp)
            r, pr, img = reward(text, ref)
            if r > best[0]:
                best = (r, text, img)
            seqs.append(row[:P].tolist() + comp)
            flags.append([0] * P + [1] * len(comp))
            rewards.append(r)
            parts.append(pr)
            texts.append(text)
        ids, _ = C.pad_batch(seqs, tok.pad_id)
        cm, _ = C.pad_batch(flags, 0)
        ids, m = ids.to(device), cm[:, 1:].float().to(device)
        rw = torch.tensor(rewards, device=device).view(args.batch, args.group)
        adv = ((rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-4)).view(-1)
        with torch.no_grad():
            old_lp = token_logps(policy, ids)
            ref_lp = token_logps(reference, ids)
        policy.train()
        for _ in range(args.mu):
            lp = token_logps(policy, ids)
            ratio = torch.exp(lp - old_lp)
            pg = -torch.min(ratio * adv[:, None], ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip) * adv[:, None])
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1
            loss = (((pg + args.beta * kl) * m).sum(1) / m.sum(1).clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            kl_mean = ((kl * m).sum() / m.sum()).item()
            clip_frac = ((((ratio - 1).abs() > args.eps_clip).float() * m).sum() / m.sum()).item()
            kl_seq = (kl * m).sum(1) / m.sum(1).clamp(min=1)
        if args.log_rollouts_every and (step % args.log_rollouts_every == 0 or step == 1):
            log_group(step, PROMPT, texts, rewards, adv, parts, kl_seq, args.group)
        C.metric(step, loss=loss.item(), reward_mean=float(np.mean(rewards)), reward_best=best[0], reward_std=float(np.std(rewards)),
                 gate_rate=float(np.mean([p["gate"] for p in parts])), similarity=float(np.mean([p["similarity"] for p in parts])),
                 kl=kl_mean, clip_frac=clip_frac, completion_len=m.sum(1).mean().item())
        if step % 10 == 0 or step == args.steps:
            C.log(f"step {step}: reward {np.mean(rewards):.3f}, best so far {best[0]:.3f}:\n{best[1]}")
    paths = save_canvas(best[2], os.path.join(args.out, "best")) if best[2] is not None else []
    paths += save_canvas(ref, os.path.join(args.out, "reference"))
    with open(os.path.join(args.out, "best_program.txt"), "w") as f:
        f.write(best[1] + "\n")
    ck = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), policy, tok, args.steps)
    C.status("done", f"best canvas {paths[0]}")
    C.result(reward_best=best[0], reward_mean_last=float(np.mean(rewards)), steps=args.steps, artifacts=paths, checkpoint=ck,
             best_program=best[1])


# --------------------------------------------------------------------------- real: TRL GRPOTrainer


def real(args, ref):
    trl = C.require("trl")
    peft = C.require("peft")
    datasets = C.require("datasets")
    rows = [{"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Draw a picture with a few overlapping soft shapes."}]}] * args.n_prompts
    ds = datasets.Dataset.from_list(rows)
    best = {"reward": -1.0, "program": "", "img": None}

    calls = {"n": 0}

    def paint_reward(completions, prompts=None, **kw):
        out, parts_all, texts = [], [], []
        for c in completions:
            text = c[-1]["content"] if isinstance(c, list) else c
            text = text.replace("```", "").strip()
            r, parts, img = reward(text, ref)
            if r > best["reward"]:
                best.update(reward=r, program=text, img=img)
            out.append(r)
            parts_all.append(parts)
            texts.append(text)
        # ROLLOUT lines for the first group (TRL keeps a prompt's completions contiguous); the advantage is
        # the group normalization computed here for display, one reward call per generation step.
        calls["n"] += 1
        if args.log_rollouts_every and calls["n"] % args.log_rollouts_every == 0:
            G = min(args.group, len(out))
            grp = torch.tensor(out[:G])
            adv = (grp - grp.mean()) / (grp.std() + 1e-4) if G > 1 else grp * 0
            for i in range(G):
                pr = prompts[i] if prompts else ""
                q = pr[-1]["content"] if isinstance(pr, list) and pr else pr
                C.rollout(step=calls["n"], group=0, idx=i, prompt=C.clip_text(q, 300), completion=C.clip_text(texts[i], 600),
                          reward=out[i], advantage=float(adv[i]), n_commands=len(parse(texts[i])), **parts_all[i])
        return out

    sig = inspect.signature(trl.GRPOConfig.__init__).parameters
    cfg_kw = dict(output_dir=os.path.join(args.out, "trainer"), max_steps=args.steps, learning_rate=args.lr,
                  per_device_train_batch_size=args.group, num_generations=args.group, max_completion_length=args.max_new,
                  beta=args.beta, logging_steps=1, save_strategy="no", report_to="none", bf16=torch.cuda.is_available(),
                  seed=args.seed, temperature=args.temperature, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.05)
    for k, v in dict(epsilon=args.eps_clip, num_iterations=args.mu).items():
        if k in sig:
            cfg_kw[k] = v
    peft_config = peft.LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = trl.GRPOTrainer(model=args.model, reward_funcs=paint_reward, args=trl.GRPOConfig(**cfg_kw), train_dataset=ds,
                              peft_config=peft_config, callbacks=[C.make_metric_callback()])
    out = trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter)
    paths = save_canvas(best["img"], os.path.join(args.out, "best")) if best["img"] is not None else []
    paths += save_canvas(ref, os.path.join(args.out, "reference"))
    C.status("done", f"adapter saved to {adapter}")
    C.result(train_loss=out.training_loss, steps=out.global_step, reward_best=best["reward"], best_program=best["program"],
             artifacts=paths, adapter=adapter)


def main():
    p = C.base_parser("paint_grpo", __doc__.split("\n")[0])
    p.add_argument("--reference", default=None, help="PNG/JPG to match instead of the procedural reference (needs Pillow)")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--group", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="groups per step")
    p.add_argument("--mu", type=int, default=2)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--warm-steps", type=int, default=300)
    p.add_argument("--n-prompts", type=int, default=512, help="real: dataset rows (all the same prompt)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--log-rollouts-every", type=int, default=5,
                   help="every N steps print one ROLLOUT line per sample of the first group (0 disables)")
    args = p.parse_args()
    d = dict(steps=60, group=4, batch=4, lr=5e-4, max_new=100) if args.smoke else dict(steps=200, group=8, batch=1, lr=5e-5, max_new=256)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ref = load_reference(args.reference) if args.reference else procedural_reference()
    (smoke if args.smoke else real)(args, ref)


if __name__ == "__main__":
    main()
