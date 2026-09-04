"""Speculative decoding: a small draft model proposes, the target verifies (Lab 17).

What this teaches
  * the accept/reject rule that makes speculative sampling exact. Let q be the
    draft's next-token distribution and p the target's, both at the same
    temperature. The draft proposes x ~ q. Accept x with probability
    min(1, p(x) / q(x)); on rejection, sample the replacement from the
    residual distribution
        p'(y) = max(0, p(y) - q(y)) / sum_y' max(0, p(y') - q(y'))
    and stop verifying further draft tokens. If all k draft tokens are
    accepted, sample one more token from the target's p at position k. The
    output distribution is then exactly p at every position, whatever q is;
    q only changes how many tokens each target forward pass yields.
  * the numbers that matter: acceptance rate (accepted draft tokens / proposed
    draft tokens), mean accepted length per verify pass, tokens produced per
    target forward (mean accepted + 1), and wall-clock tokens/s for the plain
    loop and the speculative loop. Speculation wins when the target forward
    is much more expensive than k draft forwards; in the CPU smoke both models
    are tiny, so the speedup is small or negative, and the recipe says so
    rather than hiding it.
  * a total-variation check of exactness: draw N first tokens from the
    speculative procedure and N from plain target sampling, compare the two
    empirical distributions (TV = 0.5 * sum |f1 - f2|), and compare against
    the TV between two independent plain draws so you know the noise floor.

How to run
  smoke (CPU, offline): a 1-layer draft and a 2-layer target, both trained
  briefly on the synthetic corpus (or --target-ckpt from pretrain_nano.py):
    python lab/recipes/spec_decode.py --smoke --steps 300 --k 4
    python lab/recipes/spec_decode.py --smoke --target-ckpt out/pretrain_nano/ckpt.pt --k 4
  real (RTX 5090): two models of the same family and tokenizer,
    python lab/recipes/spec_decode.py --target Qwen/Qwen2.5-1.5B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct --k 5 --prompts-jsonl prompts.jsonl
  which times transformers' generate(assistant_model=...) against plain generate
  and also runs the hand-written loop below on the same prompts.
  needs: pip install transformers
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("spec_decode", __doc__.split("\n")[0])
    p.add_argument("--k", type=int, default=4, help="draft tokens per verify pass")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--tv-samples", type=int, default=2000, help="smoke: draws for the total-variation check")
    p.add_argument("--target-ckpt", default=None, help="smoke: pretrain_nano checkpoint as the target")
    p.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompts-jsonl", default=None, help='real: rows of {"prompt": ...}')
    p.add_argument("--log-passes", type=int, default=6, help="print a ROLLOUT line for the first N verify passes (0 disables)")
    return p


# --------------------------------------------------------------------------- the algorithm (model-agnostic)


def spec_step(next_logits_target, next_logits_draft, prefix: torch.Tensor, k: int, T: float, gen: torch.Generator):
    """One verify pass, vectorized over B rows that share a prefix length.

    next_logits_*(ids) -> logits (B, L, V) for the whole sequence. Returns
    (new_tokens list-of-lists, n_accepted (B,), drafts (B, k)). Each row yields n_accepted + 1 tokens.
    """
    B, S = prefix.shape
    x = prefix
    qs = []
    for _ in range(k):                                           # draft k tokens autoregressively
        q = F.softmax(next_logits_draft(x)[:, -1, :].float() / T, -1)
        tok = torch.multinomial(q, 1, generator=gen)
        qs.append(q)
        x = torch.cat([x, tok], 1)
    drafts = x[:, S:]                                            # (B, k)
    q = torch.stack(qs, 1)                                       # (B, k, V)
    p = F.softmax(next_logits_target(x)[:, S - 1:, :].float() / T, -1)   # (B, k+1, V): p_0..p_k in ONE target pass
    p_d = p[:, :k].gather(-1, drafts[..., None])[..., 0]         # p_i(x_i)
    q_d = q.gather(-1, drafts[..., None])[..., 0]                # q_i(x_i)
    u = torch.rand(B, k, generator=gen)
    accept = u < torch.clamp(p_d / q_d.clamp(min=1e-20), max=1.0)
    n_acc = torch.where(accept.all(1), torch.full((B,), k), accept.int().argmin(1))   # first rejection index, or k
    out = []
    for b in range(B):
        n = int(n_acc[b])
        if n < k:
            resid = torch.clamp(p[b, n] - q[b, n], min=0)
            resid = resid / resid.sum() if resid.sum() > 0 else p[b, n]
            extra = torch.multinomial(resid, 1, generator=gen)
        else:
            extra = torch.multinomial(p[b, k], 1, generator=gen)
        out.append(drafts[b, :n].tolist() + [int(extra)])
    return out, n_acc, drafts


def speculative_generate(next_target, next_draft, prompt: list[int], k: int, T: float, max_new: int, eos_id, gen, seq_len=None,
                         on_pass=None):
    """Sequential loop for one prompt. Returns (tokens, stats).

    on_pass(context_ids, draft_ids, n_accepted, new_ids) is called after every verify pass (for ROLLOUT lines)."""
    ids = torch.tensor([prompt])
    produced, proposed, accepted, passes = [], 0, 0, 0
    while len(produced) < max_new:
        kk = min(k, max_new - len(produced))
        if seq_len is not None and ids.shape[1] + kk + 1 > seq_len:
            break
        new, n_acc, drafts = spec_step(next_target, next_draft, ids, kk, T, gen)
        if on_pass is not None:
            on_pass(ids[0].tolist(), drafts[0].tolist(), int(n_acc[0]), new[0])
        passes += 1
        proposed += kk
        accepted += int(n_acc[0])
        for t in new[0]:
            produced.append(t)
            if eos_id is not None and t == eos_id:
                break
        if eos_id is not None and produced[-1] == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([new[0]])], 1)
    return produced, {"passes": passes, "proposed": proposed, "accepted": accepted}


def plain_generate(next_target, prompt: list[int], T: float, max_new: int, eos_id, gen, seq_len=None):
    """Same style as the speculative loop (full re-forward per token, no KV cache) so the timing is comparable."""
    ids = torch.tensor([prompt])
    produced = []
    while len(produced) < max_new:
        if seq_len is not None and ids.shape[1] + 1 > seq_len:
            break
        p = F.softmax(next_target(ids)[:, -1, :].float() / T, -1)
        t = int(torch.multinomial(p, 1, generator=gen))
        produced.append(t)
        ids = torch.cat([ids, torch.tensor([[t]])], 1)
        if eos_id is not None and t == eos_id:
            break
    return produced


def make_pass_logger(decode, limit: int):
    """Returns (state, on_pass). on_pass prints a ROLLOUT line for the first `limit` verify passes:
    the context tail, the k draft tokens, how many were accepted, and the token that replaced the
    first rejected draft (sampled from the residual max(0, p - q)) or the bonus token from p_k."""
    state = {"n": 0, "prompt": 0}

    def on_pass(ctx_ids, draft_ids, n_acc, new_ids):
        state["n"] += 1
        if state["n"] > limit:
            return
        k = len(draft_ids)
        C.rollout(step=state["n"], prompt=state["prompt"], context_tail=C.clip_text(decode(ctx_ids[-48:]), 300),
                  draft=decode(draft_ids), draft_ids=draft_ids, proposed=k, accepted=n_acc, accepted_text=decode(draft_ids[:n_acc]),
                  corrected_token=decode([new_ids[-1]]), corrected_id=new_ids[-1], correction="residual" if n_acc < k else "bonus",
                  emitted=decode(new_ids))

    return state, on_pass


def total_variation(a: torch.Tensor, b: torch.Tensor, V: int) -> float:
    fa = torch.bincount(a, minlength=V).float() / a.numel()
    fb = torch.bincount(b, minlength=V).float() / b.numel()
    return 0.5 * (fa - fb).abs().sum().item()


# --------------------------------------------------------------------------- smoke


def train_lm(cfg, ids, steps, device, seed, tag):
    C.set_seed(seed)
    model = C.GPT(cfg).to(device)
    opt = C.make_adamw(model, 3e-3)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        lr = C.lr_at(step, steps, 3e-3, 10, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = C.random_windows(ids, 16, 64, gen)
        loss = C.lm_loss(model(x.to(device)), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        C.metric(step, **{f"{tag}_loss": loss.item()})
    return model.eval()


def smoke(args):
    device = C.pick_device(args.device)
    tok = C.CharTokenizer()
    ids = torch.tensor(tok.encode(C.synthetic_text("all")), dtype=torch.long)
    C.status("train", "target (2 layers) and draft (1 layer)")
    if args.target_ckpt:
        target, tok, _ = C.load_checkpoint(args.target_ckpt, device)
        target.eval()
    else:
        target = train_lm(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=160), ids, args.steps, device, args.seed, "target")
    draft = train_lm(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=1, d_model=64, n_head=4, seq_len=160), ids, args.steps, device, args.seed + 1, "draft")
    seq_len = min(target.cfg.seq_len, draft.cfg.seq_len)
    nt = lambda x: target(x.to(device)).cpu()  # noqa: E731
    nd = lambda x: draft(x.to(device)).cpu()   # noqa: E731
    prompts = [tok.encode(s) for s in ["the cat", "a boy named", "the old bear", "the girl", "two rabbits", "the boat was", "mia lost", "the wind"][: args.n_prompts]]

    C.status("decode", f"{len(prompts)} prompts, k={args.k}, T={args.temperature}, max_new={args.max_new}")
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    plain_tokens = sum(len(plain_generate(nt, p, args.temperature, args.max_new, tok.eos_id, gen, seq_len)) for p in prompts)
    plain_s = time.perf_counter() - t0
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    spec_tokens = passes = proposed = accepted = 0
    pl_state, on_pass = make_pass_logger(lambda ids: "".join(tok.itos[int(i)] for i in ids), args.log_passes)
    for i, p in enumerate(prompts):
        pl_state["prompt"] = i
        out, st = speculative_generate(nt, nd, p, args.k, args.temperature, args.max_new, tok.eos_id, gen, seq_len, on_pass)
        spec_tokens += len(out)
        passes += st["passes"]
        proposed += st["proposed"]
        accepted += st["accepted"]
        C.metric(i, accepted=st["accepted"], proposed=st["proposed"], passes=st["passes"], tokens=len(out))
        if i == 0:
            C.log(f"  sample: {tok.decode(p)!r} -> {tok.decode(out)!r}")
    spec_s = time.perf_counter() - t0
    acc_rate = accepted / max(1, proposed)
    C.log(f"acceptance rate {acc_rate:.3f}, mean accepted per pass {accepted / max(1, passes):.2f}, tokens per target forward {spec_tokens / max(1, passes):.2f}")
    C.log(f"plain {plain_tokens / plain_s:.1f} tok/s, speculative {spec_tokens / spec_s:.1f} tok/s (both loops re-run the full prefix, no KV cache)")

    C.status("check", f"total variation of first-token frequencies over {args.tv_samples} draws")
    N = args.tv_samples
    prefix = torch.tensor([tok.encode("the cat sat on the ")]).repeat(N, 1)
    gen = torch.Generator().manual_seed(args.seed + 7)
    new, _, _ = spec_step(nt, nd, prefix, args.k, args.temperature, gen)
    spec_first = torch.tensor([row[0] for row in new])
    p0 = F.softmax(nt(prefix[:1])[:, -1, :].float() / args.temperature, -1)[0]
    plain_a = torch.multinomial(p0, N, replacement=True, generator=gen)
    plain_b = torch.multinomial(p0, N, replacement=True, generator=gen)
    tv_spec = total_variation(spec_first, plain_a, tok.vocab_size)
    tv_plain = total_variation(plain_b, plain_a, tok.vocab_size)
    q0 = F.softmax(nd(prefix[:1])[:, -1, :].float() / args.temperature, -1)[0]
    tv_draft = 0.5 * (p0 - q0).abs().sum().item()
    C.log(f"TV(speculative, plain) = {tv_spec:.4f}; TV(plain, plain) noise floor = {tv_plain:.4f}; TV(draft dist, target dist) = {tv_draft:.4f}")
    C.status("done", "")
    C.result(k=args.k, acceptance_rate=acc_rate, mean_accepted_per_pass=accepted / max(1, passes), tokens_per_target_forward=spec_tokens / max(1, passes),
             tok_s_plain=plain_tokens / plain_s, tok_s_speculative=spec_tokens / spec_s, tv_spec_vs_plain=tv_spec, tv_plain_vs_plain=tv_plain,
             tv_draft_vs_target=tv_draft, steps=args.steps)


# --------------------------------------------------------------------------- real


def real(args):
    transformers = C.require("transformers")
    device = C.pick_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    C.status("load", f"target={args.target} draft={args.draft}")
    tok = transformers.AutoTokenizer.from_pretrained(args.target)
    target = transformers.AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()
    draft = transformers.AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(device).eval()
    prompts = [r["prompt"] for r in C.read_jsonl(args.prompts_jsonl)] if args.prompts_jsonl else \
        ["Explain why the sky is blue in three sentences.", "Write a short poem about a lighthouse.", "List five uses for a paperclip.",
         "What is the difference between a list and a tuple in Python?"]
    prompts = prompts[: args.n_prompts]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    do_sample = args.temperature > 0

    def timed_generate(assistant):
        n, t0 = 0, time.perf_counter()
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            with torch.no_grad():
                out = target.generate(**enc, max_new_tokens=args.max_new, do_sample=do_sample, temperature=args.temperature if do_sample else None,
                                      assistant_model=assistant, pad_token_id=tok.eos_token_id)
            n += out.shape[1] - enc["input_ids"].shape[1]
        sync()
        return n / (time.perf_counter() - t0)

    C.status("generate", "transformers generate: plain vs assistant_model")
    tps_plain = timed_generate(None)
    tps_assist = timed_generate(draft)
    C.log(f"transformers: plain {tps_plain:.1f} tok/s, assistant_model {tps_assist:.1f} tok/s")

    C.status("loop", f"hand-written speculative loop, k={args.k} (no KV cache, full re-forward each pass)")
    nt = lambda x: target(x.to(device)).logits.float().cpu()  # noqa: E731
    nd = lambda x: draft(x.to(device)).logits.float().cpu()   # noqa: E731
    gen = torch.Generator().manual_seed(args.seed)
    T = args.temperature if do_sample else 1e-4                 # near-greedy when temperature is 0
    passes = proposed = accepted = produced = 0
    pl_state, on_pass = make_pass_logger(tok.decode, args.log_passes)
    t0 = time.perf_counter()
    for i, t in enumerate(texts):
        ids = tok(t)["input_ids"]
        pl_state["prompt"] = i
        with torch.no_grad():
            out, st = speculative_generate(nt, nd, ids, args.k, T, args.max_new, tok.eos_token_id, gen, on_pass=on_pass)
        passes += st["passes"]
        proposed += st["proposed"]
        accepted += st["accepted"]
        produced += len(out)
        C.metric(i, accepted=st["accepted"], proposed=st["proposed"], passes=st["passes"])
        C.log(f"  {prompts[i][:40]!r} -> {tok.decode(out)[:100]!r}")
    sync()
    tps_loop = produced / (time.perf_counter() - t0)
    C.status("done", "")
    C.result(target=args.target, draft=args.draft, k=args.k, acceptance_rate=accepted / max(1, proposed), mean_accepted_per_pass=accepted / max(1, passes),
             tokens_per_target_forward=produced / max(1, passes), tok_s_transformers_plain=tps_plain, tok_s_transformers_assisted=tps_assist,
             tok_s_handwritten_loop=tps_loop)


def main():
    args = build_parser().parse_args()
    d = dict(steps=300, max_new=40, n_prompts=8) if args.smoke else dict(steps=0, max_new=128, n_prompts=4)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if args.smoke else real)(args)


if __name__ == "__main__":
    main()
