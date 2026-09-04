"""Architecture poking: print a model's module tree and derived config, then probe one string with hooks
to see residual-stream norms, attention entropy and the logit lens, layer by layer (Labs 11, 13).

What this teaches
  * where the parameter budget goes: embeddings, attention, MLP, norms, head, and what tying the
    unembedding to the embedding does to the count
  * the cost numbers a config implies: forward FLOPs per token = 2N + 2 * n_layer * ctx * d_model
    (N = non-embedding parameters; the second term is the score and value products at context ctx),
    KV-cache bytes per token = 2 * n_layer * n_kv_heads * head_dim * bytes(dtype); grouped-query
    attention shrinks the second number and not the first
  * the residual stream: in a pre-norm network every block adds onto it, so its norm grows with
    depth; attention entropy per head (nats, mean over query positions; log(position) is the
    maximum for a causal row) separates sharp heads (previous-token, induction) from diffuse ones
  * the logit lens: apply the final norm and the unembedding to each layer's residual and watch the
    next-token prediction form; agreement with the final top-1 usually switches on in the last third

How to run
  smoke (CPU, offline): trains the minimal GPT from common.py for --steps on the synthetic corpus
  and probes it; or probe a checkpoint from pretrain_nano.py / midtrain.py:
    python lab/recipes/inspect_model.py --smoke
    python lab/recipes/inspect_model.py --model out/pretrain_nano/ckpt.pt --text "the cat sat on the"
  real: any Hugging Face causal LM (loaded with eager attention so attention maps come back):
    python lab/recipes/inspect_model.py --model Qwen/Qwen2.5-0.5B --text "The capital of France is"
    python lab/recipes/inspect_model.py --model meta-llama/Llama-3.2-1B --layer 12 --n-heads 8
  needs: pip install transformers   (real mode only)

Outputs under --out: module_table.json (every parameter: name, shape, count, category),
per_layer.json (residual norms, per-head attention entropy, logit-lens top-k per layer, and the
per-position lens at --layer), attention.json (attention matrices of the first --n-heads heads for
the probe, one entry per layer, plus the token labels), summary.json (config and totals).
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402

SMOKE_TEXT = "the cat sat on the warm step. a bird sang in the"
HF_TEXT = "The capital of France is"


def build_parser():
    p = C.base_parser("inspect_model", __doc__.split("\n")[0])
    p.add_argument("--model", default=None, help="HF id, HF local path, or a common.py checkpoint (*.pt); --smoke trains a tiny GPT if omitted")
    p.add_argument("--text", default=None, help="probe string (default depends on the model kind)")
    p.add_argument("--layer", type=int, default=-1, help="layer whose per-position logit lens and all-head entropies are printed (-1 = last)")
    p.add_argument("--top-k", type=int, default=5, help="logit-lens predictions kept per layer")
    p.add_argument("--n-heads", type=int, default=4, help="attention matrices dumped per layer (first N heads)")
    p.add_argument("--ctx", type=int, default=4096, help="context length for the attention FLOPs term")
    p.add_argument("--dtype", default="auto", help="real: float32, bfloat16, float16 or auto (bf16 on cuda)")
    return p


# --------------------------------------------------------------------------- parameter accounting


def categorize(name: str) -> str:
    n = name.lower()
    if "lm_head" in n or "embed_out" in n or n.endswith("output_layer.weight"):
        return "head"
    if any(t in n for t in ("embed", "wte", "wpe", "tok_emb", "pos_emb")):
        return "embedding"
    if any(t in n for t in ("norm", "ln_", "ln1", "ln2", ".n1.", ".n2.", "layernorm")):
        return "norm"
    if any(t in n for t in ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(t in n for t in ("mlp", "feed_forward", "ffn", "fc", "dense", "experts", "router", "gate", "w_up", "w_down")):
        return "mlp"
    return "other"


def module_table(model: nn.Module, blocks_prefix: str) -> tuple[list[dict], dict, list[int]]:
    """Rows for every parameter (tied weights appear once), totals per category, params per block."""
    rows, totals = [], {}
    for name, p in model.named_parameters():
        cat = categorize(name)
        rows.append({"name": name, "shape": list(p.shape), "params": p.numel(), "category": cat, "dtype": str(p.dtype).replace("torch.", "")})
        totals[cat] = totals.get(cat, 0) + p.numel()
    per_block = {}
    for r in rows:
        if r["name"].startswith(blocks_prefix + "."):
            i = int(r["name"][len(blocks_prefix) + 1:].split(".")[0])
            per_block[i] = per_block.get(i, 0) + r["params"]
    return rows, totals, [per_block[i] for i in sorted(per_block)]


def print_table(rows: list[dict], blocks_prefix: str, per_block: list[int]) -> None:
    """Compact stdout view: everything outside the blocks, block 0 in full, the rest summarized."""
    w = max(len(r["name"]) for r in rows)
    C.log(f"{'module':{w}}  {'shape':>18}  {'params':>12}  category")
    block0 = blocks_prefix + ".0."
    shown_other = False
    for r in rows:
        inside = r["name"].startswith(blocks_prefix + ".")
        if inside and not r["name"].startswith(block0):
            if not shown_other:
                same = all(b == per_block[0] for b in per_block)
                C.log(f"{blocks_prefix}.1..{len(per_block) - 1}: {len(per_block) - 1} more blocks, "
                      + (f"{per_block[0]:,} params each" if same else "params per block " + ",".join(f"{b:,}" for b in per_block[1:])))
                shown_other = True
            continue
        C.log(f"{r['name']:{w}}  {'x'.join(map(str, r['shape'])):>18}  {r['params']:>12,}  {r['category']}")


# --------------------------------------------------------------------------- model loading: two kinds


class Probe:
    """Everything the inspection needs, the same for the minimal GPT and an HF causal LM."""

    def __init__(self, kind, model, blocks, blocks_prefix, embed, final_norm, head, encode, token_labels, config, attn_from_output):
        self.kind, self.model, self.blocks, self.blocks_prefix = kind, model, blocks, blocks_prefix
        self.embed, self.final_norm, self.head = embed, final_norm, head
        self.encode, self.token_labels, self.config = encode, token_labels, config
        self.attn_from_output = attn_from_output   # HF: attentions come from output_attentions; minimal: recomputed in a hook


def minimal_attn_weights(attn: C.Attention, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Recompute softmax(q k^T / sqrt(d)) for the minimal GPT (its forward uses SDPA and keeps no weights). Returns (H, S, S)."""
    B, S, D = x.shape
    q, k, _ = attn.qkv(x).split(D, dim=-1)
    q = q.view(B, S, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k.view(B, S, attn.n_head, attn.head_dim).transpose(1, 2)
    q, k = C.apply_rope(q, cos, sin, 0), C.apply_rope(k, cos, sin, 0)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    if attn.causal:
        scores = scores.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1), float("-inf"))
    return F.softmax(scores.float(), -1)[0]


def load_minimal(args, device) -> Probe:
    if args.model:
        C.status("load", args.model)
        model, tok, _ = C.load_checkpoint(args.model, device)
    else:
        tok = C.CharTokenizer()
        cfg = C.GPTConfig(vocab_size=tok.vocab_size, n_layer=3, d_model=64, n_head=4, seq_len=128)
        model = C.GPT(cfg).to(device)
        ids = torch.tensor(tok.encode(C.synthetic_text("all")), dtype=torch.long)
        C.status("train", f"minimal GPT, {args.steps} steps on the synthetic corpus so the lens has something to show")
        opt = C.make_adamw(model, 3e-3)
        gen = torch.Generator().manual_seed(args.seed)
        for step in range(args.steps):
            lr = C.lr_at(step, args.steps, 3e-3, 10, 0.1)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = C.random_windows(ids, 16, 64, gen)
            loss = C.lm_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0 or step == args.steps - 1:
                C.metric(step, train_loss=loss.item(), lr=lr)
    model.eval()
    c = model.cfg
    hidden = model.blocks[0].mlp.w_gate.out_features
    config = dict(kind="minimal-gpt", n_layer=c.n_layer, loop=c.loop, d_model=c.d_model, n_head=c.n_head, n_kv_head=c.n_head,
                  head_dim=c.d_model // c.n_head, vocab=c.vocab_size, mlp_width=hidden, mlp_kind="SwiGLU", seq_len=c.seq_len,
                  tied_embeddings=bool(c.tie_embeddings), dtype="float32", positional="rotary", norm="RMSNorm")
    labels = lambda ids: [tok.itos[int(i)] for i in ids]  # noqa: E731
    return Probe("minimal", model, model.blocks, "blocks", model.tok_emb, model.norm, model.lm_head, tok.encode, labels, config, False)


def find_hf_parts(model: nn.Module):
    """Locate the block list, its parent, and the final norm without knowing the architecture's names."""
    blocks, blocks_name = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and all(type(m) is type(mod[0]) for m in mod):
            has_attn = any(("attn" in n.lower() or "attention" in n.lower()) for n, _ in mod[0].named_children())
            if has_attn and (blocks is None or len(mod) > len(blocks)):
                blocks, blocks_name = mod, name
    if blocks is None:
        raise SystemExit("could not find the decoder block list (a ModuleList of identical blocks that contain an attention module)")
    parent = model.get_submodule(blocks_name.rsplit(".", 1)[0]) if "." in blocks_name else model
    final_norm = None
    for n, m in parent.named_children():
        if m is blocks:
            continue
        if "norm" in type(m).__name__.lower() or "norm" in n.lower() or n == "ln_f":
            final_norm = m
    if final_norm is None:
        raise SystemExit(f"could not find the final norm among {[n for n, _ in parent.named_children()]}")
    return blocks, blocks_name, final_norm


def load_hf(args, device) -> Probe:
    transformers = C.require("transformers")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        args.dtype, torch.bfloat16 if device.type == "cuda" else torch.float32)
    C.status("load", f"{args.model} ({str(dtype).replace('torch.', '')}, eager attention)")
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model = model.to(device).eval()
    blocks, blocks_name, final_norm = find_hf_parts(model)
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    cfg = model.config
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or embed.weight.shape[1]
    n_head = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_head
    head_dim = getattr(cfg, "head_dim", None) or d_model // n_head
    mlp_width = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or getattr(cfg, "ffn_dim", None) or 4 * d_model
    tied = head is not None and head.weight.data_ptr() == embed.weight.data_ptr()
    act = getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", None) or "?"
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    rope_theta = getattr(cfg, "rope_theta", None) or (rope_params.get("rope_theta") if isinstance(rope_params, dict) else None)
    parent = model.get_submodule(blocks_name.rsplit(".", 1)[0]) if "." in blocks_name else model
    has_rotary = rope_theta is not None or any("rotary" in n.lower() for n, _ in parent.named_children())
    positional = "rotary" if has_rotary else ("learned" if any("wpe" in n or "position_embeddings" in n for n, _ in model.named_parameters()) else "?")
    config = dict(kind=f"hf:{cfg.model_type}", n_layer=len(blocks), loop=1, d_model=d_model, n_head=n_head, n_kv_head=n_kv, head_dim=head_dim,
                  vocab=embed.weight.shape[0], mlp_width=mlp_width, mlp_kind=str(act), seq_len=getattr(cfg, "max_position_embeddings", None),
                  tied_embeddings=bool(tied), dtype=str(dtype).replace("torch.", ""), positional=positional,
                  norm=type(final_norm).__name__, rope_theta=rope_theta)
    encode = lambda s: tok(s)["input_ids"]  # noqa: E731
    labels = lambda ids: tok.convert_ids_to_tokens([int(i) for i in ids])  # noqa: E731
    return Probe("hf", model, blocks, blocks_name, embed, final_norm, head, encode, labels, config, True)


# --------------------------------------------------------------------------- derived numbers


def derived(config: dict, totals: dict, ctx: int) -> dict:
    total = sum(totals.values())
    non_embed = total - totals.get("embedding", 0) - totals.get("head", 0)
    L, d = config["n_layer"] * config.get("loop", 1), config["d_model"]
    bytes_per = {"float32": 4, "bfloat16": 2, "float16": 2}.get(config["dtype"], 2)
    kv_per_token_elems = 2 * L * config["n_kv_head"] * config["head_dim"]
    return dict(params_total=total, params_non_embedding=non_embed, flops_per_token_2N=2 * non_embed,
                flops_attention_term=2 * L * ctx * d, flops_per_token_forward=2 * non_embed + 2 * L * ctx * d,
                flops_per_token_train=3 * (2 * non_embed + 2 * L * ctx * d), ctx_for_flops=ctx,
                kv_bytes_per_token=kv_per_token_elems * bytes_per, kv_bytes_per_token_bf16=kv_per_token_elems * 2,
                kv_bytes_per_token_fp8=kv_per_token_elems, gqa_ratio=config["n_head"] / max(1, config["n_kv_head"]))


# --------------------------------------------------------------------------- the probe


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """attn (H, S, S) rows sum to 1. Entropy in nats per query row, mean over rows -> (H,)."""
    a = attn.float()
    return (-(a * torch.log(a.clamp(min=1e-12))).sum(-1)).mean(-1)


@torch.no_grad()
def run_probe(pr: Probe, text: str, device, top_k: int, n_heads: int, sel_layer: int):
    ids = pr.encode(text)
    labels = pr.token_labels(ids)
    x = torch.tensor([ids], device=device)
    S = len(ids)
    resids: list[torch.Tensor] = []
    attns: list[torch.Tensor] = []
    handles = []

    def pre0(mod, args):                       # the input to block 0 = the (scaled) embedding
        resids.append(args[0].detach())

    def post(mod, args, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        resids.append(h.detach())

    handles.append(pr.blocks[0].register_forward_pre_hook(pre0))
    for blk in pr.blocks:
        handles.append(blk.register_forward_hook(post))
    if not pr.attn_from_output:
        for blk in pr.blocks:
            def attn_hook(mod, args, out):
                attns.append(minimal_attn_weights(mod, args[0], args[1], args[2]).detach())
            handles.append(blk.attn.register_forward_hook(attn_hook))
    try:
        if pr.kind == "hf":
            out = pr.model(x, output_attentions=True)
            if getattr(out, "attentions", None) and all(a is not None for a in out.attentions):
                attns = [a[0].detach() for a in out.attentions]
            else:
                C.log("attention weights not returned (attn_implementation is not eager?); entropies will be null")
            final_logits = out.logits[0, -1].float()
        else:
            final_logits = pr.model(x)[0, -1].float()
    finally:
        for h in handles:
            h.remove()
    p_final = F.softmax(final_logits, -1)
    final_top = int(p_final.argmax())

    per_layer, attn_dump = [], []
    for l, h in enumerate(resids):
        h = h.float()
        norms = h[0].norm(dim=-1)                                     # (S,)
        lens_logits = pr.head(pr.final_norm(h[:, -1:].to(next(pr.head.parameters()).dtype)))[0, -1].float()
        p = F.softmax(lens_logits, -1)
        top = torch.topk(p, min(top_k, p.numel()))
        top_ids = top.indices.tolist()
        kl = float((p_final * (torch.log(p_final.clamp(min=1e-12)) - torch.log(p.clamp(min=1e-12)))).sum())
        rec = dict(layer=l, resid_norm_mean=float(norms.mean()), resid_norm_last=float(norms[-1]), resid_norm_per_pos=[round(float(v), 4) for v in norms],
                   lens_top=[{"token": t, "id": i, "prob": round(float(pp), 5)} for t, i, pp in zip(pr.token_labels(top_ids), top_ids, top.values)],
                   lens_top1_prob=float(top.values[0]), lens_agrees_final=int(top_ids[0] == final_top), lens_kl_to_final=kl,
                   lens_prob_of_final_top1=float(p[final_top]))
        if l == sel_layer:
            pos_logits = pr.head(pr.final_norm(h.to(next(pr.head.parameters()).dtype)))[0].float()
            pos_top = pos_logits.argmax(-1).tolist()
            rec["positions"] = [{"pos": i, "token": labels[i], "lens_top1": t} for i, t in enumerate(pr.token_labels(pos_top))]
        if l >= 1 and l - 1 < len(attns):
            a = attns[l - 1]                                          # (H, S, S)
            ent = attention_entropy(a)
            rec["attn_entropy_per_head"] = [round(float(e), 4) for e in ent]
            rec["attn_entropy_mean"] = float(ent.mean())
            rec["attn_entropy_max_possible"] = float(torch.log(torch.arange(1, S + 1).float()).mean())
            rec["attn_prev_token_share"] = round(float(a[:, 1:, :-1].diagonal(dim1=-2, dim2=-1).mean()), 4) if S > 1 else None
            attn_dump.append({"layer": l, "heads": [[[round(float(v), 4) for v in row] for row in a[hh]] for hh in range(min(n_heads, a.shape[0]))]})
        else:
            rec["attn_entropy_per_head"], rec["attn_entropy_mean"] = None, None
        per_layer.append(rec)
    return dict(text=text, tokens=labels, ids=ids, final_top=dict(token=pr.token_labels([final_top])[0], id=final_top, prob=float(p_final[final_top])),
                per_layer=per_layer, attention=attn_dump)


# --------------------------------------------------------------------------- main


def main():
    args = build_parser().parse_args()
    if args.steps is None:
        args.steps = 100
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = C.pick_device(args.device)
    is_minimal = args.smoke or (args.model or "").endswith(".pt")
    if not is_minimal and not args.model:
        raise SystemExit("give --model (HF id / path) or --smoke")
    pr = load_minimal(args, device) if is_minimal else load_hf(args, device)
    text = args.text or (SMOKE_TEXT if is_minimal else HF_TEXT)

    C.status("architecture", pr.config["kind"])
    rows, totals, per_block = module_table(pr.model, pr.blocks_prefix)
    print_table(rows, pr.blocks_prefix, per_block)
    for cat in ("embedding", "attention", "mlp", "norm", "head", "other"):
        totals.setdefault(cat, 0)
    d = derived(pr.config, totals, args.ctx)
    tied = pr.config["tied_embeddings"]
    C.log(f"totals: {d['params_total']:,} params = embedding {totals['embedding']:,} + attention {totals['attention']:,} + mlp {totals['mlp']:,}"
          f" + norms {totals['norm']:,} + head {totals['head']:,}" + (" (tied to the embedding)" if tied else "") + f" + other {totals['other']:,}")
    c = pr.config
    C.log(f"config: {c['n_layer']} layers" + (f" x loop {c['loop']}" if c.get('loop', 1) > 1 else "") + f", d_model {c['d_model']}, {c['n_head']} heads"
          f" / {c['n_kv_head']} kv heads (GQA {d['gqa_ratio']:.0f}:1), head_dim {c['head_dim']}, vocab {c['vocab']:,}, MLP width {c['mlp_width']} ({c['mlp_kind']}),"
          f" seq_len {c['seq_len']}, {c['norm']}, {c['positional']} positions, {c['dtype']}")
    C.log(f"forward FLOPs/token at ctx {args.ctx}: 2N = {d['flops_per_token_2N']:.3e} + attention {d['flops_attention_term']:.3e} = {d['flops_per_token_forward']:.3e}"
          f" (train x3 = {d['flops_per_token_train']:.3e}); KV cache {d['kv_bytes_per_token']:,} bytes/token in {c['dtype']}"
          f" ({d['kv_bytes_per_token_bf16']:,} bf16, {d['kv_bytes_per_token_fp8']:,} fp8)")
    C.metric(0, **{f"params_{k}": v for k, v in totals.items()}, params_total=d["params_total"], flops_per_token_forward=d["flops_per_token_forward"],
             kv_bytes_per_token=d["kv_bytes_per_token"])

    n_resid = c["n_layer"] * c.get("loop", 1) + 1
    sel = args.layer if args.layer >= 0 else n_resid - 1
    sel = min(sel, n_resid - 1)
    C.status("probe", f"{len(pr.encode(text))} tokens: {text!r}; lens detail at layer {sel}")
    res = run_probe(pr, text, device, args.top_k, args.n_heads, sel)
    C.log(f"final next-token: {res['final_top']['token']!r} p={res['final_top']['prob']:.3f}")
    for rec in res["per_layer"]:
        top = ", ".join(f"{t['token']!r}:{t['prob']:.2f}" for t in rec["lens_top"][:3])
        ent = "-" if rec["attn_entropy_mean"] is None else f"{rec['attn_entropy_mean']:.2f}"
        C.log(f"  layer {rec['layer']:>2}  |resid| {rec['resid_norm_mean']:8.2f}  attn H {ent:>5}  lens: {top}  agree={rec['lens_agrees_final']}")
        C.metric(rec["layer"], resid_norm=rec["resid_norm_mean"], resid_norm_last=rec["resid_norm_last"], attn_entropy_mean=rec["attn_entropy_mean"],
                 logit_lens_top1_prob=rec["lens_top1_prob"], logit_lens_agree=rec["lens_agrees_final"], logit_lens_kl_to_final=rec["lens_kl_to_final"],
                 logit_lens_prob_of_final_top1=rec["lens_prob_of_final_top1"])
    detail = res["per_layer"][sel]
    if detail.get("attn_entropy_per_head"):
        C.log(f"layer {sel} attention entropy per head (max possible {detail['attn_entropy_max_possible']:.2f}): "
              + " ".join(f"{e:.2f}" for e in detail["attn_entropy_per_head"]))
    if detail.get("positions"):
        C.log(f"layer {sel} lens per position: " + " ".join(f"{p['token']!r}->{p['lens_top1']!r}" for p in detail["positions"][-8:]))

    paths = {}
    for name, obj in [("module_table", {"rows": rows, "totals": totals, "per_block": per_block}),
                      ("per_layer", {"text": res["text"], "tokens": res["tokens"], "ids": res["ids"], "final_top": res["final_top"], "layers": res["per_layer"]}),
                      ("attention", {"tokens": res["tokens"], "n_heads": args.n_heads, "layers": res["attention"]}),
                      ("summary", {"model": args.model or "minimal-gpt (smoke)", "config": pr.config, "totals": totals, "derived": d, "text": text})]:
        paths[name] = os.path.join(args.out, name + ".json")
        with open(paths[name], "w") as f:
            json.dump(obj, f)
    C.status("done", f"wrote {', '.join(paths.values())}")
    agree_from = next((r["layer"] for r in res["per_layer"] if all(x["lens_agrees_final"] for x in res["per_layer"][r["layer"]:])), n_resid - 1)
    C.result(model=args.model or "minimal-gpt (smoke)", config=pr.config, **{f"params_{k}": v for k, v in totals.items()}, **d,
             n_tokens=len(res["ids"]), final_top=res["final_top"], lens_agrees_from_layer=agree_from,
             resid_norm_first=res["per_layer"][0]["resid_norm_mean"], resid_norm_last_layer=res["per_layer"][-1]["resid_norm_mean"],
             **{f"path_{k}": v for k, v in paths.items()})


if __name__ == "__main__":
    main()
