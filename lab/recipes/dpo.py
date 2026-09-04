"""Direct Preference Optimization with a frozen reference (Lab 05: preference learning).

What this teaches
  * the DPO loss written out. For a prompt x with a chosen response y_w and a
    rejected response y_l, with policy pi and frozen reference pi_ref,
        r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))        (implicit reward)
        L = -log sigmoid(r(x, y_w) - r(x, y_l))
    where log pi(y|x) is the SUM of token log-probabilities over the response
    tokens only. The reference keeps the policy from drifting: the reward is
    relative to what the base model already believed.
  * what to watch: reward margin (should rise), implicit reward accuracy (the
    fraction of pairs with positive margin), and log pi(y_w|x) itself, which
    can fall even while the margin rises (likelihood displacement).

How to run
  smoke (CPU, offline): the loss is implemented by hand on the minimal GPT
  with a deep-copied frozen reference, on the synthetic preference triples.
  A short SFT warm-up on the chosen answers runs first (as it would in a real
  pipeline) unless --ckpt gives an SFT checkpoint:
    python lab/recipes/dpo.py --smoke --steps 100 --beta 0.1
  real (RTX 5090): TRL's DPOTrainer with a LoRA policy; the adapter-disabled
  model serves as the reference so only one copy of the weights is loaded:
    python lab/recipes/dpo.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/ultrafeedback_binarized --max-samples 2000 --steps 200
  needs: pip install transformers trl peft datasets
"""
from __future__ import annotations

import copy
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("dpo", __doc__.split("\n")[0])
    p.add_argument("--ckpt", default=None, help="smoke: SFT checkpoint to start from (skips the warm-up)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warm-steps", type=int, default=150, help="smoke: SFT steps on chosen answers before DPO")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--split", default="train")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--loss-type", default="sigmoid", help="TRL loss_type: sigmoid, ipo, hinge, ...")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return p


# --------------------------------------------------------------------------- smoke: DPO by hand


def encode_pair(tok, prompt: str, response: str):
    p = tok.encode(f"q: {prompt}\na: ")
    r = tok.encode(response, add_eos=True)
    return p + r, len(p)


def sequence_logps(model, ids: torch.Tensor, mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token_t | tokens_<t) over response tokens, per row. ids: (B, L) right-padded."""
    logits = model(ids[:, :-1]).float()
    logp = F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]          # (B, L-1)
    pos = torch.arange(1, ids.shape[1], device=ids.device)[None]
    resp = (pos >= prompt_lens[:, None]) & mask[:, 1:]                                # response positions only
    return (logp * resp).sum(-1)


def smoke(args):
    device = C.pick_device(args.device)
    if args.ckpt:
        policy, tok, _ = C.load_checkpoint(args.ckpt, device)
    else:
        tok = C.CharTokenizer()
        policy = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=96)).to(device)
    pairs = C.PREFS
    seqs, plens = [], []
    for prompt, chosen, rejected in pairs:
        for resp in (chosen, rejected):
            s, pl = encode_pair(tok, prompt, resp)
            seqs.append(s)
            plens.append(pl)
    ids, mask = C.pad_batch(seqs, tok.pad_id)
    ids, mask = ids.to(device), mask.to(device)
    plens = torch.tensor(plens, device=device)
    n = len(pairs)
    c_idx = torch.arange(0, 2 * n, 2, device=device)
    r_idx = c_idx + 1

    if not args.ckpt and args.warm_steps > 0:
        # SFT warm-up on the chosen answers: DPO assumes the policy already puts mass on the responses it ranks
        C.status("warmup", f"{args.warm_steps} SFT steps on chosen answers")
        opt = C.make_adamw(policy, 3e-3, 0.0)
        labels = ids[c_idx][:, 1:].clone()
        pos = torch.arange(1, ids.shape[1], device=device)[None]
        labels[~((pos >= plens[c_idx][:, None]) & mask[c_idx][:, 1:])] = -100
        for step in range(args.warm_steps):
            loss = C.lm_loss(policy(ids[c_idx][:, :-1]), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        C.log(f"warm-up SFT loss {loss.item():.3f}")

    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ref_logp = sequence_logps(ref, ids, mask, plens)
    opt = C.make_adamw(policy, args.lr, 0.0)
    C.status("train", f"{args.steps} DPO steps, beta={args.beta}, {n} pairs")
    policy.train()
    for step in range(args.steps):
        lr = C.lr_at(step, args.steps, args.lr, 5, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        logp = sequence_logps(policy, ids, mask, plens)
        rew = args.beta * (logp - ref_logp)                       # implicit rewards
        margin = rew[c_idx] - rew[r_idx]
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        C.metric(step, loss=loss.item(), lr=lr, reward_chosen=rew[c_idx].mean(), reward_rejected=rew[r_idx].mean(),
                 reward_margin=margin.mean(), reward_accuracy=(margin > 0).float().mean(), logps_chosen=logp[c_idx].mean(),
                 logps_rejected=logp[r_idx].mean())
    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), policy, tok, args.steps)
    C.status("done", f"saved {path}")
    C.result(final_loss=loss.item(), reward_margin=margin.mean(), reward_accuracy=(margin > 0).float().mean(),
             logps_chosen=logp[c_idx].mean(), beta=args.beta, pairs=n, checkpoint=path)


# --------------------------------------------------------------------------- real: TRL DPOTrainer


def real(args):
    transformers = C.require("transformers")
    trl = C.require("trl")
    peft = C.require("peft")
    datasets = C.require("datasets")
    device = C.pick_device(args.device)
    C.status("load", args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    peft_config = peft.LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.target_modules.split(","),
                                  task_type="CAUSAL_LM")
    C.status("data", f"{args.dataset} split={args.split} max={args.max_samples}")
    ds = datasets.load_dataset(args.dataset, split=args.split, streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if len(rows) >= args.max_samples:
            break
        if "chosen" not in row or "rejected" not in row:
            raise SystemExit(f"dataset needs chosen/rejected columns; columns are {list(row.keys())}")
        r = {"chosen": row["chosen"], "rejected": row["rejected"]}
        if "prompt" in row:
            r["prompt"] = row["prompt"]
        rows.append(r)
    train_ds = datasets.Dataset.from_list(rows)
    C.log(f"{len(rows)} preference rows")

    sig = inspect.signature(trl.DPOConfig.__init__).parameters
    cfg_kw = dict(output_dir=os.path.join(args.out, "trainer"), max_steps=args.steps, per_device_train_batch_size=args.batch,
                  gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                  logging_steps=1, save_strategy="no", report_to="none", bf16=device.type == "cuda", seed=args.seed,
                  beta=args.beta, loss_type=args.loss_type, max_length=args.max_len)
    if "max_prompt_length" in sig:
        cfg_kw["max_prompt_length"] = args.max_prompt_len
    tr_sig = inspect.signature(trl.DPOTrainer.__init__).parameters
    tr_kw = dict(model=model, ref_model=None, args=trl.DPOConfig(**cfg_kw), train_dataset=train_ds, peft_config=peft_config,
                 callbacks=[C.make_metric_callback()])
    tr_kw["processing_class" if "processing_class" in tr_sig else "tokenizer"] = tokenizer
    trainer = trl.DPOTrainer(**tr_kw)
    out = trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    last = {k: v for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items() if isinstance(v, (int, float))}
    C.status("done", f"adapter saved to {adapter}")
    C.result(train_loss=out.training_loss, steps=out.global_step, beta=args.beta, adapter=adapter,
             **{k.replace("/", "_"): v for k, v in last.items() if k.startswith("rewards")})


def main():
    args = build_parser().parse_args()
    d = dict(steps=100, lr=1e-3, batch=None) if args.smoke else dict(steps=200, lr=5e-6, batch=2)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if args.smoke else real)(args)


if __name__ == "__main__":
    main()
