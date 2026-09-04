#!/usr/bin/env python3
"""Studio brick: build a character (Lab 20: agentic cinema): a hero image, a hero set, and a character LoRA.

Real mode (on the 5090): drives Celwright's identity script (~/identity_v3b.py; env CINEMA_IDENTITY_SCRIPT, run with
CINEMA_WAN_PYTHON) one resumable stage at a time, with the character bible's IDENTITY / STYLE / NEG text patched in:
  hero     <work>/hero_v3.png (generated from the description when no --hero-src is given) + hero_v2.png body crop + crops/
  heroset  <work>/heroset/ with captions.json: framings of the hero via IP-Adapter crops, filtered by DINO; the prototype
  lora     <work>/lora/pytorch_lora_weights.safetensors: rank-16 UNet LoRA on the hero set
Smoke mode (anywhere, no GPU, no network): synthesizes a fake hero and hero set from numpy shapes, a fake train.json,
and computes the same statistics, so the whole pipeline (run, metrics, refresh, contact sheet) runs offline in seconds.

    python cinema_character.py --character okuun --stage heroset --work ~/celwright_v3b --identity "..." --out out/characters/okuun/heroset
    python cinema_character.py --smoke --character test --stage heroset --out /tmp/x

Prints METRIC / STATUS / RESULT lines (the Cortex run protocol). RESULT carries the paths on this machine and
proto_mean / proto_min (DINO cosine of the kept hero-set frames vs the hero crop), p_own (CLIP picks the frame's own
framing), n_kept, elapsed_s, and the contact sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import metric, status, result  # type: ignore
except Exception:  # keep the brick usable standalone
    def _p(tag, **kw):
        print(tag + " " + json.dumps(kw), flush=True)
    def metric(**kw): _p("METRIC", **kw)      # noqa: E306
    def status(**kw): _p("STATUS", **kw)      # noqa: E306
    def result(**kw): _p("RESULT", **kw)      # noqa: E306

STAGES = ["hero", "heroset", "lora"]
FRAMINGS_SMOKE = ["close-up of the head", "full body, side view", "low angle from below", "waist-deep in dark water"]


def _hero_path(work: Path) -> Path | None:
    for n in ("hero_v3.png", "hero_v2.png", "hero.png"):
        if (work / n).exists():
            return work / n
    return None


# ---------------------------------------------------------------- smoke: numpy shapes stand in for the diffusion stages

def _shape(W: int, H: int, seed: int, cx: float, cy: float, scale: float, angle: float, rng):
    """A 'creature': a body disc, a head disc, four limb ellipses, all one silhouette, with soft edges and noise."""
    import numpy as np
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), dtype=np.float32)
    parts = [(0, 0, 0.30, 0.22), (0.22, -0.20, 0.13, 0.13), (-0.28, 0.15, 0.10, 0.05), (0.28, 0.15, 0.10, 0.05), (-0.20, 0.30, 0.06, 0.12), (0.20, 0.30, 0.06, 0.12)]
    ca, sa = np.cos(angle), np.sin(angle)
    for dx, dy, rx, ry in parts:
        px = cx + scale * (dx * ca - dy * sa) * min(W, H)
        py = cy + scale * (dx * sa + dy * ca) * min(W, H)
        img += np.exp(-(((xx - px) / (rx * scale * min(W, H))) ** 2 + ((yy - py) / (ry * scale * min(W, H))) ** 2) * 2)
    img = np.clip(img, 0, 1)
    img = img + 0.03 * rng.standard_normal(img.shape)
    rgb = np.stack([img * 0.15, img * 0.85 + 0.05, img * 0.95 + 0.05], -1)  # cyan-ish, on a dark field
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _desc(arr) -> "list[float]":
    import numpy as np
    g = arr.astype(np.float32).mean(-1) / 255
    m = g > 0.3
    ys, xs = np.nonzero(m) if m.any() else (np.array([0]), np.array([0]))
    return [g.mean(), g.std(), m.mean(), xs.std() / g.shape[1], ys.std() / g.shape[0], (m[: g.shape[0] // 2].mean() + 1e-3) / (m.mean() + 1e-3)]


def _cos(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def smoke_stages(work: Path, stage: str, seed: int) -> dict:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    W, H = 192, 128
    t0 = time.time()
    res: dict = {"model": "smoke-shapes"}
    # hero (resumable, like the real script)
    hero = _hero_path(work)
    if hero is None:
        arr = _shape(W, H, seed, W * 0.5, H * 0.55, 1.0, 0.0, rng)
        hero = work / "hero_v3.png"
        Image.fromarray(arr).save(hero)
        (work / "crops").mkdir(exist_ok=True)
        Image.fromarray(arr[:, W // 4: 3 * W // 4]).save(work / "crops" / "body.png")
    res["hero"] = str(hero)
    status(phase="hero", msg=f"hero at {hero.name}")
    if stage == "hero":
        res["elapsed_s"] = time.time() - t0
        return res
    # hero set: framings = the same shapes at other scales/angles; DINO stands in as a silhouette-descriptor cosine
    d = work / "heroset"
    cap = d / "captions.json"
    if not cap.exists():
        d.mkdir(exist_ok=True)
        hd = _desc(np.asarray(Image.open(hero).convert("RGB")))
        kept, rows = [], []
        for i, framing in enumerate(FRAMINGS_SMOKE):
            arr = _shape(W, H, seed, W * (0.4 + 0.2 * rng.random()), H * (0.4 + 0.3 * rng.random()), 0.6 + 0.9 * rng.random(), rng.normal(0, 0.5), rng)
            dc = _cos(_desc(arr), hd)
            row = {"framing": framing, "crop": "body", "seed": 700 + 10 * i, "dino_hero": dc, "dino_crop": dc, "adh": 0.2 + 0.1 * rng.random(), "p_own": float(rng.random() > 0.35)}
            rows.append(row)
            metric(step=i, dino_crop=dc, p_own=row["p_own"])
            if dc >= 0.45:
                name = f"hs_{i:02d}_0.png"
                Image.fromarray(arr).save(d / name)
                kept.append({**row, "file": name})
        (d / "all_candidates.json").write_text(json.dumps(rows, indent=1))
        cap.write_text(json.dumps(kept, indent=1))
        print(f"heroset: kept {len(kept)}/{len(rows)} (dino_crop>=0.45)", flush=True)
    res["heroset"] = str(d)
    status(phase="heroset", msg=f"hero set at {d}")
    if stage == "heroset":
        res["elapsed_s"] = time.time() - t0
        return res
    # lora: a fake loss curve and train.json; no weights
    ld = work / "lora"
    if not (ld / "train.json").exists():
        ld.mkdir(exist_ok=True)
        loss = 0.35
        for step in range(20):
            loss = 0.9 * loss + 0.1 * (0.12 + 0.05 * rng.random())
            metric(step=step, loss=loss)
        (ld / "train.json").write_text(json.dumps({"rank": 16, "steps": 20, "lr": 5e-5, "bs": 2, "n_images": len(json.loads(cap.read_text())) + 1, "elapsed_s": time.time() - t0, "smoke": True}, indent=1))
    res["lora"] = str(ld)
    status(phase="lora", msg=f"lora at {ld}")
    res["elapsed_s"] = time.time() - t0
    return res


# ---------------------------------------------------------------- real: drive identity_v3b.py with the bible's text patched in

WRAPPER = r'''
import sys, importlib.util, json
from pathlib import Path
cfg = json.loads(sys.argv[1])
spec = importlib.util.spec_from_file_location("identity_v3b", cfg["script"])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if cfg.get("identity"): m.IDENTITY = cfg["identity"]
if cfg.get("style"): m.STYLE = cfg["style"]
if cfg.get("negative"): m.NEG = cfg["negative"]
work = Path(cfg["work"]).expanduser(); work.mkdir(parents=True, exist_ok=True)
if cfg.get("hero_src"): m.HERO_SRC = Path(cfg["hero_src"]).expanduser()
elif (work / "hero_v3.png").exists(): m.HERO_SRC = work / "hero_v3.png"
if not m.HERO_SRC.exists():
    # no source image for this character: generate the hero from the bible (compact prompt, no adapter), 768x512 so the crop boxes apply
    from PIL import Image
    pipe = m.load_pipe(with_controlnet=False)
    img = m.gen(pipe, m.prompt(m.HERO_FRAMING), int(cfg.get("seed", 7)), ref=Image.new("RGB", (768, 512)), ip_scale=0.0)
    img.save(work / "hero_v3.png"); m.HERO_SRC = work / "hero_v3.png"
    del pipe
    import torch; torch.cuda.empty_cache()
    print("hero generated from the bible:", m.HERO_SRC, flush=True)
sys.argv = [cfg["script"], str(work), "--stage", cfg["stage"]] + (["--smoke"] if cfg.get("smoke") else [])
m.main()
'''


def real_stages(work: Path, stage: str, identity: str, style: str, negative: str, hero_src: str | None, seed: int) -> dict:
    xp = lambda v: str(Path(os.path.expandvars(v)).expanduser())
    script = Path(xp(os.environ.get("CINEMA_IDENTITY_SCRIPT", "~/identity_v3b.py")))
    if not script.exists():
        raise SystemExit(f"the identity script is not here: {script} (set CINEMA_IDENTITY_SCRIPT, or run this recipe on the 5090)")
    py = xp(os.environ.get("CINEMA_WAN_PYTHON", sys.executable))
    cfg = {"script": str(script), "work": str(work), "stage": stage, "identity": identity, "style": style, "negative": negative, "hero_src": xp(hero_src) if hero_src else None, "seed": seed}
    status(phase=stage, msg=f"{py} identity_v3b --stage {stage} work={work}")
    t0 = time.time()
    proc = subprocess.Popen([py, "-c", WRAPPER, json.dumps(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if "\r" in line:
            line = line.split("\r")[-1]
        print(line, flush=True)
        if line.startswith("lora step "):  # lora step 100/800 loss 0.1234 12s mem 9.1GB
            try:
                parts = line.split()
                metric(step=int(parts[2].split("/")[0]), loss=float(parts[4]), elapsed_s=time.time() - t0)
            except Exception:
                pass
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"identity script exited {code}")
    hero = _hero_path(work)
    res = {"model": "identity_v3b", "hero": str(hero) if hero else None, "elapsed_s": time.time() - t0}
    if stage in ("heroset", "lora"):
        res["heroset"] = str(work / "heroset")
    if stage == "lora":
        res["lora"] = str(work / "lora")
    return res


# ---------------------------------------------------------------- shared: statistics and the contact sheet

def summarize(work: Path, stage: str, out: Path, res: dict) -> dict:
    cap = work / "heroset" / "captions.json"
    if cap.exists():
        rows = json.loads(cap.read_text())
        vals = [r.get("dino_crop", r.get("dino_hero", 0.0)) for r in rows]
        if vals:
            res["n_kept"] = len(rows)
            res["proto_mean"] = sum(vals) / len(vals)
            res["proto_min"] = min(vals)
            res["p_own"] = sum(float(r.get("p_own", 0)) for r in rows) / len(rows)
            for i, v in enumerate(vals):
                metric(step=i, proto=v)
    tj = work / "lora" / "train.json"
    if tj.exists():
        try:
            t = json.loads(tj.read_text())
            res["lora_steps"] = t.get("steps")
            res["lora_train_s"] = t.get("elapsed_s")
        except Exception:
            pass
    try:  # contact sheet: the hero, then up to 7 hero-set frames
        from PIL import Image
        picks = []
        hero = _hero_path(work)
        if hero:
            picks.append(hero)
        if cap.exists():
            picks += [work / "heroset" / r["file"] for r in json.loads(cap.read_text()) if (work / "heroset" / r["file"]).exists()][:7]
        if picks:
            cell = (192, 128)
            sheet = Image.new("RGB", (cell[0] * len(picks), cell[1]))
            for k, p in enumerate(picks):
                sheet.paste(Image.open(p).convert("RGB").resize(cell), (k * cell[0], 0))
            sheet.save(out / "contact.png")
            res["contact"] = str(out / "contact.png")
        if hero:
            shutil.copyfile(hero, out / "hero.png")
    except Exception as e:
        print("contact sheet skipped:", e, flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="hero")
    ap.add_argument("--stage", default="heroset", choices=STAGES)
    ap.add_argument("--work", default=None, help="the character's folder on this machine (hero, heroset/, lora/); default out/characters/<id>")
    ap.add_argument("--identity", default="", help="the IDENTITY text from the bible")
    ap.add_argument("--style", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--hero-src", default=None, help="an existing image to crop the hero from (else it is generated from the bible)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="this run's output: results.json, contact.png, hero.png")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    work = Path(a.work or f"out/characters/{a.character}").expanduser()
    out = Path(a.out or f"out/characters/{a.character}/{a.stage}").expanduser()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    status(phase="start", msg=f"character {a.character}: stage {a.stage}")
    if a.smoke:
        res = smoke_stages(work, a.stage, a.seed)
    else:
        res = real_stages(work, a.stage, a.identity, a.style, a.negative, a.hero_src, a.seed)
    res = summarize(work, a.stage, out, res)
    res.update(character=a.character, stage=a.stage, work=str(work), out=str(out))
    status(phase="done", msg=f"{a.stage} done" + (f": proto {res['proto_mean']:.3f} (min {res['proto_min']:.3f}), p_own {res['p_own']:.2f}, {res['n_kept']} kept" if "proto_mean" in res else ""))
    result(**res)


if __name__ == "__main__":
    main()
