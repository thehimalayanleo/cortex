#!/usr/bin/env python3
"""Studio brick: render one shot to a short clip and score it (Lab 20: agentic cinema).

Real mode (on the 5090): drives the Wan 2.2 TI2V-5B image-to-video brick (~/wan_i2v.py from Celwright) with a keyframe
and a prompt, then reports its temporal critics: identity per frame against the character prototype, and flicker.
Smoke mode (anywhere, no GPU, no network): synthesizes a tiny clip of a moving blob with numpy, computes the same two
critic statistics on it, and writes clip frames plus a contact sheet, so the whole pipeline (run, metrics, take
retrieval, telemetry) can be exercised in seconds.

    python cinema_render.py --shot s01 --prompt "..." --keyframe path.png --frames 49 --size 832x480 --out out/cinema/s01
    python cinema_render.py --smoke --shot s01 --prompt "a lighthouse in a storm" --out out/cinema/s01

Prints METRIC / STATUS / RESULT lines (the Cortex run protocol). RESULT carries the clip path and the scores.
"""
from __future__ import annotations

import argparse
import json
import os
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


def smoke_render(out: Path, prompt: str, frames: int, size: tuple[int, int], seed: int) -> dict:
    """A moving soft blob on a dark field: deterministic, seconds to make, and the critics have something to measure."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    W, H = size
    W, H = max(64, W // 8), max(64, H // 8)  # small canvas: this is a pipeline test, not a render
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cx, cy = W * 0.35, H * 0.5
    vx, vy = W * 0.004, rng.normal(0, H * 0.002)
    yy, xx = np.mgrid[0:H, 0:W]
    ids, prev = [], None
    proto = None
    t0 = time.time()
    for i in range(frames):
        cx += vx; cy += vy
        r = min(W, H) * 0.18
        img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img = img + 0.02 * rng.standard_normal(img.shape)  # sensor noise, so flicker is not zero
        img = np.clip(img, 0, 1)
        arr = (img * 255).astype(np.uint8)
        Image.fromarray(arr).save(out / "frames" / f"{i:03d}.png")
        # "identity": cosine between the frame's blob descriptor and the first frame's; a stand-in for the DINO critic
        desc = np.array([img.mean(), img.std(), (img > 0.5).mean(), cx / W, cy / H])
        if proto is None:
            proto = desc
        ids.append(float(np.dot(desc, proto) / (np.linalg.norm(desc) * np.linalg.norm(proto) + 1e-9)))
        if prev is not None:
            metric(step=i, identity=ids[-1], flicker=float(np.abs(arr.astype(np.float32) - prev).mean()))
        prev = arr.astype(np.float32)
    # contact sheet: 8 frames in a row
    picks = [int(k * (frames - 1) / 7) for k in range(8)]
    sheet = Image.new("L", (W * 8, H))
    for k, p in enumerate(picks):
        sheet.paste(Image.open(out / "frames" / f"{p:03d}.png"), (k * W, 0))
    sheet.save(out / "contact.png")
    flick = [float(np.abs(np.asarray(Image.open(out / "frames" / f"{i+1:03d}.png"), dtype=np.float32) - np.asarray(Image.open(out / "frames" / f"{i:03d}.png"), dtype=np.float32)).mean()) for i in range(frames - 1)]
    return {"model": "smoke-blob", "size": [W, H], "frames": frames, "gen_s": time.time() - t0, "elapsed_s": time.time() - t0,
            "identity_mean": sum(ids) / len(ids), "identity_min": min(ids), "identity_first": ids[0], "identity_last": ids[-1],
            "flicker_mean": sum(flick) / len(flick), "flicker_max": max(flick), "clip": None, "contact": str(out / "contact.png")}


def real_render(out: Path, keyframe: str, prompt: str, frames: int, size: str, steps: int, proto: str | None) -> dict:
    """Call Celwright's Wan brick and relay its progress; parse its DONE line for the critics."""
    xp = lambda v: str(Path(os.path.expandvars(v)).expanduser())  # env values may arrive quoted, with $HOME unexpanded
    brick = Path(xp(os.environ.get("CINEMA_WAN_BRICK", "~/wan_i2v.py")))
    if not brick.exists():
        raise SystemExit(f"the Wan brick is not here: {brick} (set CINEMA_WAN_BRICK, or run this recipe on the 5090)")
    py = xp(os.environ.get("CINEMA_WAN_PYTHON", sys.executable))
    cmd = [py, str(brick), xp(keyframe), prompt, str(out), "--frames", str(frames), "--size", size, "--steps", str(steps)]
    if proto:
        cmd += ["--proto", xp(proto)]
    status(phase="render", msg=" ".join(cmd[:3]) + f" … frames={frames} size={size} steps={steps}")
    t0 = time.time()
    done = None
    # The box has 29 GB of host RAM and the pipeline stages ~24 GB while loading; right after a previous take the
    # kernel sometimes SIGKILLs the loader (-9). One retry after the caches settle recovers it.
    for attempt in range(1, 3):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if "\r" in line:  # tqdm carriage returns: keep the last state only
                line = line.split("\r")[-1]
            print(line, flush=True)
            if line.startswith("DONE "):
                try:
                    done = json.loads(line[5:])
                except Exception:
                    pass
            elif "decoded" in line:
                metric(step=1, phase_s=time.time() - t0)
        code = proc.wait()
        if code == 0:
            break
        if code == -9 and attempt == 1:
            status(phase="retry", msg="the brick was killed while loading (host memory); waiting 30 s and trying once more")
            time.sleep(30)
            continue
        raise SystemExit(f"wan brick exited {code}")
    res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else (done or {})
    for i, v in enumerate(res.get("per_frame_identity", [])):
        metric(step=i, identity=v)
    res["clip"] = str(out / "clip.mp4")
    # contact sheet from the saved frames
    try:
        from PIL import Image
        fr = sorted((out / "frames").glob("*.png"))
        if fr:
            picks = [fr[int(k * (len(fr) - 1) / 7)] for k in range(8)]
            ims = [Image.open(p).convert("RGB") for p in picks]
            w, h = ims[0].size
            sheet = Image.new("RGB", (w * 8, h))
            for k, im in enumerate(ims):
                sheet.paste(im, (k * w, 0))
            sheet.save(out / "contact.png")
            res["contact"] = str(out / "contact.png")
    except Exception as e:  # the clip is the product; a missing sheet is not a failure
        print("contact sheet skipped:", e, flush=True)
    res.pop("per_frame_identity", None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", default="s01")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--keyframe", default=None, help="image to animate (real mode)")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--proto", default=None, help="hero set folder for the identity critic (real mode)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or f"out/cinema/{a.shot}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    status(phase="start", msg=f"shot {a.shot}: {a.prompt[:80]}")
    if a.smoke:
        w, h = (int(v) for v in a.size.split("x"))
        res = smoke_render(out, a.prompt, min(a.frames, 24), (w, h), a.seed)
    else:
        if not a.keyframe:
            raise SystemExit("--keyframe is required in real mode")
        res = real_render(out, a.keyframe, a.prompt, a.frames, a.size, a.steps, a.proto)
    # one verdict the director can act on: identity held and the clip did not flicker
    res["verdict"] = "keep" if res.get("identity_min", 0) >= 0.85 * res.get("identity_first", 1) and res.get("flicker_max", 0) <= 3 * max(res.get("flicker_mean", 0), 1e-6) else "reshoot"
    res["shot"] = a.shot
    res["out"] = str(out)
    status(phase="done", msg=f"{res['verdict']}: identity {res['identity_mean']:.3f} (min {res['identity_min']:.3f}), flicker {res['flicker_mean']:.2f}")
    result(**res)


if __name__ == "__main__":
    main()
