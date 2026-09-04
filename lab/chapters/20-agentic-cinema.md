---
title: "Lab 20: Agentic cinema: a director agent over a render farm"
kind: permanent
topics: [lab]
chapter: 20
station: none
recipe: recipes/cinema_render.py
reading_time: 65 min
---

## What you will be able to do

1. Define the shot as the unit of work, write its schema, and explain why every render is a take attached to a shot rather than a file in a folder.
2. Write the director's loop (plan, render, critique, decide) as a reinforcement learning loop with no gradients, derive what rejection sampling can and cannot buy at a fixed render budget, and name the two places where gradients could enter later (the director's tokens via GRPO with a rendered reward, Lab 15; the video brick's denoising steps via Flow-GRPO, Lab 19).
3. Implement the three image critics used in Celwright (identity, camera, framing) and the two temporal critics on video (identity decay, flicker), each with its formula and its threshold, and say what each one cannot see.
4. Explain why a director must hold a world state, and why an identity lock that raises the identity score can lower the framing score at the same time.
5. Drive the Studio in Cortex from the chat: a shot list in `studio/shots.json`, a take as a `cinema_render` run on the 5090, a verdict from the recipe, and METRIC lines you can push to Grafana and query as if a render regression were an incident.

## The idea in one paragraph

A film is a list of shots, and a shot is a prompt, a keyframe, a frame count, a size, a status, and the takes that have been rendered for it. A director agent walks that list. For each shot it plans a request, hands the request to a generator it does not own (a brick), gets frames back, runs deterministic critics over the frames, and decides: keep the take, or change something and shoot again. Nothing is trained in that loop. The policy is a frozen language model reading critic feedback, the environment is a GPU running a diffusion model, and the reward is a handful of numbers a critic computed. It is the same shape as the loops in Labs 05, 15 and 19 with the gradient step removed, which is why it is worth building first: the critics and the loop are the environment you would need for the gradient version anyway, and a loop with no learning tells you exactly how much of your problem is selection (reshoot until a seed works) and how much needs a better policy. This chapter follows Ajinkya's Celwright project, where the loop, the critics, the video brick and the numbers below all come from `celwright/DIRECTOR_MODEL.md` (runs of 2026-09-01 and 2026-09-02 on one RTX 5090), and the Studio in Cortex, which wraps one brick of that pipeline behind five agent tools. Where this chapter says "the log", it means that document.

## The math

### The shot as the unit of work

The Studio stores a board as one JSON document, `studio/shots.json`, with a logline and a list of shots. A shot is

```json
{"id": "s02-storm", "title": "Storm", "prompt": "Okuun rises from the swell, low angle, lightning behind, slow push-in",
 "keyframe": "hero_v3.png", "frames": 49, "size": "832x480", "notes": "", "status": "planned", "takes": []}
```

with `status` in the set {planned, rendering, rendered, approved, reshoot}, and `takes` a list of run ids. A take is not a file; it is a run of `recipes/cinema_render.py` with its streamed log, its METRIC lines, and its RESULT record (identity and flicker statistics, a verdict, the paths of the clip and the contact sheet). The shot owns its takes so that the question "which take is the current one" and the question "how many renders has this shot cost" both have an answer without a filesystem walk. Three facts follow. A shot's status is a function of its latest finished take (keep gives `rendered`, otherwise `reshoot`), so the board can be rebuilt from the runs at any time. A reshoot is a new take on the same shot, never an edit of an old one, so the history of what was tried survives. And the id is stable across takes, so a metric series keyed by shot id is a time series of the shot's quality, which is what observability needs later.

### The loop, written as RL without gradients

Write the director as a policy $\pi$ over requests. For shot $s$ with world state $w$ (who is where, what they carry, what time it is), the director emits a request

$$
a = \pi(s, w, h) = \{\text{prompt}, \text{reference}, \text{ip\_scale}, \text{cn\_scale}, \text{seed}, \text{notes}\},
$$

where $h$ is the history of earlier takes for this shot with their critic feedback. This is the tool-call JSON the real loop uses (`remote/director_loop.py`): `reference` names a crop of the hero (body, head, torso, a claw, legs) the adapter should condition on, `ip_scale` in $[0.3, 0.7]$ is the identity adapter's strength, `cn_scale` in $[0.4, 1.0]$ is how strictly the frame follows the set's depth previz, and `notes` is the director's stated reason. A brick $\rho$ maps the request to frames, $F = \rho(a)$, and a set of critics maps frames to a vector of scores $c(F) = (\text{identity}, \text{framing}, \text{camera\_err}, \dots)$. The decision rule is a conjunction of thresholds,

$$
\text{keep}(F) = [\text{identity} \ge \tau_{\text{id}}] \wedge [\text{framing} \ge \tau_{\text{fr}}] \wedge [\text{camera\_err} \le \tau_{\text{cam}}],
$$

and the loop is: for $t = 1, \dots, n$, emit $a_t$, render, score, stop if keep. The objective the director is judged on is the pass rate at a fixed budget $n$, and secondarily the mean of a composite score over shots. The log's composite, used to pick training traces, is

$$
\text{composite} = \text{identity} + \text{framing} - 0.5 \min(1, \text{camera\_err}),
$$

which reads: two things you want, one thing you want less of, capped so a wildly wrong camera does not dominate.

This is the objective of Lab 15, $\mathbb{E}_{a \sim \pi}[R(\rho(a))]$, with $R$ the composite or the pass indicator. What is missing is the update. With $\pi$ frozen, the only way the expected reward moves is through $h$: the director reads feedback and changes its request. So the first question about any director loop is whether the policy's response to feedback does better than ignoring the feedback and rolling a new seed.

### What rejection sampling buys, and what it cannot

The cheapest director changes only the seed. If a take passes with probability $p$ independently of earlier takes, the chance that a shot passes within $n$ attempts is $1 - (1 - p)^n$ and the expected number of renders is $\frac{1 - (1-p)^n}{p}$. At $p = 0.1$ and $n = 4$ that is a pass rate of 0.344 at 3.44 renders per shot; to reach a 0.9 pass rate you need $n = \ln 0.1 / \ln 0.9 \approx 22$ attempts, which at the render times below is twenty minutes of GPU per shot.

Now read the log's numbers against this. The `fixed` arm (one render per shot, no loop) passed 0.10 of ten shots; the `reseed` arm (same plan, up to four seeds) passed 0.10 at 3.7 attempts per shot. If $p$ were 0.1 for every shot, reseeding should have passed 0.34. The observed 0.10 at 3.7 attempts is instead the signature of a different distribution: one shot (the storm) passes at every seed and nine never pass, so the expected attempts are $(1 \times 1 + 9 \times 4) / 10 = 3.7$, exactly what was measured. Pass probability is a property of the shot, not of the seed, and for the nine shots at $p \approx 0$ no number of seeds helps. That is the argument for a director that changes the request rather than the seed, and it is also the argument that is not yet won: the log's zero-shot Qwen3-VL-4B director, which sees the rejected frame and the feedback, also passed 0.10 at 3.7 attempts with identity 0.683 against reseed's 0.746. Its repairs were one-dimensional (raise `ip_scale` from 0.55 to 0.70, keep the crop, new seed). After Stage A, a LoRA on the director trained on 34 traces of successful plans and repairs, the pass rate reached 0.20 (the breach shot passed after a repair) and framing rose from 0.15 to 0.21, at identity 0.675, still below reseed. So today's honest reading is: the loop works end to end, the critics are real, and the director is not yet better than luck.

### Where gradients could enter

Two places, for two different policies. The director is a token policy, and Stage C in the log's plan is Lab 15 with the render as the environment: sample $G$ requests per shot, render each, score each with the critics, form the group-normalized advantage of Lab 05, and update the director's LoRA. Everything from Lab 15 carries over, including the cost structure: a render is tens of seconds, generation of a 100-token JSON request is well under a second, so the step is the render farm, and a gate outlier (a request the brick cannot parse) shrinks the group's useful spread exactly as derived there. The brick is a denoising policy, and Flow-GRPO (Lab 19) could train it with the temporal critics as the reward, which would move the fix for flicker from "reshoot" into the model. Neither is built. What is built is what both would need: a request format, deterministic critics with known thresholds and calibration, and a loop that emits every take's scores as METRIC lines.

### The three image critics

Identity. Let $e(\cdot)$ be the DINOv2-small class embedding of an image and $H$ the hero set, a small collection of renders of the character at varied framings. The prototype is the mean embedding $\bar e = \frac{1}{|H|} \sum_{h \in H} e(h)$, and a frame's identity is the cosine

$$
\text{identity}(f) = \frac{e(f) \cdot \bar e}{\|e(f)\| \, \|\bar e\|}.
$$

The threshold `IDENTITY_MIN` is 0.72 in the real loop's default and 0.80 in the stub critic. Why a prototype and not the hero image itself: the log's v3 hero set was built from a hero that had drifted off-model and all 40 candidates came back as the same full-body view, and a generic sea creature scored 0.59 against that set's prototype, so the prototype encoded "kaiju composition" rather than the character. The rebuilt set (crops of the original hero as framing atoms, v3b) has a null floor of 0.42 and the hero itself at 0.82, which is the check to run before trusting any prototype: score a generic stand-in and the hero, and the working range is what lies between. Against v3b, the IP-Adapter lock scored 0.90 on the first seed and text alone 0.68.

Camera. The set is a Gaussian splat (a garden scan standing in for a pier, 5.83M gaussians), and for a shot with requested pose $(\text{yaw}, \text{pitch}, \text{dist})$ the sets brick renders previz at the request and at neighbours: an outer ring at yaw and pitch $\pm 0.3$ and dist $\times 1.5$ and $\div 1.5$, and an inner ring at half those steps, 13 poses in all. The similarity between a generated frame and each previz render is the mean cosine of DINOv2 patch tokens on a 24 by 16 grid, "the same thing in the same place". Per axis, the best-matching pose wins; if it is interior, the offset is the vertex of the parabola through it and its two neighbours, and if it is the outermost sample the offset is the full step, so a one-step drift is never under-reported. The error is

$$
\text{camera\_err} = \sqrt{\Delta\text{yaw}^2 + \Delta\text{pitch}^2 + \ln^2 \big(\text{dist}_{\text{est}} / \text{dist}_{\text{req}}\big)},
$$

with `CAMERA_MAX` 0.25, the radius inside which the frame is closer to the request than to any outer neighbour. The log calibrated it with probe frames rendered at known offsets: exact poses pass 90 percent, a 0.11 drift passes 70 percent with 13 poses (40 percent with the earlier 7), a 0.15 drift 90 percent, and a full-step drift 0 percent, which is what "should fail" looks like.

Framing. Neither cosine sees "a close-up of a claw" or "low angle"; the log's CLIP-B/32 adherence probe scored about 0 on the framing-only shots (plating, plate-shift) for both arms it compared. The framing critic is a VLM judge, Qwen3-VL-4B, asked two things per frame: which of the ten shot descriptions the frame shows (`p_own`, chance 0.1), and a 0 to 10 rating of the match to its own shot, reported as `match` in $[0, 1]$. `FRAMING_MIN` is 0.4 in the documented run and 0.6 as the code default. This is Lab 09's judge with everything Lab 09 warns about, and the log's example of its value is the claw shot: text 0.35, hero lock 0.12, shot-matched crop 0.30, a difference CLIP could not see.

### The two temporal critics

A video take is $T$ frames $f_1, \dots, f_T$. Identity is computed per frame against the prototype, and the loop reports the mean, the minimum, and the decay from first to last,

$$
\text{decay} = \text{identity}(f_1) - \text{identity}(f_T).
$$

Flicker is the mean absolute difference between consecutive grey frames on the 0 to 255 scale,

$$
\text{flicker}_t = \frac{1}{HW} \sum_{x, y} \big| f_{t+1}(x, y) - f_t(x, y) \big|, \qquad t = 1, \dots, T - 1,
$$

with its mean and maximum over the clip. The recipe's verdict is

$$
\text{keep} \iff \min_t \text{identity}(f_t) \ge 0.85 \, \text{identity}(f_1) \;\wedge\; \max_t \text{flicker}_t \le 3 \, \overline{\text{flicker}}.
$$

The first clause says identity may not fall more than 15 percent below where the keyframe put it; the second says no single cut in the clip may be more than three times the typical motion, which is a spike detector for a dropped or re-lit frame. Apply it to the log's storm clip (Wan 2.2 TI2V-5B, 49 frames at 832 by 480): identity mean 0.900 and minimum 0.876, first to last 0.909 to 0.893, flicker mean 6.1 and maximum 12.1. The identity clause holds since $0.876 \ge 0.85 \times 0.909 = 0.773$, and the flicker clause holds since $12.1 \le 18.3$, so the verdict is keep. Two cautions the arithmetic exposes. The identity clause is relative to the first frame, so a keyframe that is already off-model passes on decay alone; the absolute gate belongs on the keyframe, before the clip is shot, which is what the image critics are for. And the flicker rule is relative to the mean, so a clip that strobes on every frame raises its own mean and passes; the snippet below shows this happening.

### Why the director holds a world state

The prompt for a shot is not free text; it is a function of state. If Okuun's claw was wounded in shot 4, shot 7 must show the wound; if the lantern was handed over in shot 5, shot 6 must not show it in the first character's hand; if the scene is at dusk, no shot is at noon. The log's world state is a set of atoms (who, where, carrying what, marked how, when) with transitions per shot and a deterministic continuity critic that checks a shot's description against the atoms. In the stub loop (50 seeds, a memoryless drifting generator) the director shipped 5.7 continuity, identity and camera violations per ten-shot scene against the baseline's 11.3, at 28 generator calls against 10. That ratio, half the violations for three times the renders, is the price of a director with no repair ability beyond rejection, and it is the number Stage C exists to improve. The state also feeds the sets brick: with character proxies (a solid volume per present character at its blocking position) rendered into the previz depth, the two-shot's camera error fell from 0.23 to 0.02, because "who is where" was in the depth map the generator followed.

### Identity locks, and the confound

Four ways to make the character look like itself, with what the log found. An IP-Adapter conditions the generator on a global CLIP image embedding of a reference; against the v3b prototype it scores 0.90 and it is consistent across shots, but it earns that score by pasting the hero's composition into every frame: in the cleaned-set evaluation its framing `match` is 0.18 against text's 0.38, and in the earlier v2 run no locked frame at any `ip_scale` reached the median adherence of the text-only frames. Lowering `ip_scale` from 0.75 to 0.4 did not release the composition; identity stayed flat at 0.76 first-seed. A hero set of framing crops used as references steers framing (claw `match` 0.12 to 0.30) at identity 0.86, but a crop is a weaker anchor and some crops drift off-model. A character LoRA at rank 16 on the UNet's attention was burned in the first training (learning rate 1e-4, 800 steps, fp16 autocast without a gradient scaler): identity fell monotonically with adapter scale, 0.71 to 0.33. The sweep found rank 16 at 5e-5 for 400 steps raises identity with scale to 0.806 at scale 0.7, and after cleaning the hero set the LoRA reached 0.76 identity with framing 0.24, which the log calls the first identity brick that does not pay the full composition tax; it still trails the lock's 0.85 on identity. Regional masks (conditioning the adapter only inside the proxy's silhouette) fixed the camera and did not move in-region identity (0.47 with and without). The menu today is lock for identity, LoRA for identity with freedom, and neither alone satisfies both critics.

### Observability

Every take prints METRIC lines (`{"step": t, "identity": ...}` per frame, with `flicker` alongside in smoke mode) and one RESULT line, the Cortex run protocol from `recipes/common.py`. The server's `telemetry.py` turns METRIC lines into metrics pushed to Grafana Cloud's Mimir influx-line endpoint and log lines into Loki entries, batched in a background thread, and does nothing when its environment variables are unset. Keyed by shot id, `identity_min` across takes is a time series per shot, and "which shots regressed this week" is the query: for each shot, the latest take's `identity_min` minus the best earlier take's, sorted. That is incident investigation applied to a render farm, and the same Grafana MCP an agent would use to ask "which service regressed" would answer it, given a dashboard and a query. That last step is not built.

## Build it small

The snippet is the whole loop with every piece replaced by the smallest thing of the same shape. The brick renders 24 frames of a moving blob; `drift` grows a second lobe over the clip (the character changes shape, which is identity decay) and `flicker_p` is the per-frame chance of a one-frame lighting drop; the seed decides how hard the drift bites this take. The embedding is a crop centred on the blob's centroid, pooled to 8 by 8 and centred, so it ignores where the character is and reacts to what it looks like, which is the property you want from DINOv2 here. The two temporal critics and the verdict rule are the recipe's. The director is the cheapest one, a reseed loop with a budget of four, and the board is four shots of increasing difficulty.

```python
import numpy as np, json
T, S, ATTEMPTS = 24, 48, 4                      # frames per take, frame side, reshoots allowed per shot
yy, xx = np.mgrid[0:S, 0:S]

def fake_render(seed, drift, flicker_p):
    """A 'brick': frames of a moving blob. drift morphs the blob's shape over time (identity decay);
    flicker_p is the per-frame chance of a lighting pop. The seed decides how much of each this take suffers."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.2, 1.8)                    # this take's luck: how strongly the drift bites
    cx, cy, vx, vy = S * 0.3, S * 0.5, S * 0.012, rng.normal(0, S * 0.004)
    frames = []
    for t in range(T):
        cx += vx; cy += vy
        r = S * 0.12
        img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img += drift * k * t / T * np.exp(-((xx - cx - 2.5 * r) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))  # a lobe grows: the character changes shape
        img *= 0.35 if rng.random() < flicker_p else 1.0   # lighting drop for one frame
        frames.append(np.clip(img + 0.01 * rng.standard_normal(img.shape), 0, 1))
    return np.stack(frames)

def embed(img):
    """Stand-in for a DINOv2 embedding: a 32x32 crop centred on the blob's centroid, pooled to 8x8 and centred.
    Translation-invariant (where the character is, via the crop) and shape-sensitive (what it looks like)."""
    w = img / img.sum(); cy, cx = int(round((w * yy).sum())), int(round((w * xx).sum()))
    pad = np.pad(img, 16); crop = pad[cy:cy + 32, cx:cx + 32]          # padded so the crop never leaves the frame
    d = crop.reshape(8, 4, 8, 4).mean(axis=(1, 3)).ravel()            # 4x4 average pool -> 64 numbers
    return d - d.mean()

def cos(u, v): return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

def critics(frames, proto):
    ids = [cos(embed(f), proto) for f in frames]                       # identity per frame vs the prototype
    fl = [float(np.abs(frames[i + 1] - frames[i]).mean() * 255) for i in range(T - 1)]   # mean abs grey diff
    return {"identity_first": ids[0], "identity_last": ids[-1], "identity_min": min(ids),
            "flicker_mean": float(np.mean(fl)), "flicker_max": max(fl)}

def verdict(c):                                                        # the rule in recipes/cinema_render.py
    hold = c["identity_min"] >= 0.85 * c["identity_first"]
    calm = c["flicker_max"] <= 3 * max(c["flicker_mean"], 1e-6)
    return "keep" if hold and calm else ("reshoot:identity" if not hold else "reshoot:flicker")

shots = [{"id": "s01-establish", "drift": 0.2, "flicker_p": 0.00},   # easy: nothing goes wrong
         {"id": "s02-storm", "drift": 0.9, "flicker_p": 0.00},       # identity is the risk
         {"id": "s03-lantern", "drift": 0.2, "flicker_p": 0.05},     # lighting pops are the risk
         {"id": "s04-breach", "drift": 1.2, "flicker_p": 0.06}]      # both; may run out of attempts
keyframe = fake_render(0, 0.0, 0.0)[0]
proto = embed(keyframe)                                                # the hero-set prototype (one frame here)
for n, shot in enumerate(shots):
    shot["takes"], shot["status"] = [], "planned"
    for attempt in range(ATTEMPTS):                                    # plan (fixed prompt), render, critique, decide
        seed = 100 * (attempt + 1) + 7 * n                             # a reshoot is the same plan with a new seed
        c = critics(fake_render(seed, shot["drift"], shot["flicker_p"]), proto)
        c["seed"], c["verdict"] = seed, verdict(c)
        print("METRIC", json.dumps({"shot": shot["id"], "take": attempt, **{k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()}}))
        shot["takes"].append(c)
        if c["verdict"] == "keep":
            break
    shot["status"] = "rendered" if shot["takes"][-1]["verdict"] == "keep" else "reshoot"
    print(f"{shot['id']:14s} {shot['status']:9s} takes {len(shot['takes'])}  last: identity {c['identity_first']:.2f}->{c['identity_last']:.2f} "
          f"(min {c['identity_min']:.2f}), flicker {c['flicker_mean']:.1f} max {c['flicker_max']:.1f}")
print("board:", {s["id"]: s["status"] for s in shots}, "renders:", sum(len(s["takes"]) for s in shots))
```

Output from one run (numpy 2.4, under a second on a laptop; the METRIC lines are abbreviated to the summary lines here):

```
s01-establish  rendered  takes 1  last: identity 1.00->0.98 (min 0.98), flicker 3.2 max 3.3
s02-storm      rendered  takes 2  last: identity 1.00->0.94 (min 0.94), flicker 3.5 max 3.7
s03-lantern    rendered  takes 1  last: identity 1.00->0.99 (min 0.98), flicker 3.2 max 3.3
s04-breach     reshoot   takes 4  last: identity 1.00->0.93 (min 0.93), flicker 5.7 max 17.3
board: {'s01-establish': 'rendered', 's02-storm': 'rendered', 's03-lantern': 'rendered', 's04-breach': 'reshoot'} renders: 8
```

What I observed, take by take from the METRIC lines. The establishing shot kept on its first take. The storm's first take decayed to identity 0.817, below the 0.85 line, and the second seed drew a gentler drift factor and held at 0.937: rejection sampling did its one job. The lantern shot, whose risk was a lighting pop at 5 percent per frame, drew no pop in 24 frames and kept; the risk was real and this seed did not realize it, which is the toy version of the storm shot in the log passing at every seed. The breach shot failed identity on three seeds (0.771, 0.822, 0.767), then on the fourth seed identity held at 0.931 and the flicker clause failed by a hair: maximum 17.27 against three times the mean, 17.19. It ran out of attempts and the board carries it as `reshoot`. Eight renders for four shots, a 2x cost for a 3 of 4 board, in the same regime as the log's 28 renders for 10 shots.

Two things the output teaches that the code does not say. First, the fourth breach take failed a different critic from the first three; a director that reads the verdict string knows that the seed lever is exhausted for identity and that something else (fewer pops, a different brick, a shorter clip) is needed for flicker, and a director that reads only keep or reshoot does not. Second, look at the breach shot's flicker means: 4.9, 4.8, 4.9, 5.7 against 3.2 on a calm shot. The pops themselves inflate the mean, so the 3x rule's bar rises with the disease it is meant to catch; at a higher `flicker_p` a clip that strobes throughout passes (exercise 2). A relative rule needs a baseline the defect cannot move, such as the median, or a calm reference clip.

## Build it real

The brick is `recipes/cinema_render.py`, and it has two modes with the same outputs. In smoke mode it renders a moving blob in numpy at one eighth of the requested size, computes identity (a descriptor cosine against the first frame) and flicker per frame, writes `frames/`, a `contact.png` of eight frames, and a RESULT with the same fields as the real mode, in seconds and with no GPU, so the Studio's plumbing (start a run, stream METRIC lines, fetch artifacts, set the verdict) can be exercised anywhere. In real mode, on the 5090, it calls Celwright's Wan brick (`~/wan_i2v.py`, overridable with `CINEMA_WAN_BRICK`; its interpreter with `CINEMA_WAN_PYTHON`) with a keyframe and a prompt, relays the brick's progress, parses its DONE line and `results.json`, emits one METRIC per frame with the identity, builds the contact sheet, and applies the verdict rule. Arguments: `--shot` (the id, used for the output path), `--prompt`, `--keyframe` (required in real mode), `--frames` (default 49), `--size` (default `832x480`), `--steps` (default 30), `--proto` (a hero-set folder for the identity critic; the Studio passes `CINEMA_PROTO` if set), `--seed`, `--out`, and `--smoke`.

The Wan brick as used, from the log. Wan 2.2 TI2V-5B in bf16 is loaded straight onto the GPU, because diffusers' CPU offload was killed by the box's 29 GB of system RAM. At 49 frames the VAE cannot decode beside the transformer and the text encoder, so generation runs in two phases: denoise to latents, move the transformer and text encoder to CPU and empty the cache, then decode with VAE tiling. The passing storm keyframe (identity 0.92, framing 0.7) became 49 frames at 832 by 480 in 34 seconds of generation, 54 seconds with model load and critics, with the identity and flicker numbers used in the math section. The clip kept the identity the keyframe had, across the whole clip; the temporal wall the Celwright README worried about did not appear at this length on the previz-then-render path.

Time, as a formula. One take costs

$$
T_{\text{take}} = T_{\text{load}} + T_{\text{gen}}(\text{frames}, \text{size}, \text{steps}) + T_{\text{critics}} + T_{\text{fetch}},
$$

and the log's one measurement puts the first three at 54 seconds for the defaults, of which generation is 34. Generation scales with the number of denoising steps times the DiT forward over the latent volume (Lab 19's cost section: no cache across steps, the whole latent every step), so halving `--steps` roughly halves the 34 seconds and changing `--frames` moves the latent volume linearly. Fetch is an `scp` of the clip from the box. A four-shot board with the toy's 2x reshoot ratio is eight takes, about seven minutes of GPU, plus planning by the chat model, which is seconds.

How the Studio wires it, from `server/studio.py`. Shots live in the vault at `studio/shots.json`; takes are fetched into `studio/takes/<run_id>/` (`results.json`, `contact.png`, `clip.mp4`); keyframes and references the person drops in live in `studio/assets/`. `render(sid)` builds the recipe's argument string from the shot, picks the executor (`ssh`, the 5090, when it is available; otherwise local), falls back to smoke mode when there is no keyframe or no box, copies a vault asset to `~/cortex-lab/assets/` on the box with `scp` when the keyframe is an asset name, starts the run through the same `runs.start` every other recipe uses, appends the run id to the shot's `takes`, and sets its status to `rendering`. `refresh(sid)` walks the shot's takes, fetches the artifacts of any that finished, and sets the status: a queued or running take keeps `rendering`; otherwise the latest finished verdict decides, keep gives `rendered` and anything else `reshoot`; and if the last take failed outright (the brick exited nonzero, an OOM), the shot goes to `reshoot` with a director note pointing at the run log, so an infrastructure failure is never mistaken for a critic's rejection. The person approves from the UI, which is the one status the machine never sets. The chat agent sees five tools: `studio_board` (the logline, shots, takes with their scores and verdicts, and the asset list), `plan_shots` (a logline to $n$ planned shots; the chat model is asked for a JSON array of title, a 25 to 60 word image-to-video prompt, and a note, at temperature 0.4, with an instruction to keep continuity of character and setting), `set_shot` (edit prompt, keyframe, frames, size, status, or a director note; or add a shot), `render_shot` (start a take and open the run), and `refresh_shot`. The last four are write tools and go through the same confirmation path as starting any GPU run.

What runs today and what does not. Today: the board, planning by the chat model (glm-5.3 through OpenCode Go by default), takes on the 5090 through the Wan brick, the two temporal critics and the verdict, artifact fetch, and METRIC lines in the run log. Not today: a director that reads the verdict and changes the request (the Celwright loop does this with Qwen3-VL-4B on keyframes, not on video, and the Studio's chat agent can do it by hand through `set_shot` and `render_shot` but has no autonomous loop); a Gemini or other frontier director (planning uses whatever `CORTEX_MODEL` names, and nothing has been evaluated against the Celwright arms); the image critics (camera and framing) inside the Studio, which only sees what the Wan brick reports; and the Grafana leg, where `telemetry.py` exists and is gated on `GRAFANA_*` variables, no dashboard has been built for shots, and no agent tool asks Grafana anything.

## How it goes wrong

The lock pastes the composition. Symptom: identity rises to 0.9 and stays there at every `ip_scale`, while framing falls below the text-only arm and the contact sheet shows the same full-body pose in a close-up of a claw. Cause: a global image-embedding adapter conditions on the reference's layout as much as on the character; the log's v2 run found no locked frame at any scale reaching the text arm's median adherence. Fix: shot-matched crops as references, a LoRA trained on a cleaned, framing-varied hero set, or a two-pass render (layout from depth, then the character inpainted into its region), which the log names as the next lever and has not built.

The prototype encodes the wrong thing. Symptom: a generic stand-in scores nearly as high against the prototype as the hero does. Cause: the hero set was generated with one composition, so its mean embedding is that composition; the log's v3 null floor was 0.59. Fix: the two-anchor check before any run (null stand-in and hero itself; v3b gave 0.42 and 0.82), and a hero set built from crops with different framings.

The director repairs one knob. Symptom: every repair raises `ip_scale`, keeps the crop, and changes the seed; identity at equal budget lands below reseed. Cause: a zero-shot backbone with a capability card and no examples of good repairs. The log found two further defects and guarded them in code: the director dropped the shot's framing words from the prompt (now enforced verbatim), and returned an identical request while claiming a change (now forced to a new seed). Fix: Stage A traces from loops that improved the composite by more than 0.05 (34 traces moved pass from 0.10 to 0.20), then Stage C.

The spike rule is blinded by a constant strobe. Symptom: a clip that flickers on every frame passes the flicker clause. Cause: the rule compares the maximum with the clip's own mean, and the defect raises the mean; the toy's breach takes show the mean climbing from 3.2 to 5.7. Fix: a baseline the defect cannot move (the median difference, or a calm clip's mean at the same size and motion), plus an absolute ceiling on the mean.

Sub-step camera drift is rejected. Symptom: frames that are visibly at the requested pose fail the camera critic about half the time. Cause: with a seven-pose neighbourhood a 0.1 radian drift is extrapolated from the outer ring and a neighbour sometimes wins on one axis. Fix: the inner ring at half a step (13 poses), which took the 0.11 drift's pass rate from 40 to 70 percent without loosening the one-step catch.

The LoRA is burned. Symptom: identity falls monotonically as the adapter's scale rises, frames go from coherent to haloed to noise. Cause: training dynamics, not loading: learning rate 1e-4 for 800 steps in fp16 autocast with no gradient scaler. Fix: the sweep's rank 16 at 5e-5 for 400 steps, and the check that identity rises with scale before any claim about the adapter.

The keyframe was already off-model. Symptom: a clip keeps on the verdict rule with identity 0.8 to 0.9 relative to its own first frame, and the character is wrong. Cause: the video verdict is relative to `identity_first`; nothing in it compares the first frame with the prototype. Fix: gate the keyframe with the image critics first (the Celwright order), and add an absolute clause on `identity_first` when a prototype is passed.

The brick will not fit. Symptom: the process is killed with no traceback during load, or the decode fails at the end after a full generation. Cause: CPU offload against 29 GB of system RAM, and the VAE beside the transformer and text encoder at 49 frames. Fix: the two-phase decode in the brick; do not lower frames or size until you have moved the two big modules off the card.

## Measure it

For a director, the numbers that matter are pass rate at a fixed budget and attempts per pass, reported for three arms at the same budget: fixed (one render), reseed (same plan, new seeds), and the director (repairs that see feedback). The director earns its name only when it beats reseed on pass rate or on the composite at equal renders; today it does not. For identity: mean and minimum against the prototype, with the two anchors (null stand-in, hero itself) printed beside them so the reader knows where 0.75 sits; the log's v3b range is 0.42 to 0.82, and a lock at 0.90 above the hero's own 0.82 is itself a sign that the score is rewarding something other than the character. For framing: the judge's `match` and `p_own`, with `p_own` against chance (0.1 for ten shots), and a per-shot breakdown, because the storm passes at 0.7 and the two-shot scores 0 for every arm on a set that cannot stage two characters. For camera: the pass rate at 0.25 and the calibration table on probe frames, re-run whenever the neighbourhood or the similarity changes. For video: identity mean, minimum and decay, flicker mean and maximum, at a fixed size and length, against a baseline clip; the storm's 6.1 and 12.1 are the one baseline the log has, and the log is explicit that this is a baseline, not yet a threshold. For continuity: violations shipped per scene and generator calls per scene, side by side, because the stub loop's 5.7 against 11.3 came at 28 calls against 10. What is good: a board whose reshoot count falls over the run, a director arm above reseed at equal budget, identity above the null floor by a margin comparable to the hero's own gap above it, and a calibration table whose "should fail" rows are at zero.

## Exercises

1. Under the independent-seed model, compute the pass rate at budgets of 1, 4 and 8 for $p = 0.1$, then compute the attempts per shot for a board where one shot in ten has $p = 1$ and nine have $p = 0$. Check: 0.10, 0.34, 0.57; and 3.7 attempts at budget 4, which is the log's reseed arm.

2. Run the snippet with `flicker_p` at 0.15 on the lantern shot. Check: the pops raise the flicker mean enough that a clip with several visible drops passes the 3x rule. Replace the mean with the median in the verdict and report which takes change verdict.

3. Give the toy director a second lever. On `reshoot:identity`, halve the shot's `drift` before the next take (a stand-in for a stronger lock); on `reshoot:flicker`, halve `flicker_p` (a stand-in for a calmer brick or fewer steps). Check: the breach shot passes within the budget, and the total render count falls below 8.

4. Add an absolute clause: `identity_first >= 0.9` against the prototype. Render a keyframe from `fake_render(0, 1.5, 0)` at its last frame and use it as the key for a shot. Check: the relative rule keeps the clip and the absolute clause rejects it.

5. Run `recipes/cinema_render.py --smoke --shot s01 --prompt "a lighthouse in a storm"` locally and read the RESULT line; then, on the 5090 with a hero keyframe in `studio/assets`, render one take through `render_shot` and `refresh_shot`. Check: the real take reports identity and flicker of the same shape, and your storm-like shot lands near the log's 0.90 and 6.1 or you can say why not.

6. Save the toy's METRIC lines to a CSV over five runs with different base seeds, and write the regression query: per shot, the latest take's `identity_min` minus the best earlier take's, sorted ascending. Check: the breach shot is at the top of the list, which is the answer a Grafana dashboard would give.

## Test yourself

1. Why is the take, and not the frame or the clip file, the record the Studio keeps?

<details><summary>Answer</summary>
A take is a run: it has a request (the shot's prompt, keyframe, frames, size and the seed), a stream of METRIC lines, a RESULT with the critic statistics and a verdict, and an id the shot lists. That makes the shot's status a function of its takes, the render cost a count, and the quality a time series keyed by shot id. A clip file has none of that and a frame is a fraction of one measurement.
</details>

2. The reseed arm passed 0.10 at 3.7 attempts with a budget of 4. What does that tell you about the per-shot pass probabilities, and what lever does it rule out?

<details><summary>Answer</summary>
If every shot had $p = 0.1$ per seed, four seeds would pass 0.34 of shots. Passing 0.10 at 3.7 attempts is what you get when one shot passes on every seed (1 attempt) and nine never pass (4 attempts each): $(1 + 36)/10 = 3.7$. Pass probability is a property of the shot, so more seeds cannot help the nine; the request has to change (reference crop, conditioning, blocking, a different brick).
</details>

3. Write the loop's objective as an RL objective and say what is missing for it to be RL.

<details><summary>Answer</summary>
$J = \mathbb{E}_{a \sim \pi(\cdot \mid s, w, h)}[R(\rho(a))]$ with $R$ the pass indicator or the composite. The policy, the environment (the brick) and the reward (the critics) are all present; what is missing is an update to $\pi$. The expected reward moves only through the history $h$ the frozen policy reads. Stage C would add the GRPO update from Lab 05 with the critics as the reward, and Flow-GRPO could update the brick instead.
</details>

4. Why does the identity critic use the mean embedding of a hero set rather than the embedding of the hero image?

<details><summary>Answer</summary>
A single hero image carries one composition, and a cosine to it rewards copying that composition as much as the character. A set with varied framings averages composition out and keeps what is shared, which is the character. The log's v3 set, built with one framing, gave a generic stand-in 0.59; the crop-built v3b set gave it 0.42 with the hero at 0.82. The check is those two anchors.
</details>

5. Derive when the flicker spike rule fails, using the definition of the mean.

<details><summary>Answer</summary>
Let $m$ of the $T - 1$ differences be pops of size $P$ and the rest motion of size $b < P$. The mean is $(m P + (T - 1 - m) b) / (T - 1)$ and the rule passes when $P \le 3 \times$ that mean; rearranging, $m \ge (T - 1)(P - 3b) / (3(P - b))$, so the pops need only make up a fraction $(P - 3b) / (3(P - b))$ of the frames. At $P = 5b$ that is one sixth, about 17 percent of frames: a clip that strobes that often passes. A median or an external baseline does not move with $m$.
</details>

6. A locked frame scores identity 0.90 while the hero image itself scores 0.82 against the same prototype. What should you conclude?

<details><summary>Answer</summary>
The score is measuring something the hero does not maximize, which is the set's shared composition; the locked frame is closer to the set's average look than the original hero is. Identity above the hero's own score is a warning about the prototype, not a success. Pair it with the framing judge, which in the log dropped from 0.38 to 0.18 for the lock.
</details>

7. Explain the camera critic's parabola rule and why an outermost winner gets the full step.

<details><summary>Answer</summary>
Per axis, similarity is sampled at the request and at neighbours; if the best sample is interior, the peak of the parabola through it and its two neighbours estimates the achieved offset between the samples. If the best sample is the outermost, the true offset may be beyond it, and the critic reports the full step rather than interpolating inward, so a one-step drift is never under-reported. The cost is false rejection of sub-step drift, which the inner ring reduced.
</details>

8. Spot the problem in this verdict for a video take:

```python
keep = identity_min >= 0.85 * identity_first and flicker_max <= 3 * flicker_mean
```

<details><summary>Answer</summary>
Both clauses are relative to the clip itself. A keyframe already off-model passes the first clause on decay alone, and a strobing clip raises its own mean and passes the second. Add an absolute gate on the keyframe against the prototype (or gate the keyframe with the image critics before shooting) and a baseline for flicker the defect cannot move. The recipe also guards the mean with a floor so a still clip does not divide by zero; that is fine.
</details>

9. Where does the step time go if you turn this loop into Stage C with GRPO, and what does Lab 15 tell you to expect from a request the brick cannot parse?

<details><summary>Answer</summary>
Rendering: 34 seconds of generation per take against under a second to generate a JSON request, so a group of $G$ requests costs $G$ renders and the render farm is the step, as in Lab 15's reproduction. A malformed request is a gate rejection: near-zero reward in a group of honest takes, which under group normalization inflates the standard deviation and shrinks every other advantage. Exclude infrastructure failures, and use `scale_rewards none` or a small gate weight for the policy's own failures.
</details>

10. The stub loop shipped half the violations at three times the generator calls. Which number does Stage C target and why is the other one a ceiling?

<details><summary>Answer</summary>
Violations shipped is the product metric; generator calls are the price. A director that only rejects can lower violations only by spending more renders, so the calls-per-scene number is what a policy that repairs correctly would reduce: fewer takes to reach the same verdicts. The ceiling on violations is set by the critics (a violation no critic sees ships regardless), so improving the director beyond the critics' coverage needs new critics, not more renders.
</details>

## What will change, what will not

The durable part is the decomposition, which is Lab 15's with the gradient removed: a policy over requests, a brick you do not differentiate through, deterministic critics on the render, and a decision. That structure survives any change of generator, because the brick contract (a card, an input and output adapter, a critic) is what lets you swap Wan for the next model without touching the loop. The rejection-sampling arithmetic is arithmetic; it will keep telling you, for any director, whether a pass rate is a property of the seed or of the shot. The two-anchor check on a prototype and the calibration table on probe frames are the kinds of measurement that outlive the critics they were built for.

What will change: the critics themselves. DINOv2 cosine against a prototype is a stand-in for an identity model that knows what a character is; a VLM judge rated 0 to 10 is a stand-in for a framing critic with a calibration set; the flicker rule is a baseline, not a threshold, on one clip. Expect the image-to-video brick to absorb the temporal critics as training signal (Lab 19), at which point flicker stops being a reshoot reason. Expect the identity confound to be resolved by conditioning that separates layout from appearance, whether that is a two-pass render, a better adapter, or a video model that takes the previz and the reference as separate inputs; the regional-mask null says that the current adapter cannot do it inside a depth-dictated layout. And expect the director to become worth training only once the loop has produced a few thousand traces, because 34 traces moved the pass rate from 0.10 to 0.20 and nothing more.

What is open: whether a language model reading critic feedback can beat rejection sampling at equal budget on shots the seeds cannot fix, which is the whole bet; whether critics that are cheap enough to run on every take can be made hard enough to resist a policy optimized against them (Lab 15's judge exploitation, applied to a framing judge); and whether a world state of atoms scales from a ten-shot scene to a ninety-minute film without a continuity critic of its own that is as calibrated as the camera critic is today.

## Read next

1. DINOv2: Learning Robust Visual Features without Supervision, Oquab, 2023. The embedding behind the identity and camera critics; read it for what the patch tokens carry and why a class-token cosine measures appearance.
2. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models, Ye, 2023. The identity lock, and the reason a global image embedding carries composition with it.
3. Adding Conditional Control to Text-to-Image Diffusion Models, Zhang, 2023. ControlNet, the depth conditioning that took the camera pass rate from 5 to 80 percent.
4. 3D Gaussian Splatting for Real-Time Radiance Field Rendering, Kerbl, 2023. The set representation the previz and the camera critic are built on.
5. Wan: Open and Advanced Large-Scale Video Generative Models, Wan Team, 2025. The video brick; read the VAE and DiT sections for the memory arithmetic behind the two-phase decode.
6. ReAct: Synergizing Reasoning and Acting in Language Models, Yao, 2022. The plan, act, observe loop the director runs, with tool calls as actions.
7. Reflexion: Language Agents with Verbal Reinforcement Learning, Shinn, 2023. Repair from feedback without gradients, which is what the director does and what Stage C would replace.
8. Flow-GRPO: Training Flow Matching Models via Online RL, Liu, 2025. Where gradients would enter the brick; read with Lab 19.
