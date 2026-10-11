"""The tabletop the arm works on, and the checks that decide whether a plan is physically sound.

A model never moves the arm directly. It writes a short skill program over named objects:

    {"op": "move", "to": "red" | [x, y, z], "above": 0.08}   tip to an object (plus a height) or to a point, metres
    {"op": "open"} / {"op": "close"}                          gripper
    {"op": "home"}                                            the home pose
    {"op": "refuse", "reason": "..."}                         the task cannot be done safely

compile_program() turns it into joint waypoints with inverse kinematics; simulate() replays any waypoints (from a
program or from a model that wrote joint angles itself) against the scene: joint limits, reach, links through the
table, a closed gripper driven into a block, grasps, drops, stacking. check_goal() scores the final scene against a
task's hidden goal. Everything here is deterministic; no model is a judge.

Frame: x forward, y left, z up, metres, origin at the base on the table.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-100, 100), "elbow_flex": (-100, 100), "wrist_flex": (-100, 100), "wrist_roll": (-160, 160), "gripper": (0, 100)}
LINKS = {"base": 0.0585, "upper": 0.1165, "forearm": 0.1350, "wrist": 0.0600, "gripper": 0.0900}
HOME = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 30}
OPEN, CLOSED, GRIP_T = 80.0, 5.0, 40.0   # gripper values; >= GRIP_T counts as open
GRASP_R = 0.02                           # tip must be this close to a block's centre when the gripper closes
COLORS = {"red": "#d0453f", "green": "#4db39d", "yellow": "#c9962b", "purple": "#8a6bc9", "blue": "#3b6fb6"}


def fk(j: dict[str, float]) -> list[list[float]]:
    """Joint angles (deg) → base, shoulder, elbow, wrist, tip positions. Angles are measured from vertical."""
    pan = math.radians(j.get("shoulder_pan", 0))
    a1 = math.radians(j.get("shoulder_lift", 0))
    a2 = a1 + math.radians(j.get("elbow_flex", 0))
    a3 = a2 + math.radians(j.get("wrist_flex", 0))
    pts = [(0.0, 0.0), (0.0, LINKS["base"])]
    r, z = 0.0, LINKS["base"]
    for ang, L in ((a1, LINKS["upper"]), (a2, LINKS["forearm"]), (a3, LINKS["wrist"] + LINKS["gripper"])):
        r += L * math.sin(ang)
        z += L * math.cos(ang)
        pts.append((r, z))
    return [[rr * math.cos(pan), rr * math.sin(pan), zz] for rr, zz in pts]


def _in_limits(j: dict[str, float]) -> bool:
    return all(LIMITS[k][0] - 1e-6 <= v <= LIMITS[k][1] + 1e-6 for k, v in j.items() if k in LIMITS)


def ik(x: float, y: float, z: float, gripper: float = OPEN) -> dict[str, float] | None:
    """Tip to (x, y, z). Prefers the tool pointing straight down, tilts toward horizontal until a solution fits the
    joint limits and keeps every link above the table. None when no such solution exists."""
    L1, L2, Lt, base = LINKS["upper"], LINKS["forearm"], LINKS["wrist"] + LINKS["gripper"], LINKS["base"]
    r = math.hypot(x, y)
    pan = math.degrees(math.atan2(y, x)) if r > 1e-6 else 0.0
    for pitch in range(180, 85, -5):  # tool angle from vertical: 180 = pointing down
        a3 = math.radians(pitch)
        dr, dz = r - Lt * math.sin(a3), z - Lt * math.cos(a3) - base
        D = math.hypot(dr, dz)
        if D > L1 + L2 or D < abs(L1 - L2):
            continue
        ce = max(-1.0, min(1.0, (D * D - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
        for e in (math.acos(ce), -math.acos(ce)):
            a1 = math.atan2(dr, dz) - math.atan2(L2 * math.sin(e), L1 + L2 * math.cos(e))
            j = {"shoulder_pan": pan, "shoulder_lift": math.degrees(a1), "elbow_flex": math.degrees(e),
                 "wrist_flex": math.degrees(a3 - a1 - e), "wrist_roll": 0.0, "gripper": gripper}
            if _in_limits(j) and min(p[2] for p in fk(j)[2:]) >= 0.004:
                return {k: round(v, 3) for k, v in j.items()}
    return None


# ---------------------------------------------------------------- scenes

def make_scene(seed: int = 0, n_blocks: int = 3, far: str | None = None) -> dict[str, Any]:
    """Blocks (3 cm cubes) and one bin, placed where the arm can reach them, at least 7 cm apart."""
    rng = random.Random(seed)
    objs: list[dict[str, Any]] = [{"name": "bin", "kind": "bin", "xyz": [round(rng.uniform(0.14, 0.2), 3), round(rng.choice([-1, 1]) * rng.uniform(0.13, 0.17), 3), 0.0], "size": 0.09, "height": 0.04, "color": COLORS["blue"]}]
    for name in ["red", "green", "yellow"][:n_blocks]:
        for _ in range(200):
            p = [round(rng.uniform(0.15, 0.23), 3), round(rng.uniform(-0.14, 0.14), 3)]
            if all(math.hypot(p[0] - o["xyz"][0], p[1] - o["xyz"][1]) > (0.09 if o["kind"] == "bin" else 0.07) for o in objs):
                break
        objs.append({"name": name, "kind": "block", "xyz": [p[0], p[1], 0.015], "size": 0.03, "color": COLORS[name]})
    if far:
        objs.append({"name": far, "kind": "block", "xyz": [0.47, round(rng.uniform(-0.05, 0.05), 3), 0.015], "size": 0.03, "color": COLORS.get(far, "#888")})
    return {"objects": objs, "held": None, "seed": seed}


def norm(name: Any) -> str:
    """'The Red block' → 'red'."""
    n = str(name).lower().strip()
    for w in ("the ", " block", " cube"):
        n = n.replace(w, "")
    return n.strip()


def obj(world: dict[str, Any], name: str) -> dict[str, Any] | None:
    n = norm(name)
    return next((o for o in world["objects"] if o["name"] == n), None)


def describe(world: dict[str, Any]) -> str:
    rows = []
    for o in world["objects"]:
        x, y, z = o["xyz"]
        if o["kind"] == "bin":
            rows.append(f"- bin: open box, centre ({x:.3f}, {y:.3f}), floor at z 0, footprint {o['size']*100:.0f} cm square, walls {o['height']*100:.0f} cm high")
        else:
            rows.append(f"- {o['name']} block: {o['size']*100:.0f} cm cube, centre ({x:.3f}, {y:.3f}, {z:.3f})" + (" (held)" if world.get("held") == o["name"] else ""))
    return "\n".join(rows)


# ---------------------------------------------------------------- program → waypoints

def compile_program(world: dict[str, Any], program: list[dict[str, Any]], start: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Skill program → joint waypoints. Returns (waypoints, issues, refused). Targets are resolved against the scene as it
    will be at that step, so "move to red" after red was dropped in the bin goes to the bin."""
    wps: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    cur = dict(start)
    w = copy.deepcopy(world)
    for i, step in enumerate(program[:24], 1):
        op = str(step.get("op", "")).lower()
        if op == "refuse":
            return wps, issues, True
        if op in ("open", "close"):
            cur = {**cur, "gripper": OPEN if op == "open" else CLOSED}
            wps.append({"name": op, "joints": dict(cur), "seconds": 0.5, "step": i})
        elif op == "home":
            cur = {**HOME, "gripper": cur["gripper"]}
            wps.append({"name": "home", "joints": dict(cur), "seconds": 1.4, "step": i})
        elif op == "move":
            to, above = step.get("to"), float(step.get("above", 0.0) or 0.0)
            if isinstance(to, str):
                o = obj(w, to)
                if not o:
                    issues.append({"step": i, "level": "error", "msg": f"there is no object called {to!r}"})
                    continue
                if w.get("held") == o["name"]:
                    issues.append({"step": i, "level": "error", "msg": f"cannot move to {to}: the gripper is holding it"})
                    continue
                x, y, z = o["xyz"]
                if o["kind"] == "bin":
                    z = o["height"]
                target = [x, y, z + above]
            elif isinstance(to, (list, tuple)) and len(to) == 3:
                target = [float(v) for v in to]
                target[2] += above
            else:
                issues.append({"step": i, "level": "error", "msg": f"move needs an object name or [x, y, z], got {to!r}"})
                continue
            j = ik(*target, gripper=cur["gripper"])
            if j is None:
                r = math.hypot(target[0], target[1])
                issues.append({"step": i, "level": "error", "msg": f"cannot reach ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}): {r:.2f} m out from the base, no joint solution inside the limits"})
                continue
            cur = j
            wps.append({"name": f"move {to if isinstance(to, str) else 'point'}" + (f" +{above*100:.0f}cm" if above else ""), "joints": dict(cur), "seconds": 1.2, "step": i})
        else:
            issues.append({"step": i, "level": "error", "msg": f"unknown op {op!r}"})
            continue
        # keep the scene in step so later object references see earlier grasps and drops
        w = simulate(w, [wps[-1]], wps[-2]["joints"] if len(wps) > 1 else start)["world"]
    return wps, issues, False


# ---------------------------------------------------------------- physics

def _inside(p: list[float], o: dict[str, Any], pad: float = 0.0) -> bool:
    h = o["size"] / 2 + pad
    c = o["xyz"]
    return abs(p[0] - c[0]) < h and abs(p[1] - c[1]) < h and abs(p[2] - c[2]) < h


def _support_z(w: dict[str, Any], held: dict[str, Any]) -> float:
    """Where a released block comes to rest: on the highest block under it, else the table (or the bin floor)."""
    z = 0.0
    x, y = held["xyz"][0], held["xyz"][1]
    for o in w["objects"]:
        if o is held or o["kind"] != "block":
            continue
        if abs(o["xyz"][0] - x) < o["size"] * 0.7 and abs(o["xyz"][1] - y) < o["size"] * 0.7 and o["xyz"][2] < held["xyz"][2]:
            z = max(z, o["xyz"][2] + o["size"] / 2)
    return z + held["size"] / 2


def simulate(world: dict[str, Any], waypoints: list[dict[str, Any]], start: dict[str, float]) -> dict[str, Any]:
    """Replay waypoints (joint space, linear interpolation like the executor) and report what happens.
    Gripper changes take effect at the end of a segment: closing grasps the nearest block within GRASP_R of the tip,
    opening drops whatever is held onto what is below it."""
    w = copy.deepcopy(world)
    issues: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    prev = {k: float(start.get(k, 0)) for k in JOINTS}
    for n, wp in enumerate(waypoints, 1):
        step = wp.get("step", n)
        nxt = {k: float(wp["joints"].get(k, prev[k])) for k in JOINTS}
        for k, v in nxt.items():
            lo, hi = LIMITS[k]
            if not lo - 0.5 <= v <= hi + 0.5:
                issues.append({"step": step, "level": "error", "msg": f"{k} {v:.0f}° is outside its limits [{lo}, {hi}]"})
                nxt[k] = max(lo, min(hi, v))
        held = obj(w, w["held"]) if w.get("held") else None
        hit = None
        for s in range(1, 13):
            e = s / 12
            j = {k: prev[k] + (nxt[k] - prev[k]) * e for k in JOINTS}
            j["gripper"] = prev["gripper"]
            pts = fk(j)
            tip = pts[-1]
            if min(p[2] for p in pts[2:]) < -0.002:
                hit = hit or "the arm passes through the table"
            if held is not None:
                held["xyz"] = [round(v, 4) for v in tip]
                if tip[2] - held["size"] / 2 < -0.002:
                    hit = hit or f"the held {held['name']} block is dragged through the table"
                for o in w["objects"]:
                    if o is not held and o["kind"] == "block" and _inside(tip, o, held["size"] / 2 - 0.004):
                        hit = hit or f"the held {held['name']} block hits the {o['name']} block"
            elif j["gripper"] < GRIP_T:
                for o in w["objects"]:
                    if o["kind"] == "block" and _inside(tip, o, -0.002):
                        hit = hit or f"the closed gripper is driven into the {o['name']} block (open it first)"
        if hit:
            issues.append({"step": step, "level": "error", "msg": hit})
        tip = fk(nxt)[-1]
        if prev["gripper"] >= GRIP_T and nxt["gripper"] < GRIP_T and held is None:  # close
            cands = sorted(((math.dist(tip, o["xyz"]), o) for o in w["objects"] if o["kind"] == "block"), key=lambda t: t[0])
            if cands and cands[0][0] <= GRASP_R:
                w["held"] = cands[0][1]["name"]
                cands[0][1]["xyz"] = [round(v, 4) for v in tip]
            else:
                near = f"; nearest is {cands[0][1]['name']} at {cands[0][0]*100:.1f} cm" if cands else ""
                issues.append({"step": step, "level": "warn", "msg": f"gripper closed on nothing at ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}){near}"})
        elif prev["gripper"] < GRIP_T and nxt["gripper"] >= GRIP_T and held is not None:  # open → drop
            z0 = held["xyz"][2]
            held["xyz"][2] = round(_support_z(w, held), 4)
            w["held"] = None
            if z0 - held["xyz"][2] > 0.05:
                issues.append({"step": step, "level": "warn", "msg": f"{held['name']} dropped from {100*(z0-held['xyz'][2]):.0f} cm"})
        prev = nxt
        timeline.append(copy.deepcopy(w))
    return {"world": w, "issues": issues, "timeline": timeline}


# ---------------------------------------------------------------- goals (hidden from the planner)

def check_goal(world: dict[str, Any], goal: list[dict[str, Any]], refused: bool) -> tuple[bool, list[str]]:
    fails = []
    for g in goal:
        k = g["pred"]
        if k == "refused":
            if not refused:
                fails.append("should have refused")
            continue
        if refused:
            fails.append("refused a doable task")
            break
        o = obj(world, g.get("obj", ""))
        b = obj(world, g.get("target", "")) if k in ("in", "on") else None
        if o is None or (k in ("in", "on") and b is None) or k not in ("in", "on", "at"):
            fails.append(f"cannot check {g}")
            continue
        if world.get("held") == o["name"]:
            fails.append(f"{o['name']} still in the gripper")
            continue
        x, y, z = o["xyz"]
        if k == "in":
            if not (abs(x - b["xyz"][0]) < b["size"] / 2 and abs(y - b["xyz"][1]) < b["size"] / 2 and z < b["height"] + o["size"]):
                fails.append(f"{o['name']} not in the {b['name']}")
        elif k == "on":
            if not (math.hypot(x - b["xyz"][0], y - b["xyz"][1]) < 0.015 and abs(z - (b["xyz"][2] + b["size"])) < 0.006):
                fails.append(f"{o['name']} not on {b['name']}")
        elif k == "at":
            if math.hypot(x - g["xy"][0], y - g["xy"][1]) > g.get("tol", 0.02) or z > 0.02:
                fails.append(f"{o['name']} not at ({g['xy'][0]:.2f}, {g['xy'][1]:.2f}) on the table")
    return (not fails), fails


def tasks(seeds: tuple[int, ...] = (1, 2)) -> list[dict[str, Any]]:
    """Seven task families × seeds. Each: scene, instruction, hidden goal."""
    out = []
    for s in seeds:
        sc = make_scene(s)
        pos = {o["name"]: o["xyz"] for o in sc["objects"]}
        bn = pos["bin"]
        nearest = min(("red", "green", "yellow"), key=lambda c: math.dist(pos[c][:2], bn[:2]))
        side, shift = 1, None
        for sd in ((1, -1) if pos["yellow"][1] < 0 else (-1, 1)):  # a side whose target is free and reachable
            cand = [pos["yellow"][0], round(pos["yellow"][1] + sd * 0.08, 3)]
            clear = all(math.dist(cand, pos[c][:2]) > 0.05 for c in ("red", "green")) and math.dist(cand, bn[:2]) > 0.08
            if clear and ik(cand[0], cand[1], 0.015) and ik(cand[0], cand[1], 0.075):
                side, shift = sd, cand
                break
        out += [
            {"id": f"bin-{s}", "scene": sc, "instruction": "Put the red block in the bin.", "goal": [{"pred": "in", "obj": "red", "target": "bin"}]},
            {"id": f"stack-{s}", "scene": sc, "instruction": "Stack the green block on top of the red block.", "goal": [{"pred": "on", "obj": "green", "target": "red"}]},
        ] + ([{"id": f"shift-{s}", "scene": sc, "instruction": f"Slide the yellow block 8 cm to the {'left' if side > 0 else 'right'} (left is +y).", "goal": [{"pred": "at", "obj": "yellow", "xy": shift}]}] if shift else []) + [
            {"id": f"two-{s}", "scene": sc, "instruction": "Put the green and yellow blocks in the bin.", "goal": [{"pred": "in", "obj": "green", "target": "bin"}, {"pred": "in", "obj": "yellow", "target": "bin"}]},
            {"id": f"nearest-{s}", "scene": sc, "instruction": "Put whichever block is closest to the bin into the bin.", "goal": [{"pred": "in", "obj": nearest, "target": "bin"}]},
            {"id": f"swap-{s}", "scene": sc, "instruction": "Swap the positions of the red and green blocks.", "goal": [{"pred": "at", "obj": "red", "xy": pos["green"][:2]}, {"pred": "at", "obj": "green", "xy": pos["red"][:2]}]},
            {"id": f"far-{s}", "scene": make_scene(s, far="purple"), "instruction": "Put the purple block in the bin.", "goal": [{"pred": "refused"}]},
        ]
    return out


def oracle_program(task: dict[str, Any]) -> list[dict[str, Any]]:
    """A hand-written planner that knows the goal: the ceiling, and a check that every task is solvable here."""
    def carry(src: list[float], dest: list[dict[str, Any]]) -> list[dict[str, Any]]:
        x, y, z = src
        return ([{"op": "open"}, {"op": "move", "to": [x, y, z + 0.06]}, {"op": "move", "to": [x, y, z]}, {"op": "close"},
                 {"op": "move", "to": [x, y, z + 0.06]}] + dest + [{"op": "open"}])

    def onto(x: float, y: float) -> list[dict[str, Any]]:
        return [{"op": "move", "to": [x, y, 0.075]}, {"op": "move", "to": [x, y, 0.016]}]

    sc = task["scene"]
    if any(g["pred"] == "refused" for g in task["goal"]):
        return [{"op": "refuse", "reason": "out of reach"}]
    if task["id"].startswith("swap"):
        r, gr = obj(sc, "red")["xyz"], obj(sc, "green")["xyz"]
        tmp = [0.1, -0.1 if r[1] + gr[1] > 0 else 0.1, 0.015]  # a free spot near the base
        return (carry(r, onto(*tmp[:2])) + carry(gr, onto(r[0], r[1])) + carry(tmp, onto(gr[0], gr[1])) + [{"op": "home"}])
    prog: list[dict[str, Any]] = []
    for g in task["goal"]:
        src = obj(sc, g["obj"])["xyz"]
        if g["pred"] == "in":
            prog += carry(src, [{"op": "move", "to": "bin", "above": 0.05}])
        elif g["pred"] == "on":
            prog += carry(src, [{"op": "move", "to": g["target"], "above": 0.06}, {"op": "move", "to": g["target"], "above": 0.031}])
        elif g["pred"] == "at":
            prog += carry(src, onto(*g["xy"]))
    return prog + [{"op": "home"}]
