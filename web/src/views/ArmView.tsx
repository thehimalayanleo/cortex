/**
 * Physical AI: the SO-ARM101 in the Lab. A 3D view of the arm from its joint angles (forward kinematics on the
 * server, drawn here), sliders and named poses, episode recording and playback (LeRobot-style frames), and a planner
 * that turns an instruction into waypoints (Nemotron on Nebius when configured). Without an arm the bridge simulates
 * the servos, so everything here works on a laptop; with one, the same bridge drives the real Feetech bus.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { ArmBench, ArmEpisode, ArmPlanOut, ArmStatus, ArmTrain, NebiusStatus } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

type P3 = [number, number, number];
function hull(pts: P3[]): P3[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o: P3, a: P3, b: P3) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo: P3[] = [], up: P3[] = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const pct = (v?: number) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const MODE_LABEL: Record<string, string> = { oracle: "oracle (ceiling)", joints: "model writes joint angles", skills: "skill program, no repair", "skills+repair": "skill program + simulator repair" };

const LABELS: Record<string, string> = { shoulder_pan: "pan", shoulder_lift: "lift", elbow_flex: "elbow", wrist_flex: "wrist", wrist_roll: "roll", gripper: "grip" };

export function ArmView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [st, setSt] = useState<ArmStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [eps, setEps] = useState<ArmEpisode[]>([]);
  const [task, setTask] = useState("");
  const [instr, setInstr] = useState("put the red block in the bin");
  const [mode, setMode] = useState("skills");
  const [bench, setBench] = useState<ArmBench | null>(null);
  const [neb, setNeb] = useState<NebiusStatus | null>(null);
  const [keyIn, setKeyIn] = useState("");
  const [seed, setSeed] = useState(1);
  const [train, setTrain] = useState<ArmTrain | null>(null);
  const [trainSeed, setTrainSeed] = useState(6);
  const [trainSafety, setTrainSafety] = useState(true);
  const [plan, setPlan] = useState<ArmPlanOut | null>(null);
  const [planning, setPlanning] = useState(false);
  const [port, setPort] = useState("");
  const [poseName, setPoseName] = useState("");
  const [trail, setTrail] = useState<number[][]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cam = useRef({ yaw: 0.7, pitch: 0.45, drag: null as null | [number, number] });
  const pending = useRef<Record<string, number>>({});
  const timer = useRef(0);

  const load = useCallback(async () => {
    try {
      const s = await api.arm.status();
      setSt(s);
      setErr(null);
      setTrail((t) => (s.moving || s.recording ? [...t.slice(-240), s.fk.tip] : t));
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
    api.arm.episodes().then(setEps).catch(() => undefined);
    api.arm.bench().then(setBench).catch(() => undefined);
    api.arm.nebius().then(setNeb).catch(() => undefined);
    api.arm.train().then(setTrain).catch(() => undefined);
  }, [load, refresh]);
  useEffect(() => {
    if (train?.status !== "running" && train?.status !== "queued") return;
    const t = window.setInterval(() => api.arm.train().then(setTrain).catch(() => undefined), 4000);
    return () => window.clearInterval(t);
  }, [train?.status]);
  useEffect(() => {
    if (bench?.status !== "running") return;
    const t = window.setInterval(() => api.arm.bench().then(setBench).catch(() => undefined), 3000);
    return () => window.clearInterval(t);
  }, [bench?.status]);
  useEffect(() => {
    const t = window.setInterval(() => void load(), st?.moving || st?.recording ? 150 : 1000);
    return () => window.clearInterval(t);
  }, [load, st?.moving, st?.recording]);

  // sliders: coalesce fast changes into one request every 80 ms
  const setJoint = (j: string, v: number) => {
    if (!st) return;
    setSt({ ...st, target: { ...st.target, [j]: v }, joints: st.mode === "offline" ? { ...st.joints, [j]: v } : st.joints });
    pending.current[j] = v;
    if (!timer.current) timer.current = window.setTimeout(() => {
      const js = pending.current;
      pending.current = {};
      timer.current = 0;
      api.arm.joints(js).then(setSt).catch((e) => toast(errorMessage(e), "error"));
    }, 80);
  };

  // draw the arm
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !st) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth, H = c.clientHeight;
      if (c.width !== W * dpr || c.height !== H * dpr) {
        c.width = W * dpr;
        c.height = H * dpr;
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const { yaw, pitch } = cam.current;
      const S = Math.min(W, H) * 1.6; // pixels per metre
      const proj = (x: number, y: number, z: number) => {
        // world: x forward, y left, z up. camera orbits the base.
        const cx = x * Math.cos(yaw) - y * Math.sin(yaw), cy = x * Math.sin(yaw) + y * Math.cos(yaw);
        const depth = cy * Math.cos(pitch) + z * Math.sin(pitch);
        const sy = -cy * Math.sin(pitch) + z * Math.cos(pitch);
        const f = 1 / (1.6 + depth * 0.8);
        return [W / 2 + cx * S * f, H * 0.72 - sy * S * f, f] as [number, number, number];
      };
      // ground grid (10 cm)
      ctx.strokeStyle = "rgba(140,160,170,0.18)";
      ctx.lineWidth = 1;
      for (let i = -4; i <= 4; i++) {
        const a = proj(i * 0.1, -0.4, 0), b = proj(i * 0.1, 0.4, 0), c2 = proj(-0.4, i * 0.1, 0), d = proj(0.4, i * 0.1, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c2[0], c2[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
      }
      // reach circle for the current lift plane
      ctx.strokeStyle = "rgba(77,179,157,0.25)";
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      const R = st.links.upper + st.links.forearm + st.links.wrist + st.links.gripper;
      for (let k = 0; k <= 64; k++) {
        const a = (k / 64) * Math.PI * 2;
        const p = proj(R * Math.cos(a), R * Math.sin(a), 0);
        if (k === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // trail of the tip
      if (trail.length > 1) {
        ctx.strokeStyle = "rgba(201,150,43,0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        trail.forEach((t, i) => {
          const p = proj(t[0], t[1], t[2]);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        });
        ctx.stroke();
      }
      // the tabletop: bins as open boxes, blocks as cubes; a held block rides on the tip
      const tipNow = st.fk.tip;
      for (const o of st.world?.objects ?? []) {
        const [x, y, z] = st.world.held === o.name ? tipNow : o.xyz;
        const h = o.size / 2;
        if (o.kind === "bin") {
          const H = o.height ?? 0.04;
          const ring = (zz: number) => [[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]].map(([dx, dy]) => proj(x + dx, y + dy, zz));
          ctx.fillStyle = o.color + "22";
          const f = ring(0);
          ctx.beginPath(); f.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.fill();
          ctx.strokeStyle = o.color + "cc";
          ctx.lineWidth = 1.5;
          for (const zz of [0, H]) { const r = ring(zz); ctx.beginPath(); r.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.stroke(); }
          for (const [dx, dy] of [[-h, -h], [h, -h], [h, h], [-h, h]]) { const a = proj(x + dx, y + dy, 0), b = proj(x + dx, y + dy, H); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
        } else {
          const corners: P3[] = [];
          for (const dx of [-h, h]) for (const dy of [-h, h]) for (const dz of [-h, h]) corners.push(proj(x + dx, y + dy, z + dz));
          const hl = hull(corners);
          ctx.fillStyle = o.color;
          ctx.beginPath(); hl.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.closePath(); ctx.fill();
          const top = [[-h, -h], [h, -h], [h, h], [-h, h]].map(([dx, dy]) => proj(x + dx, y + dy, z + h));
          ctx.fillStyle = "rgba(255,255,255,0.18)";
          ctx.beginPath(); top.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.closePath(); ctx.fill();
        }
      }
      // the planned tip path
      if (plan?.waypoints.length) {
        ctx.strokeStyle = plan.ok ? "rgba(77,179,157,0.8)" : "rgba(208,69,63,0.8)";
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        plan.waypoints.forEach((w, i) => { const q = proj(w.tip[0], w.tip[1], w.tip[2]); if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); });
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // shadow
      const pts = st.fk.points;
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 6;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const q = proj(p[0], p[1], 0);
        if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      });
      ctx.stroke();
      // links
      const cols = ["#8a94a0", "#4db39d", "#3b6fb6", "#c2632b"];
      ctx.lineCap = "round";
      for (let i = 1; i < pts.length; i++) {
        const a = proj(pts[i - 1][0], pts[i - 1][1], pts[i - 1][2]), b = proj(pts[i][0], pts[i][1], pts[i][2]);
        ctx.strokeStyle = cols[i - 1] ?? "#c2632b";
        ctx.lineWidth = Math.max(4, 14 * b[2] - (i - 1) * 2);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        ctx.fillStyle = "#e6ede9";
        ctx.beginPath(); ctx.arc(a[0], a[1], Math.max(3, 6 * a[2]), 0, Math.PI * 2); ctx.fill();
      }
      // gripper jaws at the tip (open angle from the gripper joint)
      const tip = pts[pts.length - 1], prev = pts[pts.length - 2];
      const open = (st.joints.gripper ?? 0) / 100;
      const dx = tip[0] - prev[0], dy = tip[1] - prev[1], dz = tip[2] - prev[2];
      const n = Math.hypot(dx, dy, dz) || 1;
      const ux = dx / n, uy = dy / n, uz = dz / n;
      // a vector perpendicular to the link, in the pan plane
      const px = -uy, py = ux, pz = 0;
      const gap = 0.012 + 0.03 * open;
      for (const sgn of [-1, 1]) {
        const a = proj(tip[0] - ux * 0.035 + sgn * px * gap, tip[1] - uy * 0.035 + sgn * py * gap, tip[2] - uz * 0.035 + sgn * pz * gap);
        const b = proj(tip[0] + sgn * px * gap, tip[1] + sgn * py * gap, tip[2] + sgn * pz * gap);
        ctx.strokeStyle = "#d0453f";
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      // base
      const b0 = proj(0, 0, 0);
      ctx.fillStyle = "rgba(140,150,160,0.5)";
      ctx.beginPath(); ctx.ellipse(b0[0], b0[1], 22 * b0[2], 9 * b0[2], 0, 0, Math.PI * 2); ctx.fill();
      // readout
      ctx.font = "11px IBM Plex Mono, monospace";
      ctx.fillStyle = "rgba(180,195,200,0.85)";
      const t = st.fk.tip;
      ctx.fillText(`tip x ${t[0].toFixed(3)}  y ${t[1].toFixed(3)}  z ${t[2].toFixed(3)} m · reach ${st.fk.reach_m.toFixed(3)} m`, 10, H - 10);
      ctx.fillText(`${st.mode}${st.bridge ? ` · bridge ${st.bridge}` : ""}${st.moving ? ` · ${st.moving}` : ""}${st.recording ? ` · REC ${st.recording}` : ""}`, 10, 18);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    (c as unknown as { __draw?: () => void }).__draw = draw;
    return () => ro.disconnect();
  }, [st, trail, plan]);
  const redraw = () => (canvasRef.current as unknown as { __draw?: () => void } | null)?.__draw?.();

  if (err) return <EmptyState title="Could not reach the arm service" hint={err} />;
  if (!st) return <EmptyState title="Loading the arm" />;
  const offline = st.mode === "offline";
  const act = <T,>(p: Promise<T>, then?: (v: T) => void) => p.then((v) => { then?.(v); void load(); }).catch((e) => toast(errorMessage(e), "error"));

  return (
    <div className="arm">
      <div className="arm-top">
        <div className="arm-stage">
          <canvas
            ref={canvasRef}
            className="arm-canvas"
            onPointerDown={(e) => { cam.current.drag = [e.clientX, e.clientY]; try { (e.target as HTMLElement).setPointerCapture(e.pointerId); } catch { /* synthetic */ } }}
            onPointerUp={() => { cam.current.drag = null; }}
            onPointerMove={(e) => {
              const d = cam.current.drag;
              if (!d) return;
              cam.current.yaw += (e.clientX - d[0]) * 0.008;
              cam.current.pitch = Math.max(0.05, Math.min(1.4, cam.current.pitch + (e.clientY - d[1]) * 0.006));
              cam.current.drag = [e.clientX, e.clientY];
              redraw();
            }}
          />
          <div className="arm-status">
            <span className={`pill ${offline ? "" : st.mode === "hardware" ? "ok" : "sim"}`}>{offline ? "offline" : st.mode === "hardware" ? `hardware · ${st.port}` : "simulated servos"}</span>
            {st.bridge && <button className="lnk" onClick={() => navigate({ kind: "lab", run: st.bridge! })}>bridge log</button>}
          </div>
        </div>
        <div className="arm-panel">
          <div className="arm-card">
            <div className="row">
              <b>Nemotron on Nebius Token Factory</b>
              <span className="grow" />
              <span className={`verdict ${neb?.connected ? "ok" : "warn"}`}>{neb?.connected ? "connected" : "not connected"}</span>
            </div>
            {neb?.connected ? (
              <div className="muted small mono">fast {neb.fast} · deep {neb.deep}</div>
            ) : (
              <div className="row">
                <input className="input sm" type="password" placeholder="paste a Token Factory API key" value={keyIn} onChange={(e) => setKeyIn(e.target.value)} autoComplete="off" />
                <button className="btn sm primary" disabled={keyIn.trim().length < 10} onClick={() => act(api.arm.setNebiusKey(keyIn.trim()), (n) => { setNeb(n); setKeyIn(""); toast(n.connected ? "Token Factory connected" : n.error ?? "Key saved, but no models listed", n.connected ? undefined : "error"); })}>Connect</button>
              </div>
            )}
            {neb && !neb.connected && <div className="muted small">Until then plans come from {neb.provider === "selfhost" ? "self-hosted Nemotron 3 Nano (vLLM on the 5090)" : "the app's chat model"}. The key stays on this machine (~/.config/cortex/nebius.key, mode 600).</div>}
          </div>
          <div className="arm-card">
            <div className="row">
              <b>Connection</b>
              <span className="grow" />
              {offline ? (
                <>
                  <select className="select sm" value={port} onChange={(e) => setPort(e.target.value)} aria-label="Serial port">
                    <option value="">simulate (no arm)</option>
                    {st.ports.map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                  <button className="btn sm primary" onClick={() => act(api.arm.connect(port, !port), () => toast(port ? `Connecting ${port}` : "Simulated servos started"))}>{port ? "Connect arm" : "Start simulator"}</button>
                </>
              ) : (
                <button className="btn sm" onClick={() => act(api.arm.disconnect())}>Disconnect</button>
              )}
            </div>
            <div className="muted small">
              {st.ports.length ? `${st.ports.length} serial port${st.ports.length > 1 ? "s" : ""} seen on ${st.host}.` : `No serial port seen on ${st.host}; plug the SO-ARM101's USB in (or run the bridge on the GPU box) and this list fills in.`} The bridge is an ordinary run: same log, metrics and stop button.
            </div>
          </div>
          <div className="arm-card">
            <b>Joints</b>
            {st.joint_names.map((j) => (
              <label key={j} className="arm-slider">
                <span className="jn">{LABELS[j] ?? j}</span>
                <input type="range" min={st.limits[j][0]} max={st.limits[j][1]} step={1} value={Math.round(st.target[j] ?? 0)} onChange={(e) => setJoint(j, Number(e.target.value))} aria-label={j} />
                <span className="jv mono">{Math.round(st.joints[j] ?? 0)}°</span>
              </label>
            ))}
            <div className="row">
              {Object.keys(st.poses).map((p) => (
                <button key={p} className="btn sm" onClick={() => act(api.arm.goto(p))} title={JSON.stringify(st.poses[p])}>{p}</button>
              ))}
              <button className="btn sm" onClick={() => act(api.arm.stop())}>stop</button>
            </div>
            <div className="row">
              <input className="input sm" placeholder="save current as pose…" value={poseName} onChange={(e) => setPoseName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && poseName.trim()) act(api.arm.savePose(poseName.trim()), () => setPoseName("")); }} />
            </div>
          </div>
        </div>
      </div>
      <div className="arm-bottom">
        <div className="arm-card">
          <div className="row">
            <b>Give it a task</b>
            <span className="grow" />
            <div className="seg" role="group" aria-label="Planner">
              {(["skills", "joints"] as const).map((m) => <button key={m} className={`btn sm${mode === m ? " primary" : ""}`} onClick={() => setMode(m)} title={MODE_LABEL[m]}>{m === "skills" ? "skill program" : "joint angles"}</button>)}
            </div>
          </div>
          <div className="row">
            <input className="input" value={instr} onChange={(e) => setInstr(e.target.value)} placeholder="put the red block in the bin" />
            <button className="btn sm" disabled={planning} onClick={() => { setPlanning(true); api.arm.plan(instr, false, mode).then(setPlan).catch((e) => toast(errorMessage(e), "error")).finally(() => setPlanning(false)); }}>{planning ? "planning…" : "Plan"}</button>
            <button className="btn sm primary" disabled={planning || offline} onClick={() => { setPlanning(true); setTrail([]); api.arm.plan(instr, true, mode).then((p) => { setPlan(p); if (!p.executed && !p.refused) toast("Not run: the simulator rejected the plan", "error"); void load(); }).catch((e) => toast(errorMessage(e), "error")).finally(() => setPlanning(false)); }}>Plan and run</button>
          </div>
          <div className="row muted small">
            <span>scene</span>
            <input className="input sm arm-seed" type="number" min={1} value={seed} onChange={(e) => setSeed(Number(e.target.value) || 1)} aria-label="Scene seed" />
            <button className="btn sm" onClick={() => act(api.arm.resetWorld(seed, false), () => setPlan(null))}>reset scene</button>
            <button className="btn sm" onClick={() => act(api.arm.resetWorld(seed, true), () => setPlan(null))} title="adds a purple block 47 cm out, beyond reach">+ unreachable block</button>
            <span className="grow" />
            <span>{st.planner}</span>
          </div>
          {plan && (
            <div className="arm-plan">
              <div className="row">
                <span className={`verdict ${plan.refused ? "warn" : plan.ok ? "ok" : "bad"}`}>{plan.refused ? "refused" : plan.ok ? "verified by the simulator" : "rejected by the simulator"}</span>
                <span className="muted small">{plan.executed ? "running on the arm" : plan.ok && !plan.refused ? "not run" : "nothing moved"}</span>
              </div>
              {plan.goal && plan.goal.length > 0 && (
                <div className="small"><span className="muted">declared goal </span><span className="mono">{plan.goal.map((g) => `${g.pred}(${g.obj}${g.target ? `, ${g.target}` : ""}${g.xy ? `, ${g.xy.map((v) => v.toFixed(2)).join(", ")}` : ""})`).join(" ∧ ")}</span></div>
              )}
              <div className="arm-attempts">
                {plan.attempts.map((a, i) => (
                  <div key={i} className={`att ${a.ok ? "ok" : "bad"}`}>
                    <span className="mono small">#{i + 1} {a.tier}</span>
                    <span className="mono small grow">{a.model}</span>
                    <span className="muted small">{a.seconds}s · {a.issues.filter((x) => x.level === "error").length} errors</span>
                  </div>
                ))}
              </div>
              {plan.issues.length > 0 && (
                <ul className="arm-issues">
                  {plan.issues.map((x, i) => <li key={i} className={x.level}><span className="mono">step {x.step}</span> {x.msg}</li>)}
                </ul>
              )}
              {plan.program && plan.program.length > 0 && (
                <details>
                  <summary className="muted small">program ({plan.program.length} steps)</summary>
                  <pre className="arm-prog">{plan.program.map((op, i) => `${String(i + 1).padStart(2)} ${JSON.stringify(op)}`).join("\n")}</pre>
                </details>
              )}
              <details>
                <summary className="muted small">{plan.waypoints.length} joint waypoints</summary>
                {plan.waypoints.map((w, i) => (
                  <button key={i} className="wp" onClick={() => act(api.arm.joints(w.joints, w.seconds))} title="move to this waypoint">
                    <span className="mono muted">{i + 1}</span> <b>{w.name || `waypoint ${i + 1}`}</b>
                    <span className="mono small">{st.joint_names.map((j) => `${LABELS[j]} ${Math.round(w.joints[j])}`).join(" · ")}</span>
                  </button>
                ))}
              </details>
              {plan.note && <div className="muted small">{plan.note}</div>}
            </div>
          )}
          {offline && <div className="muted small">Start the simulator or connect the arm to run plans.</div>}
        </div>
        <div className="arm-card">
          <div className="row">
            <b>Episodes</b>
            <span className="muted small">LeRobot-style frames: joints at {10} Hz with the task text</span>
            <span className="grow" />
            {st.recording ? (
              <button className="btn sm danger" onClick={() => act(api.arm.recordStop(), () => api.arm.episodes().then(setEps))}>Stop recording</button>
            ) : (
              <>
                <input className="input sm" placeholder="task, e.g. pick up the red block" value={task} onChange={(e) => setTask(e.target.value)} />
                <button className="btn sm" disabled={offline} onClick={() => { setTrail([]); act(api.arm.recordStart(task)); }}>Record</button>
              </>
            )}
          </div>
          {eps.length === 0 ? (
            <div className="muted small">No episodes yet. Record while you move the sliders, run a plan, or teleoperate; play one back to see the arm repeat it.</div>
          ) : (
            <div className="arm-episodes">
              {eps.map((e) => (
                <div key={e.id} className="ep">
                  <span className="mono small">{e.id}</span>
                  <span className="grow">{e.task || <span className="muted">no task text</span>}</span>
                  <span className="muted small">{e.frames} frames{e.seconds ? ` · ${e.seconds}s` : ""} · {e.mode}</span>
                  <button className="btn sm" disabled={offline} onClick={() => { setTrail([]); act(api.arm.play(e.id)); }}>play</button>
                  <button className="btn sm" onClick={() => act(api.arm.removeEpisode(e.id), () => setEps((l) => l.filter((x) => x.id !== e.id)))}>delete</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="arm-card">
        <div className="row">
          <b>Train the planner</b>
          <span className="muted small">LoRA on simulator-verified programs for Nemotron 3 Nano 4B, on the 5090; then it becomes the planner</span>
          <span className="grow" />
          <label className="muted small arm-check"><input type="checkbox" checked={trainSafety} onChange={(e) => setTrainSafety(e.target.checked)} /> safety examples</label>
          <span className="muted small">seed</span>
          <input className="input sm arm-seed" type="number" min={1} value={trainSeed} onChange={(e) => setTrainSeed(Number(e.target.value) || 1)} aria-label="Training seed" />
          <button className="btn sm primary" disabled={train?.status === "running" || train?.status === "queued"}
            onClick={() => act(api.arm.trainStart(trainSeed, trainSafety), () => api.arm.train().then(setTrain))}>
            {train?.status === "running" || train?.status === "queued" ? "training…" : "Train on the 5090"}
          </button>
        </div>
        {train?.run && (
          <div className="row muted small">
            <span>run {train.run} · {train.status}</span>
            {train.metrics && train.metrics.length > 0 && <span className="mono">step {train.metrics[train.metrics.length - 1].step} · loss {train.metrics[train.metrics.length - 1].loss.toFixed(3)}</span>}
            <button className="lnk" onClick={() => navigate({ kind: "lab", run: train.run! })}>log</button>
          </div>
        )}
        {train?.result ? (
          <div className="arm-bench">
            {([
              ["plans that work on new tables", train.result.base.plan_success, train.result.tuned.plan_success, true],
              ["refuses unsafe arm requests (for a safety reason)", null, train.result.tuned.unsafe_arm_safe_refusal ?? null, true],
              ["refuses harmful chat requests", train.result.base.advbench_refusal, train.result.tuned.advbench_refusal, true],
              ["refuses ordinary tasks that mention people", null, train.result.tuned.benign_people_refusal ?? null, false],
            ] as [string, number | null, number | null, boolean][]).map(([label, before, after, upGood]) => (
              <div key={label} className={`bm${label.startsWith("plans") ? " hero" : ""}`}>
                <div className="big">{pct(after ?? undefined)}</div>
                <div className="small">{label}</div>
                <div className="muted small">{before != null ? `base model ${pct(before)}` : upGood ? "higher is better" : "lower is better"}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="muted small">Trains one seed (about 2 minutes), then scores it on held-out tables 31-40 and on refusal checks (about 15 minutes); the planner switches to the new adapter when it finishes.</div>
        )}
      </div>
      <div className="arm-card">
        <div className="row">
          <b>Benchmark</b>
          <span className="muted small">{bench?.n_tasks ?? 21} tabletop tasks with hidden goals, scored by the simulator (never by a model)</span>
          <span className="grow" />
          {bench?.status === "running" && <span className="muted small">{bench.rows?.length ?? 0}/{bench.total}</span>}
          <button className="btn sm primary" disabled={bench?.status === "running"} onClick={() => act(api.arm.benchStart(), setBench)}>{bench?.status === "running" ? "running…" : "Run benchmark"}</button>
        </div>
        {bench?.summary ? (
          <>
            <div className="arm-bench">
              {(bench.modes ?? []).map((m) => {
                const r = bench.summary?.[m];
                return (
                  <div key={m} className={`bm${m === "skills+repair" ? " hero" : ""}`}>
                    <div className="big">{pct(r?.success)}</div>
                    <div className="small">{MODE_LABEL[m] ?? m}</div>
                    <div className="muted small">{r ? `valid ${pct(r.valid)} · ${r.calls} calls · ${r.seconds}s` : "waiting"}</div>
                  </div>
                );
              })}
            </div>
            <div className="muted small">Success = the hidden goal holds in the simulated scene after the plan runs (a plan the simulator rejects never runs). Valid = the simulator found no errors. {bench.planner}{bench.finished ? ` · finished ${bench.finished.replace("T", " ")}` : ""}</div>
            <details>
              <summary className="muted small">per task</summary>
              <table className="arm-grid">
                <thead><tr><th>task</th>{(bench.modes ?? []).map((m) => <th key={m}>{m}</th>)}</tr></thead>
                <tbody>
                  {[...new Set((bench.rows ?? []).map((r) => r.task))].sort().map((t) => (
                    <tr key={t}>
                      <td className="mono">{t}</td>
                      {(bench.modes ?? []).map((m) => {
                        const r = bench.rows?.find((x) => x.task === t && x.mode === m);
                        return <td key={m} className={r ? (r.success ? "ok" : "bad") : ""} title={r ? [...r.fails, ...r.errors].join("\n") || "ok" : "pending"}>{r ? (r.success ? "✓" : r.refused ? "refused" : "✗") : "·"}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </>
        ) : (
          <div className="muted small">Four planners on the same tasks (bin, stack, shift, two blocks, nearest, swap, and an out-of-reach block that should be refused): the oracle ceiling, the model writing joint angles, one skill program, and skill programs repaired from simulator feedback.</div>
        )}
      </div>
    </div>
  );
}
