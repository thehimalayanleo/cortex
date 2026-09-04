/**
 * The Galaxy: the whole library as a strategy-game map. Zoomed out you see universes (territories); zoom in and
 * solar systems (cities) appear, then districts (blocks of 5-10 papers), then the papers themselves with their titles,
 * so a PDF can be chosen from the map. Click a territory, city, or district to fly into it; click a paper to open it.
 * Built from lab/recipes/galaxy_index.py (bge-small + DBSCAN + k-means districts + circle packing).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Galaxy, GalaxyPaper } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const PALETTE = ["#4db39d", "#c2632b", "#3b6fb6", "#8a5fb0", "#c05a9a", "#c9962b", "#3fa46a", "#d0453f", "#5aa0c8", "#a3b34d", "#b06f4d", "#6d8bcf"];
type Level = "universe" | "system" | "district";
interface Circle { level: Level; id: number; label: string; size: number; cx: number; cy: number; r: number; color: string; parent?: Circle }

export function GalaxyView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [g, setG] = useState<Galaxy | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [hover, setHover] = useState<GalaxyPaper | null>(null);
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const [open, setOpen] = useState<Set<string>>(new Set()); // expanded tree nodes "u:1", "s:3", "d:12"
  const [here, setHere] = useState<Circle[]>([]); // breadcrumb: circles containing the view centre
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef({ x: 0, y: 0, zoom: 1, drag: null as null | [number, number], moved: false, rx: 0.4, ry: 0.5, anim: 0 });
  const projRef = useRef<{ x: number; y: number; p: GalaxyPaper }[]>([]);
  const circlesRef = useRef<Circle[]>([]);

  const load = useCallback(async () => {
    try {
      setG(await api.galaxy.get());
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load, refresh]);
  useEffect(() => {
    if (!g?.building) return;
    const t = window.setInterval(() => void load(), 5000);
    return () => window.clearInterval(t);
  }, [g?.building, load]);

  const ql = q.trim().toLowerCase();
  const matches = useMemo(() => {
    if (!g) return new Set<string>();
    if (!ql) return new Set(g.papers.map((p) => p.id));
    return new Set(g.papers.filter((p) => `${p.title} ${p.authors ?? ""} ${(p.topics ?? []).join(" ")} ${p.year ?? ""}`.toLowerCase().includes(ql)).map((p) => p.id));
  }, [g, ql]);
  const color = (cid: number) => (cid < 0 ? "#6b7570" : PALETTE[cid % PALETTE.length]);
  const clusterOf = useMemo(() => new Map((g?.clusters ?? []).map((c) => [c.id, c])), [g]);
  const hoodOf = useMemo(() => new Map((g?.hoods ?? []).map((h) => [h.id, h])), [g]);

  // circles of every level, with parents, in world units ([-1, 1])
  const circles = useMemo<Circle[]>(() => {
    if (!g) return [];
    const out: Circle[] = [];
    const uni = new Map<number, Circle>();
    for (const u of g.universes) {
      if (u.r == null || u.cx == null || u.cy == null) continue;
      const c: Circle = { level: "universe", id: u.id, label: u.label, size: u.size, cx: u.cx, cy: u.cy, r: u.r, color: "#8fa0aa" };
      uni.set(u.id, c);
      out.push(c);
    }
    const sys = new Map<number, Circle>();
    for (const s of g.clusters) {
      if (s.id < 0 || !s.r) continue;
      const c: Circle = { level: "system", id: s.id, label: s.label, size: s.size, cx: s.cx, cy: s.cy, r: s.r, color: color(s.id), parent: uni.get(s.universe) };
      sys.set(s.id, c);
      out.push(c);
    }
    for (const h of g.hoods ?? []) {
      out.push({ level: "district", id: h.id, label: h.label, size: h.size, cx: h.cx, cy: h.cy, r: h.r, color: color(h.cluster), parent: sys.get(h.cluster) });
    }
    circlesRef.current = out;
    return out;
  }, [g]);
  const hasMap = circles.length > 0;
  // first look: fit the whole map to the stage
  const fitted = useRef(false);
  useEffect(() => {
    if (!hasMap || fitted.current || !canvasRef.current) return;
    fitted.current = true;
    const ext = Math.max(...circles.filter((c) => c.level === "universe").map((c) => Math.hypot(c.cx, c.cy) + c.r), 0.5);
    const v = view.current;
    v.zoom = Math.max(0.5, Math.min(4, 0.95 / ext));
    v.x = 0;
    v.y = 0;
    redraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasMap, circles]);

  const geometry = () => {
    const c = canvasRef.current!;
    const W = c.clientWidth, H = c.clientHeight, v = view.current;
    const base = Math.min(W, H) * 0.45 * v.zoom;
    return { W, H, base, toScreen: (x: number, y: number) => [W / 2 + v.x + x * base, H / 2 + v.y - y * base] as [number, number], toWorld: (sx: number, sy: number) => [(sx - W / 2 - v.x) / base, -(sy - H / 2 - v.y) / base] as [number, number] };
  };

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !g) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { W, H, base, toScreen } = geometry();
      if (c.width !== W * dpr || c.height !== H * dpr) {
        c.width = W * dpr;
        c.height = H * dpr;
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const v = view.current;
      const rot = (p: GalaxyPaper) => {
        if (mode === "2d") return [p.x, p.y, 0];
        const x = p.x3, y = p.y3, z = p.z3;
        const y1 = y * Math.cos(v.rx) - z * Math.sin(v.rx), z1 = y * Math.sin(v.rx) + z * Math.cos(v.rx);
        const x2 = x * Math.cos(v.ry) + z1 * Math.sin(v.ry), z2 = -x * Math.sin(v.ry) + z1 * Math.cos(v.ry);
        return [x2, y1, z2];
      };
      const proj = (p: GalaxyPaper) => {
        const [x, y, z] = rot(p);
        const f = mode === "2d" ? 1 : 1 / (1.9 - z * 0.6);
        return { x: W / 2 + v.x + x * base * f, y: H / 2 + v.y - y * base * f, z, f };
      };
      const onScreen = (x: number, y: number, r: number) => x + r > -20 && x - r < W + 20 && y + r > -20 && y - r < H + 20;
      const shown = { universe: 0, system: 0, district: 0, paper: 0 };
      if (mode === "2d" && hasMap) {
        // territories, cities, districts: drawn big to small, labelled when big enough on screen
        for (const level of ["universe", "system", "district"] as Level[]) {
          for (const ci of circles) {
            if (ci.level !== level) continue;
            const [x, y] = toScreen(ci.cx, ci.cy);
            const r = ci.r * base;
            if (!onScreen(x, y, r)) continue;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            if (level === "universe") {
              const grad = ctx.createRadialGradient(x, y, r * 0.3, x, y, r);
              grad.addColorStop(0, "rgba(130,150,170,0.10)");
              grad.addColorStop(1, "rgba(130,150,170,0.03)");
              ctx.fillStyle = grad;
              ctx.fill();
              ctx.setLineDash([6, 6]);
              ctx.strokeStyle = "rgba(150,170,185,0.35)";
              ctx.lineWidth = 1;
              ctx.stroke();
              ctx.setLineDash([]);
              if (r > 34) {
                shown.universe++;
                const fs = Math.max(11, Math.min(22, r / 8));
                ctx.font = `700 ${fs}px IBM Plex Mono, monospace`;
                ctx.fillStyle = "rgba(175,190,200,0.75)";
                const t = ci.label.toUpperCase();
                ctx.fillText(t, x - ctx.measureText(t).width / 2, y - r - 6);
              }
            } else if (level === "system") {
              ctx.fillStyle = ci.color;
              ctx.globalAlpha = 0.09;
              ctx.fill();
              ctx.globalAlpha = 0.5;
              ctx.strokeStyle = ci.color;
              ctx.lineWidth = r > 60 ? 1.5 : 1;
              ctx.stroke();
              ctx.globalAlpha = 1;
              if (r > 18) {
                shown.system++;
                const fs = Math.max(10.5, Math.min(15, r / 6));
                ctx.font = `600 ${fs}px IBM Plex Sans, sans-serif`;
                const t = r > 40 ? `${ci.label} · ${ci.size}` : ci.label.split(" · ")[0];
                const w = ctx.measureText(t).width + 10;
                const lx = x - w / 2, ly = r > 90 ? y - r + 4 : y - r - fs - 6;
                ctx.fillStyle = "rgba(8,12,10,0.7)";
                ctx.beginPath();
                ctx.roundRect(lx, ly, w, fs + 6, 5);
                ctx.fill();
                ctx.fillStyle = ci.color;
                ctx.fillText(t, lx + 5, ly + fs + 1);
              }
            } else if (r > 26) {
              ctx.strokeStyle = ci.color;
              ctx.globalAlpha = 0.28;
              ctx.setLineDash([3, 4]);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.globalAlpha = 1;
              if (r > 60) {
                shown.district++;
                ctx.font = "500 10.5px IBM Plex Mono, monospace";
                const t = ci.label;
                const w = ctx.measureText(t).width + 8;
                ctx.fillStyle = "rgba(8,12,10,0.6)";
                ctx.beginPath();
                ctx.roundRect(x - w / 2, y + r - 16, w, 15, 4);
                ctx.fill();
                ctx.fillStyle = "rgba(200,210,215,0.7)";
                ctx.fillText(t, x - w / 2 + 4, y + r - 5);
              }
            }
          }
        }
      }
      // planets (buildings). Titles appear only once a district is roomy on screen, and never on top of each other.
      const order = g.papers.map((p) => ({ ...proj(p), p })).sort((a, b) => a.z - b.z);
      projRef.current = order;
      const hoodPx = new Map<number, number>();
      if (mode === "2d") for (const ci of circles) if (ci.level === "district") hoodPx.set(ci.id, ci.r * base);
      const wantLabel = (p: GalaxyPaper) => mode === "2d" && (((hoodPx.get(p.hood ?? -1) ?? 0) > 105) || (!!ql && matches.size <= 40));
      const candidates: { x: number; y: number; r: number; p: GalaxyPaper; pri: number }[] = [];
      for (const t of order) {
        const p = t.p;
        if (!onScreen(t.x, t.y, 8)) continue;
        const on = matches.has(p.id);
        const r = (p.status === "read" ? 4.6 : p.status === "reading" ? 4 : 3.2) * (mode === "2d" ? 1 : t.f) * Math.min(2.2, Math.sqrt(v.zoom));
        ctx.beginPath();
        ctx.arc(t.x, t.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color(p.cluster);
        ctx.globalAlpha = on ? (p.cluster < 0 ? 0.55 : 0.95) : 0.08;
        ctx.fill();
        if (on && p.status === "read") {
          ctx.strokeStyle = "rgba(255,255,255,0.75)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        if (on && wantLabel(p)) candidates.push({ x: t.x, y: t.y, r, p, pri: (p.status === "read" ? 0 : p.status === "reading" ? 1 : 2) + Math.hypot(t.x - W / 2, t.y - H / 2) / Math.max(W, H) });
      }
      if (candidates.length) {
        const fs = v.zoom >= 8 ? 12.5 : 11.5;
        ctx.font = `${fs}px IBM Plex Sans, sans-serif`;
        const placed: [number, number, number, number][] = [];
        const free = (x: number, y: number, w: number, h: number) => !placed.some(([px, py, pw, ph]) => x < px + pw && x + w > px && y < py + ph && y + h > py);
        candidates.sort((a, b) => a.pri - b.pri);
        for (const c of candidates) {
          const max = v.zoom >= 8 ? 60 : 36;
          const t = c.p.title.length > max ? c.p.title.slice(0, max - 2) + "…" : c.p.title;
          const w = ctx.measureText(t).width + 8, h = fs + 5;
          // try right, left, below, above
          const spots: [number, number][] = [[c.x + c.r + 3, c.y - h / 2], [c.x - c.r - 3 - w, c.y - h / 2], [c.x - w / 2, c.y + c.r + 2], [c.x - w / 2, c.y - c.r - 2 - h]];
          const spot = spots.find(([x, y]) => x >= 0 && x + w <= W && y >= 0 && y + h <= H && free(x, y, w, h));
          if (!spot) continue;
          const [x, y] = spot;
          placed.push([x, y, w, h]);
          shown.paper++;
          ctx.fillStyle = "rgba(8,12,10,0.78)";
          ctx.beginPath();
          ctx.roundRect(x, y, w, h, 4);
          ctx.fill();
          ctx.fillStyle = c.p.status === "read" ? "#f1f5f2" : "rgba(220,228,224,0.9)";
          ctx.fillText(t, x + 4, y + fs);
          if (placed.length >= 220) break;
        }
      }
      if (hover) {
        const t = order.find((o) => o.p.id === hover.id);
        if (t) {
          ctx.font = "12.5px IBM Plex Sans, sans-serif";
          const txt = `${hover.title}${hover.year ? ` (${hover.year})` : ""}`;
          const w = ctx.measureText(txt).width + 12;
          const x = Math.min(W - w - 4, t.x + 10), y = Math.max(16, t.y - 10);
          ctx.fillStyle = "rgba(10,14,12,0.94)";
          ctx.fillRect(x, y - 13, w, 20);
          ctx.strokeStyle = color(hover.cluster);
          ctx.strokeRect(x, y - 13, w, 20);
          ctx.fillStyle = "#e6ede9";
          ctx.fillText(txt, x + 6, y + 2);
        }
      }
      // scale hint: what the current zoom shows
      ctx.font = "10.5px IBM Plex Mono, monospace";
      ctx.fillStyle = "rgba(160,175,180,0.7)";
      const hint = mode === "3d" ? "3D · drag to rotate, shift-drag to pan" : v.zoom < 1.6 ? "territories · scroll or click a universe to zoom in" : v.zoom < 3.2 ? "cities · click a system or district to fly in" : "districts · zoom into one to read its titles; hover any planet";
      ctx.fillText(hint, 10, H - 10);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    (c as unknown as { __draw?: () => void }).__draw = draw;
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [g, matches, hover, mode, circles, ql]);

  const redraw = () => (canvasRef.current as unknown as { __draw?: () => void } | null)?.__draw?.();

  // where is the view centre? (breadcrumb)
  const updateHere = useCallback(() => {
    if (!canvasRef.current) return;
    const { W, H, toWorld } = geometry();
    const [wx, wy] = toWorld(W / 2, H / 2);
    const chain: Circle[] = [];
    for (const level of ["universe", "system", "district"] as Level[]) {
      const hit = circlesRef.current.filter((ci) => ci.level === level && Math.hypot(ci.cx - wx, ci.cy - wy) <= ci.r && (!chain.length || ci.parent === chain[chain.length - 1]));
      if (!hit.length) break;
      chain.push(hit.sort((a, b) => Math.hypot(a.cx - wx, a.cy - wy) - Math.hypot(b.cx - wx, b.cy - wy))[0]);
    }
    setHere((old) => (old.length === chain.length && old.every((c, i) => c === chain[i]) ? old : chain));
  }, []);

  const flyTo = useCallback((cx: number, cy: number, r: number) => {
    const c = canvasRef.current;
    if (!c) return;
    const W = c.clientWidth, H = c.clientHeight, m = Math.min(W, H);
    const v = view.current;
    const zoom = Math.max(0.5, Math.min(40, (m * 0.42) / (r * m * 0.45)));
    const base = m * 0.45 * zoom;
    const target = { x: -cx * base, y: cy * base, zoom };
    const from = { x: v.x, y: v.y, zoom: v.zoom };
    const t0 = performance.now(), dur = 520, token = ++v.anim;
    const step = (now: number) => {
      if (v.anim !== token) return;
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      // interpolate zoom geometrically so the motion feels even
      v.zoom = from.zoom * Math.pow(target.zoom / from.zoom, e);
      const b = m * 0.45 * v.zoom;
      const fx = (from.x / (m * 0.45 * from.zoom)), fy = (from.y / (m * 0.45 * from.zoom));
      v.x = (fx + (-cx - fx) * e) * b;
      v.y = (fy + (cy - fy) * e) * b;
      redraw();
      if (k < 1) requestAnimationFrame(step);
      else updateHere();
    };
    requestAnimationFrame(step);
  }, [updateHere]);
  const flyCircle = (ci: Circle) => {
    flyTo(ci.cx, ci.cy, ci.r * 1.08);
    setOpen((o) => {
      const n = new Set(o);
      for (let c: Circle | undefined = ci; c; c = c.parent) n.add(`${c.level[0]}:${c.id}`);
      return n;
    });
  };
  const flyHome = () => {
    const ext = Math.max(...circles.filter((c) => c.level === "universe").map((c) => Math.hypot(c.cx, c.cy) + c.r), 0.5);
    flyTo(0, 0, ext / 0.95);
  };

  const pos = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as [number, number];
  };
  const pick = (mx: number, my: number) => {
    let best: GalaxyPaper | null = null, bd = 9;
    for (const o of projRef.current) {
      const d = Math.hypot(o.x - mx, o.y - my);
      if (d < bd && matches.has(o.p.id)) {
        bd = d;
        best = o.p;
      }
    }
    return best;
  };
  /** The deepest circle under the pointer that does not already fill the screen: the thing to fly into next. */
  const pickCircle = (mx: number, my: number): Circle | null => {
    if (mode !== "2d") return null;
    const { W, H, base, toWorld } = geometry();
    const [wx, wy] = toWorld(mx, my);
    const limit = Math.min(W, H) * 0.4;
    let best: Circle | null = null;
    for (const level of ["universe", "system", "district"] as Level[]) {
      const hit = circlesRef.current.filter((ci) => ci.level === level && Math.hypot(ci.cx - wx, ci.cy - wy) <= ci.r && (!best || ci.parent === best));
      if (!hit.length) break;
      const ci = hit.sort((a, b) => Math.hypot(a.cx - wx, a.cy - wy) - Math.hypot(b.cx - wx, b.cy - wy))[0];
      if (ci.r * base >= limit) {
        best = ci;
        continue;
      }
      return ci;
    }
    return null;
  };

  if (err) return <EmptyState title="Could not load the galaxy" hint={err} />;
  if (!g) return <EmptyState title="Loading the galaxy" />;
  const systems = g.clusters.filter((c) => c.id >= 0).sort((a, b) => b.size - a.size);
  const hoods = g.hoods ?? [];
  const isOpen = (k: string) => open.has(k) || (!!ql && matches.size <= 80);
  const toggle = (k: string) => setOpen((o) => {
    const n = new Set(o);
    if (n.has(k)) n.delete(k);
    else n.add(k);
    return n;
  });
  const circleFor = (level: Level, id: number) => circles.find((c) => c.level === level && c.id === id);
  const papersIn = (pred: (p: GalaxyPaper) => boolean) => g.papers.filter((p) => pred(p) && matches.has(p.id));
  const drifting = g.papers.filter((p) => p.cluster < 0);

  return (
    <div className="galaxy">
      <div className="galaxy-bar">
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the map: title, author, topic, year…" aria-label="Search" spellCheck={false} />
        <div className="seg">
          <button className={mode === "2d" ? "on" : ""} onClick={() => setMode("2d")}>Map</button>
          <button className={mode === "3d" ? "on" : ""} onClick={() => setMode("3d")}>3D</button>
        </div>
        <span className="muted small">
          {g.n ? `${g.n} papers · ${hoods.length} districts · ${systems.length} systems · ${g.universes.length} universes` : "no index yet"}
          {ql ? ` · ${matches.size} match` : ""}
          {g.stale ? " · library changed since" : ""}
          {g.building ? " · rebuilding…" : ""}
        </span>
        <span className="grow" />
        <button className="btn sm" disabled={!!g.building} onClick={() => api.galaxy.rebuild().then((r) => { toast("Rebuilding the galaxy"); navigate({ kind: "lab", run: r.id }); }).catch((e) => toast(errorMessage(e), "error"))}>
          {g.building ? "Rebuilding…" : "Rebuild"}
        </button>
      </div>
      <div className="galaxy-body">
        <div className="galaxy-stage">
          <canvas
            ref={canvasRef}
            className="galaxy-canvas"
            onPointerDown={(e) => {
              view.current.drag = pos(e);
              view.current.moved = false;
              view.current.anim++;
              try { (e.target as HTMLElement).setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
            }}
            onPointerUp={(e) => {
              const [mx, my] = pos(e);
              const d = view.current.drag;
              view.current.drag = null;
              if (d && Math.hypot(mx - d[0], my - d[1]) < 4) {
                const p = pick(mx, my);
                if (p) {
                  navigate({ kind: "paper", id: p.id });
                  return;
                }
                const ci = pickCircle(mx, my);
                if (ci) flyCircle(ci);
              } else updateHere();
            }}
            onDoubleClick={() => flyHome()}
            onPointerMove={(e) => {
              const [mx, my] = pos(e);
              const v = view.current;
              if (v.drag) {
                if (mode === "2d" || e.shiftKey) {
                  v.x += mx - v.drag[0];
                  v.y += my - v.drag[1];
                } else {
                  v.ry += (mx - v.drag[0]) * 0.008;
                  v.rx += (my - v.drag[1]) * 0.008;
                }
                v.drag = [mx, my];
                redraw();
              } else {
                const p = pick(mx, my);
                if (p?.id !== hover?.id) setHover(p);
              }
            }}
            onWheel={(e) => {
              e.preventDefault();
              const v = view.current;
              v.anim++;
              const [mx, my] = pos(e as unknown as React.PointerEvent);
              const { W, H } = geometry();
              const k = e.deltaY < 0 ? 1.15 : 0.87;
              const nz = Math.max(0.5, Math.min(40, v.zoom * k));
              const f = nz / v.zoom;
              // zoom about the cursor
              v.x = (mx - W / 2) - ((mx - W / 2) - v.x) * f;
              v.y = (my - H / 2) - ((my - H / 2) - v.y) * f;
              v.zoom = nz;
              redraw();
              updateHere();
            }}
          />
          {mode === "2d" && hasMap && (
            <div className="galaxy-crumbs">
              <button className="crumb" onClick={flyHome}>whole map</button>
              {here.map((c) => (
                <button key={`${c.level}:${c.id}`} className="crumb" style={{ borderColor: c.color }} onClick={() => flyCircle(c)}>
                  <span className="muted">{c.level}</span> {c.label} <span className="muted">{c.size}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <aside className="galaxy-side">
          {hover ? (
            <div className="galaxy-card">
              <b>{hover.title}</b>
              <div className="muted small">{hover.authors}{hover.year ? ` · ${hover.year}` : ""} · {hover.status}</div>
              <div className="small">
                {hover.cluster >= 0 ? `${clusterOf.get(hover.cluster)?.label}` : "drifting (no close neighbours)"}
                {hover.hood != null && hoodOf.get(hover.hood) ? ` › ${hoodOf.get(hover.hood)?.label}` : ""}
              </div>
              {hover.near?.length ? (
                <div className="small near">
                  nearest planets
                  <ul>
                    {hover.near.slice(0, 4).map(([id, sim]) => {
                      const n = g.papers.find((p) => p.id === id);
                      return (
                        <li key={id}>
                          <button className="lnk" onClick={() => navigate({ kind: "paper", id })}>{n?.title.slice(0, 60) ?? id}</button> <span className="muted">{sim.toFixed(2)}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="galaxy-card muted small">
              Territories hold cities, cities hold districts, districts hold papers. Click a territory, city, or district on the map or in the list to fly into it; scroll to zoom; double-click for the whole map. Bigger, ringed planets are papers you have read.
            </div>
          )}
          <div className="galaxy-tree">
            {g.universes.map((u) => {
              const uk = `u:${u.id}`;
              const uc = circleFor("universe", u.id);
              const usys = systems.filter((c) => c.universe === u.id);
              const un = papersIn((p) => p.universe === u.id).length;
              if (ql && !un) return null;
              return (
                <div key={u.id} className="tree-u">
                  <div className="tree-row u" onClick={() => { toggle(uk); if (uc) flyCircle(uc); }}>
                    <span className="tw">{isOpen(uk) ? "▾" : "▸"}</span>
                    <span className="lab">{u.label}</span>
                    <span className="muted">{ql ? `${un}/${u.size}` : u.size}</span>
                  </div>
                  {isOpen(uk) && usys.map((s) => {
                    const sk = `s:${s.id}`;
                    const sc = circleFor("system", s.id);
                    const shoods = hoods.filter((h) => h.cluster === s.id).sort((a, b) => b.size - a.size);
                    const sn = papersIn((p) => p.cluster === s.id).length;
                    if (ql && !sn) return null;
                    return (
                      <div key={s.id} className="tree-s">
                        <div className="tree-row s" style={{ borderLeftColor: color(s.id) }} onClick={() => { toggle(sk); if (sc) flyCircle(sc); }}>
                          <span className="tw">{isOpen(sk) ? "▾" : "▸"}</span>
                          <span className="lab">{s.label}</span>
                          <span className="muted">{ql ? `${sn}/${s.size}` : s.size}</span>
                        </div>
                        {isOpen(sk) && shoods.map((h) => {
                          const hk = `d:${h.id}`;
                          const hc = circleFor("district", h.id);
                          const hp = papersIn((p) => p.hood === h.id);
                          if (ql && !hp.length) return null;
                          return (
                            <div key={h.id} className="tree-d">
                              <div className="tree-row d" onClick={() => { toggle(hk); if (hc) flyCircle(hc); }}>
                                <span className="tw">{isOpen(hk) ? "▾" : "▸"}</span>
                                <span className="lab">{h.label}</span>
                                <span className="muted">{hp.length}</span>
                              </div>
                              {isOpen(hk) && hp.map((p) => (
                                <button key={p.id} className={`tree-p ${p.status ?? ""}`} onClick={() => navigate({ kind: "paper", id: p.id })} onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} title={p.title}>
                                  <span className="dot" style={{ background: color(p.cluster) }} />
                                  {p.title}
                                  {p.year ? <span className="muted"> {p.year}</span> : null}
                                </button>
                              ))}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              );
            })}
            {drifting.length > 0 && <div className="muted small">{drifting.length} papers drift on the rim (no close neighbours yet).</div>}
          </div>
        </aside>
      </div>
    </div>
  );
}
