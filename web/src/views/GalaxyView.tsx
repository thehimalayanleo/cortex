/**
 * The Galaxy: the whole library as a map. Universes are broad areas (halos), solar systems are tight clusters
 * (labelled groups), papers are planets. Search dims everything that does not match; click a planet to open it.
 * Built from lab/recipes/galaxy_index.py (bge-small + DBSCAN), rebuilt when the library changes.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Galaxy, GalaxyPaper } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const PALETTE = ["#4db39d", "#c2632b", "#3b6fb6", "#8a5fb0", "#c05a9a", "#c9962b", "#3fa46a", "#d0453f", "#5aa0c8", "#a3b34d", "#b06f4d", "#6d8bcf"];

export function GalaxyView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [g, setG] = useState<Galaxy | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [focus, setFocus] = useState<number | null>(null); // cluster id
  const [hover, setHover] = useState<GalaxyPaper | null>(null);
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef({ x: 0, y: 0, zoom: 1, drag: null as null | [number, number], rx: 0.4, ry: 0.5 });
  const projRef = useRef<{ x: number; y: number; p: GalaxyPaper }[]>([]);

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
  const clusterColor = (cid: number) => (cid < 0 ? "#6b7570" : PALETTE[cid % PALETTE.length]);
  const clusterOf = useMemo(() => new Map((g?.clusters ?? []).map((c) => [c.id, c])), [g]);

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !g) return;
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
      const v = view.current;
      const base = Math.min(W, H) * 0.45 * v.zoom;
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
      // universes: soft halos around their systems' centroids
      const byUniverse = new Map<number, GalaxyPaper[]>();
      for (const p of g.papers) if (p.universe >= 0) byUniverse.set(p.universe, [...(byUniverse.get(p.universe) ?? []), p]);
      ctx.font = "600 12px IBM Plex Mono, monospace";
      for (const u of g.universes) {
        const ps = byUniverse.get(u.id) ?? [];
        if (!ps.length) continue;
        const pts = ps.map(proj);
        const cx = pts.reduce((s, t) => s + t.x, 0) / pts.length, cy = pts.reduce((s, t) => s + t.y, 0) / pts.length;
        const r = Math.max(40, Math.sqrt(pts.reduce((s, t) => s + (t.x - cx) ** 2 + (t.y - cy) ** 2, 0) / pts.length) * 1.6);
        const grad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
        grad.addColorStop(0, "rgba(120,140,160,0.10)");
        grad.addColorStop(1, "rgba(120,140,160,0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(160,175,185,0.55)";
        ctx.fillText(u.label.toUpperCase(), cx - ctx.measureText(u.label.toUpperCase()).width / 2, cy - r - 6);
      }
      // solar systems: a faint ring + label
      const bySystem = new Map<number, GalaxyPaper[]>();
      for (const p of g.papers) if (p.cluster >= 0) bySystem.set(p.cluster, [...(bySystem.get(p.cluster) ?? []), p]);
      ctx.font = "11.5px IBM Plex Sans, sans-serif";
      for (const cl of g.clusters) {
        if (cl.id < 0) continue;
        const ps = bySystem.get(cl.id) ?? [];
        if (!ps.length) continue;
        const pts = ps.map(proj);
        const cx = pts.reduce((s, t) => s + t.x, 0) / pts.length, cy = pts.reduce((s, t) => s + t.y, 0) / pts.length;
        const r = Math.max(14, Math.sqrt(pts.reduce((s, t) => s + (t.x - cx) ** 2 + (t.y - cy) ** 2, 0) / pts.length) * 1.5 + 8);
        const col = clusterColor(cl.id);
        const dim = focus != null && focus !== cl.id;
        ctx.strokeStyle = col;
        ctx.globalAlpha = dim ? 0.12 : 0.35;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = dim ? 0.25 : 0.9;
        ctx.fillStyle = col;
        const label = `${cl.label} · ${cl.size}`;
        ctx.fillText(label, cx - ctx.measureText(label).width / 2, cy + r + 13);
        ctx.globalAlpha = 1;
      }
      // planets
      const order = g.papers.map((p) => ({ ...proj(p), p })).sort((a, b) => a.z - b.z);
      projRef.current = order;
      for (const t of order) {
        const p = t.p;
        const on = matches.has(p.id) && (focus == null || p.cluster === focus);
        const r = (p.status === "read" ? 4.2 : p.status === "reading" ? 3.8 : 3) * (mode === "2d" ? 1 : t.f) * Math.min(1.6, Math.sqrt(v.zoom));
        ctx.beginPath();
        ctx.arc(t.x, t.y, r, 0, Math.PI * 2);
        ctx.fillStyle = clusterColor(p.cluster);
        ctx.globalAlpha = on ? (p.cluster < 0 ? 0.55 : 0.95) : 0.08;
        ctx.fill();
        if (on && p.status === "read") {
          ctx.strokeStyle = "rgba(255,255,255,0.7)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if (hover) {
        const t = order.find((o) => o.p.id === hover.id);
        if (t) {
          ctx.font = "12px IBM Plex Sans, sans-serif";
          const txt = `${hover.title}${hover.year ? ` (${hover.year})` : ""}`;
          const w = ctx.measureText(txt).width + 12;
          const x = Math.min(W - w - 4, t.x + 10), y = Math.max(16, t.y - 10);
          ctx.fillStyle = "rgba(10,14,12,0.92)";
          ctx.fillRect(x, y - 13, w, 20);
          ctx.strokeStyle = clusterColor(hover.cluster);
          ctx.strokeRect(x, y - 13, w, 20);
          ctx.fillStyle = "#e6ede9";
          ctx.fillText(txt, x + 6, y + 2);
        }
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(c);
    (c as unknown as { __draw?: () => void }).__draw = draw;
    return () => ro.disconnect();
  }, [g, matches, focus, hover, mode, clusterOf]);

  const redraw = () => (canvasRef.current as unknown as { __draw?: () => void } | null)?.__draw?.();
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

  if (err) return <EmptyState title="Could not load the galaxy" hint={err} />;
  if (!g) return <EmptyState title="Loading the galaxy" />;
  const systems = g.clusters.filter((c) => c.id >= 0).sort((a, b) => b.size - a.size);
  return (
    <div className="galaxy">
      <div className="galaxy-bar">
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search planets: title, author, topic, year…" aria-label="Search" spellCheck={false} />
        <select className="select sm" value={focus ?? ""} onChange={(e) => setFocus(e.target.value === "" ? null : Number(e.target.value))} aria-label="Solar system">
          <option value="">all solar systems</option>
          {systems.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label} · {c.size}
            </option>
          ))}
        </select>
        <div className="seg">
          <button className={mode === "2d" ? "on" : ""} onClick={() => setMode("2d")}>2D</button>
          <button className={mode === "3d" ? "on" : ""} onClick={() => setMode("3d")}>3D</button>
        </div>
        <span className="muted small">
          {g.n ? `${g.n} papers · ${systems.length} solar systems · ${g.universes.length} universes · ${g.model?.replace("BAAI/", "")}` : "no index yet"}
          {g.stale ? " · library changed since" : ""}
          {g.building ? " · rebuilding…" : ""}
        </span>
        <span className="grow" />
        <button className="btn sm" disabled={!!g.building} onClick={() => api.galaxy.rebuild().then((r) => { toast("Rebuilding the galaxy"); navigate({ kind: "lab", run: r.id }); }).catch((e) => toast(errorMessage(e), "error"))}>
          {g.building ? "Rebuilding…" : "Rebuild"}
        </button>
      </div>
      <div className="galaxy-body">
        <canvas
          ref={canvasRef}
          className="galaxy-canvas"
          onPointerDown={(e) => {
            view.current.drag = pos(e);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
          }}
          onPointerUp={(e) => {
            const [mx, my] = pos(e);
            const d = view.current.drag;
            view.current.drag = null;
            if (d && Math.hypot(mx - d[0], my - d[1]) < 4) {
              const p = pick(mx, my);
              if (p) navigate({ kind: "paper", id: p.id });
            }
          }}
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
            view.current.zoom = Math.max(0.5, Math.min(8, view.current.zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
            redraw();
          }}
        />
        <aside className="galaxy-side">
          {hover ? (
            <div className="galaxy-card">
              <b>{hover.title}</b>
              <div className="muted small">{hover.authors}{hover.year ? ` · ${hover.year}` : ""} · {hover.status}</div>
              <div className="small">{hover.cluster >= 0 ? `solar system: ${clusterOf.get(hover.cluster)?.label}` : "unclustered (no close neighbours)"}</div>
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
              Hover a planet for its title and nearest neighbours; click to open. Drag to pan (2D) or rotate (3D, hold Shift to pan); scroll to zoom. Halos are universes, rings are solar systems, bigger planets are papers you have read.
            </div>
          )}
          <div className="galaxy-list">
            {g.universes.map((u) => (
              <div key={u.id} className="galaxy-universe">
                <div className="u-label">{u.label} <span className="muted">· {u.size}</span></div>
                {systems.filter((c) => c.universe === u.id).map((c) => (
                  <button key={c.id} className={`sys ${focus === c.id ? "on" : ""}`} style={{ borderLeftColor: clusterColor(c.id) }} onClick={() => setFocus(focus === c.id ? null : c.id)}>
                    {c.label} <span className="muted">{c.size}</span>
                  </button>
                ))}
              </div>
            ))}
            {g.clusters.some((c) => c.id === -1) && (
              <div className="muted small">{g.clusters.find((c) => c.id === -1)?.size} papers float between systems (no close neighbours yet).</div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
