/**
 * The Studio: a shot list, takes rendered on the GPU box, the critics' scores, and the director's verdict.
 * Same buttons for the person, the chat, a WebMCP agent, and the ADK director (cortex/agent/director).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Shot, StudioBoard, Take } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const STATUS_LABEL: Record<Shot["status"], string> = { planned: "planned", rendering: "rendering", rendered: "rendered", approved: "approved", reshoot: "reshoot" };

export function StudioView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [board, setBoard] = useState<StudioBoard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [logline, setLogline] = useState("");
  const [planning, setPlanning] = useState(false);
  const [gpu, setGpu] = useState<{ ready: boolean; host: string | null; name?: string } | null>(null);
  const [tele, setTele] = useState<{ metrics: boolean; logs: boolean; mcp: boolean; grafana_url: string | null; metrics_sent: number } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const b = await api.studio.board();
      setBoard(b);
      setLogline((l) => l || b.logline || "");
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
    api.lab.gpu().then((g) => setGpu({ ready: g.ready, host: g.host, name: g.gpu?.name })).catch(() => setGpu({ ready: false, host: null }));
    api.telemetry().then((t) => setTele({ metrics: t.metrics, logs: t.logs, mcp: t.mcp, grafana_url: t.grafana_url, metrics_sent: t.metrics_sent })).catch(() => undefined);
  }, [load, refresh]);
  // Poll while something renders.
  useEffect(() => {
    if (!board?.shots.some((s) => s.status === "rendering")) return;
    const t = window.setInterval(async () => {
      for (const s of board.shots.filter((x) => x.status === "rendering")) {
        try {
          await api.studio.refresh(s.id);
        } catch {
          /* still running */
        }
      }
      void load();
    }, 5000);
    return () => window.clearInterval(t);
  }, [board, load]);

  const plan = async () => {
    if (!logline.trim()) return;
    setPlanning(true);
    try {
      await api.studio.plan(logline.trim(), 4);
      toast("Shots planned");
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setPlanning(false);
    }
  };
  const render = async (s: Shot, smoke?: boolean) => {
    try {
      const r = await api.studio.render(s.id, { smoke });
      toast(`Rendering ${s.title} on ${r.executor}`);
      await load();
      navigate({ kind: "lab", run: r.id });
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const patch = async (s: Shot, p: Partial<Shot> & { director_note?: string }) => {
    try {
      await api.studio.update(s.id, p);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const remove = async (s: Shot) => {
    if (!window.confirm(`Delete shot "${s.title}"?`)) return;
    await api.studio.remove(s.id);
    await load();
  };
  const upload = async (f: File) => {
    try {
      const r = await api.studio.upload(f);
      toast(`Keyframe ${r.name} added`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };

  if (err) return <EmptyState title="Could not load the studio" hint={err} />;
  if (!board) return <EmptyState title="Loading the studio" />;
  return (
    <div className="studio">
      <div className="studio-head">
        <div className="studio-logline">
          <label className="muted small">Logline</label>
          <textarea value={logline} onChange={(e) => setLogline(e.target.value)} rows={2} placeholder="A kaiju of black coral rises from a storm and, for one moment, hesitates." spellCheck={false} />
          <div className="row">
            <button className="primary" onClick={() => void plan()} disabled={planning || !logline.trim()}>
              {planning ? "Planning…" : "Plan 4 shots"}
            </button>
            <button className="btn" onClick={() => api.studio.logline(logline).then(() => toast("Logline saved"))} disabled={!logline.trim()}>
              Save logline
            </button>
            <span className="grow" />
            <span className={`pill ${gpu?.ready ? "ok" : "bad"}`} title="Where takes render">
              {gpu?.ready ? `render farm · ${gpu.name ?? gpu.host}` : "render farm offline · smoke brick only"}
            </span>
            <span className={`pill ${tele?.metrics ? "ok" : ""}`} title="Take metrics pushed to Grafana Cloud; the director queries them through the Grafana MCP">
              {tele?.metrics ? `Grafana · ${tele.metrics_sent} points` : "Grafana off"}
            </span>
          </div>
        </div>
        <div className="studio-assets">
          <label className="muted small">Keyframes</label>
          <div className="asset-row">
            {board.assets.map((a) => (
              <img key={a} src={api.studio.assetUrl(a)} alt={a} title={a} />
            ))}
            <button className="btn sm" onClick={() => fileRef.current?.click()}>
              Add image
            </button>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
          </div>
          <span className="muted small">Or use a path on the box, e.g. ~/celwright_v3b/hero_v3.png</span>
        </div>
      </div>

      {board.shots.length === 0 ? (
        <EmptyState title="No shots yet" hint="Write a logline and press Plan, or ask the chat: 'plan four shots for …'." />
      ) : (
        <div className="shot-grid">
          {board.shots.map((s) => (
            <ShotCard key={s.id} shot={s} assets={board.assets} gpuReady={!!gpu?.ready} onRender={render} onPatch={patch} onRemove={remove} />
          ))}
        </div>
      )}
    </div>
  );
}

function ShotCard({ shot, assets, gpuReady, onRender, onPatch, onRemove }: { shot: Shot; assets: string[]; gpuReady: boolean; onRender: (s: Shot, smoke?: boolean) => void; onPatch: (s: Shot, p: Partial<Shot> & { director_note?: string }) => void; onRemove: (s: Shot) => void }) {
  const [prompt, setPrompt] = useState(shot.prompt);
  const [kf, setKf] = useState(shot.keyframe ?? "");
  useEffect(() => {
    setPrompt(shot.prompt);
    setKf(shot.keyframe ?? "");
  }, [shot.prompt, shot.keyframe]);
  const last = shot.takes[shot.takes.length - 1];
  const kfUrl = shot.keyframe && !shot.keyframe.startsWith("~") && !shot.keyframe.startsWith("/") ? api.studio.assetUrl(shot.keyframe) : null;
  return (
    <article className={`shot st-${shot.status}`}>
      <header>
        <span className="shot-id">{shot.id.split("-")[0]}</span>
        <input className="shot-title" value={shot.title} onChange={(e) => onPatch(shot, { title: e.target.value })} />
        <span className={`pill st-${shot.status}`}>{STATUS_LABEL[shot.status]}</span>
        <button className="icon-btn" onClick={() => onRemove(shot)} title="Delete shot" aria-label="Delete shot">
          ×
        </button>
      </header>
      <div className="shot-body">
        <div className="shot-kf">
          {last?.contact ? <img src={last.contact} alt="contact sheet of the latest take" className="contact" /> : kfUrl ? <img src={kfUrl} alt="keyframe" /> : <div className="kf-empty">no keyframe</div>}
          {last?.clip && (
            <video src={last.clip} controls loop muted playsInline />
          )}
        </div>
        <div className="shot-fields">
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onBlur={() => prompt !== shot.prompt && onPatch(shot, { prompt })} rows={4} spellCheck={false} aria-label="Prompt" />
          <div className="row">
            <select className="select sm" value={kf} onChange={(e) => { setKf(e.target.value); onPatch(shot, { keyframe: e.target.value || null }); }} aria-label="Keyframe">
              <option value="">keyframe: none (smoke)</option>
              {assets.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
              <option value="~/celwright_v3b/hero_v3.png">~/celwright_v3b/hero_v3.png (on the box)</option>
              {kf && !assets.includes(kf) && !kf.startsWith("~/celwright_v3b/hero_v3") && <option value={kf}>{kf}</option>}
            </select>
            <select className="select sm" value={String(shot.frames ?? 49)} onChange={(e) => onPatch(shot, { frames: Number(e.target.value) })} aria-label="Frames">
              {[17, 33, 49, 81].map((n) => (
                <option key={n} value={n}>
                  {n} frames
                </option>
              ))}
            </select>
            <select className="select sm" value={shot.size ?? "832x480"} onChange={(e) => onPatch(shot, { size: e.target.value })} aria-label="Size">
              {["832x480", "640x384", "480x832"].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          {shot.notes && <p className="muted small">{shot.notes}</p>}
          {shot.director_note && <p className="small director-note">Director: {shot.director_note}</p>}
        </div>
      </div>
      <footer>
        <button className="primary sm" onClick={() => onRender(shot)} disabled={shot.status === "rendering"} title={gpuReady && shot.keyframe ? "Render on the GPU box" : "No keyframe or no GPU: runs the smoke brick"}>
          {shot.status === "rendering" ? "Rendering…" : gpuReady && shot.keyframe ? "Render on the 5090" : "Render (smoke)"}
        </button>
        {shot.status === "rendered" && (
          <button className="btn sm" onClick={() => onPatch(shot, { status: "approved" })}>
            Approve
          </button>
        )}
        {shot.status === "reshoot" && (
          <button className="btn sm" onClick={() => onRender(shot)}>
            Reshoot
          </button>
        )}
        <span className="grow" />
        <Takes takes={shot.takes} />
      </footer>
    </article>
  );
}

function Takes({ takes }: { takes: Take[] }) {
  if (!takes.length) return <span className="muted small">no takes yet</span>;
  return (
    <span className="takes">
      {takes.slice(-4).map((t, i) => (
        <button key={t.id} className={`take ${t.verdict ?? t.status}`} onClick={() => navigate({ kind: "lab", run: t.id })} title={`${t.status}${t.verdict ? ` · ${t.verdict}` : ""}${t.identity_mean != null ? ` · identity ${t.identity_mean.toFixed(2)} (min ${(t.identity_min ?? 0).toFixed(2)})` : ""}${t.flicker_mean != null ? ` · flicker ${t.flicker_mean.toFixed(2)}` : ""}${t.gen_s != null ? ` · ${Math.round(t.gen_s)} s` : ""}${t.origin && t.origin !== "ui" ? ` · by agent` : ""}`}>
          T{takes.length - Math.min(4, takes.length) + i + 1}
          {t.identity_mean != null ? ` ${t.identity_mean.toFixed(2)}` : ""}
        </button>
      ))}
    </span>
  );
}
