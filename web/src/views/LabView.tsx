/**
 * The Training Lab inside Cortex: the in-browser stations (an embedded page that trains a tiny transformer with
 * tf.js), the chapters (vault notes with topic "lab"), and real runs on a GPU (this machine, the user's box over
 * SSH, or Modal) with live metrics. The chat on the right and the browser's agent (WebMCP) drive the same controls.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { GpuStatus, LabChapter, LabPlan, LabRecipe, LabRun, LabRunDetail, PlanCard, PlanCol } from "../api";
import { navigate } from "../lib/router";
import { useAsync } from "../lib/hooks";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

export const LAB_STATIONS = ["overview", "data", "pretrain", "midtrain", "posttrain", "encoder", "cluster", "paint", "speculative", "moe"] as const;

/** Ask the embedded lab page to do something; resolves with its reply. Used by the WebMCP tools too. */
export function labMessage(msg: Record<string, unknown>, timeoutMs = 4000): Promise<Record<string, unknown>> {
  const frame = document.getElementById("lab-frame") as HTMLIFrameElement | null;
  if (!frame?.contentWindow) return Promise.reject(new Error("the lab is not open; navigate to the Lab first"));
  const id = Math.random().toString(36).slice(2);
  return new Promise((resolve, reject) => {
    const t = window.setTimeout(() => {
      window.removeEventListener("message", on);
      reject(new Error("the lab page did not answer"));
    }, timeoutMs);
    const on = (e: MessageEvent) => {
      const d = e.data as Record<string, unknown> | null;
      if (!d || d.id !== id) return;
      window.clearTimeout(t);
      window.removeEventListener("message", on);
      resolve(d);
    };
    window.addEventListener("message", on);
    frame.contentWindow!.postMessage({ ...msg, id }, "*");
  });
}

export function LabView({ station, runId, plan, refresh }: { station?: string; runId?: string; plan?: boolean; refresh: number }) {
  const [tab, setTab] = useState<"stations" | "chapters" | "runs" | "plan">(plan ? "plan" : runId != null ? "runs" : "stations");
  useEffect(() => {
    if (plan) setTab("plan");
    else if (runId != null) setTab("runs");
    else if (station) setTab("stations");
  }, [runId, plan, station]);
  const chapters = useAsync(() => api.lab.chapters(), [], [refresh]);

  return (
    <section className="lab-view">
      <header className="lab-head">
        <div className="lab-tabs" role="tablist">
          {(["plan", "stations", "chapters", "runs"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              className={`tab ${tab === t ? "on" : ""}`}
              aria-selected={tab === t}
              onClick={() => {
                setTab(t);
                navigate(t === "plan" ? { kind: "lab", plan: true } : t === "runs" ? { kind: "lab", run: "" } : t === "stations" ? { kind: "lab", station: station ?? "overview" } : { kind: "lab" });
              }}
            >
              {t === "plan" ? "My plan" : t === "stations" ? "In the browser" : t === "chapters" ? `Chapters${chapters.data ? ` · ${chapters.data.length}` : ""}` : "GPU runs"}
            </button>
          ))}
        </div>
        <span className="muted small">Train small models here; run the real thing on a GPU; the agent gets the same buttons.</span>
      </header>
      {tab === "stations" && <Stations station={station} />}
      {tab === "chapters" && <Chapters chapters={chapters.data} loading={chapters.loading} error={chapters.error} />}
      {tab === "runs" && <Runs runId={runId || undefined} refresh={refresh} />}
      {tab === "plan" && <Plan refresh={refresh} />}
    </section>
  );
}

function Stations({ station }: { station?: string }) {
  const st = station && (LAB_STATIONS as readonly string[]).includes(station) ? station : "overview";
  const src = useMemo(() => `/lab/?embed=1#${st}`, [st]);
  const ref = useRef<HTMLIFrameElement>(null);
  // Change the station without reloading the page (state inside the lab survives).
  useEffect(() => {
    const w = ref.current?.contentWindow;
    if (w) w.postMessage({ type: "lab:show", station: st }, "*");
  }, [st]);
  return (
    <div className="lab-stations">
      <nav className="lab-stnav" aria-label="Stations">
        {LAB_STATIONS.map((s) => (
          <button key={s} className={`lnk ${s === st ? "on" : ""}`} onClick={() => navigate({ kind: "lab", station: s })}>
            {s === "overview" ? "Map" : s === "posttrain" ? "post-train" : s === "midtrain" ? "mid-train" : s}
          </button>
        ))}
        <span className="grow" />
        <a className="lnk" href="/lab/" target="_blank" rel="noreferrer">
          Open full page ↗
        </a>
      </nav>
      <iframe id="lab-frame" ref={ref} title="Training Lab" src={src} className="lab-frame" />
    </div>
  );
}

function Chapters({ chapters, loading, error }: { chapters: LabChapter[] | null; loading: boolean; error: string | null }) {
  if (error) return <EmptyState title="Could not load the chapters" hint={error} />;
  if (loading && !chapters) return <EmptyState title="Loading chapters" />;
  if (!chapters?.length) return <EmptyState title="No chapters yet" hint="Chapters live in cortex/lab/chapters and are synced into the vault at startup." />;
  return (
    <ol className="lab-chapters">
      {chapters.map((c) => (
        <li key={c.slug}>
          <button className="lab-chapter" onClick={() => navigate({ kind: "note", slug: c.slug })}>
            <span className="num">{String(c.chapter ?? "").padStart(2, "0")}</span>
            <span className="ttl">{c.title ?? c.slug}</span>
            <span className="meta">
              {c.station && c.station !== "none" ? `station ${c.station} · ` : ""}
              {c.recipe && c.recipe !== "none" ? `${c.recipe} · ` : ""}
              {c.reading_time ?? ""}
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}

function Runs({ runId, refresh }: { runId?: string; refresh: number }) {
  const { toast } = useToast();
  const ex = useAsync(() => api.lab.executors(), [], []);
  const recipes = useAsync(() => api.lab.recipes(), [], []);
  const runs = useAsync(() => api.lab.runs(), [], [refresh]);
  const [recipe, setRecipe] = useState("pretrain_nano");
  const [args, setArgs] = useState("--smoke --steps 200");
  const [executor, setExecutor] = useState<"local" | "ssh" | "modal">("local");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(runId ?? null);
  useEffect(() => {
    if (runId) setSelected(runId);
  }, [runId]);

  // Pick the best available executor once we know what exists.
  useEffect(() => {
    const e = ex.data;
    if (!e) return;
    if (e.ssh.available) setExecutor("ssh");
    else if (e.modal.available) setExecutor("modal");
    else setExecutor("local");
  }, [ex.data]);

  const start = async () => {
    setBusy(true);
    try {
      const r = await api.lab.start({ recipe, args, executor });
      toast(`Started ${r.recipe} on ${r.executor}`);
      setSelected(r.id);
      navigate({ kind: "lab", run: r.id });
      runs.reload();
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setBusy(false);
    }
  };

  // Poll the list while something is running.
  useEffect(() => {
    const anyLive = runs.data?.some((r) => r.status === "running" || r.status === "queued");
    if (!anyLive) return;
    const t = window.setInterval(() => runs.reload(), 3000);
    return () => window.clearInterval(t);
  }, [runs.data, runs]);

  const e = ex.data;
  return (
    <div className="lab-runs">
      <form
        className="lab-launch"
        onSubmit={(ev) => {
          ev.preventDefault();
          void start();
        }}
      >
        <label>
          <span>Recipe</span>
          <select className="select sm" value={recipe} onChange={(ev) => setRecipe(ev.target.value)}>
            {(recipes.data ?? [{ name: "pretrain_nano", file: "", doc: "" }]).map((r: LabRecipe) => (
              <option key={r.name} value={r.name}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
        <label className="grow">
          <span>Arguments</span>
          <input className="input sm" value={args} onChange={(ev) => setArgs(ev.target.value)} spellCheck={false} placeholder="--smoke --steps 200" />
        </label>
        <label>
          <span>Where</span>
          <select className="select sm" value={executor} onChange={(ev) => setExecutor(ev.target.value as typeof executor)}>
            <option value="local" disabled={e ? !e.local.available : false}>
              this machine
            </option>
            <option value="ssh" disabled={e ? !e.ssh.available : true}>
              {e?.ssh.host ? `GPU box (${e.ssh.host})` : "GPU box over SSH"}
            </option>
            <option value="modal" disabled={e ? !e.modal.available : true}>
              Modal
            </option>
          </select>
        </label>
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Starting…" : "Run"}
        </button>
      </form>
      {recipes.data && (
        <p className="muted small lab-doc">{recipes.data.find((r) => r.name === recipe)?.doc || "Pick a recipe. Each one is a short, readable script under lab/recipes."}</p>
      )}
      {e?.demo && !e.ssh.available && !e.modal.available && (
        <p className="muted small">This is the hosted demo: runs execute on the demo server's CPU with --smoke. Point CORTEX_SSH_HOST at your own GPU box, or sign in to Modal, for the real thing.</p>
      )}
      <GpuPanel onReady={() => setExecutor("ssh")} />
      <div className="lab-runs-body">
        <ul className="lab-runlist" aria-label="Runs">
          {(runs.data ?? []).map((r) => (
            <li key={r.id}>
              <button className={`lab-run ${selected === r.id ? "on" : ""}`} onClick={() => { setSelected(r.id); navigate({ kind: "lab", run: r.id }); }}>
                <span className={`dot ${r.status}`} aria-hidden="true" />
                <span className="ttl">
                  {r.recipe} <span className="muted">{r.args}</span>
                </span>
                <span className="meta">
                  {r.executor} · {r.status}
                  {r.last && typeof r.last.loss === "number" ? ` · loss ${r.last.loss.toFixed(3)}` : ""}
                </span>
              </button>
            </li>
          ))}
          {runs.data && runs.data.length === 0 && <li className="muted small">No runs yet. Start with pretrain_nano --smoke; it finishes in about a minute on a CPU.</li>}
        </ul>
        {selected ? <RunDetail id={selected} onGone={() => { setSelected(null); runs.reload(); }} /> : <EmptyState title="Pick a run" hint="Metrics and logs stream here while it trains." />}
      </div>
    </div>
  );
}

function RunDetail({ id, onGone }: { id: string; onGone: () => void }) {
  const { toast } = useToast();
  const [run, setRun] = useState<LabRunDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement>(null);

  const load = useCallback(async () => {
    try {
      setRun(await api.lab.run(id, 400));
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  // Stream while live.
  useEffect(() => {
    if (!run || !(run.status === "running" || run.status === "queued")) return;
    const es = new EventSource(api.lab.eventsUrl(id));
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data) as { type: string; lines?: string[]; rows?: Record<string, number>[]; status?: LabRun["status"] };
        setRun((r) => {
          if (!r) return r;
          if (d.type === "log" && d.lines) return { ...r, log: [...r.log, ...d.lines].slice(-800) };
          if (d.type === "metrics" && d.rows) return { ...r, metrics: [...r.metrics, ...d.rows] };
          if (d.type === "status" && d.status) return { ...r, status: d.status };
          return r;
        });
        if (d.type === "status") {
          es.close();
          void load();
        }
      } catch {
        /* ignore malformed events */
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [id, run?.status, load]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run?.log.length]);

  if (err) return <EmptyState title="Could not load the run" hint={err} />;
  if (!run) return <EmptyState title="Loading run" />;
  const keys = metricKeys(run.metrics);
  return (
    <div className="lab-rundetail">
      <header>
        <h2>
          {run.recipe} <span className="muted">{run.args}</span>
        </h2>
        <span className={`pill ${run.status === "done" ? "ok" : run.status === "failed" ? "bad" : ""}`}>{run.status}</span>
        <span className="muted small">
          {run.executor} · {run.started}
          {run.ended ? ` → ${run.ended}` : ""}
        </span>
        <span className="grow" />
        {(run.status === "running" || run.status === "queued") && (
          <button className="btn sm" onClick={() => api.lab.stop(id).then(() => toast("Stopping"))}>
            Stop
          </button>
        )}
        <button
          className="btn sm"
          onClick={() => {
            if (!window.confirm("Delete this run and its logs?")) return;
            api.lab.remove(id).then(onGone).catch((e) => toast(errorMessage(e), "error"));
          }}
        >
          Delete
        </button>
      </header>
      {keys.length > 0 && (
        <div className="lab-charts">
          {keys.slice(0, 4).map((k) => (
            <MetricChart key={k} name={k} rows={run.metrics} />
          ))}
        </div>
      )}
      {run.result && (
        <details className="lab-result" open>
          <summary>Result</summary>
          <pre>{JSON.stringify(run.result, null, 2)}</pre>
        </details>
      )}
      <pre className="lab-log" ref={logRef}>
        {run.log.join("\n") || "(no output yet)"}
      </pre>
    </div>
  );
}

function metricKeys(rows: Record<string, number>[]): string[] {
  const seen = new Map<string, number>();
  for (const r of rows) for (const [k, v] of Object.entries(r)) if (typeof v === "number" && k !== "step" && k !== "t") seen.set(k, (seen.get(k) ?? 0) + 1);
  return Array.from(seen.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => k);
}

function MetricChart({ name, rows }: { name: string; rows: Record<string, number>[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const pts = rows.filter((r) => typeof r[name] === "number" && Number.isFinite(r[name])).map((r) => ({ x: r.step ?? 0, y: r[name] }));
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth || 300;
    const h = c.clientHeight || 120;
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
    g.font = "11px " + (css("--mono") || "monospace");
    g.fillStyle = css("--faint");
    g.fillText(name, 6, 12);
    if (pts.length < 2) return;
    const pad = { l: 42, r: 8, t: 18, b: 16 };
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = Math.min(...ys), ymax = Math.max(...ys);
    if (ymax - ymin < 1e-9) ymax = ymin + 1;
    const X = (x: number) => pad.l + ((x - xmin) / Math.max(1e-9, xmax - xmin)) * (w - pad.l - pad.r);
    const Y = (y: number) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
    g.strokeStyle = css("--line");
    g.lineWidth = 1;
    for (let k = 0; k <= 2; k++) {
      const v = ymin + (k / 2) * (ymax - ymin);
      g.beginPath();
      g.moveTo(pad.l, Y(v));
      g.lineTo(w - pad.r, Y(v));
      g.stroke();
      g.fillText(v.toPrecision(3), 4, Y(v) + 4);
    }
    g.strokeStyle = css("--accent") || "#4db39d";
    g.lineWidth = 1.8;
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))));
    g.stroke();
    g.fillStyle = css("--muted");
    g.fillText(`last ${pts[pts.length - 1].y.toPrecision(4)} · step ${pts[pts.length - 1].x}`, pad.l, h - 4);
  }, [name, rows]);
  return <canvas ref={ref} className="lab-chart" aria-label={`${name} over steps`} />;
}


/** The GPU box: the app checks it and, if PyTorch is missing, installs everything over SSH itself. */
export function GpuPanel({ onReady }: { onReady?: () => void }) {
  const [st, setSt] = useState<GpuStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [log, setLog] = useState<string[] | null>(null);
  const [running, setRunning] = useState(false);
  const check = useCallback(async () => {
    setChecking(true);
    try {
      const g = await api.lab.gpu();
      setSt(g);
      if (g.ready) onReady?.();
    } catch (e) {
      setSt({ host: null, reachable: false, ready: false, message: errorMessage(e) });
    } finally {
      setChecking(false);
    }
  }, [onReady]);
  useEffect(() => {
    void check();
  }, [check]);
  const setup = async () => {
    setRunning(true);
    setLog([]);
    try {
      await api.lab.gpuSetup((ev) => {
        if (ev.type === "log") setLog((l) => [...(l ?? []), ...ev.lines].slice(-200));
        if (ev.type === "error") setLog((l) => [...(l ?? []), "error: " + ev.message]);
        if (ev.type === "status") {
          setLog((l) => [...(l ?? []), ev.status === "done" ? "ready" : `setup failed (exit ${ev.exit})`]);
          if (ev.gpu) {
            setSt(ev.gpu);
            if (ev.gpu.ready) onReady?.();
          }
        }
      });
    } finally {
      setRunning(false);
      void check();
    }
  };
  const tone = !st ? "" : st.ready ? "ok" : st.reachable ? "warn" : "bad";
  return (
    <div className="lab-gpu">
      <span className={`pill ${tone}`}>{checking && !st ? "checking GPU…" : st?.ready ? `GPU ready · ${st.gpu?.name ?? st.host}` : st?.reachable ? `GPU reachable · ${st.gpu?.name ?? st.host}` : `GPU offline · ${st?.host ?? "not configured"}`}</span>
      <span className="muted small">
        {st?.message}
        {st?.gpu ? ` · ${st.gpu.memory_used} of ${st.gpu.memory_total} used${st.busy ? " · a run is in progress" : ""}` : ""}
      </span>
      <span className="grow" />
      <button className="btn sm" onClick={() => void check()} disabled={checking}>
        {checking ? "Checking…" : "Re-check"}
      </button>
      {st?.reachable && !st.ready && (
        <button className="primary sm" onClick={() => void setup()} disabled={running}>
          {running ? "Setting up…" : "Set up PyTorch on the box"}
        </button>
      )}
      {log && <pre className="lab-log lab-gpu-log">{log.join("\n")}</pre>}
    </div>
  );
}


/** The learning plan: one kanban over the chapters. Cards move with the arrows or through the agent's lab_plan_move tool. */
function Plan({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [plan, setPlan] = useState<LabPlan | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<number | "all">("all");
  const load = useCallback(async () => {
    try {
      setPlan(await api.lab.plan());
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load, refresh]);
  const move = async (c: PlanCard, col: PlanCol) => {
    try {
      setPlan(await api.lab.planMove(c.id, col));
      if (col === "done") toast(`Done: ${c.title}`);
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const open = (c: PlanCard) => {
    if (c.kind === "station" && c.station) navigate({ kind: "lab", station: c.station });
    else if (c.kind === "recipe") navigate({ kind: "lab", run: "" });
    else if (c.note) navigate({ kind: "note", slug: c.note });
  };
  if (err) return <EmptyState title="Could not load the plan" hint={err} />;
  if (!plan) return <EmptyState title="Loading plan" />;
  const chaptersIn = Array.from(new Set(plan.cards.map((c) => c.chapter))).sort((a, b) => a - b);
  const cards = plan.cards.filter((c) => filter === "all" || c.chapter === filter);
  const cols: { id: PlanCol; title: string }[] = [
    { id: "todo", title: "To learn" },
    { id: "doing", title: "In progress" },
    { id: "done", title: "Done" },
  ];
  const pct = plan.total ? Math.round((100 * plan.done) / plan.total) : 0;
  return (
    <div className="lab-plan">
      <div className="lab-plan-head">
        <div className="lab-progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} title={`${plan.done} of ${plan.total} cards done`}>
          <span style={{ width: `${pct}%` }} />
        </div>
        <span className="muted small">
          {plan.done} of {plan.total} done · {pct}%
        </span>
        <span className="score" title={`XP by card: read ${plan.xp_by_kind.read}, station ${plan.xp_by_kind.station}, snippet ${plan.xp_by_kind.build}, recipe ${plan.xp_by_kind.recipe}, self-test ${plan.xp_by_kind.quiz}`}>
          <b>{plan.xp} XP</b> · level {plan.level} {plan.level_name} · {plan.next_level_xp - plan.xp} to next
        </span>
        <span className={`score ${plan.streak > 0 ? "hot" : ""}`} title="Days in a row with at least one card done">
          streak {plan.streak}
          {plan.done_today ? ` · ${plan.done_today} today` : ""}
        </span>
        <span className="grow" />
        <label className="muted small">
          Chapter{" "}
          <select className="select sm" value={String(filter)} onChange={(e) => setFilter(e.target.value === "all" ? "all" : Number(e.target.value))}>
            <option value="all">all</option>
            {chaptersIn.map((n) => (
              <option key={n} value={n}>
                {String(n).padStart(2, "0")}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="kanban">
        {cols.map((col) => (
          <section key={col.id} className="kanban-col" aria-label={col.title}>
            <h3>
              {col.title} <span className="muted">{cards.filter((c) => c.col === col.id).length}</span>
            </h3>
            <ul>
              {cards
                .filter((c) => c.col === col.id)
                .map((c) => (
                  <li key={c.id} className={`kcard k-${c.kind}`}>
                    <button className="kcard-main" onClick={() => open(c)} title="Open">
                      <span className="kchap">{String(c.chapter).padStart(2, "0")}</span>
                      <span className="ktitle">{c.title}</span>
                      {c.comment && <span className="kcomment">{c.comment}</span>}
                    </button>
                    <span className="kmoves">
                      {col.id !== "todo" && (
                        <button className="icon-btn" onClick={() => void move(c, col.id === "done" ? "doing" : "todo")} title="Move back" aria-label="Move back">
                          ←
                        </button>
                      )}
                      {col.id !== "done" && (
                        <button className="icon-btn" onClick={() => void move(c, col.id === "todo" ? "doing" : "done")} title={col.id === "todo" ? "Start" : "Mark done"} aria-label={col.id === "todo" ? "Start" : "Mark done"}>
                          {col.id === "todo" ? "→" : "✓"}
                        </button>
                      )}
                    </span>
                  </li>
                ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
