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

/** Arguments that make a first click succeed on any machine; edit them for the real thing. */
const DEFAULT_ARGS: Record<string, string> = {
  pretrain_nano: "--smoke --steps 300",
  midtrain: "--smoke --steps 200",
  sft_lora: "--smoke --steps 200",
  dpo: "--smoke --steps 150",
  grpo_tool: "--smoke --steps 60",
  paint_grpo: "--smoke --steps 80",
  embed_contrastive: "--smoke --steps 200",
  embed_vault: "--smoke",
  eval_suite: "--smoke",
  redteam_suite: "--smoke",
  kernel_bench: "--smoke",
  optim_bench: "--smoke --steps 100",
  lean_eval: "--smoke",
  spec_decode: "--smoke --steps 60",
  moe_nano: "--smoke --steps 150",
  inspect_model: "--smoke",
  scratch: "",
  shell: "",
};

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

export function LabView({ station, runId, plan, terminal, refresh }: { station?: string; runId?: string; plan?: boolean; terminal?: boolean; refresh: number }) {
  const [tab, setTab] = useState<"stations" | "chapters" | "runs" | "plan" | "terminal">(terminal ? "terminal" : plan ? "plan" : runId != null ? "runs" : "stations");
  useEffect(() => {
    if (terminal) setTab("terminal");
    else if (plan) setTab("plan");
    else if (runId != null) setTab("runs");
    else if (station) setTab("stations");
  }, [runId, plan, station, terminal]);
  const chapters = useAsync(() => api.lab.chapters(), [], [refresh]);

  return (
    <section className="lab-view">
      <header className="lab-head">
        <div className="lab-tabs" role="tablist">
          {(["plan", "stations", "chapters", "runs", "terminal"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              className={`tab ${tab === t ? "on" : ""}`}
              aria-selected={tab === t}
              onClick={() => {
                setTab(t);
                navigate(t === "plan" ? { kind: "lab", plan: true } : t === "terminal" ? { kind: "lab", terminal: true } : t === "runs" ? { kind: "lab", run: "" } : t === "stations" ? { kind: "lab", station: station ?? "overview" } : { kind: "lab" });
              }}
            >
              {t === "plan" ? "My plan" : t === "stations" ? "In the browser" : t === "chapters" ? `Chapters${chapters.data ? ` · ${chapters.data.length}` : ""}` : t === "terminal" ? "Terminal" : "GPU runs"}
            </button>
          ))}
        </div>
        <details className="lab-howto">
          <summary className="muted small">How to use this</summary>
          <ol>
            <li><b>My plan</b> is the path: cards in order, one chapter at a time. Move a card when you finish it; the chat can do it for you after a quiz.</li>
            <li><b>In the browser</b> is where you watch training happen. Every station has one Train button and one number to watch, named in its first paragraph.</li>
            <li><b>Chapters</b> are the theory, written to be read start to finish. Each has a short snippet you can run with the "Run on GPU" button, and a self-test.</li>
            <li><b>GPU runs</b> is the real thing: the same ideas as full scripts on your 5090. Loss curves and exact rollouts stream in while it trains.</li>
          </ol>
        </details>
      </header>
      {tab === "stations" && <Stations station={station} />}
      {tab === "chapters" && <Chapters chapters={chapters.data} loading={chapters.loading} error={chapters.error} />}
      {tab === "runs" && <Runs runId={runId || undefined} refresh={refresh} />}
      {tab === "plan" && <Plan refresh={refresh} />}
      {tab === "terminal" && <Terminal />}
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
  // Keep the embedded page on the same palette as Cortex: send the live tokens on load and whenever the theme changes.
  useEffect(() => {
    const send = () => {
      const w = ref.current?.contentWindow;
      if (!w) return;
      const cs = getComputedStyle(document.documentElement);
      const vars: Record<string, string> = {};
      for (const v of ["--bg", "--surface", "--surface-2", "--border", "--text", "--text-2", "--text-3", "--accent", "--accent-soft", "--on-accent", "--accent-text"]) {
        const val = cs.getPropertyValue(v).trim();
        if (val) vars[v] = val;
      }
      const theme = document.documentElement.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      w.postMessage({ type: "lab:theme", vars, theme }, "*");
    };
    const frame = ref.current;
    frame?.addEventListener("load", send);
    const mo = new MutationObserver(send);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["style", "data-theme", "data-palette"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", send);
    send();
    return () => {
      frame?.removeEventListener("load", send);
      mo.disconnect();
      mq.removeEventListener("change", send);
    };
  }, []);
  return (
    <div className="lab-stations">
      <a className="lab-fullpage" href={`/lab/#${st}`} target="_blank" rel="noreferrer" title="Open the lab in its own tab">
        Open full page ↗
      </a>
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
  const [recipe, setRecipeState] = useState("pretrain_nano");
  const [args, setArgs] = useState("--smoke --steps 200");
  const setRecipe = (r: string) => {
    setRecipeState(r);
    setArgs(DEFAULT_ARGS[r] ?? "--smoke --steps 200");
  };
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
                  {r.recipe} <span className="muted">{r.recipe === "scratch" ? r.code_preview : r.args}</span>
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
      {run.rollouts && run.rollouts.length > 0 && <Rollouts rows={run.rollouts} />}
      {run.script && <ScriptView id={run.id} preview={run.code_preview} />}
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
        {st?.tailscale?.ip ? ` · Tailscale ${st.tailscale.ip} (${st.tailscale.link ?? "up"}) · ssh ${st.ssh_round_trip_ms ?? "?"} ms` : ""}
      </span>
      {st?.ready && (
        <button className="btn sm" onClick={() => api.lab.start({ recipe: "kernel_bench", args: "", executor: "ssh" }).then((r) => navigate({ kind: "lab", run: r.id })).catch((e) => toast(errorMessage(e), "error"))} title="Run kernel_bench on the box and watch the numbers">
          Benchmark the link
        </button>
      )}
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
  const [newTitle, setNewTitle] = useState("");
  const [query, setQuery] = useState("");
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
  const q = query.trim().toLowerCase();
  const cards = plan.cards.filter((c) => (filter === "all" || c.chapter === filter || (c.custom && filter === 0)) && (!q || `${String(c.chapter).padStart(2, "0")} ${c.title} ${c.kind} ${c.comment ?? ""}`.toLowerCase().includes(q)));
  const cols: { id: PlanCol; title: string }[] = [
    { id: "todo", title: "To learn" },
    { id: "doing", title: "In progress" },
    { id: "done", title: "Done" },
  ];
  const pct = plan.total ? Math.round((100 * plan.done) / plan.total) : 0;
  // Units: the To learn column grouped by chapter, in order, like a course path. The current unit is the first
  // one with something left to do; it opens by default and the others collapse.
  const unitTitle = (n: number) => {
    const read = plan.cards.find((c) => c.chapter === n && c.kind === "read");
    return read ? read.title.replace(/^Read:\s*/, "") : `Chapter ${n}`;
  };
  const units = (() => {
    const byKey = new Map<string, { key: string; chapter: number; title: string; cards: PlanCard[]; done: number; total: number }>();
    for (const c of cards) {
      const key = c.custom ? "custom" : String(c.chapter);
      if (!byKey.has(key)) byKey.set(key, { key, chapter: c.custom ? 999 : c.chapter, title: c.custom ? "Your own cards" : unitTitle(c.chapter), cards: [], done: 0, total: 0 });
      const u = byKey.get(key)!;
      u.total += 1;
      if (c.col === "done") u.done += 1;
      if (c.col === "todo") u.cards.push(c);
    }
    return Array.from(byKey.values()).filter((u) => u.cards.length > 0).sort((a, b) => a.chapter - b.chapter);
  })();
  const currentUnit = units.find((u) => u.cards.length > 0)?.key;
  const nextUp = units.find((u) => u.key === currentUnit)?.cards[0];
  const add = async () => {
    const t = newTitle.trim();
    if (!t) return;
    try {
      setPlan(await api.lab.planAdd({ title: t, kind: "custom" }));
      setNewTitle("");
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const remove = async (c: PlanCard) => {
    if (!window.confirm(`Delete "${c.title}"?`)) return;
    try {
      setPlan(await api.lab.planRemove(c.id));
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  return (
    <div className="lab-plan">
      <div className="lab-plan-head">
        <div className="stat-block">
          <div className="lab-progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} title={`${plan.done} of ${plan.total} cards done`}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <span className="muted small">{plan.done} of {plan.total} done</span>
        </div>
        <div className="stat-block" title={`XP by card: read ${plan.xp_by_kind.read}, station ${plan.xp_by_kind.station}, snippet ${plan.xp_by_kind.build}, recipe ${plan.xp_by_kind.recipe}, self-test ${plan.xp_by_kind.quiz}`}>
          <b>{plan.xp} XP</b>
          <span className="muted small">level {plan.level} · {plan.level_name} · {plan.next_level_xp - plan.xp} to next</span>
        </div>
        <div className={`stat-block ${plan.streak > 0 ? "hot" : ""}`} title="Days in a row with at least one card done">
          <b>{plan.streak} day{plan.streak === 1 ? "" : "s"}</b>
          <span className="muted small">streak{plan.done_today ? ` · ${plan.done_today} today` : ""}</span>
        </div>
      </div>
      <div className="lab-plan-tools">
        <input
          className="input plan-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search your plan (title, chapter number, kind)…"
          aria-label="Search cards"
          spellCheck={false}
        />
        <label className="muted small">
          Chapter{" "}
          <select className="select" value={String(filter)} onChange={(e) => setFilter(e.target.value === "all" ? "all" : Number(e.target.value))}>
            <option value="all">all</option>
            {chaptersIn.map((n) => (
              <option key={n} value={n}>
                {String(n).padStart(2, "0")}
              </option>
            ))}
          </select>
        </label>
        <form
          className="lab-plan-add"
          onSubmit={(e) => {
            e.preventDefault();
            void add();
          }}
        >
          <input className="input" value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="Add something to learn…" aria-label="New card" />
          <button className="btn primary" type="submit" disabled={!newTitle.trim()}>
            Add
          </button>
        </form>
      </div>
      {nextUp && (
        <div className="next-up">
          <span className="next-label">Next up</span>
          <span className="next-chap">{nextUp.custom ? "you" : `Unit ${String(nextUp.chapter).padStart(2, "0")}`}</span>
          <span className="next-title">{nextUp.title}</span>
          <span className="grow" />
          <button className="btn sm" onClick={() => open(nextUp)}>
            Open
          </button>
          <button className="primary sm" onClick={() => void move(nextUp, "doing")}>
            Start
          </button>
        </div>
      )}
      <div className="kanban">
        {cols.map((col) => (
          <section key={col.id} className="kanban-col" aria-label={col.title}>
            <h3>
              {col.title} <span className="muted">{cards.filter((c) => c.col === col.id).length}</span>
            </h3>
            {col.id === "todo" ? (
              <div className="units">
                {units.map((u) => (
                  <details key={u.key} className={`unit ${u.key === currentUnit ? "current" : ""}`} open={u.key === currentUnit}>
                    <summary>
                      <span className="unit-num">{u.key === "custom" ? "you" : String(u.chapter).padStart(2, "0")}</span>
                      <span className="unit-title">{u.title}</span>
                      <span className="unit-count">
                        {u.done}/{u.total}
                      </span>
                      <span className="unit-bar">
                        <i style={{ width: `${u.total ? (100 * u.done) / u.total : 0}%` }} />
                      </span>
                    </summary>
                    <ul>
                      {u.cards.map((c) => (
                        <li key={c.id} className={`kcard k-${c.kind}`}>
                          <button className="kcard-main" onClick={() => open(c)} title="Open">
                            <span className="ktitle">{c.title}</span>
                            {c.comment && <span className="kcomment">{c.comment}</span>}
                          </button>
                          <span className="kmoves">
                            {c.custom && (
                              <button className="icon-btn" onClick={() => void remove(c)} title="Delete this card" aria-label="Delete this card">
                                ×
                              </button>
                            )}
                            <button className="icon-btn" onClick={() => void move(c, "doing")} title="Start" aria-label="Start">
                              →
                            </button>
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                ))}
              </div>
            ) : (
            <ul>
              {cards
                .filter((c) => c.col === col.id)
                .map((c) => (
                  <li key={c.id} className={`kcard k-${c.kind}`}>
                    <button className="kcard-main" onClick={() => open(c)} title="Open">
                      <span className="kchap">{c.custom ? "you" : String(c.chapter).padStart(2, "0")}</span>
                      <span className="ktitle">{c.title}</span>
                      {c.comment && <span className="kcomment">{c.comment}</span>}
                    </button>
                    <span className="kmoves">
                      {c.custom && (
                        <button className="icon-btn" onClick={() => void remove(c)} title="Delete this card" aria-label="Delete this card">
                          ×
                        </button>
                      )}
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
            )}
          </section>
        ))}
      </div>
    </div>
  );
}


/** Exact rollouts: what the policy actually sampled, what it scored, and the advantage it was trained on. */
function Rollouts({ rows }: { rows: Record<string, unknown>[] }) {
  const steps = Array.from(new Set(rows.map((r) => Number(r.step ?? 0)))).sort((a, b) => b - a);
  const [step, setStep] = useState<number | null>(null);
  const cur = step ?? steps[0];
  const shown = rows.filter((r) => Number(r.step ?? 0) === cur);
  const skip = new Set(["step", "group", "idx", "prompt", "completion", "chosen", "rejected", "advantage", "reward"]);
  const extra = Array.from(new Set(shown.flatMap((r) => Object.keys(r).filter((k) => !skip.has(k) && typeof r[k] === "number"))));
  const fmt = (v: unknown) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v ?? ""));
  return (
    <details className="lab-result lab-rollouts" open>
      <summary>
        Rollouts · step{" "}
        <select className="select sm" value={cur} onChange={(e) => setStep(Number(e.target.value))} onClick={(e) => e.stopPropagation()}>
          {steps.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>{" "}
        <span className="muted">{shown.length} samples; sorted by advantage</span>
      </summary>
      <div className="rollout-list">
        {shown
          .slice()
          .sort((a, b) => Number(b.advantage ?? b.reward ?? 0) - Number(a.advantage ?? a.reward ?? 0))
          .map((r, i) => (
            <div key={i} className={`rollout ${Number(r.advantage ?? 0) >= 0 ? "up" : "down"}`}>
              <div className="rollout-head">
                <b>reward {fmt(r.reward)}</b>
                {r.advantage != null && <span>adv {fmt(r.advantage)}</span>}
                {extra.map((k) => (
                  <span key={k} className="muted">
                    {k} {fmt(r[k])}
                  </span>
                ))}
              </div>
              {r.prompt != null && <pre className="rollout-prompt">{String(r.prompt)}</pre>}
              {r.completion != null && <pre className="rollout-completion">{String(r.completion)}</pre>}
              {r.chosen != null && (
                <div className="rollout-pair">
                  <pre className="rollout-completion">chosen: {String(r.chosen)}</pre>
                  <pre className="rollout-completion">rejected: {String(r.rejected ?? "")}</pre>
                </div>
              )}
            </div>
          ))}
      </div>
    </details>
  );
}

function ScriptView({ id, preview }: { id: string; preview?: string }) {
  const [code, setCode] = useState<string | null>(null);
  return (
    <details className="lab-result" onToggle={(e) => { if ((e.target as HTMLDetailsElement).open && code == null) api.lab.script(id).then((r) => setCode(r.code)).catch(() => setCode("(could not load)")); }}>
      <summary>Script · {preview}</summary>
      <pre>{code ?? "…"}</pre>
    </details>
  );
}


/** A terminal: one command at a time on this machine or the GPU box, output streamed. Not a PTY (no vim), but
 *  everything a training loop needs: nvidia-smi, ls out, python recipes/x.py, tail -f. Commands are runs, so they are kept. */
function Terminal() {
  const { toast } = useToast();
  const ex = useAsync(() => api.lab.executors(), [], []);
  const [executor, setExecutor] = useState<"local" | "ssh">("ssh");
  useEffect(() => {
    if (ex.data && !ex.data.ssh.available) setExecutor("local");
  }, [ex.data]);
  const [cmd, setCmd] = useState("nvidia-smi");
  const [history, setHistory] = useState<{ id: string; cmd: string; executor: string; lines: string[]; status: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const outRef = useRef<HTMLDivElement>(null);
  const prevCmds = useRef<string[]>([]);
  const histIdx = useRef(-1);

  useEffect(() => {
    const onPrefill = (e: Event) => {
      setCmd(String((e as CustomEvent).detail?.cmd ?? ""));
      inputRef.current?.focus();
    };
    window.addEventListener("cortex:terminal-prefill", onPrefill);
    return () => window.removeEventListener("cortex:terminal-prefill", onPrefill);
  }, []);
  useEffect(() => {
    // older shell runs come back as history
    api.lab.runs(30).then((rs) => {
      const shells = rs.filter((r) => r.recipe === "shell").reverse();
      prevCmds.current = shells.map((r) => r.cmd ?? "").filter(Boolean);
    }).catch(() => undefined);
  }, []);
  useEffect(() => {
    const el = outRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [history]);

  const run = async (command?: string) => {
    const c = (command ?? cmd).trim();
    if (!c || busy) return;
    setBusy(true);
    setCmd("");
    prevCmds.current.push(c);
    histIdx.current = -1;
    try {
      const r = await api.lab.start({ recipe: "shell", cmd: c, executor });
      setHistory((h) => [...h, { id: r.id, cmd: c, executor, lines: [], status: "running" }]);
      const es = new EventSource(api.lab.eventsUrl(r.id));
      await new Promise<void>((resolve) => {
        es.onmessage = (ev) => {
          try {
            const d = JSON.parse(ev.data) as { type: string; lines?: string[]; status?: string };
            if (d.type === "log" && d.lines) setHistory((h) => h.map((x) => (x.id === r.id ? { ...x, lines: [...x.lines, ...d.lines!.filter((l) => !l.startsWith("[cortex] $"))].slice(-2000) } : x)));
            if (d.type === "status" || d.type === "error") {
              setHistory((h) => h.map((x) => (x.id === r.id ? { ...x, status: d.status ?? "failed" } : x)));
              es.close();
              resolve();
            }
          } catch {
            /* ignore */
          }
        };
        es.onerror = () => { es.close(); resolve(); };
      });
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { e.preventDefault(); void run(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); const n = prevCmds.current.length; if (!n) return; histIdx.current = histIdx.current === -1 ? n - 1 : Math.max(0, histIdx.current - 1); setCmd(prevCmds.current[histIdx.current]); }
    else if (e.key === "ArrowDown") { e.preventDefault(); const n = prevCmds.current.length; if (histIdx.current === -1) return; histIdx.current = Math.min(n - 1, histIdx.current + 1); setCmd(prevCmds.current[histIdx.current]); }
  };
  const host = ex.data?.ssh.host ?? "gpu";
  return (
    <div className="lab-term">
      <div className="lab-term-out" ref={outRef}>
        {history.length === 0 && (
          <div className="muted small">
            One command at a time; output streams here and is kept as a run. On the GPU box the working directory is ~/cortex-lab with the lab venv on PATH. Try <code>nvidia-smi</code>, <code>ls recipes</code>, or <code>python recipes/pretrain_nano.py --smoke --steps 50</code>. Bash blocks in the chapters have a "Run in terminal" button that lands here.
          </div>
        )}
        {history.map((h) => (
          <div key={h.id} className={`term-entry ${h.status}`}>
            <div className="term-cmd">
              <span className="prompt">{h.executor === "ssh" ? `${host} ~/cortex-lab $` : "local $"}</span> {h.cmd}
              <span className="grow" />
              <button className="lnk" onClick={() => navigate({ kind: "lab", run: h.id })} title="Open as a run">
                {h.status}
              </button>
            </div>
            <pre>{h.lines.join("\n") || (h.status === "running" ? "…" : "")}</pre>
          </div>
        ))}
      </div>
      <div className="lab-term-in">
        <select className="select sm" value={executor} onChange={(e) => setExecutor(e.target.value as "local" | "ssh")} aria-label="Where">
          <option value="ssh" disabled={ex.data ? !ex.data.ssh.available : false}>{host}</option>
          <option value="local">this machine</option>
        </select>
        <span className="prompt">$</span>
        <input ref={inputRef} className="input sm" value={cmd} onChange={(e) => setCmd(e.target.value)} onKeyDown={onKey} spellCheck={false} autoFocus placeholder="command" aria-label="Command" />
        <button className="primary sm" onClick={() => void run()} disabled={busy || !cmd.trim()}>
          {busy ? "Running…" : "Run"}
        </button>
      </div>
    </div>
  );
}
