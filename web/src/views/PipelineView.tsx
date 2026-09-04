/**
 * The Pipeline tab: the training pie. A template becomes a chain of recipe runs (data -> pretrain -> midtrain ->
 * sft -> rl -> eval), each stage feeding the next its checkpoint. The flow is drawn as SVG stage cards with live
 * status, the corpus composition as a pie from the data stage's RESULT, and the eval stage's before/after report.
 * The chat and the browser's agent (WebMCP) have the same buttons (list_pipelines, start_pipeline, read_pipeline, retry_stage).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage } from "../api";
import type { Pipeline, PipelineStage, PipelineTemplate } from "../api";
import { navigate } from "../lib/router";
import { emitCommand } from "../lib/events";
import { useAsync } from "../lib/hooks";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const PIE_COLORS = ["#2f8f7b", "#6b8cc7", "#c9962b", "#c05a9a", "#8b5cf6", "#3fa46a", "#d0453f", "#7a8a99"];

/** The one number to watch per recipe once its stage is done; before that, the last METRIC line. */
const HEADLINE: Record<string, string[]> = {
  data_prep: ["total_tokens"],
  pretrain_nano: ["val_loss", "train_loss"],
  midtrain: ["val_b_after", "val_a_after"],
  sft_lora: ["exact_match_heldout", "final_loss"],
  grpo_reason: ["reward_greedy_after", "reward_greedy_before"],
  grpo_tool: ["reward_greedy_after"],
  eval_suite: ["custom_exact_match"],
  embed_vault: ["n_pairs", "n_chunks"],
  embed_contrastive: ["recall_at_1_after", "recall_at_1_before"],
};

function fmtNum(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v ?? "");
  if (Number.isInteger(v)) return v.toLocaleString();
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
}

function fmtElapsed(s: number | null | undefined): string {
  if (s == null) return "";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m ? `${m}m ${sec}s` : `${sec}s`;
}

function headline(st: PipelineStage): string {
  if (st.status === "done" && st.result) {
    const keys = HEADLINE[st.recipe] ?? [];
    for (const k of keys) {
      const v = st.result[k];
      if (typeof v === "number") return `${k} ${fmtNum(v)}`;
    }
    const first = Object.entries(st.result).find(([k, v]) => typeof v === "number" && k !== "steps");
    if (first) return `${first[0]} ${fmtNum(first[1])}`;
  }
  const last = st.last;
  if (last) {
    for (const k of ["loss", "reward_mean", "val_loss", "exact_match", "tokens", "chunks_embedded"]) {
      if (typeof last[k] === "number") return `${k} ${fmtNum(last[k])}${typeof last.step === "number" ? ` @${last.step}` : ""}`;
    }
    const first = Object.entries(last).find(([k, v]) => typeof v === "number" && k !== "step" && k !== "t");
    if (first) return `${first[0]} ${fmtNum(first[1])}`;
  }
  return st.status === "pending" ? "waiting" : "";
}

export function PipelineView({ pipelineId, refresh }: { pipelineId?: string; refresh: number }) {
  const { toast } = useToast();
  const templates = useAsync(() => api.pipelines.templates(), [], []);
  const ex = useAsync(() => api.lab.executors(), [], []);
  const list = useAsync(() => api.pipelines.list(), [], [refresh]);
  const [template, setTemplate] = useState("reasoning-nano");
  const [executor, setExecutor] = useState<"local" | "ssh" | "modal">("local");
  const [smoke, setSmoke] = useState(true);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(pipelineId ?? null);
  const [active, setActive] = useState<Pipeline | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (pipelineId) setSelected(pipelineId);
  }, [pipelineId]);
  useEffect(() => {
    const e = ex.data;
    if (!e) return;
    if (e.ssh.available) setExecutor("ssh");
    else if (e.modal.available) setExecutor("modal");
    else setExecutor("local");
  }, [ex.data]);
  // With nothing chosen, show the latest pipeline.
  useEffect(() => {
    if (!selected && list.data && list.data.length) setSelected(list.data[0].id);
  }, [list.data, selected]);

  const load = useCallback(async () => {
    if (!selected) {
      setActive(null);
      return;
    }
    try {
      setActive(await api.pipelines.get(selected));
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, [selected]);
  useEffect(() => {
    void load();
  }, [load, refresh]);
  // Poll while it runs (the runner ticks every few seconds on the server).
  useEffect(() => {
    if (!active || active.status !== "running") return;
    const t = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(t);
  }, [active, load]);

  const start = async () => {
    setBusy(true);
    try {
      const p = await api.pipelines.create({ template, executor, smoke, start: true });
      toast(`Started ${p.template} on ${p.executor}${smoke ? " (smoke)" : ""}`);
      setSelected(p.id);
      navigate({ kind: "lab", pipeline: true, pipelineId: p.id });
      list.reload();
      emitCommand("vault-changed");
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setBusy(false);
    }
  };
  const act = async (fn: () => Promise<unknown>, msg: string) => {
    try {
      await fn();
      toast(msg);
      await load();
      list.reload();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };

  const tdoc = templates.data?.find((t: PipelineTemplate) => t.name === template);
  const e = ex.data;
  const failedStage = active?.stages.find((s) => s.status === "failed");
  return (
    <div className="pipeline">
      <form
        className="pipeline-launch"
        onSubmit={(ev) => {
          ev.preventDefault();
          void start();
        }}
      >
        <label>
          <span>Template</span>
          <select className="select sm" value={template} onChange={(ev) => setTemplate(ev.target.value)}>
            {(templates.data ?? [{ name: "reasoning-nano", title: "Reasoning nano", doc: "", stages: [] }]).map((t: PipelineTemplate) => (
              <option key={t.name} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Where</span>
          <select className="select sm" value={executor} onChange={(ev) => setExecutor(ev.target.value as typeof executor)}>
            <option value="local" disabled={e ? !e.local.available : false}>this machine</option>
            <option value="ssh" disabled={e ? !e.ssh.available : true}>{e?.ssh.host ? `GPU box (${e.ssh.host})` : "GPU box over SSH"}</option>
            <option value="modal" disabled={e ? !e.modal.available : true}>Modal</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={smoke} onChange={(ev) => setSmoke(ev.target.checked)} />
          <span>smoke (CPU-sized, minutes)</span>
        </label>
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "Starting…" : "Start pipeline"}
        </button>
        <button className="btn" type="button" onClick={() => navigate({ kind: "note", slug: "lab-21-the-training-pie" })} title="The chapter that walks through this pipeline stage by stage">
          Read Lab 21
        </button>
        {tdoc && (
          <p className="doc">
            {tdoc.title}: {tdoc.doc} Stages: {tdoc.stages.map((s) => `${s.name} (${s.recipe})`).join(" → ")}.
          </p>
        )}
      </form>

      {list.data && list.data.length > 0 && (
        <div className="pipeline-list" aria-label="Pipelines">
          {list.data.map((p) => (
            <button key={p.id} type="button" className={`chip ${selected === p.id ? "on" : ""}`} onClick={() => { setSelected(p.id); navigate({ kind: "lab", pipeline: true, pipelineId: p.id }); }} title={`${p.title} · ${p.executor}${p.smoke ? " · smoke" : ""}`}>
              <span className={`dot ${p.status}`} aria-hidden="true" />
              {p.template} · {p.id.slice(0, 15)} · {p.progress.done}/{p.progress.total}
              {p.current ? ` · ${p.current}` : ""}
            </button>
          ))}
        </div>
      )}

      {err && <p className="muted small">{err}</p>}
      {!active && !err && <EmptyState title="No pipeline yet" hint="Pick a template, keep smoke on, and press Start: six stages run one after another on this machine in a few minutes." />}
      {active && (
        <section className="pipeline-active">
          <header>
            <h2>{active.title}</h2>
            <span className="status-pill" data-status={active.status}>{active.status}</span>
            <span className="muted small">
              {active.executor}{active.smoke ? " · smoke" : " · real"} · {active.progress.done}/{active.progress.total} stages · {active.id}
            </span>
            {active.status === "created" && <button className="btn sm" type="button" onClick={() => void act(() => api.pipelines.start(active.id), "Started")}>Start</button>}
            {active.status === "running" && <button className="btn sm" type="button" onClick={() => void act(() => api.pipelines.pause(active.id), "Paused: the running stage finishes, nothing new starts")}>Pause</button>}
            {active.status === "paused" && <button className="btn sm" type="button" onClick={() => void act(() => api.pipelines.start(active.id), "Resumed")}>Resume</button>}
            {failedStage && <button className="btn sm" type="button" onClick={() => void act(() => api.pipelines.retry(active.id, failedStage.name), `Retrying ${failedStage.name}`)}>Retry {failedStage.name}</button>}
            <button className="btn sm danger" type="button" onClick={() => { if (window.confirm("Delete this pipeline? Its runs are kept.")) void act(async () => { await api.pipelines.remove(active.id); setSelected(null); setActive(null); }, "Deleted"); }}>Delete</button>
            {active.error && <span className="small" style={{ color: "#d0453f" }}>{active.error}</span>}
          </header>
          <div className="pipeline-flow">
            <Flow stages={active.stages} />
          </div>
          <div className="pipeline-grid">
            <div className="pipeline-pie">
              <h3>Corpus composition</h3>
              {active.data?.sources ? <Pie sources={active.data.sources} total={active.data.total_tokens} unit={`${active.data.tokenizer} tokens`} /> : <p className="muted small">The pie fills in when the data stage prints its RESULT (tokens per source).</p>}
            </div>
            <div className="pipeline-report">
              <h3>Report</h3>
              <Report p={active} />
            </div>
          </div>
          <StageTable stages={active.stages} />
        </section>
      )}
    </div>
  );
}

/** Stage cards left to right in template order; solid arrows for the primary dependency, dashed arcs for the others. */
function Flow({ stages }: { stages: PipelineStage[] }) {
  const W = 168, H = 96, GAP = 40, PAD = 12, TOP = 10;
  const n = stages.length;
  const width = PAD * 2 + n * W + (n - 1) * GAP;
  const height = TOP + H + 44;
  const idx = useMemo(() => new Map(stages.map((s, i) => [s.name, i])), [stages]);
  const x = (i: number) => PAD + i * (W + GAP);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="Pipeline stages">
      <defs>
        <marker id="pipe-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" className="pipe-arrow" />
        </marker>
      </defs>
      {stages.map((s, i) =>
        s.deps.map((d, k) => {
          const j = idx.get(d);
          if (j == null) return null;
          if (k === 0 && j === i - 1) {
            return <line key={`${d}-${s.name}`} className="pipe-edge" x1={x(j) + W} y1={TOP + H / 2} x2={x(i)} y2={TOP + H / 2} markerEnd="url(#pipe-arrowhead)" />;
          }
          const x0 = x(j) + W / 2, x1 = x(i) + W / 2, y0 = TOP + H;
          const dip = y0 + 18 + 6 * Math.min(3, i - j);
          return <path key={`${d}-${s.name}`} className="pipe-edge far" d={`M ${x0} ${y0} C ${x0} ${dip}, ${x1} ${dip}, ${x1} ${y0}`} markerEnd="url(#pipe-arrowhead)" />;
        }),
      )}
      {stages.map((s, i) => (
        <g key={s.name} className={`pipe-card ${s.status}`} transform={`translate(${x(i)}, ${TOP})`}>
          <rect width={W} height={H} rx={10} />
          <text className="name" x={12} y={22}>{s.name}</text>
          <text className="st" x={W - 12} y={22} textAnchor="end">{s.status}</text>
          <text className="recipe" x={12} y={38}>{s.recipe}</text>
          <text className="metric" x={12} y={58}>{headline(s).slice(0, 26)}</text>
          {s.run_id ? (
            <a href={`#/lab/run/${encodeURIComponent(s.run_id)}`} onClick={(ev) => { ev.preventDefault(); navigate({ kind: "lab", run: s.run_id! }); }}>
              <text className="meta" x={12} y={78}>run {s.run_id.slice(0, 15)}</text>
            </a>
          ) : (
            <text className="meta" x={12} y={78}>{s.error ? s.error.slice(0, 24) : `after ${s.deps.join(", ") || "start"}`}</text>
          )}
          <text className="meta" x={W - 12} y={78} textAnchor="end">{fmtElapsed(s.elapsed_s)}</text>
        </g>
      ))}
    </svg>
  );
}

function Pie({ sources, total, unit }: { sources: Record<string, number>; total: number; unit: string }) {
  const entries = Object.entries(sources).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const sum = entries.reduce((a, [, v]) => a + v, 0) || 1;
  const R = 54, C = 60;
  let angle = -Math.PI / 2;
  const slices = entries.map(([k, v], i) => {
    const frac = v / sum;
    const a0 = angle;
    const a1 = angle + frac * 2 * Math.PI;
    angle = a1;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const p0 = [C + R * Math.cos(a0), C + R * Math.sin(a0)];
    const p1 = [C + R * Math.cos(a1), C + R * Math.sin(a1)];
    const d = frac >= 0.9999 ? `M ${C} ${C - R} A ${R} ${R} 0 1 1 ${C - 0.01} ${C - R} Z` : `M ${C} ${C} L ${p0[0]} ${p0[1]} A ${R} ${R} 0 ${large} 1 ${p1[0]} ${p1[1]} Z`;
    return { k, v, frac, d, color: PIE_COLORS[i % PIE_COLORS.length] };
  });
  return (
    <div className="pie">
      <svg viewBox="0 0 120 120" width={120} height={120} role="img" aria-label="Tokens per source">
        {slices.map((s) => (
          <path key={s.k} d={s.d} fill={s.color} stroke="var(--panel)" strokeWidth={1}>
            <title>{`${s.k}: ${s.v.toLocaleString()} (${(s.frac * 100).toFixed(1)}%)`}</title>
          </path>
        ))}
      </svg>
      <ul className="legend">
        {slices.map((s) => (
          <li key={s.k}>
            <i style={{ background: s.color }} />
            <b>{s.k}</b> {s.v.toLocaleString()} ({(s.frac * 100).toFixed(1)}%)
          </li>
        ))}
        <li className="muted">total {total.toLocaleString()} {unit}</li>
      </ul>
    </div>
  );
}

function Report({ p }: { p: Pipeline }) {
  const evalStage = p.stages.find((s) => s.recipe === "eval_suite");
  const r = evalStage?.result;
  if (r && typeof r.custom_exact_match === "number") {
    const after = r.custom_exact_match as number;
    const before = typeof r.baseline_exact_match === "number" ? (r.baseline_exact_match as number) : null;
    const delta = before == null ? null : after - before;
    return (
      <>
        <div className="big">
          {(after * 100).toFixed(1)}%
          <small>exact match, n={String(r.custom_n ?? "?")}, 95% CI [{fmtNum(r.custom_ci95_lo)}, {fmtNum(r.custom_ci95_hi)}]</small>
        </div>
        {before != null && (
          <div className="small">
            before (midtrain checkpoint) {(before * 100).toFixed(1)}% → after (RL checkpoint) {(after * 100).toFixed(1)}%{" "}
            <span className={`delta ${delta! >= 0 ? "up" : "down"}`}>{delta! >= 0 ? "+" : ""}{(delta! * 100).toFixed(1)} pts</span>
          </div>
        )}
        <pre>{JSON.stringify(r, null, 1)}</pre>
      </>
    );
  }
  const last = [...p.stages].reverse().find((s) => s.status === "done" && s.result);
  if (last) {
    return (
      <>
        <div className="small muted">latest finished stage: {last.name} ({last.recipe})</div>
        <pre>{JSON.stringify(last.result, null, 1)}</pre>
      </>
    );
  }
  return <p className="muted small">The before/after report appears when the eval stage finishes: exact match on held-out reasoning items for the RL checkpoint against the midtrain checkpoint, with bootstrap intervals.</p>;
}

function StageTable({ stages }: { stages: PipelineStage[] }) {
  return (
    <div className="pipeline-stages">
      <h3>Stages</h3>
      <table>
        <thead>
          <tr>
            <th>stage</th><th>recipe</th><th>status</th><th>run</th><th>headline</th><th>elapsed</th><th>arguments</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((s) => (
            <tr key={s.name}>
              <td className="name">{s.name}</td>
              <td>{s.recipe}</td>
              <td className={s.status === "failed" ? "err" : ""}>{s.status}{s.attempts > 1 ? ` (try ${s.attempts})` : ""}</td>
              <td>{s.run_id ? <a href={`#/lab/run/${encodeURIComponent(s.run_id)}`} onClick={(ev) => { ev.preventDefault(); navigate({ kind: "lab", run: s.run_id! }); }}>{s.run_id}</a> : ""}</td>
              <td>{headline(s)}</td>
              <td>{fmtElapsed(s.elapsed_s)}</td>
              <td className={s.error ? "err" : "args"}>{s.error ?? s.args ?? s.args_template}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
