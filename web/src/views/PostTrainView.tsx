/**
 * Post-train: a guided page for someone who has never post-trained a model. One line says what it is; a quick
 * start runs a complete example; three plain choices (skill, model, method) start a job; the job trains, tests on
 * new items with the skill's checker, tests that refusal holds, and says if the model got better. "Try it" asks the
 * same question to the model before and after training. UI text is ASD-STE100 style.
 */
import { useCallback, useEffect, useState } from "react";
import { api, errorMessage } from "../api";
import type { PtJob, PtOverview, PtPreview } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const pct = (v?: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const SKILL: Record<string, { title: string; line: string; check: string; example: string }> = {
  code: { title: "Python functions", line: "Write a short function from a description.", check: "running its unit tests", example: "Write a function that returns the second largest number in a list." },
  gsm8k: { title: "Math", line: "Solve short word problems.", check: "the final number", example: "A shop sells 3 pens for $2. How much do 12 pens cost?" },
  arm: { title: "Robot arm plans", line: "Write a plan that moves blocks on a table.", check: "the arm simulator", example: "Put the red block in the bin." },
};
const METHOD: Record<string, { title: string; line: string }> = {
  sft: { title: "Show it answers", line: "SFT. The model copies correct answers. Start here." },
  grpo: { title: "Let it try", line: "GRPO. The model tries; the checker scores each try. Use after SFT." },
};

type Score = { task_score: number; refusal: number; over_refusal?: number };
type Check = { base: Score; tuned: Score; ok: boolean; why: string };
function checks(j: PtJob): Check | null {
  const r = j.result as Record<string, any> | null;
  if (!r || !r.base || !r.tuned) return null;
  const arm = "plan_success" in r.tuned;  // the 5090 arm runs report their own field names
  const base = arm ? { task_score: r.base.plan_success, refusal: r.base.advbench_refusal } : r.base;
  const tuned = arm ? { task_score: r.tuned.plan_success, refusal: r.tuned.advbench_refusal } : r.tuned;
  const better = tuned.task_score > base.task_score, safe = tuned.refusal >= base.refusal - 0.1;
  const overUp = (tuned.over_refusal ?? 0) > (base.over_refusal ?? 0) + 0.1;
  // the lab's own verdict (lab/recipes/mlx_post.py) decides; this only words it
  const ok = r.verdict ? !!r.verdict.promote : better && safe && !overUp;
  const why = !better ? "The skill score did not increase." : !safe ? "The model now refuses fewer harmful requests."
    : overUp ? "The model now refuses normal tasks too. It learned to say no to everything." : "The skill score increased, and the model still refuses harmful requests.";
  return { base, tuned, ok, why };
}
const stage = (j: PtJob) => (j.status === "done" ? 2 : j.log.some((l) => l.includes('"phase": "eval"')) ? 1 : 0);

function Curve({ pts }: { pts: number[] }) {
  if (pts.length < 2) return <div className="pt-curve muted small">The loss curve shows here when training starts.</div>;
  const lo = Math.min(...pts), hi = Math.max(...pts), W = 560, H = 90;
  const d = pts.map((v, i) => `${i ? "L" : "M"}${(i / (pts.length - 1)) * W},${H - ((v - lo) / (hi - lo || 1)) * (H - 6) - 3}`).join(" ");
  return (
    <div className="pt-curve">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"><path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.6" vectorEffect="non-scaling-stroke" /></svg>
      <span className="muted small mono">loss {pts[0].toFixed(2)} → {pts[pts.length - 1].toFixed(2)} (lower is better)</span>
    </div>
  );
}

function Option({ on, disabled, onClick, title, line, foot }: { on: boolean; disabled?: boolean; onClick: () => void; title: string; line: string; foot?: string }) {
  return (
    <button className={`pt-opt${on ? " on" : ""}`} disabled={disabled} onClick={onClick}>
      <b>{title}</b>
      <span className="small">{line}</span>
      {foot && <span className="muted small">{foot}</span>}
    </button>
  );
}

function DataView({ task, model }: { task: string; model: string }) {
  const { toast } = useToast();
  const [view, setView] = useState<"sft" | "rl">("sft");
  const [data, setData] = useState<PtPreview | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setData(null); }, [task]);
  const show = () => {
    setBusy(true);
    api.post.preview(task, model).then(async (r) => {
      for (let i = 0; i < 120; i++) {
        await new Promise((ok) => setTimeout(ok, 2000));
        const run = await api.lab.run(r.id, 5);
        if (run.status === "done") { setData(run.result as unknown as PtPreview); return; }
        if (run.status === "failed") throw new Error("The preview failed. Open the run log.");
      }
    }).catch((e) => toast(errorMessage(e), "error")).finally(() => setBusy(false));
  };
  return (
    <section className="pt2-step">
      <div className="row">
        <h3>See the data</h3>
        <div className="seg">
          <button className={`btn sm${view === "sft" ? " primary" : ""}`} onClick={() => setView("sft")}>SFT data</button>
          <button className={`btn sm${view === "rl" ? " primary" : ""}`} onClick={() => setView("rl")}>RL data</button>
        </div>
        <span className="grow" />
        <button className="btn sm" disabled={busy} onClick={show}>{busy ? "Sampling…" : data ? "Show new examples" : "Show the data"}</button>
      </div>
      <div className="muted small">
        {view === "sft" ? "SFT: each example is one question and one correct answer. The model reads the question. It learns to write the answer." :
          "RL: the model writes several answers to one question. The checker scores each answer. Answers above the group average get pushed up; answers below get pushed down."}
      </div>
      {!data ? <div className="muted small">Push "Show the data". The lab takes examples from the task and, for RL, samples 4 answers per question on this Mac (about 15 seconds).</div>
        : view === "sft" ? (
          <div className="pt-data">
            {data.sft.map((r, i) => (
              <div key={i} className="pt-sft">
                <div className="pt-sft-in"><span className="muted small">The model reads ({r.prompt_tokens} tokens, no training signal)</span><pre>{r.prompt}</pre></div>
                <div className="pt-sft-out"><span className="small">The model learns to write ({r.answer_tokens} tokens, trained)</span><pre>{r.answer}</pre></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="pt-data">
            {data.rl.map((g, i) => (
              <div key={i} className="pt-group">
                <pre className="pt-q">{g.prompt}</pre>
                <div className="muted small">Group average reward {g.mean.toFixed(2)}.{g.std === 0 ? " All answers got the same score, so this question teaches nothing this time." : ""}</div>
                <div className="pt-rolls">
                  {g.rollouts.map((o, k) => {
                    const dir = o.advantage > 0.05 ? "up" : o.advantage < -0.05 ? "down" : "flat";
                    return (
                      <div key={k} className={`pt-roll ${dir}`}>
                        <div className="row"><b className="small">Answer {k + 1}</b><span className="grow" />
                          <span className="mono small">{o.tests ? `${o.tests.filter((t) => t.passed).length}/${o.tests.length} tests` : `reward ${o.reward}`}</span></div>
                        <div className="pt-adv"><span style={{ width: `${Math.min(100, Math.abs(o.advantage) * 40)}%` }} /></div>
                        <div className="small">{dir === "up" ? `Pushed up (advantage +${o.advantage.toFixed(2)})` : dir === "down" ? `Pushed down (advantage ${o.advantage.toFixed(2)})` : "No change (advantage 0)"}</div>
                        {o.tests && <div className="pt-tests">{o.tests.map((t, j) => <span key={j} className={t.passed ? "ok" : "bad"} title={t.test}>{t.passed ? "✓" : "✗"}</span>)}</div>}
                        <details><summary className="muted small">code</summary><pre>{o.completion}</pre></details>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
    </section>
  );
}

export function PostTrainView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [ov, setOv] = useState<PtOverview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [task, setTask] = useState("arm");
  const [method, setMethod] = useState("sft");
  const [model, setModel] = useState("mlx-community/Qwen3-0.6B-4bit");
  const [iters, setIters] = useState(60);
  const [safety, setSafety] = useState(true);
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [q, setQ] = useState("");
  const [tryOut, setTryOut] = useState<{ before: string; after: string } | null>(null);
  const [trying, setTrying] = useState(false);

  const load = useCallback(() => api.post.overview().then((o) => { setOv(o); setErr(null); }).catch((e) => setErr(errorMessage(e))), []);
  useEffect(() => { void load(); }, [load, refresh]);
  const running = ov?.jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(t);
  }, [running, load]);
  useEffect(() => { setQ(SKILL[task]?.example ?? ""); setTryOut(null); }, [task, sel]);

  if (err) return <EmptyState title="The lab is not available" hint={err} />;
  if (!ov) return <EmptyState title="Loading the lab" />;
  const job = ov.jobs.find((j) => j.id === sel) ?? ov.jobs[0];
  const c = job ? checks(job) : null;
  const where = (m: PtOverview["models"][number]) => m.engine !== "mlx" ? "on the 5090" : m.fits?.[method] ? `on this Mac · ~${m.estimate_gb?.[method]} GB` : "too big for this Mac";
  const usable = (m: PtOverview["models"][number]) => m.engine === "mlx" ? !!m.fits?.[method] : task === "arm" && method === "sft";
  const cur = ov.models.find((m) => m.id === model);

  const start = (o?: { task: string; method: string; model: string; iters: number }) => {
    setBusy(true);
    api.post.start({ task: o?.task ?? task, method: o?.method ?? method, model: o?.model ?? model, iters: o?.iters ?? iters, safety })
      .then((j) => { setSel(j.id); toast("Training started"); return load(); })
      .catch((e) => toast(errorMessage(e), "error")).finally(() => setBusy(false));
  };
  const ask = () => {
    if (!job || !q.trim()) return;
    setTrying(true);
    api.post.try(job.id, q).then(setTryOut).catch((e) => toast(errorMessage(e), "error")).finally(() => setTrying(false));
  };

  return (
    <div className="pt2">
      <header className="pt2-head">
        <h2>Post-train a model</h2>
        <p className="muted">Post-training teaches a model one more skill. The lab trains the model, tests it on new questions, and tells you if it got better.</p>
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={() => { const o = { task: "arm", method: "sft", model: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", iters: 75 }; setTask(o.task); setMethod(o.method); setModel(o.model); start(o); }}>
            Quick start: teach a 4B NVIDIA model to plan robot-arm moves
          </button>
          <span className="muted small">Runs on the 5090. About 10 minutes.</span>
        </div>
        <div className="muted small">The small models on this Mac are good for learning the steps. They are too small to learn a skill well and stay safe.</div>
      </header>

      <section className="pt2-step">
        <h3><span className="pt-n">1</span> What should it learn?</h3>
        <div className="pt-opts">
          {["code", "gsm8k", "arm"].map((t) => <Option key={t} on={task === t} onClick={() => setTask(t)} title={SKILL[t].title} line={SKILL[t].line} foot={`Checked by ${SKILL[t].check}`} />)}
        </div>
      </section>

      <section className="pt2-step">
        <h3><span className="pt-n">2</span> Which model?</h3>
        <div className="pt-opts">
          {ov.models.map((m) => <Option key={m.id} on={model === m.id} disabled={!usable(m)} onClick={() => setModel(m.id)} title={m.label} line={where(m)} />)}
        </div>
      </section>

      <section className="pt2-step">
        <h3><span className="pt-n">3</span> How should it learn?</h3>
        <div className="pt-opts">
          {["sft", "grpo"].map((x) => <Option key={x} on={method === x} onClick={() => setMethod(x)} title={METHOD[x].title} line={METHOD[x].line} />)}
        </div>
        <div className="row">
          <button className="btn primary" disabled={busy || !cur || !usable(cur)} onClick={() => start()}>{busy ? "Starting…" : "Start training"}</button>
          <details className="pt-more">
            <summary className="muted small">Settings</summary>
            <label className="small">Steps <input className="input sm arm-seed" type="number" min={4} value={iters} onChange={(e) => setIters(Number(e.target.value) || 20)} /></label>
            {task === "arm" && <label className="small arm-check"><input type="checkbox" checked={safety} onChange={(e) => setSafety(e.target.checked)} /> Also teach it to refuse unsafe arm requests</label>}
          </details>
        </div>
      </section>

      <DataView task={task} model={model} />

      {job && (
        <section className="pt2-step pt2-run">
          <div className="row">
            <h3>{SKILL[job.task]?.title ?? job.task} · {METHOD[job.method]?.title ?? job.method} · {job.model.split("/").pop()}</h3>
            <span className="grow" />
            <button className="lnk small" onClick={() => navigate({ kind: "lab", run: job.run })}>log</button>
          </div>
          <div className="pt-flow">
            {["1 Train", "2 Test on new questions", "3 Decide"].map((s, i) => {
              const at = job.status === "failed" ? -1 : stage(job);
              return <span key={s} className={`pt-flow-s ${at > i || (at === 2 && i === 2) ? "done" : at === i ? "now" : ""}`}>{s}</span>;
            })}
          </div>
          <Curve pts={job.metrics.filter((x) => x.loss != null).map((x) => x.loss!)} />
          {job.status === "failed" && <div className="pt-bad">The job failed. Open the log.</div>}
          {c ? (
            <>
              <div className={`pt-verdict ${c.ok ? "ok" : "bad"}`}>{c.ok ? "Better. You can use this model." : "Not better. Do not use this model."}</div>
              <div className="small">{c.why}</div>
              <div className="pt-nums">
                <div><div className="muted small">Skill score on new questions</div><div className="big">{pct(c.base.task_score)} → {pct(c.tuned.task_score)}</div></div>
                <div><div className="muted small">Refuses harmful requests</div><div className="big">{pct(c.base.refusal)} → {pct(c.tuned.refusal)}</div></div>
              </div>
              <div className="muted small">Left number: before training. Right number: after training.</div>
              {job.where.includes("Mac") && (
                <div className="pt-try">
                  <b>Try it</b>
                  <div className="row">
                    <input className="input" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") ask(); }} />
                    <button className="btn sm primary" disabled={trying} onClick={ask}>{trying ? "Asking…" : "Ask both"}</button>
                  </div>
                  {tryOut && (
                    <div className="pt-cmp">
                      <div><div className="muted small">Before training</div><pre>{tryOut.before}</pre></div>
                      <div><div className="muted small">After training</div><pre>{tryOut.after}</pre></div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : job.status !== "failed" && <div className="muted small">{stage(job) === 0 ? "The model is training." : "The lab is testing the model on new questions."}</div>}
        </section>
      )}

      {ov.jobs.length > 1 && (
        <details className="pt2-step">
          <summary className="muted small">Earlier jobs ({ov.jobs.length})</summary>
          <table className="arm-grid pt-jobs">
            <tbody>
              {ov.jobs.map((j) => {
                const cc = checks(j);
                return (
                  <tr key={j.id} className={j.id === job?.id ? "sel" : ""} onClick={() => setSel(j.id)}>
                    <td className="mono">{j.started.replace("T", " ").slice(5, 16)}</td>
                    <td>{SKILL[j.task]?.title ?? j.task} · {j.method}</td>
                    <td className="mono">{j.model.split("/").pop()}</td>
                    <td>{cc ? `${pct(cc.base.task_score)} → ${pct(cc.tuned.task_score)}` : j.status}</td>
                    <td className={cc ? (cc.ok ? "ok" : "bad") : ""}>{cc ? (cc.ok ? "better" : "not better") : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
