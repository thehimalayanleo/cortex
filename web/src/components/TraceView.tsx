/**
 * One plan in plain words: a summary line, then one row per try with a pass/fail mark for each check and the first
 * problem. Raw model reply and program sit behind "details". Spans come from server/arm.py (Trace).
 */
import { useState } from "react";
import type { TraceSpan } from "../api";

const STEP: Record<string, string> = {
  "model call": "Model writes a plan",
  "parse reply": "Plan is readable",
  compile: "Arm can reach every point",
  simulate: "No collision or drop",
  "check goal": "Ends in the stated goal",
  "start motion": "Arm moves",
};
const secs = (s: TraceSpan) => `${(((s.end ?? s.start) - s.start) / 1000).toFixed(1)} s`;
const errs = (s: TraceSpan) => (Array.isArray(s.attrs.errors) ? (s.attrs.errors as string[]) : [])
  .concat(s.attrs.error ? [s.kind === "parse" ? `the reply is not valid JSON, so the simulator cannot read the plan (${s.attrs.error})` : String(s.attrs.error)] : []);
const clean = (e: string) => e.replace(/^step 0: /, "").replace(/^step (\d+): /, "plan step $1: ").replace(/^your declared goal does not hold after the program: /, "goal not met: ");

function Try({ n, span, steps }: { n: number; span: TraceSpan; steps: TraceSpan[] }) {
  const [open, setOpen] = useState(false);
  const ok = span.status === "ok";
  const problem = steps.flatMap(errs)[0];
  const llm = steps.find((s) => s.kind === "llm");
  const parse = steps.find((s) => s.kind === "parse");
  return (
    <div className={`tr-try ${ok ? "ok" : "bad"}`}>
      <div className="tr-head">
        <b>{n === 1 ? "Try 1" : `Try ${n} (errors sent back to the model)`}</b>
        <span className={ok ? "tr-ok" : "tr-bad"}>{ok ? "passed" : "failed"}</span>
        <span className="muted">{secs(span)}</span>
        <span className="grow" />
        <button className="lnk small" onClick={() => setOpen((v) => !v)}>{open ? "Hide details" : "Details"}</button>
      </div>
      <div className="tr-steps">
        {steps.filter((s) => STEP[s.name]).map((s) => (
          <span key={s.id} className={`tr-step ${s.status}`}>{s.status === "ok" ? "✓" : "✗"} {STEP[s.name]}</span>
        ))}
      </div>
      {problem && <div className="tr-problem"><b>Problem:</b> {clean(problem)}</div>}
      {open && (
        <div className="tr-more">
          {steps.flatMap(errs).length > 1 && <ul className="tr-errs">{steps.flatMap(errs).map((e, i) => <li key={i}>{clean(e)}</li>)}</ul>}
          {llm && <div className="muted small">{String(llm.attrs.model ?? "")} · {secs(llm)}</div>}
          {n > 1 && llm?.attrs.input != null && <><div className="muted small">Message to the model</div><pre className="tr-pre">{String(llm.attrs.input)}</pre></>}
          {Array.isArray(parse?.attrs.program) && (
            <><div className="muted small">Plan from the model</div>
              <pre className="tr-pre">{(parse!.attrs.program as unknown[]).map((op, i) => `${String(i + 1).padStart(2)} ${JSON.stringify(op)}`).join("\n")}</pre></>
          )}
          {!parse && llm?.attrs.output != null && <><div className="muted small">Reply from the model</div><pre className="tr-pre">{String(llm.attrs.output)}</pre></>}
        </div>
      )}
    </div>
  );
}

export function TraceView({ spans }: { spans: TraceSpan[] }) {
  const root = spans.find((s) => s.parent === null);
  if (!root) return null;
  const kids = (id: number) => spans.filter((s) => s.parent === id);
  const tries = kids(root.id).filter((s) => s.kind === "attempt");
  const groups = tries.length ? tries.map((t) => ({ span: t, steps: kids(t.id) })) : [{ span: root, steps: kids(root.id).filter((s) => s.kind !== "exec") }];
  const moved = kids(root.id).some((s) => s.kind === "exec");
  const ok = root.status === "ok";
  const last = groups[groups.length - 1];
  const lastProblem = last.steps.flatMap(errs)[0];
  const n = groups.length;
  return (
    <div className="tr">
      <div className={`tr-sum ${ok ? "ok" : "bad"}`}>
        {ok
          ? `Passed on try ${n} of 3. ${moved ? "The arm moved." : "The arm did not move yet."} Time: ${secs(root)}.`
          : `The simulator rejected ${n} ${n === 1 ? "try" : "tries"}. The arm did not move. Last problem: ${lastProblem ? clean(lastProblem) : "unknown"}.`}
      </div>
      <div className="tr-note">The model writes a plan, and the simulator checks it. If a check fails, the lab sends the errors back to the model, for a maximum of 3 tries. Only a plan that passes all checks can move the arm.</div>
      {groups.map((g, i) => <Try key={g.span.id} n={i + 1} span={g.span} steps={g.steps} />)}
    </div>
  );
}
