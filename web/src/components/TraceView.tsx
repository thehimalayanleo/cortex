/**
 * One plan as a trace: span tree on the left with a waterfall bar per span, the picked span's details on the right.
 * Spans come from server/arm.py (Trace), shaped like OpenTelemetry: parent, start/end ms, status, attributes.
 */
import { useEffect, useMemo, useState } from "react";
import type { TraceSpan } from "../api";

const KIND: Record<string, string> = { root: "plan", attempt: "try", llm: "model", parse: "parse", sim: "sim", exec: "run" };
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);

function order(spans: TraceSpan[]): { s: TraceSpan; depth: number }[] {
  const kids = new Map<number | null, TraceSpan[]>();
  for (const s of spans) kids.set(s.parent, [...(kids.get(s.parent) ?? []), s]);
  const out: { s: TraceSpan; depth: number }[] = [];
  const walk = (p: number | null, d: number) => (kids.get(p) ?? []).forEach((s) => { out.push({ s, depth: d }); walk(s.id, d + 1); });
  walk(null, 0);
  return out;
}

function Value({ k, v }: { k: string; v: unknown }) {
  if (typeof v === "string" && (v.length > 80 || v.includes("\n"))) return <pre className="tr-pre">{v}</pre>;
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
    if (!v.length) return <span className="muted">none</span>;
    return <ul className={k === "errors" ? "tr-errs" : "tr-list"}>{v.map((x, i) => <li key={i}>{x}</li>)}</ul>;
  }
  if (typeof v === "object" && v !== null) {
    const lines = Array.isArray(v) ? v.map((x, i) => `${String(i + 1).padStart(2)} ${JSON.stringify(x)}`).join("\n") : JSON.stringify(v, null, 1);
    return <pre className="tr-pre">{lines || "none"}</pre>;
  }
  return <span className="mono">{String(v)}</span>;
}

export function TraceView({ spans }: { spans: TraceSpan[] }) {
  const rows = useMemo(() => order(spans), [spans]);
  const total = Math.max(1, ...spans.map((s) => s.end ?? s.start));
  const firstBad = rows.find((r) => r.s.status === "error" && r.s.kind !== "root" && r.s.kind !== "attempt")?.s.id;
  const [pick, setPick] = useState<number>(firstBad ?? 0);
  useEffect(() => setPick(firstBad ?? 0), [spans, firstBad]);
  const cur = spans.find((s) => s.id === pick) ?? spans[0];
  if (!cur) return null;
  const dur = (cur.end ?? cur.start) - cur.start;
  return (
    <div className="tr">
      <div className="tr-tree" role="listbox" aria-label="Spans">
        {rows.map(({ s, depth }) => {
          const d = (s.end ?? s.start) - s.start;
          return (
            <button key={s.id} role="option" aria-selected={s.id === pick} className={`tr-row ${s.status} ${s.id === pick ? "on" : ""}`} onClick={() => setPick(s.id)}>
              <span className="tr-name" style={{ paddingLeft: depth * 14 }}>
                <span className={`tr-dot ${s.status}`} />
                {s.name}
                <span className="tr-kind">{KIND[s.kind] ?? s.kind}</span>
              </span>
              <span className="tr-bar"><i style={{ left: `${(s.start / total) * 100}%`, width: `max(2px, ${(d / total) * 100}%)` }} /></span>
              <span className="tr-ms mono">{ms(d)}</span>
            </button>
          );
        })}
      </div>
      <div className="tr-detail">
        <div className="row">
          <b>{cur.name}</b>
          <span className={`verdict ${cur.status === "ok" ? "ok" : "bad"}`}>{cur.status === "ok" ? "passed" : "failed"}</span>
          <span className="grow" />
          <span className="muted small mono">{ms(dur)} · at {ms(cur.start)}</span>
        </div>
        {Object.entries(cur.attrs).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => (
          <div key={k} className="tr-attr">
            <div className="muted small">{k.replace(/_/g, " ")}</div>
            <Value k={k} v={v} />
          </div>
        ))}
      </div>
    </div>
  );
}
