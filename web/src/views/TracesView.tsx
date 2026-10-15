/**
 * Collect: the long-horizon collector. Anything goes in (notes, preferences as chosen/rejected, prompt/response
 * pairs, ratings, files) and nothing is ever rewritten. Export as SFT or DPO rows when it is time to train.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { TraceRec, TraceStats } from "../api";
import { useToast } from "../components/Toast";

const KINDS = ["note", "preference", "pair", "rating", "decision", "taste", "feedback", "idea", "link"];
// what the form offers; older kinds stay readable and filterable
const ADD: { kind: string; label: string; hint: string }[] = [
  { kind: "note", label: "Note", hint: "A thought, a decision, a link, or how you like things done." },
  { kind: "pair", label: "Good answer", hint: "A question and the answer you want a model to give. Exports as SFT data." },
  { kind: "preference", label: "A is better than B", hint: "Two answers to the same question, and which one you prefer. Exports as DPO data." },
  { kind: "rating", label: "Rating", hint: "A score from 1 to 5 for something, with a note on why." },
];

export function TracesView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [stats, setStats] = useState<TraceStats | null>(null);
  const [rows, setRows] = useState<TraceRec[]>([]);
  const [kind, setKind] = useState("note");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState("");
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [chosen, setChosen] = useState("");
  const [rejected, setRejected] = useState("");
  const [rating, setRating] = useState("");
  const [q, setQ] = useState("");
  const [filterKind, setFilterKind] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const load = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([api.traces.stats(), api.traces.list({ limit: 100, kind: filterKind || undefined, q: q || undefined })]);
      setStats(s);
      setRows(r);
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  }, [filterKind, q, toast]);
  useEffect(() => {
    void load();
  }, [load, refresh]);

  const add = async () => {
    try {
      await api.traces.add({
        kind, content: content || undefined, tags: tags.split(",").map((t) => t.trim()).filter(Boolean), source: "ui",
        prompt: prompt || undefined, response: response || undefined, chosen: chosen || undefined, rejected: rejected || undefined, rating: rating ? Number(rating) : undefined,
      });
      setContent(""); setPrompt(""); setResponse(""); setChosen(""); setRejected(""); setRating("");
      toast("Collected");
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const upload = async (f: File) => {
    try {
      await api.traces.upload(f, "file", tags, content);
      toast(`Collected ${f.name}`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    for (const f of Array.from(e.dataTransfer.files)) void upload(f);
  };

  const can = !!(content.trim() || (chosen && rejected) || (prompt && response));
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && can) { e.preventDefault(); void add(); }
  };
  const cur = ADD.find((x) => x.kind === kind) ?? ADD[0];

  return (
    <div className={`traces ${drag ? "cl-drop" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}>
      <div className="cl-head">
        <h2>Collect</h2>
        <p>Save what you know, prefer and like. Later, export it as training data for the Post-train tab. Files can be dropped anywhere on this page.</p>
      </div>

      <div className="cl-card">
        <div className="cl-kinds" role="radiogroup" aria-label="What to save">
          {ADD.map((x) => (
            <button key={x.kind} role="radio" aria-checked={kind === x.kind} className={`cl-kind ${kind === x.kind ? "on" : ""}`} onClick={() => setKind(x.kind)}>{x.label}</button>
          ))}
        </div>
        <div className="cl-hint">{cur.hint}</div>
        {kind === "pair" && (
          <div className="cl-two">
            <label>Question<textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={onKey} rows={3} /></label>
            <label>Answer you want<textarea value={response} onChange={(e) => setResponse(e.target.value)} onKeyDown={onKey} rows={3} /></label>
          </div>
        )}
        {kind === "preference" && (
          <>
            <label className="cl-field">Question (optional)<textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={onKey} rows={2} /></label>
            <div className="cl-two">
              <label>Better answer (A)<textarea value={chosen} onChange={(e) => setChosen(e.target.value)} onKeyDown={onKey} rows={3} /></label>
              <label>Worse answer (B)<textarea value={rejected} onChange={(e) => setRejected(e.target.value)} onKeyDown={onKey} rows={3} /></label>
            </div>
          </>
        )}
        {kind === "rating" && (
          <label className="cl-field">Score (1 to 5)<input className="cl-in" style={{ maxWidth: 120 }} value={rating} onChange={(e) => setRating(e.target.value)} inputMode="numeric" /></label>
        )}
        <label className="cl-field">{kind === "note" ? "Note" : "Why (optional)"}<textarea value={content} onChange={(e) => setContent(e.target.value)} onKeyDown={onKey} rows={kind === "note" ? 4 : 2} /></label>
        <label className="cl-field">Tags (optional, comma separated)<input className="cl-in" value={tags} onChange={(e) => setTags(e.target.value)} /></label>
        <div className="cl-actions">
          <button className="btn primary" onClick={() => void add()} disabled={!can}>Collect</button>
          <span className="cl-hint">or press Cmd+Enter</span>
          <span className="grow" />
          <button className="btn" onClick={() => fileRef.current?.click()}>Add a file</button>
          <input ref={fileRef} type="file" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
        </div>
      </div>

      <div className="cl-stats">
        <span><b>{stats?.total ?? 0}</b>saved</span>
        <span><b>{stats?.sft_pairs ?? 0}</b>good answers (SFT)</span>
        <span><b>{stats?.dpo_pairs ?? 0}</b>A-over-B pairs (DPO)</span>
        <span className="grow" />
        <a className="btn sm" href={api.traces.exportUrl("sft")} target="_blank" rel="noreferrer">Export SFT</a>
        <a className="btn sm" href={api.traces.exportUrl("dpo")} target="_blank" rel="noreferrer">Export DPO</a>
        <a className="btn sm" href={api.traces.exportUrl("all")} target="_blank" rel="noreferrer">Export all</a>
      </div>

      <div className="cl-filter">
        <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search what you saved" aria-label="Search" />
        <select className="select" value={filterKind} onChange={(e) => setFilterKind(e.target.value)} aria-label="Show">
          <option value="">everything</option>
          {KINDS.concat(["file", "chat", "screenshot"]).map((k) => (
            <option key={k} value={k}>{ADD.find((x) => x.kind === k)?.label ?? k}{stats?.by_kind[k] ? ` · ${stats.by_kind[k]}` : ""}</option>
          ))}
        </select>
      </div>
      <ul className="trace-list">
        {rows.map((r) => (
          <li key={r.id} className={`trace k-${r.kind}`}>
            <div className="trace-head">
              <span className="kind">{ADD.find((x) => x.kind === r.kind)?.label ?? r.kind}</span>
              <span className="muted small">{r.when}{r.source ? ` · ${r.source}` : ""}{r.tags?.length ? ` · ${r.tags.join(", ")}` : ""}</span>
            </div>
            {r.content && <div className="trace-body">{r.content}</div>}
            {r.prompt && <div className="trace-pair"><span className="lbl">Question</span>{r.prompt}</div>}
            {r.response && <div className="trace-pair"><span className="lbl">Answer</span>{r.response}</div>}
            {r.chosen && <div className="trace-pair chosen"><span className="lbl">Better (A)</span>{r.chosen}</div>}
            {r.rejected && <div className="trace-pair rejected"><span className="lbl">Worse (B)</span>{r.rejected}</div>}
            {r.rating != null && <div className="trace-pair"><span className="lbl">Score</span>{r.rating}</div>}
            {r.file && <div className="trace-pair"><span className="lbl">File</span>{r.file}</div>}
          </li>
        ))}
        {rows.length === 0 && <li className="cl-hint">{q || filterKind ? "Nothing matches." : "Nothing saved yet. Write a note above, or drop a file on this page."}</li>}
      </ul>
    </div>
  );
}
