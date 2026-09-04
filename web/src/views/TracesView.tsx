/**
 * Traces: the long-horizon collector. Anything goes in (notes, preferences as chosen/rejected, prompt/response
 * pairs, ratings, files) and nothing is ever rewritten. Export as SFT or DPO rows when it is time to train.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { TraceRec, TraceStats } from "../api";
import { useToast } from "../components/Toast";

const KINDS = ["note", "preference", "pair", "rating", "decision", "taste", "feedback", "idea", "link"];

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
    for (const f of Array.from(e.dataTransfer.files)) void upload(f);
  };

  return (
    <div className="traces" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <div className="traces-stats">
        <div className="stat-block"><b>{stats?.total ?? 0}</b><span className="muted small">traces{stats?.since ? ` since ${stats.since}` : ""}</span></div>
        <div className="stat-block"><b>{stats?.sft_pairs ?? 0}</b><span className="muted small">SFT pairs</span></div>
        <div className="stat-block"><b>{stats?.dpo_pairs ?? 0}</b><span className="muted small">DPO preferences</span></div>
        <div className="stat-block"><b>{stats ? Object.keys(stats.by_kind).length : 0}</b><span className="muted small">kinds</span></div>
        <span className="grow" />
        <a className="btn sm" href={api.traces.exportUrl("sft")} target="_blank" rel="noreferrer">Export SFT</a>
        <a className="btn sm" href={api.traces.exportUrl("dpo")} target="_blank" rel="noreferrer">Export DPO</a>
        <a className="btn sm" href={api.traces.exportUrl("all")} target="_blank" rel="noreferrer">Export all</a>
      </div>
      <p className="muted small">Everything lands as one JSON line per record in your vault under traces/YYYY-MM, files alongside. Send from here, from the chat ("collect this: …"), from an agent, or by dropping files onto this panel. Years from now this is the dataset.</p>

      <div className="trace-form">
        <div className="row">
          <select className="select sm" value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Kind">
            {KINDS.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <input className="input sm grow" value={tags} onChange={(e) => setTags(e.target.value)} placeholder="tags, comma separated" aria-label="Tags" />
          {kind === "rating" && <input className="input sm" style={{ width: 80 }} value={rating} onChange={(e) => setRating(e.target.value)} placeholder="1-5" aria-label="Rating" />}
          <button className="btn sm" onClick={() => fileRef.current?.click()}>Attach file</button>
          <input ref={fileRef} type="file" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
        </div>
        {kind === "preference" ? (
          <div className="pair">
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} placeholder="Context or prompt (optional)" />
            <textarea value={chosen} onChange={(e) => setChosen(e.target.value)} rows={3} placeholder="What I preferred" />
            <textarea value={rejected} onChange={(e) => setRejected(e.target.value)} rows={3} placeholder="What I rejected" />
          </div>
        ) : kind === "pair" ? (
          <div className="pair">
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} placeholder="Prompt or question" />
            <textarea value={response} onChange={(e) => setResponse(e.target.value)} rows={3} placeholder="The answer I want a model to give" />
          </div>
        ) : null}
        <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={3} placeholder={kind === "preference" ? "Why (optional)" : "Anything: a thought, a decision, a link, how you like things done…"} />
        <div className="row">
          <span className="muted small">Cmd+Enter collects</span>
          <span className="grow" />
          <button className="primary sm" onClick={() => void add()} disabled={!(content.trim() || (chosen && rejected) || (prompt && response))}>Collect</button>
        </div>
      </div>

      <div className="row traces-filter">
        <input className="input sm grow" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search traces" aria-label="Search traces" />
        <select className="select sm" value={filterKind} onChange={(e) => setFilterKind(e.target.value)} aria-label="Filter kind">
          <option value="">all kinds</option>
          {KINDS.concat(["file", "chat", "screenshot"]).map((k) => (
            <option key={k} value={k}>{k}{stats?.by_kind[k] ? ` · ${stats.by_kind[k]}` : ""}</option>
          ))}
        </select>
      </div>
      <ul className="trace-list">
        {rows.map((r) => (
          <li key={r.id} className={`trace k-${r.kind}`}>
            <div className="trace-head">
              <span className="kind">{r.kind}</span>
              <span className="muted small">{r.when}{r.source ? ` · ${r.source}` : ""}{r.tags?.length ? ` · ${r.tags.join(", ")}` : ""}</span>
            </div>
            {r.content && <div className="trace-body">{r.content}</div>}
            {r.prompt && <div className="trace-pair"><span className="lbl">prompt</span>{r.prompt}</div>}
            {r.response && <div className="trace-pair"><span className="lbl">response</span>{r.response}</div>}
            {r.chosen && <div className="trace-pair chosen"><span className="lbl">chosen</span>{r.chosen}</div>}
            {r.rejected && <div className="trace-pair rejected"><span className="lbl">rejected</span>{r.rejected}</div>}
            {r.rating != null && <div className="trace-pair"><span className="lbl">rating</span>{r.rating}</div>}
            {r.file && <div className="trace-pair"><span className="lbl">file</span>{r.file}</div>}
          </li>
        ))}
        {rows.length === 0 && <li className="muted small">Nothing collected yet.</li>}
      </ul>
    </div>
  );
}
