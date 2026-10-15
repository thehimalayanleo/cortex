/**
 * Chat with two models side by side: any Mac base model or any trained job. One message goes to both; each side
 * keeps its own conversation. Served by one local mlx_lm server that loads one model at a time
 * (server/posttrain.py chat), so the two replies come one after the other.
 */
import { useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { PtChatTarget } from "../api";

type Msg = { role: "user" | "assistant"; content: string; seconds?: number };

export function CompareChat({ left, right, onPick }: { left: string; right: string; onPick: (side: "left" | "right", id: string) => void }) {
  const [targets, setTargets] = useState<PtChatTarget[]>([]);
  const [hist, setHist] = useState<{ left: Msg[]; right: Msg[] }>({ left: [], right: [] });
  const [text, setText] = useState("");
  const [busy, setBusy] = useState<"" | "left" | "right">("");
  const [err, setErr] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { api.post.chatTargets().then(setTargets).catch((e) => setErr(errorMessage(e))); }, [left, right]);
  useEffect(() => { setHist({ left: [], right: [] }); }, [left, right]);
  useEffect(() => { void endRef.current?.scrollIntoView({ block: "nearest" }); }, [hist]);  // braces: newer browsers return a Promise here, and React would call it as a cleanup

  const send = async () => {
    const q = text.trim();
    if (!q || busy) return;
    setText(""); setErr("");
    let h = { left: [...hist.left, { role: "user", content: q } as Msg], right: [...hist.right, { role: "user", content: q } as Msg] };
    setHist(h);
    for (const side of ["left", "right"] as const) {
      setBusy(side);
      try {
        const r = await api.post.chat(side === "left" ? left : right, h[side].map(({ role, content }) => ({ role, content })));
        h = { ...h, [side]: [...h[side], { role: "assistant", content: r.reply, seconds: r.seconds }] };
      } catch (e) {
        h = { ...h, [side]: [...h[side], { role: "assistant", content: `(no reply: ${errorMessage(e)})` }] };
      }
      setHist(h);
    }
    setBusy("");
  };

  const col = (side: "left" | "right") => (
    <div className="cc-col">
      <select className="select" value={side === "left" ? left : right} onChange={(e) => onPick(side, e.target.value)} aria-label={`${side} model`}>
        {targets.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
      </select>
      <div className="cc-log">
        {hist[side].length === 0 && <div className="cc-empty">Ask something below. Both models get the same message.</div>}
        {hist[side].map((m, i) => (
          <div key={i} className={`cc-msg ${m.role}`}>
            <div className="cc-text">{m.content}</div>
            {m.seconds != null && <div className="cc-meta">{m.seconds} s</div>}
          </div>
        ))}
        {busy === side && <div className="cc-msg assistant cc-wait">Thinking… (the first reply loads the model)</div>}
      </div>
    </div>
  );

  return (
    <div className="cc">
      <div className="cc-cols">{col("left")}{col("right")}</div>
      <div ref={endRef} />
      {err && <div className="pt-bad small">{err}</div>}
      <div className="cc-input">
        <textarea value={text} rows={2} placeholder="Message both models (Enter sends, Shift+Enter for a new line)" onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }} />
        <div className="cc-actions">
          <button className="btn primary" disabled={!!busy || !text.trim()} onClick={() => void send()}>{busy ? "Waiting…" : "Send"}</button>
          <button className="btn" disabled={!!busy || (!hist.left.length && !hist.right.length)} onClick={() => setHist({ left: [], right: [] })}>Clear</button>
        </div>
      </div>
    </div>
  );
}
