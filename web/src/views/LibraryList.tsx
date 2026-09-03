import { useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { api, errorMessage } from "../api";
import type { PaperMeta, PaperStatus } from "../types";
import { PAPER_STATUSES } from "../types";
import { useAsync } from "../lib/hooks";
import { emitCommand } from "../lib/events";
import { navigate } from "../lib/router";
import { authorsLine, relDate, titleCase, truncate } from "../lib/format";
import { EmptyState, ErrorState, Loading } from "../components/States";
import { useToast } from "../components/Toast";

export function LibraryList({ status, topic, refresh }: { status?: string; topic?: string; refresh: number }) {
  const { toast } = useToast();
  const papers = useAsync(() => api.library.list(topic ? { topic } : {}), [topic], [refresh]);
  const all = papers.data ?? [];
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of all) c[p.status] = (c[p.status] ?? 0) + 1;
    return c;
  }, [all]);
  const list = useMemo(
    () => all.filter((p) => !status || p.status === status).sort((a, b) => String(b.added ?? "").localeCompare(String(a.added ?? ""))),
    [all, status],
  );

  // --- add paper (ingest / upload) ---
  const [src, setSrc] = useState<"arxiv" | "url" | "path">("arxiv");
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const ingest = async (e: FormEvent) => {
    e.preventDefault();
    const v = value.trim();
    if (!v || busy) return;
    setBusy(true);
    try {
      const meta = await api.library.ingest(src === "arxiv" ? { arxiv: v } : src === "url" ? { url: v } : { path: v });
      setValue("");
      toast(`Filed “${truncate(meta.title ?? meta.id, 60)}”`);
      emitCommand("vault-changed");
      papers.reload();
      navigate({ kind: "paper", id: meta.id });
    } catch (err) {
      toast(`Ingest failed: ${errorMessage(err)}`, "error");
    } finally {
      setBusy(false);
    }
  };
  const upload = async (file: File | undefined) => {
    if (!file || busy) return;
    setBusy(true);
    try {
      const meta = await api.library.upload(file);
      toast(`Uploaded “${truncate(meta.title ?? meta.id, 60)}”`);
      emitCommand("vault-changed");
      papers.reload();
      navigate({ kind: "paper", id: meta.id });
    } catch (err) {
      toast(`Upload failed: ${errorMessage(err)}`, "error");
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <>
      <div className="pane-head">
        <h1>Library</h1>
        <form className="add-paper" onSubmit={(e) => void ingest(e)}>
          <select className="select sm" value={src} onChange={(e) => setSrc(e.target.value as typeof src)} aria-label="Source type" style={{ width: "auto" }}>
            <option value="arxiv">arXiv id</option>
            <option value="url">URL</option>
            <option value="path">Local path</option>
          </select>
          <input
            className="input sm"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={src === "arxiv" ? "2406.01234" : src === "url" ? "https://…/paper.pdf" : "/path/to/paper.pdf"}
            aria-label="Paper source"
            spellCheck={false}
          />
          <button className="btn sm primary" type="submit" disabled={!value.trim() || busy}>
            {busy ? "Working…" : "Add"}
          </button>
          <input ref={fileRef} type="file" accept="application/pdf" className="visually-hidden" id="lib-upload" onChange={(e) => void upload(e.target.files?.[0])} />
          <label htmlFor="lib-upload" className="btn sm" style={{ cursor: busy ? "wait" : "pointer" }}>
            Upload PDF
          </label>
        </form>
      </div>
      <div className="list-toolbar">
        <div className="filter-tabs" role="tablist" aria-label="Status">
          <button role="tab" aria-pressed={!status} onClick={() => navigate({ kind: "library", topic })}>
            All <span className="n">{all.length}</span>
          </button>
          {PAPER_STATUSES.map((s: PaperStatus) => (
            <button key={s} role="tab" aria-pressed={status === s} onClick={() => navigate({ kind: "library", status: s, topic })}>
              {titleCase(s)} <span className="n">{counts[s] ?? 0}</span>
            </button>
          ))}
        </div>
        {topic && (
          <span className="chip">
            topic: {topic}
            <button onClick={() => navigate({ kind: "library", status })} aria-label="Clear topic filter">
              ×
            </button>
          </span>
        )}
        <span className="count">{list.length} shown</span>
      </div>
      <div className="pane-body">
        {papers.loading && <Loading label="Loading library" />}
        {papers.error && <ErrorState error={papers.error} onRetry={papers.reload} />}
        {!papers.loading && !papers.error && list.length === 0 && (
          <EmptyState title={status ? `Nothing in ${titleCase(status)}` : "Library is empty"} hint="Add a paper by arXiv id, URL, local path, or upload a PDF." />
        )}
        <div className="rows">
          {list.map((p) => (
            <PaperRow key={p.id} paper={p} />
          ))}
        </div>
      </div>
    </>
  );
}

/** Warm the browser cache for a PDF the user is hovering, so opening it is instant. Once per id. */
const prefetched = new Set<string>();
function prefetchPdf(paper: PaperMeta) {
  if (!paper.has_pdf || prefetched.has(paper.id)) return;
  prefetched.add(paper.id);
  try {
    void fetch(api.library.pdfUrl(paper.id), { priority: "low" } as RequestInit).catch(() => prefetched.delete(paper.id));
  } catch {
    prefetched.delete(paper.id);
  }
}

function PaperRow({ paper }: { paper: PaperMeta }) {
  return (
    <button
      className="row"
      onClick={() => navigate({ kind: "paper", id: paper.id })}
      onMouseEnter={() => prefetchPdf(paper)}
      onFocus={() => prefetchPdf(paper)}
    >
      <span className="title">{paper.title || paper.id}</span>
      <span className="meta">
        <span className="tag">{titleCase(paper.status)}</span>
        {paper.rating ? <span className="tag accent num">{paper.rating}/5</span> : null}
        {authorsLine(paper.authors) && <span>{authorsLine(paper.authors)}</span>}
        {paper.year ? <span className="num">{paper.year}</span> : null}
        {paper.takeaway && <span className="preview">{truncate(paper.takeaway, 120)}</span>}
      </span>
      <span className="right">{relDate(paper.added)}</span>
    </button>
  );
}
