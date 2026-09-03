import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { PaperDetail, PaperMeta, PaperStatus, Topic } from "../types";
import { PAPER_STATUSES } from "../types";
import { useAsync, useAutosave, useLocalStorage } from "../lib/hooks";
import { emitCommand, onCommand } from "../lib/events";
import { asStringArray } from "../lib/frontmatter";
import { navigate } from "../lib/router";
import { authorsLine, relDate, titleCase } from "../lib/format";
import { MarkdownEditor } from "../components/MarkdownEditor";
import { MarkdownPreview } from "../components/MarkdownPreview";
import { ModeSwitch } from "../components/NoteEditor";
import type { ViewMode } from "../components/NoteEditor";
import { TopicChips } from "../components/TopicChips";
import { ErrorState, Loading, SaveStatus } from "../components/States";
import { Resizer } from "../components/Resizer";

/**
 * The PDF viewer keeps keyboard focus once clicked, so app shortcuts (Cmd+K, Cmd+/, Cmd+N, Cmd+S)
 * never reach the page. The frame is same-origin, so re-dispatch those chords on the parent window.
 */
function forwardShortcutsFromFrame(e: React.SyntheticEvent<HTMLIFrameElement>) {
  try {
    const win = e.currentTarget.contentWindow;
    if (!win) return;
    win.addEventListener("keydown", (ke) => {
      const mod = ke.metaKey || ke.ctrlKey;
      if (!mod || !["k", "n", "s", "/"].includes(ke.key.toLowerCase())) return;
      ke.preventDefault();
      window.dispatchEvent(new KeyboardEvent("keydown", { key: ke.key, metaKey: ke.metaKey, ctrlKey: ke.ctrlKey, shiftKey: ke.shiftKey, bubbles: true, cancelable: true }));
    });
  } catch {
    /* cross-origin or plugin frame: nothing to forward */
  }
}

export function PaperView({ id, topics, refresh }: { id: string; topics: Topic[] | null; refresh: number }) {
  const paper = useAsync(() => api.library.get(id), [id], [refresh]);
  if (paper.loading) return <Loading label="Opening paper" />;
  if (paper.error) return <ErrorState title="Paper not found" error={paper.error} onRetry={paper.reload} />;
  if (!paper.data) return <ErrorState title="Paper not found" error="Empty response" onRetry={paper.reload} />;
  return <PaperEditor detail={paper.data} topics={topics} />;
}

type MetaDraft = Pick<PaperMeta, "title" | "status" | "rating" | "takeaway" | "topics">;

function PaperEditor({ detail, topics }: { detail: PaperDetail; topics: Topic[] | null }) {
  const id = detail.meta.id;
  const [tab, setTab] = useState<"pdf" | "notes">("pdf");
  const [mode, setMode] = useLocalStorage<ViewMode>("cortex.paper.mode", "split");
  const [pdfDark, setPdfDark] = useLocalStorage<"on" | "off">("cortex.paper.pdfDark", "on");

  const initialMeta = useMemo<MetaDraft>(
    () => ({
      title: detail.meta.title ?? "",
      status: detail.meta.status ?? "inbox",
      rating: detail.meta.rating ?? null,
      takeaway: detail.meta.takeaway ?? "",
      topics: asStringArray(detail.meta.topics),
    }),
    [detail],
  );
  const [meta, setMeta] = useState<MetaDraft>(initialMeta);
  const metaRef = useRef(meta);
  const [notes, setNotes] = useState(detail.notes ?? "");
  const notesRef = useRef(notes);

  const metaSave = useAutosave<MetaDraft>(async (d) => {
    await api.library.update(id, d);
    emitCommand("vault-changed");
  });
  const notesSave = useAutosave<string>(async (n) => {
    await api.library.update(id, { notes: n });
  });
  const resetMeta = metaSave.reset;
  const resetNotes = notesSave.reset;
  const flushMeta = metaSave.flush;
  const flushNotes = notesSave.flush;

  const metaStatus = useRef(metaSave.status);
  metaStatus.current = metaSave.status;
  const notesStatus = useRef(notesSave.status);
  notesStatus.current = notesSave.status;
  const lastId = useRef<string | null>(null);
  useEffect(() => {
    const same = lastId.current === detail.meta.id;
    lastId.current = detail.meta.id;
    const quiet = (s: string) => s === "idle" || s === "saved";
    if (!same || (quiet(metaStatus.current) && JSON.stringify(metaRef.current) !== JSON.stringify(initialMeta))) {
      setMeta(initialMeta);
      metaRef.current = initialMeta;
      resetMeta();
    }
    const serverNotes = detail.notes ?? "";
    if (!same || (quiet(notesStatus.current) && notesRef.current !== serverNotes)) {
      setNotes(serverNotes);
      notesRef.current = serverNotes;
      resetNotes();
    }
  }, [initialMeta, detail, resetMeta, resetNotes]);

  useEffect(
    () =>
      onCommand("save", () => {
        void flushMeta();
        void flushNotes();
      }),
    [flushMeta, flushNotes],
  );

  const updateMeta = (patch: Partial<MetaDraft>) => {
    const next = { ...metaRef.current, ...patch };
    metaRef.current = next;
    setMeta(next);
    metaSave.schedule(next);
  };
  const updateNotes = (n: string) => {
    notesRef.current = n;
    setNotes(n);
    notesSave.schedule(n);
  };

  const m = detail.meta;
  const arxivUrl = m.arxiv ? `https://arxiv.org/abs/${m.arxiv}` : null;
  const status: "idle" | "dirty" | "saving" | "saved" | "error" =
    metaSave.status === "error" || notesSave.status === "error"
      ? "error"
      : metaSave.status === "saving" || notesSave.status === "saving"
        ? "saving"
        : metaSave.status === "dirty" || notesSave.status === "dirty"
          ? "dirty"
          : metaSave.status === "saved" || notesSave.status === "saved"
            ? "saved"
            : "idle";
  const savedAt = [metaSave.savedAt, notesSave.savedAt].filter(Boolean).sort((a, b) => (b as Date).getTime() - (a as Date).getTime())[0] ?? null;

  return (
    <div className="paper">
      <div className="paper-main">
        <Resizer cssVar="--meta-w" storageKey="cortex.w.meta" defaultPx={300} min={240} max={560} grows="left" label="Resize paper details" className="at-right" />
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "pdf"} onClick={() => setTab("pdf")}>
            PDF
          </button>
          <button role="tab" aria-selected={tab === "notes"} onClick={() => setTab("notes")}>
            Notes
          </button>
          <div className="right">
            <SaveStatus status={status} savedAt={savedAt} error={metaSave.error ?? notesSave.error} />
            {tab === "notes" && <ModeSwitch mode={mode} onChange={setMode} />}
            {tab === "pdf" && (
              <>
                <button
                  type="button"
                  className="btn ghost sm"
                  aria-pressed={pdfDark === "on"}
                  title="Invert the PDF so pages are dark; figures keep their hue"
                  onClick={() => setPdfDark(pdfDark === "on" ? "off" : "on")}
                >
                  {pdfDark === "on" ? "Dark PDF: on" : "Dark PDF: off"}
                </button>
                <a className="btn ghost sm" href={api.library.pdfUrl(id)} target="_blank" rel="noopener noreferrer">
                  Open PDF in tab
                </a>
              </>
            )}
          </div>
        </div>
        {tab === "pdf" ? (
          <iframe
            className={`pdf-frame${pdfDark === "on" ? " pdf-dark" : ""}`}
            src={api.library.pdfUrl(id)}
            title={`PDF: ${m.title}`}
            onLoad={forwardShortcutsFromFrame}
          />
        ) : (
          <div className={`paper-notes mode-${mode}`}>
            {mode !== "preview" && (
              <div className="editor-col">
                <MarkdownEditor value={notes} onChange={updateNotes} docKey={`paper:${id}`} placeholder="Reading notes. Math renders in the preview." autoFocus />
              </div>
            )}
            {mode !== "editor" && (
              <div className="preview-col">
                <MarkdownPreview source={notes} emptyText="No reading notes yet." />
              </div>
            )}
          </div>
        )}
      </div>

      <aside className="paper-meta" aria-label="Paper metadata">
        <textarea
          className="title"
          value={meta.title ?? ""}
          rows={3}
          onChange={(e) => updateMeta({ title: e.target.value })}
          aria-label="Title"
        />
        {authorsLine(m.authors, 12) && <div className="authors">{authorsLine(m.authors, 12)}</div>}
        <dl className="kv">
          {m.year ? (
            <>
              <dt>year</dt>
              <dd>{m.year}</dd>
            </>
          ) : null}
          {m.arxiv ? (
            <>
              <dt>arxiv</dt>
              <dd>
                <a href={arxivUrl ?? "#"} target="_blank" rel="noopener noreferrer">
                  {m.arxiv}
                </a>
              </dd>
            </>
          ) : null}
          {m.link ? (
            <>
              <dt>link</dt>
              <dd>
                <a href={m.link} target="_blank" rel="noopener noreferrer" title={m.link}>
                  {m.link.replace(/^https?:\/\//, "")}
                </a>
              </dd>
            </>
          ) : null}
          {m.type ? (
            <>
              <dt>type</dt>
              <dd>{m.type}</dd>
            </>
          ) : null}
          {m.pages ? (
            <>
              <dt>pages</dt>
              <dd>{m.pages}</dd>
            </>
          ) : null}
          {m.added ? (
            <>
              <dt>added</dt>
              <dd>{relDate(m.added)}</dd>
            </>
          ) : null}
          <dt>id</dt>
          <dd title={m.id}>{m.id}</dd>
        </dl>

        <div className="field">
          <label htmlFor="pm-status">Status</label>
          <select id="pm-status" className="select sm" value={meta.status} onChange={(e) => updateMeta({ status: e.target.value as PaperStatus })}>
            {PAPER_STATUSES.map((s) => (
              <option key={s} value={s}>
                {titleCase(s)}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label id="pm-rating">Rating</label>
          <div className="rating" role="radiogroup" aria-labelledby="pm-rating">
            {[1, 2, 3, 4, 5].map((n) => (
              <button key={n} type="button" role="radio" aria-checked={meta.rating === n} aria-pressed={meta.rating === n} onClick={() => updateMeta({ rating: meta.rating === n ? null : n })} aria-label={`${n} of 5`}>
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label htmlFor="pm-takeaway">Takeaway</label>
          <textarea id="pm-takeaway" className="textarea" rows={4} value={meta.takeaway ?? ""} onChange={(e) => updateMeta({ takeaway: e.target.value })} placeholder="One or two lines: what this paper changes." />
        </div>

        <div className="field">
          <label>Topics</label>
          <TopicChips value={meta.topics} onChange={(t) => updateMeta({ topics: t })} suggestions={topics} />
        </div>

        {detail.text_preview && (
          <div className="field">
            <label>Extracted text</label>
            <p className="faint" style={{ fontSize: "var(--fs-xs)", lineHeight: 1.45, maxHeight: 160, overflow: "auto" }}>
              {detail.text_preview}
            </p>
          </div>
        )}

        <div style={{ marginTop: "auto" }}>
          <button className="btn ghost sm" onClick={() => navigate({ kind: "library", status: meta.status })}>
            Back to {titleCase(meta.status)}
          </button>
        </div>
      </aside>
    </div>
  );
}
