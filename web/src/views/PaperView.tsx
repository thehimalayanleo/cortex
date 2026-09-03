import { useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Highlights, PaperDetail, PaperMeta, PaperStatus, Project } from "../types";
import { PAPER_STATUSES } from "../types";
import { useAsync, useAutosave, useLocalStorage } from "../lib/hooks";
import { emitCommand, onCommand } from "../lib/events";
import { authorsLine, titleCase } from "../lib/format";
import { MarkdownEditor } from "../components/MarkdownEditor";
import { MarkdownPreview } from "../components/MarkdownPreview";
import { ModeSwitch } from "../components/NoteEditor";
import type { ViewMode } from "../components/NoteEditor";
import { InlineEdit } from "../components/InlineEdit";
import { Popover } from "../components/Popover";
import { ErrorState, Loading, SaveStatus } from "../components/States";
import { useToast } from "../components/Toast";

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

export function PaperView({ id, projects, refresh }: { id: string; projects: Project[] | null; refresh: number }) {
  const paper = useAsync(() => api.library.get(id), [id], [refresh]);
  if (paper.loading && !paper.data) return <Loading label="Opening paper" />;
  if (paper.error) return <ErrorState title="Paper not found" error={paper.error} onRetry={paper.reload} />;
  if (!paper.data) return <ErrorState title="Paper not found" error="Empty response" onRetry={paper.reload} />;
  return <PaperPage key={id} detail={paper.data} projects={projects ?? []} />;
}

function PaperPage({ detail, projects }: { detail: PaperDetail; projects: Project[] }) {
  const { toast } = useToast();
  const id = detail.meta.id;
  const [meta, setMeta] = useState<PaperMeta>(detail.meta);
  const [notesOn, setNotesOn] = useState(false);
  const [mode, setMode] = useLocalStorage<ViewMode>("cortex.paper.mode", "split");

  // Key passages: theorems, results, claims, quoted verbatim with the page they sit on.
  const [hlOn, setHlOn] = useState(false);
  const [hl, setHl] = useState<Highlights | null>(null);
  const [hlBusy, setHlBusy] = useState(false);
  const [hlError, setHlError] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState<number | null>(null);
  const loadHighlights = async (refresh = false) => {
    setHlBusy(true);
    setHlError(null);
    try {
      let h = refresh ? null : await api.library.highlights(id);
      if (!h || !h.items) h = await api.library.makeHighlights(id, refresh);
      setHl(h);
    } catch (e) {
      setHlError(errorMessage(e));
    } finally {
      setHlBusy(false);
    }
  };
  const openHighlights = () => {
    setNotesOn(false);
    setHlOn(true);
    if (!hl) void loadHighlights();
  };
  const jumpTo = (page: number) => {
    setPdfPage(page);
    setHlOn(false);
  };
  const [pdfDark, setPdfDark] = useLocalStorage<"on" | "off">("cortex.paper.pdfDark", "on");

  // Adopt server metadata on background refetches (chat tools, agents).
  useEffect(() => setMeta(detail.meta), [detail.meta]);

  const patch = async (p: Partial<PaperMeta>) => {
    setMeta((m) => ({ ...m, ...p }));
    try {
      await api.library.update(id, p);
      emitCommand("vault-changed");
    } catch (e) {
      toast(`Save failed: ${errorMessage(e)}`, "error");
    }
  };

  // Reading notes: debounced autosave, Cmd+S flushes.
  const [notes, setNotes] = useState(detail.notes ?? "");
  const notesRef = useRef(notes);
  const notesSave = useAutosave<string>(async (n) => {
    await api.library.update(id, { notes: n });
  });
  const { flush, status: saveStatus } = notesSave;
  const saveRef = useRef(saveStatus);
  saveRef.current = saveStatus;
  useEffect(() => {
    const server = detail.notes ?? "";
    if ((saveRef.current === "idle" || saveRef.current === "saved") && notesRef.current !== server) {
      notesRef.current = server;
      setNotes(server);
    }
  }, [detail.notes]);
  useEffect(() => onCommand("save", () => void flush()), [flush]);
  const updateNotes = (n: string) => {
    notesRef.current = n;
    setNotes(n);
    notesSave.schedule(n);
  };

  const assigned = meta.projects ?? [];
  const spaceTitle = (slug: string) => String(projects.find((p) => p.slug === slug)?.frontmatter.title ?? slug);
  const spaceLabel = assigned.length === 0 ? "none" : assigned.length === 1 ? spaceTitle(assigned[0]) : `${spaceTitle(assigned[0])} +${assigned.length - 1}`;
  const toggleSpace = (slug: string) => {
    const next = assigned.includes(slug) ? assigned.filter((s) => s !== slug) : [...assigned, slug];
    void patch({ projects: next });
  };

  const authors = authorsLine(meta.authors, 4);
  const arxivUrl = meta.arxiv ? `https://arxiv.org/abs/${meta.arxiv}` : null;
  const link = meta.link && !arxivUrl ? meta.link : null;

  return (
    <div className="paper-page">
      <header className="paper-head">
        <h1 className="paper-title" title={meta.id}>
          {meta.title || meta.id}
        </h1>
        <div className="paper-line">
          {authors && (
            <span className="authors" title={authorsLine(meta.authors, 50)}>
              {authors}
            </span>
          )}
          {meta.year ? <span className="num">{meta.year}</span> : null}
          {arxivUrl && (
            <a href={arxivUrl} target="_blank" rel="noopener noreferrer">
              arXiv {meta.arxiv}
            </a>
          )}
          {link && (
            <a href={link} target="_blank" rel="noopener noreferrer" title={link}>
              link
            </a>
          )}
          <select className="select sm" value={meta.status} onChange={(e) => void patch({ status: e.target.value as PaperStatus })} aria-label="Status">
            {PAPER_STATUSES.map((s) => (
              <option key={s} value={s}>
                {titleCase(s)}
              </option>
            ))}
          </select>
          <Rating value={meta.rating ?? null} onChange={(r) => void patch({ rating: r })} />
          <Popover
            render={(open, toggle) => (
              <button className={`chip-btn ${assigned.length ? "on" : ""}`} onClick={toggle} aria-expanded={open} title="Spaces this paper belongs to">
                <span className="k">Space:</span>
                <span className="v">{spaceLabel}</span>
                <span className="caret">▾</span>
              </button>
            )}
          >
            {projects.length === 0 ? (
              <div className="menu-row">No spaces yet</div>
            ) : (
              projects.map((p) => (
                <button key={p.slug} className="menu-item" role="menuitemcheckbox" aria-checked={assigned.includes(p.slug)} onClick={() => toggleSpace(p.slug)}>
                  <span className="check">{assigned.includes(p.slug) ? "✓" : ""}</span>
                  {String(p.frontmatter.title ?? p.slug)}
                </button>
              ))
            )}
          </Popover>
          <span className="grow" />
          {notesOn ? (
            <>
              <SaveStatus status={notesSave.status} savedAt={notesSave.savedAt} error={notesSave.error} />
              <ModeSwitch mode={mode} onChange={setMode} />
            </>
          ) : (
            <button type="button" className="btn ghost sm" aria-pressed={pdfDark === "on"} title="Invert the PDF so pages are dark; figures keep their hue" onClick={() => setPdfDark(pdfDark === "on" ? "off" : "on")}>
              Dark PDF
            </button>
          )}
          <button
            type="button"
            className="btn sm"
            title="Open the chat with this paper as context"
            onClick={() => window.dispatchEvent(new CustomEvent("cortex:ask", { detail: "What is this paper about? Give the core claim, the method, and what I should take from it." }))}
          >
            Ask about this paper
          </button>
          <button type="button" className="btn sm" aria-pressed={hlOn} onClick={() => (hlOn ? setHlOn(false) : openHighlights())} title={hlOn ? "Back to the PDF" : "Theorems, results, and key claims, quoted with page numbers"}>
            Highlights
          </button>
          <button type="button" className="btn sm" aria-pressed={notesOn} onClick={() => { setHlOn(false); setNotesOn(!notesOn); }} title={notesOn ? "Back to the PDF" : "Reading notes"}>
            Notes
          </button>
        </div>
        <InlineEdit value={meta.takeaway ?? ""} placeholder="One-line takeaway" label="Takeaway" onSave={(v) => void patch({ takeaway: v })} className="takeaway" />
      </header>

      {hlOn ? (
        <div className="highlights" aria-label="Key passages">
          <div className="hl-head">
            <span>{hl?.items ? `${hl.items.length} passages` : hlBusy ? "Reading the paper…" : "Key passages"}</span>
            <span className="grow" />
            <button type="button" className="btn ghost sm" disabled={hlBusy} onClick={() => void loadHighlights(true)} title="Extract again">
              {hlBusy ? "Working…" : "Refresh"}
            </button>
          </div>
          {hlError && <div className="hl-error">{hlError}</div>}
          {hl?.items && hl.items.length === 0 && !hlBusy && <div className="hl-empty">No quotable passages were found in the extracted text.</div>}
          {hl?.items?.map((h, i) => (
            <div key={i} className={`hl kind-${h.kind}`}>
              <div className="hl-meta">
                <span className="hl-kind">{h.kind}</span>
                <button type="button" className="hl-page" onClick={() => jumpTo(h.page)} title="Jump to this page in the PDF">
                  p. {h.page}
                </button>
              </div>
              <blockquote>{h.quote}</blockquote>
              {h.why && <div className="hl-why">{h.why}</div>}
            </div>
          ))}
        </div>
      ) : notesOn ? (
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
      ) : (
        <iframe
          key={pdfPage ?? 0}
          className={`pdf-frame${pdfDark === "on" ? " pdf-dark" : ""}`}
          src={api.library.pdfUrl(id) + (pdfPage ? `#page=${pdfPage}` : "")}
          title={`PDF: ${meta.title}`}
          onLoad={forwardShortcutsFromFrame}
        />
      )}
    </div>
  );
}

function Rating({ value, onChange }: { value: number | null; onChange: (r: number | null) => void }) {
  return (
    <span className="rating" role="radiogroup" aria-label="Rating">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          role="radio"
          aria-checked={value === n}
          className={value != null && n <= value ? "on" : ""}
          onClick={() => onChange(value === n ? null : n)}
          aria-label={`${n} of 5`}
          title={`${n} of 5`}
        />
      ))}
    </span>
  );
}
