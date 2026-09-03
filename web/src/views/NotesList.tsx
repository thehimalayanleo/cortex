import { useMemo } from "react";
import { api } from "../api";
import type { NoteKind, NoteSummary } from "../types";
import { NOTE_KINDS } from "../types";
import { useAsync } from "../lib/hooks";
import { navigate } from "../lib/router";
import { relDate, truncate } from "../lib/format";
import { EmptyState, ErrorState, Loading } from "../components/States";

export function NotesList({ noteKind, onNewNote, refresh }: { noteKind?: NoteKind; onNewNote: () => void; refresh: number }) {
  const notes = useAsync(() => api.notes.list({ limit: 1000 }), [], [refresh]);
  const all = notes.data ?? [];
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const n of all) c[n.kind] = (c[n.kind] ?? 0) + 1;
    return c;
  }, [all]);
  const list = useMemo(
    () =>
      all
        .filter((n) => !noteKind || n.kind === noteKind)
        .sort((a, b) => String(b.updated ?? "").localeCompare(String(a.updated ?? ""))),
    [all, noteKind],
  );

  return (
    <>
      <div className="pane-head">
        <h1>Notes</h1>
        <button className="btn sm primary" onClick={onNewNote}>
          New note <kbd>⌘N</kbd>
        </button>
      </div>
      <div className="list-toolbar">
        <div className="filter-tabs" role="tablist" aria-label="Kind">
          <button role="tab" aria-pressed={!noteKind} onClick={() => navigate({ kind: "notes" })}>
            All <span className="n">{all.length}</span>
          </button>
          {NOTE_KINDS.map((k) => (
            <button key={k} role="tab" aria-pressed={noteKind === k} onClick={() => navigate({ kind: "notes", noteKind: k })}>
              {k} <span className="n">{counts[k] ?? 0}</span>
            </button>
          ))}
        </div>
        <span className="count">{list.length} shown</span>
      </div>
      <div className="pane-body">
        {notes.loading && <Loading label="Loading notes" />}
        {notes.error && <ErrorState error={notes.error} onRetry={notes.reload} />}
        {!notes.loading && !notes.error && list.length === 0 && (
          <EmptyState title={noteKind ? `No ${noteKind} notes` : "No notes yet"} hint="Notes are markdown files in the vault's notes/ folder.">
            <button className="btn primary" onClick={onNewNote}>
              New note
            </button>
          </EmptyState>
        )}
        <div className="rows">
          {list.map((n) => (
            <NoteRow key={n.slug} note={n} />
          ))}
        </div>
      </div>
    </>
  );
}

function NoteRow({ note }: { note: NoteSummary }) {
  return (
    <button className="row" onClick={() => navigate({ kind: "note", slug: note.slug })}>
      <span className="title">{note.title || note.slug}</span>
      <span className="meta">
        <span className="tag">{note.kind}</span>
        {(note.topics ?? []).slice(0, 4).map((t) => (
          <span key={t} className="tag accent">
            {t}
          </span>
        ))}
        {note.preview && <span className="preview">{truncate(note.preview, 140)}</span>}
      </span>
      <span className="right">{relDate(note.updated)}</span>
    </button>
  );
}
