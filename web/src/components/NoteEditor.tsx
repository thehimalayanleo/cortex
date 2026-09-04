import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api";
import type { Note, NoteKind, Topic } from "../types";
import { NOTE_KINDS } from "../types";
import { asStringArray, splitFrontmatter } from "../lib/frontmatter";
import { useAutosave, useLocalStorage } from "../lib/hooks";
import { emitCommand, onCommand } from "../lib/events";
import { relDate } from "../lib/format";
import { navigate } from "../lib/router";
import { MarkdownEditor } from "./MarkdownEditor";
import { MarkdownPreview } from "./MarkdownPreview";
import { TopicChips } from "./TopicChips";
import { SaveStatus } from "./States";
import { useToast } from "./Toast";

export type ViewMode = "editor" | "split" | "preview";

export function ModeSwitch({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div className="seg" role="group" aria-label="View mode">
      {(["editor", "split", "preview"] as ViewMode[]).map((m) => (
        <button key={m} type="button" aria-pressed={mode === m} onClick={() => onChange(m)}>
          {m[0].toUpperCase() + m.slice(1)}
        </button>
      ))}
    </div>
  );
}

interface Draft {
  title: string;
  kind: NoteKind;
  topics: string[];
  body: string;
}

interface Props {
  note: Note;
  topics?: Topic[] | null;
  banner?: ReactNode;
  allowDelete?: boolean;
  autoFocus?: boolean;
}

/** Note editor with frontmatter strip, split/editor/preview, and 800 ms autosave. */
function draftsEqual(a: Draft, b: Draft): boolean {
  return a.title === b.title && a.kind === b.kind && a.body === b.body && a.topics.join("\u0000") === b.topics.join("\u0000");
}

export function NoteEditor({ note, topics, banner, allowDelete = true, autoFocus }: Props) {
  const { toast } = useToast();
  // Lab chapters open in reading mode; ordinary notes remember their own split.
  const isLab = note.slug.startsWith("lab-");
  const [mode, setMode] = useLocalStorage<ViewMode>(isLab ? "cortex.lab.mode" : "cortex.note.mode", isLab ? "preview" : "split");

  // The API splits frontmatter and body; strip a stray inline block defensively.
  const initial = useMemo(() => {
    const { frontmatter: inline, body } = splitFrontmatter(note.body ?? "");
    const fm = { ...(inline ?? {}), ...(note.frontmatter ?? {}) };
    const draft: Draft = {
      title: String(fm.title ?? note.slug),
      kind: (NOTE_KINDS.includes(fm.kind as NoteKind) ? (fm.kind as NoteKind) : "fleeting") as NoteKind,
      topics: asStringArray(fm.topics),
      body,
    };
    return { draft, fm };
  }, [note]);

  const [draft, setDraft] = useState<Draft>(initial.draft);
  const draftRef = useRef(draft);
  const [serverNote, setServerNote] = useState<Note>(note);

  const autosave = useAutosave<Draft>(async (d) => {
    const saved = await api.notes.update(note.slug, {
      frontmatter: { ...initial.fm, title: d.title, kind: d.kind, topics: d.topics },
      body: d.body,
    });
    if (saved) setServerNote(saved);
    emitCommand("vault-changed");
  });
  const { reset, flush } = autosave;
  const statusRef = useRef(autosave.status);
  statusRef.current = autosave.status;
  const lastSlug = useRef<string | null>(null);

  // Adopt server content on open / slug change, or on a background refetch (agent edits) only
  // when there are no local edits in flight and the content actually differs.
  useEffect(() => {
    const same = lastSlug.current === note.slug;
    lastSlug.current = note.slug;
    if (same && statusRef.current !== "idle" && statusRef.current !== "saved") return;
    setServerNote(note);
    if (same && draftsEqual(draftRef.current, initial.draft)) return;
    setDraft(initial.draft);
    draftRef.current = initial.draft;
    reset();
  }, [initial, note, reset]);

  useEffect(() => onCommand("save", () => void flush()), [flush]);

  const update = (patch: Partial<Draft>) => {
    const next = { ...draftRef.current, ...patch };
    draftRef.current = next;
    setDraft(next);
    autosave.schedule(next);
  };

  const remove = async () => {
    if (!window.confirm(`Delete "${draft.title}"? This removes the file from the vault.`)) return;
    try {
      reset();
      await api.notes.remove(note.slug);
      toast("Note deleted");
      emitCommand("vault-changed");
      navigate({ kind: "notes" });
    } catch (e) {
      toast(`Delete failed: ${(e as Error).message}`, "error");
    }
  };

  const fm = serverNote.frontmatter ?? {};

  return (
    <div className="doc">
      <header className="doc-head">
        {banner && <div className="line">{banner}</div>}
        <input
          className="input bare title"
          value={draft.title}
          onChange={(e) => update({ title: e.target.value })}
          placeholder="Untitled"
          aria-label="Title"
        />
        <div className="line">
          <select
            className="select sm kind-select"
            value={draft.kind}
            onChange={(e) => update({ kind: e.target.value as NoteKind })}
            aria-label="Kind"
          >
            {NOTE_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <TopicChips value={draft.topics} onChange={(t) => update({ topics: t })} suggestions={topics} />
          <div className="doc-tools">
            <SaveStatus status={autosave.status} savedAt={autosave.savedAt} error={autosave.error} />
            <ModeSwitch mode={mode} onChange={setMode} />
            {allowDelete && (
              <button className="btn ghost sm danger" onClick={() => void remove()} title="Delete note">
                Delete
              </button>
            )}
          </div>
        </div>
        <div className="line meta">
          <span title="slug">{note.slug}</span>
          {fm.created ? <span title="created">created {relDate(fm.created)}</span> : null}
          {fm.updated ? <span title="updated">updated {relDate(fm.updated)}</span> : null}
          {asStringArray(fm.sources).length > 0 && (
            <span>
              sources{" "}
              {asStringArray(fm.sources).map((s, i) => (
                <span key={s}>
                  {i > 0 && ", "}
                  <button className="mono" style={{ color: "var(--accent-text)" }} onClick={() => navigate({ kind: "paper", id: s })}>
                    {s}
                  </button>
                </span>
              ))}
            </span>
          )}
          {asStringArray(fm.projects).length > 0 && (
            <span>
              projects{" "}
              {asStringArray(fm.projects).map((s, i) => (
                <span key={s}>
                  {i > 0 && ", "}
                  <button className="mono" style={{ color: "var(--accent-text)" }} onClick={() => navigate({ kind: "project", slug: s })}>
                    {s}
                  </button>
                </span>
              ))}
            </span>
          )}
        </div>
      </header>
      <div className={`doc-body mode-${mode}`}>
        {mode !== "preview" && (
          <div className="editor-col">
            <MarkdownEditor
              value={draft.body}
              onChange={(body) => update({ body })}
              docKey={note.slug}
              autoFocus={autoFocus}
              onFiles={async (files) => {
                const out: string[] = [];
                for (const f of files) {
                  try {
                    const a = await api.notes.attach(note.slug, f);
                    out.push(a.markdown);
                  } catch (e) {
                    toast(`Could not attach ${f.name || "file"}: ${e instanceof Error ? e.message : String(e)}`, "error");
                  }
                }
                if (out.length) toast(out.length === 1 ? "Attached" : `Attached ${out.length} files`);
                return out;
              }}
            />
          </div>
        )}
        {mode !== "editor" && (
          <div className="preview-col">
            <MarkdownPreview source={draft.body} emptyText="Preview appears here as you write." />
          </div>
        )}
      </div>
    </div>
  );
}
