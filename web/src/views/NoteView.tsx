import { api } from "../api";
import { useAsync } from "../lib/hooks";
import { navigate } from "../lib/router";
import { NoteEditor } from "../components/NoteEditor";
import { EmptyState, ErrorState, Loading } from "../components/States";
import type { Topic } from "../types";

export function NoteView({ slug, topics, refresh }: { slug: string; topics: Topic[] | null; refresh: number }) {
  const note = useAsync(() => api.notes.get(slug), [slug], [refresh]);
  if (note.loading) return <Loading label="Opening note" />;
  if (note.error) {
    return <ErrorState title="Note not found" error={note.error} onRetry={note.reload} />;
  }
  if (!note.data) return <EmptyState title="Nothing here" hint="The server returned no note." />;
  return (
    <NoteEditor
      note={note.data}
      topics={topics}
      autoFocus
      banner={
        <span className="crumbs">
          <button onClick={() => navigate({ kind: "notes" })}>notes</button>
          <span>/</span>
          <button onClick={() => navigate({ kind: "notes", noteKind: note.data?.frontmatter?.kind })}>{note.data.frontmatter?.kind ?? "note"}</button>
        </span>
      }
    />
  );
}
