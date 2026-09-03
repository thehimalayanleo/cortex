import { api } from "../api";
import { useAsync } from "../lib/hooks";
import { longDate } from "../lib/format";
import { NoteEditor } from "../components/NoteEditor";
import { ErrorState, Loading } from "../components/States";
import type { Topic } from "../types";

// TODO(spec): daily notes live in daily/YYYY-MM-DD.md but are saved through PUT /api/notes/{slug}.
// We assume the server resolves the slug it returned from /api/daily/today.
export function DailyView({ topics, refresh }: { topics: Topic[] | null; refresh: number }) {
  const note = useAsync(() => api.daily.today(), [], [refresh]);
  if (note.loading) return <Loading label="Opening today" />;
  if (note.error) return <ErrorState title="Could not open today's note" error={note.error} onRetry={note.reload} />;
  if (!note.data) return <ErrorState title="Could not open today's note" error="Empty response" onRetry={note.reload} />;
  const created = note.data.frontmatter?.created ?? new Date();
  return (
    <NoteEditor
      note={note.data}
      topics={topics}
      allowDelete={false}
      autoFocus
      banner={<span className="daily-banner">Daily · {longDate(created) || longDate(new Date())}</span>}
    />
  );
}
