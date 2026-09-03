import { useMemo } from "react";
import { api } from "../api";
import type { Topic } from "../types";
import { useAsync } from "../lib/hooks";
import { navigate } from "../lib/router";
import { authorsLine, relDate, titleCase, truncate } from "../lib/format";
import { EmptyState, ErrorState, Loading } from "../components/States";

export function TopicsList({ topics, loading, error, reload }: { topics: Topic[] | null; loading: boolean; error: string | null; reload: () => void }) {
  return (
    <>
      <div className="pane-head">
        <h1>Topics</h1>
        <span className="count mono faint num">{topics?.length ?? 0}</span>
      </div>
      <div className="pane-body">
        {loading && <Loading label="Loading topics" />}
        {error && <ErrorState error={error} onRetry={reload} />}
        {!loading && !error && (topics?.length ?? 0) === 0 && <EmptyState title="No topics yet" hint="Topics live in topics.json at the vault root." />}
        <div className="topic-grid">
          {(topics ?? []).map((t) => (
            <button key={t.slug} className="topic-card" onClick={() => navigate({ kind: "topic", slug: t.slug })}>
              <span className="name">{t.name}</span>
              {t.kind && <span className="kind">{t.kind}</span>}
              {t.one_liner && <span className="one">{t.one_liner}</span>}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

// TODO(spec): /api/notes has no topic filter and /api/projects has none either; we filter
// client-side from the full lists. /api/library does accept ?topic=.
export function TopicView({ slug, topics, refresh }: { slug: string; topics: Topic[] | null; refresh: number }) {
  const topic = topics?.find((t) => t.slug === slug);
  const notes = useAsync(() => api.notes.list({ limit: 1000 }), [slug], [refresh]);
  const papers = useAsync(() => api.library.list({ topic: slug }), [slug], [refresh]);
  const projects = useAsync(() => api.projects.list(), [slug], [refresh]);

  const noteHits = useMemo(() => (notes.data ?? []).filter((n) => (n.topics ?? []).includes(slug)), [notes.data, slug]);
  const paperHits = useMemo(() => (papers.data ?? []).filter((p) => !p.topics || p.topics.length === 0 || p.topics.includes(slug)), [papers.data, slug]);
  const projectHits = useMemo(() => (projects.data ?? []).filter((p) => (p.frontmatter?.topics ?? []).includes(slug)), [projects.data, slug]);

  const anyLoading = notes.loading || papers.loading || projects.loading;
  const total = noteHits.length + paperHits.length + projectHits.length;

  return (
    <>
      <div className="pane-head">
        <span className="crumbs mono faint">
          <button onClick={() => navigate({ kind: "topics" })} style={{ color: "inherit" }}>
            topics
          </button>{" "}
          /
        </span>
        <h1>{topic?.name ?? slug}</h1>
        {topic?.kind && <span className="tag">{topic.kind}</span>}
      </div>
      {topic?.one_liner && (
        <div className="list-toolbar">
          <span className="muted">{topic.one_liner}</span>
          <span className="count">{total} items</span>
        </div>
      )}
      <div className="pane-body">
        {anyLoading && <Loading label="Collecting" />}
        {notes.error && <ErrorState title="Notes unavailable" error={notes.error} onRetry={notes.reload} compact />}
        {papers.error && <ErrorState title="Library unavailable" error={papers.error} onRetry={papers.reload} compact />}
        {projects.error && <ErrorState title="Projects unavailable" error={projects.error} onRetry={projects.reload} compact />}
        {!anyLoading && total === 0 && !notes.error && !papers.error && !projects.error && (
          <EmptyState title="Nothing tagged yet" hint={`Add "${slug}" to a note, paper, or project to see it here.`} />
        )}

        {noteHits.length > 0 && (
          <>
            <div className="group-head">
              Notes <span className="n">{noteHits.length}</span>
            </div>
            <div className="rows">
              {noteHits.map((n) => (
                <button key={n.slug} className="row" onClick={() => navigate({ kind: "note", slug: n.slug })}>
                  <span className="title">{n.title || n.slug}</span>
                  <span className="meta">
                    <span className="tag">{n.kind}</span>
                    {n.preview && <span className="preview">{truncate(n.preview, 120)}</span>}
                  </span>
                  <span className="right">{relDate(n.updated)}</span>
                </button>
              ))}
            </div>
          </>
        )}
        {paperHits.length > 0 && (
          <>
            <div className="group-head">
              Papers <span className="n">{paperHits.length}</span>
            </div>
            <div className="rows">
              {paperHits.map((p) => (
                <button key={p.id} className="row" onClick={() => navigate({ kind: "paper", id: p.id })}>
                  <span className="title">{p.title || p.id}</span>
                  <span className="meta">
                    <span className="tag">{titleCase(p.status)}</span>
                    {authorsLine(p.authors) && <span>{authorsLine(p.authors)}</span>}
                    {p.year ? <span className="num">{p.year}</span> : null}
                  </span>
                  <span className="right">{relDate(p.added)}</span>
                </button>
              ))}
            </div>
          </>
        )}
        {projectHits.length > 0 && (
          <>
            <div className="group-head">
              Projects <span className="n">{projectHits.length}</span>
            </div>
            <div className="rows">
              {projectHits.map((p) => (
                <button key={p.slug} className="row" onClick={() => navigate({ kind: "project", slug: p.slug })}>
                  <span className="title">{p.frontmatter?.title ?? p.slug}</span>
                  <span className="meta">
                    <span className="tag">{p.frontmatter?.status ?? "active"}</span>
                    {p.frontmatter?.next_action && <span className="preview">next: {truncate(String(p.frontmatter.next_action), 120)}</span>}
                  </span>
                  <span className="right">{relDate(p.frontmatter?.updated)}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
}
