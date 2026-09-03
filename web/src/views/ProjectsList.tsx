import { useMemo } from "react";
import { api } from "../api";
import type { Project, ProjectStatus } from "../types";
import { PROJECT_STATUSES } from "../types";
import { useAsync } from "../lib/hooks";
import { navigate } from "../lib/router";
import { relDate, truncate } from "../lib/format";
import { EmptyState, ErrorState, Loading } from "../components/States";

export function ProjectsList({ status, onNewProject, refresh }: { status?: string; onNewProject: () => void; refresh: number }) {
  const projects = useAsync(() => api.projects.list(), [], [refresh]);
  const all = projects.data ?? [];
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of all) {
      const s = p.frontmatter?.status ?? "active";
      c[s] = (c[s] ?? 0) + 1;
    }
    return c;
  }, [all]);
  const groups = useMemo(() => {
    const filtered = all.filter((p) => !status || (p.frontmatter?.status ?? "active") === status);
    const byStatus = new Map<string, Project[]>();
    for (const p of filtered) {
      const s = p.frontmatter?.status ?? "active";
      if (!byStatus.has(s)) byStatus.set(s, []);
      byStatus.get(s)!.push(p);
    }
    const order: string[] = [...PROJECT_STATUSES.filter((s) => byStatus.has(s))];
    for (const k of byStatus.keys()) if (!order.includes(k)) order.push(k);
    return order.map((s) => ({
      status: s,
      items: byStatus.get(s)!.sort((a, b) => String(b.frontmatter?.updated ?? "").localeCompare(String(a.frontmatter?.updated ?? ""))),
    }));
  }, [all, status]);
  const shown = groups.reduce((n, g) => n + g.items.length, 0);

  return (
    <>
      <div className="pane-head">
        <h1>Projects</h1>
        <button className="btn sm primary" onClick={onNewProject}>
          New project
        </button>
      </div>
      <div className="list-toolbar">
        <div className="filter-tabs" role="tablist" aria-label="Status">
          <button role="tab" aria-pressed={!status} onClick={() => navigate({ kind: "projects" })}>
            All <span className="n">{all.length}</span>
          </button>
          {PROJECT_STATUSES.map((s: ProjectStatus) => (
            <button key={s} role="tab" aria-pressed={status === s} onClick={() => navigate({ kind: "projects", status: s })}>
              {s} <span className="n">{counts[s] ?? 0}</span>
            </button>
          ))}
        </div>
        <span className="count">{shown} shown</span>
      </div>
      <div className="pane-body">
        {projects.loading && <Loading label="Loading projects" />}
        {projects.error && <ErrorState error={projects.error} onRetry={projects.reload} />}
        {!projects.loading && !projects.error && shown === 0 && (
          <EmptyState title={status ? `No ${status} projects` : "No projects yet"} hint="Projects are markdown files with a status, verdict, and next action.">
            <button className="btn primary" onClick={onNewProject}>
              New project
            </button>
          </EmptyState>
        )}
        {groups.map((g) => (
          <div key={g.status}>
            {!status && (
              <div className="group-head">
                {g.status} <span className="n">{g.items.length}</span>
              </div>
            )}
            <div className="rows">
              {g.items.map((p) => (
                <ProjectRow key={p.slug} project={p} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

function ProjectRow({ project }: { project: Project }) {
  const fm = project.frontmatter ?? {};
  const deadline = fm.deadline ? String(fm.deadline) : "";
  const overdue = deadline && new Date(deadline).getTime() < Date.now() && fm.status === "active";
  return (
    <button className="row" onClick={() => navigate({ kind: "project", slug: project.slug })}>
      <span className="title">{fm.title ?? project.slug}</span>
      <span className="meta">
        {fm.type && <span className="tag">{fm.type}</span>}
        {deadline && <span className={`tag num ${overdue ? "danger" : ""}`}>due {deadline}</span>}
        {fm.next_action ? <span className="preview">next: {truncate(String(fm.next_action), 120)}</span> : fm.verdict ? <span className="preview">{truncate(String(fm.verdict), 120)}</span> : null}
      </span>
      <span className="right">{relDate(fm.updated)}</span>
    </button>
  );
}
