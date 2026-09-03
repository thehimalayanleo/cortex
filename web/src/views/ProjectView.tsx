import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { Project, ProjectFrontmatter, ProjectStatus, ProjectType, Topic } from "../types";
import { PROJECT_STATUSES, PROJECT_TYPES } from "../types";
import { useAsync, useAutosave, useLocalStorage } from "../lib/hooks";
import { emitCommand, onCommand } from "../lib/events";
import { asStringArray, splitFrontmatter } from "../lib/frontmatter";
import { navigate } from "../lib/router";
import { relDate } from "../lib/format";
import { MarkdownEditor } from "../components/MarkdownEditor";
import { MarkdownPreview } from "../components/MarkdownPreview";
import { ModeSwitch } from "../components/NoteEditor";
import type { ViewMode } from "../components/NoteEditor";
import { TopicChips } from "../components/TopicChips";
import { ErrorState, Loading, SaveStatus } from "../components/States";

export function ProjectView({ slug, topics, refresh }: { slug: string; topics: Topic[] | null; refresh: number }) {
  const project = useAsync(() => api.projects.get(slug), [slug], [refresh]);
  if (project.loading) return <Loading label="Opening project" />;
  if (project.error) return <ErrorState title="Project not found" error={project.error} onRetry={project.reload} />;
  if (!project.data) return <ErrorState title="Project not found" error="Empty response" onRetry={project.reload} />;
  return <ProjectEditor project={project.data} topics={topics} />;
}

interface Draft {
  fm: ProjectFrontmatter;
  body: string;
}

function ProjectEditor({ project, topics }: { project: Project; topics: Topic[] | null }) {
  const [mode, setMode] = useLocalStorage<ViewMode>("cortex.project.mode", "split");
  const initial = useMemo<Draft>(() => {
    const { frontmatter: inline, body } = splitFrontmatter(project.body ?? "");
    const fm = { ...(inline ?? {}), ...(project.frontmatter ?? {}) } as ProjectFrontmatter;
    return { fm: { ...fm, title: fm.title ?? project.slug, topics: asStringArray(fm.topics) }, body };
  }, [project]);
  const [draft, setDraft] = useState<Draft>(initial);
  const draftRef = useRef(draft);
  const [server, setServer] = useState(project);

  const autosave = useAutosave<Draft>(async (d) => {
    const saved = await api.projects.update(project.slug, { frontmatter: d.fm, body: d.body });
    if (saved) setServer(saved);
    emitCommand("vault-changed");
  });
  const { reset, flush } = autosave;
  const statusRef = useRef(autosave.status);
  statusRef.current = autosave.status;
  const lastSlug = useRef<string | null>(null);
  useEffect(() => {
    const same = lastSlug.current === project.slug;
    lastSlug.current = project.slug;
    if (same && statusRef.current !== "idle" && statusRef.current !== "saved") return;
    setServer(project);
    if (same && JSON.stringify(draftRef.current) === JSON.stringify(initial)) return;
    setDraft(initial);
    draftRef.current = initial;
    reset();
  }, [initial, project, reset]);
  useEffect(() => onCommand("save", () => void flush()), [flush]);

  const update = (patch: Partial<Draft>) => {
    const next = { ...draftRef.current, ...patch };
    draftRef.current = next;
    setDraft(next);
    autosave.schedule(next);
  };
  const setFm = (patch: Partial<ProjectFrontmatter>) => update({ fm: { ...draftRef.current.fm, ...patch } });

  const fm = draft.fm;
  const sfm = server.frontmatter ?? {};

  return (
    <div className="doc">
      <header className="doc-head">
        <div className="line">
          <span className="crumbs">
            <button onClick={() => navigate({ kind: "projects" })}>projects</button>
            <span>/</span>
            <button onClick={() => navigate({ kind: "projects", status: fm.status })}>{fm.status ?? "active"}</button>
          </span>
          <div className="doc-tools">
            <SaveStatus status={autosave.status} savedAt={autosave.savedAt} error={autosave.error} />
            <ModeSwitch mode={mode} onChange={setMode} />
          </div>
        </div>
        <input className="input bare title" value={fm.title ?? ""} onChange={(e) => setFm({ title: e.target.value })} placeholder="Untitled project" aria-label="Title" />
        <div className="line meta">
          <span title="slug">{project.slug}</span>
          {sfm.created ? <span>created {relDate(sfm.created)}</span> : null}
          {sfm.updated ? <span>updated {relDate(sfm.updated)}</span> : null}
        </div>
      </header>
      <div className="project-form">
        <div className="form-grid">
          <div className="field">
            <label htmlFor="pf-status">Status</label>
            <select id="pf-status" className="select sm" value={fm.status ?? "active"} onChange={(e) => setFm({ status: e.target.value as ProjectStatus })}>
              {PROJECT_STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="pf-type">Type</label>
            <select id="pf-type" className="select sm" value={fm.type ?? "research"} onChange={(e) => setFm({ type: e.target.value as ProjectType })}>
              {PROJECT_TYPES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="pf-deadline">Deadline</label>
            <input id="pf-deadline" className="input sm" type="date" value={fm.deadline ?? ""} onChange={(e) => setFm({ deadline: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="pf-repo">Repo</label>
            <input id="pf-repo" className="input sm mono" value={fm.repo ?? ""} onChange={(e) => setFm({ repo: e.target.value })} placeholder="github.com/… or path" spellCheck={false} />
          </div>
          <div className="field span-2">
            <label htmlFor="pf-verdict">Verdict</label>
            <input id="pf-verdict" className="input sm" value={fm.verdict ?? ""} onChange={(e) => setFm({ verdict: e.target.value })} placeholder="What do we currently believe?" />
          </div>
          <div className="field span-2">
            <label htmlFor="pf-next">Next action</label>
            <input id="pf-next" className="input sm" value={fm.next_action ?? ""} onChange={(e) => setFm({ next_action: e.target.value })} placeholder="The one concrete next step" />
          </div>
          <div className="field span-2">
            <label>Topics</label>
            <TopicChips value={asStringArray(fm.topics)} onChange={(t) => setFm({ topics: t })} suggestions={topics} />
          </div>
        </div>
      </div>
      <div className={`doc-body mode-${mode}`}>
        {mode !== "preview" && (
          <div className="editor-col">
            <MarkdownEditor value={draft.body} onChange={(body) => update({ body })} docKey={project.slug} placeholder="Free notes for this project." />
          </div>
        )}
        {mode !== "editor" && (
          <div className="preview-col">
            <MarkdownPreview source={draft.body} emptyText="No project notes yet." />
          </div>
        )}
      </div>
    </div>
  );
}
