import { useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Project, ProjectFrontmatter, ProjectStatus } from "../types";
import { PROJECT_STATUSES } from "../types";
import { useAsync, useAutosave, useLocalStorage } from "../lib/hooks";
import { emitCommand, onCommand } from "../lib/events";
import { splitFrontmatter } from "../lib/frontmatter";
import { relDate } from "../lib/format";
import { MarkdownEditor } from "../components/MarkdownEditor";
import { MarkdownPreview } from "../components/MarkdownPreview";
import { ModeSwitch } from "../components/NoteEditor";
import type { ViewMode } from "../components/NoteEditor";
import { InlineEdit } from "../components/InlineEdit";
import { ErrorState, Loading, SaveStatus } from "../components/States";
import { useToast } from "../components/Toast";

export function ProjectView({ slug, refresh }: { slug: string; refresh: number }) {
  const project = useAsync(() => api.projects.get(slug), [slug], [refresh]);
  if (project.loading && !project.data) return <Loading label="Opening space" />;
  if (project.error) return <ErrorState title="Space not found" error={project.error} onRetry={project.reload} />;
  if (!project.data) return <ErrorState title="Space not found" error="Empty response" onRetry={project.reload} />;
  return <ProjectPage key={slug} project={project.data} />;
}

/** A space page: title, status, verdict, next action, free notes. Frontmatter fields save on change; the body autosaves. */
function ProjectPage({ project }: { project: Project }) {
  const { toast } = useToast();
  const [mode, setMode] = useLocalStorage<ViewMode>("cortex.project.mode", "split");
  const [fm, setFm] = useState<ProjectFrontmatter>(project.frontmatter ?? {});
  useEffect(() => setFm(project.frontmatter ?? {}), [project.frontmatter]);

  const setField = async (patch: Partial<ProjectFrontmatter>) => {
    setFm((f) => ({ ...f, ...patch }));
    try {
      await api.projects.update(project.slug, { frontmatter: patch });
      emitCommand("vault-changed");
    } catch (e) {
      toast(`Save failed: ${errorMessage(e)}`, "error");
    }
  };

  const initialBody = splitFrontmatter(project.body ?? "").body;
  const [body, setBody] = useState(initialBody);
  const bodyRef = useRef(body);
  const autosave = useAutosave<string>(async (b) => {
    await api.projects.update(project.slug, { body: b });
    emitCommand("vault-changed");
  });
  const { flush, status } = autosave;
  const statusRef = useRef(status);
  statusRef.current = status;
  useEffect(() => {
    const server = splitFrontmatter(project.body ?? "").body;
    if ((statusRef.current === "idle" || statusRef.current === "saved") && bodyRef.current !== server) {
      bodyRef.current = server;
      setBody(server);
    }
  }, [project.body]);
  useEffect(() => onCommand("save", () => void flush()), [flush]);
  const updateBody = (b: string) => {
    bodyRef.current = b;
    setBody(b);
    autosave.schedule(b);
  };

  const st = (fm.status ?? "active") as ProjectStatus;
  return (
    <div className="doc">
      <header className="project-head">
        <div className="line">
          <h1>{String(fm.title ?? project.slug)}</h1>
          <select className="status-pill" data-status={st} value={st} onChange={(e) => void setField({ status: e.target.value as ProjectStatus })} aria-label="Status">
            {PROJECT_STATUSES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <div className="doc-tools">
            <SaveStatus status={autosave.status} savedAt={autosave.savedAt} error={autosave.error} />
            <ModeSwitch mode={mode} onChange={setMode} />
          </div>
        </div>
        <div className="kv-line">
          <span className="k">Verdict</span>
          <InlineEdit value={String(fm.verdict ?? "")} placeholder="What do we currently believe?" label="Verdict" onSave={(v) => void setField({ verdict: v })} />
        </div>
        <div className="kv-line">
          <span className="k">Next</span>
          <InlineEdit value={String(fm.next_action ?? "")} placeholder="The one concrete next step" label="Next action" onSave={(v) => void setField({ next_action: v })} />
        </div>
        <div className="line meta">
          <span title="slug">{project.slug}</span>
          {fm.updated ? <span>updated {relDate(fm.updated)}</span> : null}
        </div>
      </header>
      <div className={`doc-body mode-${mode}`}>
        {mode !== "preview" && (
          <div className="editor-col">
            <MarkdownEditor value={body} onChange={updateBody} docKey={`project:${project.slug}`} placeholder="Notes for this space." />
          </div>
        )}
        {mode !== "editor" && (
          <div className="preview-col">
            <MarkdownPreview source={body} emptyText="No notes yet." />
          </div>
        )}
      </div>
    </div>
  );
}
