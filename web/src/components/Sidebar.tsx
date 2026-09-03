import { useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { api, errorMessage } from "../api";
import type { PaperMeta, Project } from "../types";
import { navigate } from "../lib/router";
import type { Route } from "../lib/router";
import { useAsync, useLocalStorage } from "../lib/hooks";
import type { ThemePref } from "../lib/theme";
import { authorsLine } from "../lib/format";
import { Popover } from "./Popover";
import { useToast } from "./Toast";

interface Props {
  route: Route;
  space: string;
  projects: Project[] | null;
  onSpace: (slug: string) => void;
  onNewSpace: () => void;
  onFiled: (metas: PaperMeta[]) => void;
  refresh: number;
  chatOpen: boolean;
  onToggleChat: () => void;
  onOpenPalette: () => void;
  themePref: ThemePref;
  onTheme: (p: ThemePref) => void;
  agentReady: boolean;
  agentToolCount: number;
}

type StatusFilter = "all" | "inbox" | "reading" | "read";
const FILTERS: { id: StatusFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "inbox", label: "Inbox" },
  { id: "reading", label: "Reading" },
  { id: "read", label: "Read" },
];

/** Warm the browser cache for a PDF the user is hovering, so opening it is instant. Once per id. */
const prefetched = new Set<string>();
export function prefetchPdf(paper: PaperMeta) {
  if (!paper.has_pdf || prefetched.has(paper.id)) return;
  prefetched.add(paper.id);
  try {
    void fetch(api.library.pdfUrl(paper.id), { priority: "low" } as RequestInit).catch(() => prefetched.delete(paper.id));
  } catch {
    prefetched.delete(paper.id);
  }
}

/** An arXiv id or URL, any other URL, or a local path. */
export function detectSource(raw: string): { arxiv: string } | { url: string } | { path: string } | null {
  const s = raw.trim();
  if (!s) return null;
  const id = /(\d{4}\.\d{4,5})(?:v\d+)?/.exec(s);
  if (id && (/arxiv\.org/i.test(s) || /^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$/i.test(s))) return { arxiv: id[1] };
  if (/^[~/]/.test(s)) return { path: s };
  if (/^https?:\/\//i.test(s)) return { url: s };
  return { url: `https://${s}` };
}

export function Sidebar({ route, space, projects, onSpace, onNewSpace, onFiled, refresh, chatOpen, onToggleChat, onOpenPalette, themePref, onTheme, agentReady, agentToolCount }: Props) {
  const [status, setStatus] = useLocalStorage<StatusFilter>("cortex.rail.status", "all");
  const [q, setQ] = useState("");
  const papers = useAsync(
    () => api.library.list({ status: status === "all" ? undefined : status, project: space === "all" ? undefined : space }),
    [status, space],
    [refresh],
  );

  const list = useMemo(() => {
    const all = papers.data ?? [];
    const needle = q.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((p) => `${p.title} ${authorsLine(p.authors, 50)} ${p.year ?? ""} ${p.id}`.toLowerCase().includes(needle));
  }, [papers.data, q]);

  const spaces = useMemo(() => {
    const ps = projects ?? [];
    const active = ps.filter((p) => (p.frontmatter.status ?? "active") === "active");
    const rest = ps.filter((p) => (p.frontmatter.status ?? "active") !== "active");
    const byTitle = (a: Project, b: Project) => String(a.frontmatter.title ?? a.slug).localeCompare(String(b.frontmatter.title ?? b.slug));
    return [...active.sort(byTitle), ...rest.sort(byTitle)];
  }, [projects]);

  const openId = route.kind === "paper" ? route.id : null;
  const themeLabel = themePref === "system" ? "Theme: system" : themePref === "dark" ? "Theme: dark" : "Theme: light";
  const cycleTheme = () => onTheme(themePref === "system" ? "light" : themePref === "light" ? "dark" : "system");

  return (
    <nav className="rail" aria-label="Papers">
      <div className="rail-top">
        <div className="brand">
          <span className="name">Cortex</span>
        </div>
        <div className="space-row">
          <select
            className="select sm space-select"
            value={space}
            aria-label="Space"
            onChange={(e) => {
              const v = e.target.value;
              if (v === "__new") onNewSpace();
              else onSpace(v);
            }}
          >
            <option value="all">All papers</option>
            {spaces.map((p) => (
              <option key={p.slug} value={p.slug}>
                {String(p.frontmatter.title ?? p.slug)}
              </option>
            ))}
            <option value="__new">New space…</option>
          </select>
          {space !== "all" && (
            <button
              className="icon-btn"
              onClick={() => navigate({ kind: "project", slug: space })}
              title="Open space page"
              aria-label="Open space page"
              aria-current={route.kind === "project" && route.slug === space ? true : undefined}
            >
              <ArrowIcon />
            </button>
          )}
        </div>
        <div className="rail-search">
          <input className="input sm" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter papers" aria-label="Filter papers" spellCheck={false} />
          <button className="k" onClick={onOpenPalette} title="Search everything (Cmd+K)" aria-label="Search everything">
            ⌘K
          </button>
        </div>
        <div className="seg" role="group" aria-label="Status">
          {FILTERS.map((f) => (
            <button key={f.id} type="button" aria-pressed={status === f.id} onClick={() => setStatus(f.id)}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-list" aria-busy={papers.loading}>
        {papers.loading && !papers.data && <div className="rail-empty">loading…</div>}
        {papers.error && (
          <div className="rail-empty" title={papers.error}>
            unavailable ·{" "}
            <button style={{ color: "var(--accent-text)" }} onClick={papers.reload}>
              retry
            </button>
          </div>
        )}
        {papers.data && list.length === 0 && <div className="rail-empty">{q ? "No matches." : "No papers here."}</div>}
        {list.map((p) => (
          <PaperRow key={p.id} paper={p} current={p.id === openId} />
        ))}
      </div>

      <div className="rail-foot">
        <AddPaper onFiled={onFiled} />
        <div className="links">
          <button className="lnk" onClick={() => navigate({ kind: "daily" })} aria-current={route.kind === "daily" || undefined}>
            Today
          </button>
          <button className="lnk" onClick={() => navigate({ kind: "notes" })} aria-current={route.kind === "notes" || route.kind === "note" || undefined}>
            Notes
          </button>
          <span className="grow" />
          <button className="icon-btn" onClick={cycleTheme} title={`${themeLabel} (click to change)`} aria-label={themeLabel}>
            {themePref === "dark" ? <MoonIcon /> : themePref === "light" ? <SunIcon /> : <AutoIcon />}
          </button>
          <button className="icon-btn" onClick={onToggleChat} title={chatOpen ? "Hide chat (Cmd+/)" : "Show chat (Cmd+/)"} aria-label={chatOpen ? "Hide chat" : "Show chat"} aria-pressed={chatOpen}>
            <PanelIcon />
          </button>
        </div>
        <span
          className={`pill ${agentReady ? "ok" : ""}`}
          title={agentReady ? "This page exposes its tools to the browser's agent via WebMCP" : "WebMCP not detected: open in Chrome 149+ with WebMCP or ChatGPT's browser"}
        >
          {agentReady ? `Agent tools on · ${agentToolCount}` : "Agent tools off"}
        </span>
      </div>
    </nav>
  );
}

function PaperRow({ paper, current }: { paper: PaperMeta; current: boolean }) {
  const authors = authorsLine(paper.authors, 2);
  const meta = [paper.year ? String(paper.year) : "", authors].filter(Boolean).join(" · ") || paper.id;
  return (
    <button
      className="paper-row"
      aria-current={current || undefined}
      onClick={() => navigate({ kind: "paper", id: paper.id })}
      onMouseEnter={() => prefetchPdf(paper)}
      onFocus={() => prefetchPdf(paper)}
      title={paper.title}
    >
      <span className="dot" data-status={paper.status} aria-hidden="true" />
      <span className="t">{paper.title || paper.id}</span>
      <span className="m">{meta}</span>
    </button>
  );
}

function AddPaper({ onFiled }: { onFiled: (metas: PaperMeta[]) => void }) {
  const { toast } = useToast();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const add = async (close: () => void) => {
    const src = detectSource(value);
    if (!src || busy) return;
    setBusy(true);
    try {
      const meta = await api.library.ingest(src);
      setValue("");
      close();
      onFiled([meta]);
    } catch (e) {
      toast(`Could not add: ${errorMessage(e)}`, "error");
    } finally {
      setBusy(false);
    }
  };
  const upload = async (files: FileList | null, close: () => void) => {
    const list = Array.from(files ?? []);
    if (list.length === 0 || busy) return;
    setBusy(true);
    const out: PaperMeta[] = [];
    try {
      for (const f of list) out.push(await api.library.upload(f));
      close();
    } catch (e) {
      toast(`Upload failed: ${errorMessage(e)}`, "error");
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
      if (out.length) onFiled(out);
    }
  };

  return (
    <Popover
      up
      className="add-paper"
      panelClassName="add-panel"
      render={(open, toggle) => (
        <button className="btn sm primary" onClick={toggle} aria-expanded={open} style={{ width: "100%", justifyContent: "center" }}>
          Add paper
        </button>
      )}
    >
      {(close) => (
        <>
          <input
            className="input sm"
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void add(close);
              }
            }}
            placeholder="arXiv id, arXiv URL, or any URL"
            aria-label="Paper source"
            spellCheck={false}
            disabled={busy}
          />
          <div className="row">
            <input ref={fileRef} type="file" accept="application/pdf" multiple className="visually-hidden" id="rail-upload" onChange={(e) => void upload(e.target.files, close)} />
            <label htmlFor="rail-upload" className="btn sm" style={{ cursor: busy ? "wait" : "pointer" }}>
              Upload PDF
            </label>
            <span className="grow" />
            <button className="btn sm primary" onClick={() => void add(close)} disabled={!detectSource(value) || busy}>
              {busy ? "Working…" : "Add"}
            </button>
          </div>
        </>
      )}
    </Popover>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}
function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" />
    </svg>
  );
}
function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
    </svg>
  );
}
function AutoIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" stroke="none" />
    </svg>
  );
}
function PanelIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M15 4v16" />
    </svg>
  );
}
