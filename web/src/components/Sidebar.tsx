import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api";
import type { NoteKind, NoteSummary, PaperStatus, Project, ProjectStatus } from "../types";
import { PAPER_STATUSES, PROJECT_STATUSES } from "../types";
import { onCommand } from "../lib/events";
import { navigate, routeToHash } from "../lib/router";
import type { Route } from "../lib/router";
import { useAsync, useLocalStorage } from "../lib/hooks";
import type { ThemePref } from "../lib/theme";
import { titleCase } from "../lib/format";

interface Props {
  route: Route;
  chatOpen: boolean;
  onToggleChat: () => void;
  onOpenPalette: () => void;
  onNewNote: () => void;
  themePref: ThemePref;
  onTheme: (p: ThemePref) => void;
  agentReady: boolean;
  agentToolCount: number;
}

const NOTE_KIND_ORDER: NoteKind[] = ["fleeting", "permanent", "literature", "meeting", "daily"];
const PER_GROUP = 6;

function useVaultTick() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let t: number | null = null;
    return onCommand("vault-changed", () => {
      if (t) window.clearTimeout(t);
      t = window.setTimeout(() => setTick((n) => n + 1), 400);
    });
  }, []);
  return tick;
}

export function Sidebar({ route, chatOpen, onToggleChat, onOpenPalette, onNewNote, themePref, onTheme, agentReady, agentToolCount }: Props) {
  const tick = useVaultTick();
  const health = useAsync(() => api.health(), [tick]);
  const notes = useAsync(() => api.notes.list({ limit: 500 }), [tick]);
  const library = useAsync(() => api.library.list(), [tick]);
  const projects = useAsync(() => api.projects.list(), [tick]);
  const topics = useAsync(() => api.topics(), [tick]);

  const current = routeToHash(route);
  const is = (r: Route) => routeToHash(r) === current;

  const noteGroups = useMemo(() => {
    const list = [...(notes.data ?? [])].sort((a, b) => String(b.updated ?? "").localeCompare(String(a.updated ?? "")));
    const groups = new Map<NoteKind, NoteSummary[]>();
    for (const n of list) {
      const k = (n.kind ?? "fleeting") as NoteKind;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(n);
    }
    const ordered = NOTE_KIND_ORDER.filter((k) => groups.has(k));
    for (const k of groups.keys()) if (!ordered.includes(k)) ordered.push(k);
    return ordered.map((k) => ({ kind: k, items: groups.get(k)! }));
  }, [notes.data]);

  const libCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of library.data ?? []) c[p.status] = (c[p.status] ?? 0) + 1;
    return c;
  }, [library.data]);

  const projectGroups = useMemo(() => {
    const groups = new Map<string, Project[]>();
    for (const p of projects.data ?? []) {
      const s = (p.frontmatter?.status ?? "active") as string;
      if (!groups.has(s)) groups.set(s, []);
      groups.get(s)!.push(p);
    }
    const ordered: string[] = [...PROJECT_STATUSES.filter((s) => groups.has(s))];
    for (const k of groups.keys()) if (!ordered.includes(k)) ordered.push(k);
    return ordered.map((s) => ({ status: s as ProjectStatus, items: groups.get(s)! }));
  }, [projects.data]);

  const themeLabel = themePref === "system" ? "Theme: system" : themePref === "dark" ? "Theme: dark" : "Theme: light";
  const cycleTheme = () => onTheme(themePref === "system" ? "light" : themePref === "light" ? "dark" : "system");

  return (
    <nav className="rail" aria-label="Vault">
      <div className="rail-top">
        <div className="brand">
          <span className="name">Cortex</span>
          {health.data?.vault && (
            <span className="vault" title={health.data.vault}>
              {health.data.vault}
            </span>
          )}
        </div>
        <button className="search-btn" onClick={onOpenPalette} aria-label="Search (Cmd+K)">
          <SearchIcon />
          <span>Search</span>
          <span className="k">⌘K</span>
        </button>
      </div>

      <div className="rail-body">
        <RailSection id="daily" title="Daily" active={is({ kind: "daily" })} onOpen={() => navigate({ kind: "daily" })}>
          <button className="rail-item" aria-current={is({ kind: "daily" }) || undefined} onClick={() => navigate({ kind: "daily" })}>
            <span className="t">Today</span>
          </button>
        </RailSection>

        <RailSection
          id="notes"
          title="Notes"
          count={notes.data?.length}
          active={route.kind === "notes" && !route.noteKind}
          onOpen={() => navigate({ kind: "notes" })}
          action={
            <button className="icon-btn" onClick={onNewNote} title="New note (Cmd+N)" aria-label="New note">
              <PlusIcon />
            </button>
          }
        >
          <Status state={notes} empty="No notes yet" />
          {noteGroups.map((g) => (
            <div key={g.kind}>
              <button
                className="sub-head"
                aria-current={route.kind === "notes" && route.noteKind === g.kind ? true : undefined}
                onClick={() => navigate({ kind: "notes", noteKind: g.kind })}
              >
                {g.kind}
                <span className="n">{g.items.length}</span>
              </button>
              {g.items.slice(0, PER_GROUP).map((n) => (
                <button
                  key={n.slug}
                  className="rail-item"
                  aria-current={is({ kind: "note", slug: n.slug }) || undefined}
                  onClick={() => navigate({ kind: "note", slug: n.slug })}
                  title={n.title}
                >
                  <span className="t">{n.title || n.slug}</span>
                </button>
              ))}
              {g.items.length > PER_GROUP && (
                <button className="rail-item more" onClick={() => navigate({ kind: "notes", noteKind: g.kind })}>
                  <span className="t">+{g.items.length - PER_GROUP} more</span>
                </button>
              )}
            </div>
          ))}
        </RailSection>

        <RailSection
          id="library"
          title="Library"
          count={library.data?.length}
          active={route.kind === "library" && !route.status && !route.topic}
          onOpen={() => navigate({ kind: "library" })}
        >
          <Status state={library} empty="No papers yet" />
          {library.data &&
            PAPER_STATUSES.map((s: PaperStatus) => (
              <button
                key={s}
                className="rail-item"
                aria-current={route.kind === "library" && route.status === s ? true : undefined}
                onClick={() => navigate({ kind: "library", status: s })}
              >
                <span className="t">{titleCase(s)}</span>
                <span className="n">{libCounts[s] ?? 0}</span>
              </button>
            ))}
        </RailSection>

        <RailSection
          id="projects"
          title="Projects"
          count={projects.data?.length}
          active={route.kind === "projects" && !route.status}
          onOpen={() => navigate({ kind: "projects" })}
        >
          <Status state={projects} empty="No projects yet" />
          {projectGroups.map((g) => {
            const limit = g.status === "active" ? 12 : 3;
            return (
              <div key={g.status}>
                <button
                  className="sub-head"
                  aria-current={route.kind === "projects" && route.status === g.status ? true : undefined}
                  onClick={() => navigate({ kind: "projects", status: g.status })}
                >
                  {g.status}
                  <span className="n">{g.items.length}</span>
                </button>
                {g.items.slice(0, limit).map((p) => (
                  <button
                    key={p.slug}
                    className="rail-item"
                    aria-current={is({ kind: "project", slug: p.slug }) || undefined}
                    onClick={() => navigate({ kind: "project", slug: p.slug })}
                    title={p.frontmatter?.title ?? p.slug}
                  >
                    <span className="t">{p.frontmatter?.title ?? p.slug}</span>
                  </button>
                ))}
                {g.items.length > limit && (
                  <button className="rail-item more" onClick={() => navigate({ kind: "projects", status: g.status })}>
                    <span className="t">+{g.items.length - limit} more</span>
                  </button>
                )}
              </div>
            );
          })}
        </RailSection>

        <RailSection id="topics" title="Topics" count={topics.data?.length} active={route.kind === "topics"} onOpen={() => navigate({ kind: "topics" })}>
          <Status state={topics} empty="No topics yet" />
          {(topics.data ?? []).slice(0, 14).map((t) => (
            <button
              key={t.slug}
              className="rail-item"
              aria-current={is({ kind: "topic", slug: t.slug }) || undefined}
              onClick={() => navigate({ kind: "topic", slug: t.slug })}
              title={t.one_liner ?? t.name}
            >
              <span className="t">{t.name}</span>
            </button>
          ))}
          {(topics.data?.length ?? 0) > 14 && (
            <button className="rail-item more" onClick={() => navigate({ kind: "topics" })}>
              <span className="t">+{(topics.data?.length ?? 0) - 14} more</span>
            </button>
          )}
        </RailSection>
      </div>

      <div className="rail-agent" title={agentReady ? "This page exposes its tools to the browser's agent via WebMCP" : "WebMCP not detected in this browser"}>
        <span className={`pill ${agentReady ? "ok" : ""}`}>
          {agentReady ? `Agent-ready · ${agentToolCount} tools` : "Agent tools: open in Chrome 149+ with WebMCP or ChatGPT's browser"}
        </span>
      </div>
      <div className="rail-foot">
        <span
          className={`status-dot ${health.data?.ok ? "ok" : ""}`}
          title={health.error ? `Server unreachable: ${health.error}` : health.data?.ok ? "Server connected" : "Connecting"}
          aria-hidden="true"
        />
        <span className="grow" title={health.error ?? health.data?.vault ?? ""}>
          {health.loading ? "connecting…" : health.error ? "server offline" : "connected"}
        </span>
        <button className="icon-btn" onClick={cycleTheme} title={`${themeLabel} (click to change)`} aria-label={themeLabel}>
          {themePref === "dark" ? <MoonIcon /> : themePref === "light" ? <SunIcon /> : <AutoIcon />}
        </button>
        <button
          className="icon-btn"
          onClick={onToggleChat}
          title={chatOpen ? "Hide chat (Cmd+/)" : "Show chat (Cmd+/)"}
          aria-label={chatOpen ? "Hide chat" : "Show chat"}
          aria-pressed={chatOpen}
        >
          <PanelIcon />
        </button>
      </div>
    </nav>
  );
}

function RailSection({
  id,
  title,
  count,
  active,
  onOpen,
  action,
  children,
}: {
  id: string;
  title: string;
  count?: number;
  active?: boolean;
  onOpen: () => void;
  action?: ReactNode;
  children: ReactNode;
}) {
  const [state, setState] = useLocalStorage<"open" | "closed">(`cortex.rail.${id}`, "open");
  const collapsed = state === "closed";
  return (
    <section className={`rail-section ${collapsed ? "collapsed" : ""}`}>
      <div className="sec-head" aria-current={active || undefined}>
        <button
          className="caret"
          onClick={() => setState(collapsed ? "open" : "closed")}
          aria-label={collapsed ? `Expand ${title}` : `Collapse ${title}`}
          aria-expanded={!collapsed}
        >
          ▾
        </button>
        <button onClick={onOpen} style={{ flex: 1, textAlign: "left", font: "inherit", color: "inherit", letterSpacing: "inherit", textTransform: "inherit" }}>
          {title}
        </button>
        {action}
        {typeof count === "number" && <span className="n">{count}</span>}
      </div>
      <div className="sec-body">{children}</div>
    </section>
  );
}

function Status<T>({ state, empty }: { state: { loading: boolean; error: string | null; data: T[] | null; reload: () => void }; empty: string }) {
  if (state.loading && !state.data) return <div className="rail-empty">loading…</div>;
  if (state.error)
    return (
      <div className="rail-empty" title={state.error}>
        unavailable ·{" "}
        <button style={{ color: "var(--accent-text)" }} onClick={state.reload}>
          retry
        </button>
      </div>
    );
  if (state.data && state.data.length === 0) return <div className="rail-empty">{empty}</div>;
  return null;
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.5-3.5" />
    </svg>
  );
}
function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
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
