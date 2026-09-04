import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage } from "./api";
import { NOTE_KINDS } from "./types";
import type { NoteKind, PaperMeta } from "./types";
import { useRoute } from "./lib/router";
import { navigate } from "./lib/router";
import { useAsync, useLocalStorage } from "./lib/hooks";
import { emitCommand, onCommand } from "./lib/events";
import { useTheme } from "./lib/theme";
import { installWebMCP, onWebMCPCall, webmcpTools } from "./lib/webmcp";
import type { WebMCPCall } from "./lib/webmcp";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { SearchPalette } from "./components/SearchPalette";
import { PromptDialog } from "./components/Dialog";
import { ToastProvider, useToast } from "./components/Toast";
import { EmptyState } from "./components/States";
import { NoteView } from "./views/NoteView";
import { DailyView } from "./views/DailyView";
import { PaperView } from "./views/PaperView";
import { ProjectView } from "./views/ProjectView";
import { NotesList } from "./views/NotesList";
import { TopicsList, TopicView } from "./views/TopicsView";
import { LabView } from "./views/LabView";
import { Resizer } from "./components/Resizer";

export default function App() {
  return (
    <ToastProvider>
      <Shell />
    </ToastProvider>
  );
}

function Shell() {
  const { toast } = useToast();
  const [route] = useRoute();
  const [themePref, setThemePref] = useTheme();
  const [chatPref, setChatPref] = useLocalStorage<"open" | "closed">("cortex.chat", "open");
  const chatOpen = chatPref === "open";
  const [spacePref, setSpacePref] = useLocalStorage<string>("cortex.space", "all");
  const [palette, setPalette] = useState(false);
  const [prompt, setPrompt] = useState<null | "note" | "space">(null);
  const [newKind, setNewKind] = useState<NoteKind>("fleeting");
  const [chatFocus, setChatFocus] = useState(0);

  // Vault-change ticks: fired by editors after saves, by chat tools, and by WebMCP agent tools.
  const [vaultTick, setVaultTick] = useState(0);
  useEffect(() => {
    let t: number | null = null;
    return onCommand("vault-changed", () => {
      if (t) window.clearTimeout(t);
      t = window.setTimeout(() => setVaultTick((n) => n + 1), 400);
    });
  }, []);

  // WebMCP: register the brain's tools once and keep a ledger of agent calls.
  const [agentReady, setAgentReady] = useState(false);
  const [agentCalls, setAgentCalls] = useState<WebMCPCall[]>([]);
  useEffect(() => {
    setAgentReady(installWebMCP());
    return onWebMCPCall((c) => setAgentCalls((xs) => [...xs.slice(-199), c]));
  }, []);

  const topics = useAsync(() => api.topics(), [], [vaultTick]);
  const projects = useAsync(() => api.projects.list(), [], [vaultTick]);

  // The active space: a project slug or "all". A slug that no longer exists falls back to all.
  const space = useMemo(() => {
    if (spacePref === "all" || !projects.data) return spacePref;
    return projects.data.some((p) => p.slug === spacePref) ? spacePref : "all";
  }, [spacePref, projects.data]);
  const spaceProject = useMemo(() => projects.data?.find((p) => p.slug === space) ?? null, [projects.data, space]);
  const spaceName = space === "all" ? "All papers" : String(spaceProject?.frontmatter.title ?? space);

  const toggleChat = useCallback(() => {
    setChatPref(chatOpen ? "closed" : "open");
    if (!chatOpen) setChatFocus((n) => n + 1);
  }, [chatOpen, setChatPref]);

  const openNewNote = useCallback(() => setPrompt("note"), []);
  const openNewSpace = useCallback(() => setPrompt("space"), []);

  // A paper added while a space is active joins that space.
  const onFiled = useCallback(
    async (metas: PaperMeta[]) => {
      if (space !== "all") {
        for (const m of metas) {
          try {
            await api.library.update(m.id, { projects: Array.from(new Set([...(m.projects ?? []), space])) });
          } catch {
            /* the paper is filed; the space link can be set from its header */
          }
        }
      }
      emitCommand("vault-changed");
      toast(metas.length === 1 ? "Filed 1 paper" : `Filed ${metas.length} papers`);
      if (metas[0]) navigate({ kind: "paper", id: metas[0].id });
    },
    [space, toast],
  );

  // Drop PDFs anywhere: upload each, file it, open the first. Drops inside the note editor are handled there.
  const [dropping, setDropping] = useState(false);
  const dragDepth = useRef(0);
  const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
  const onDragEnter = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    dragDepth.current += 1;
    setDropping(true);
  };
  const onDragOver = (e: React.DragEvent) => {
    if (hasFiles(e)) e.preventDefault();
  };
  const onDragLeave = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDropping(false);
  };
  const onDrop = async (e: React.DragEvent) => {
    dragDepth.current = 0;
    setDropping(false);
    if ((e.target as HTMLElement | null)?.closest?.(".cm-editor")) return; // the editor attaches these itself
    const files = Array.from(e.dataTransfer?.files ?? []).filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
    const text = e.dataTransfer?.getData("text/uri-list") || e.dataTransfer?.getData("text/plain") || "";
    const arxiv = text.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})/)?.[1];
    if (!files.length && !arxiv) return;
    e.preventDefault();
    const metas: PaperMeta[] = [];
    try {
      if (arxiv) metas.push(await api.library.ingest({ arxiv }));
      for (const f of files) metas.push(await api.library.upload(f));
    } catch (err) {
      toast(errorMessage(err), "error");
    }
    if (metas.length) void onFiled(metas);
  };

  // The inbox folder (~/Cortex/inbox) is filed by the server; notice new papers and refresh.
  useEffect(() => {
    let last: number | null = null;
    const tick = async () => {
      try {
        const h = await api.health();
        const n = h.counts.papers;
        if (last != null && n > last) {
          emitCommand("vault-changed");
          toast(n - last === 1 ? "1 new paper from the inbox folder" : `${n - last} new papers from the inbox folder`);
        }
        last = n;
      } catch {
        /* server away; try again next tick */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 15000);
    return () => window.clearInterval(id);
  }, [toast]);

  // "Run on GPU" buttons inside rendered markdown (the chapters' snippets): a one-off scratch run on the best executor.
  useEffect(() => {
    const onRun = async (e: Event) => {
      const code = String((e as CustomEvent).detail?.code ?? "");
      const lang = String((e as CustomEvent).detail?.lang ?? "python");
      if (!code.trim()) return;
      try {
        const ex = await api.lab.executors();
        if (lang === "bash") {
          // a shell block goes to the terminal: prefill it and let the person press Run
          navigate({ kind: "lab", terminal: true });
          window.setTimeout(() => window.dispatchEvent(new CustomEvent("cortex:terminal-prefill", { detail: { cmd: code.trim() } })), 200);
          return;
        }
        const executor = ex.ssh.available ? "ssh" : ex.modal.available ? "modal" : "local";
        const r = await api.lab.start({ recipe: "scratch", code, executor });
        toast(`Running the snippet on ${executor === "ssh" ? ex.ssh.host ?? "your GPU box" : executor}`);
        emitCommand("vault-changed");
        navigate({ kind: "lab", run: r.id });
      } catch (err) {
        toast(errorMessage(err), "error");
      }
    };
    window.addEventListener("cortex:run-code", onRun);
    return () => window.removeEventListener("cortex:run-code", onRun);
  }, [toast]);

  useEffect(() => onCommand("show-ledger", () => { if (chatPref !== "open") setChatPref("open"); }), [chatPref, setChatPref]);

  // "Ask about this paper": make sure the chat is visible before the panel takes the draft.
  useEffect(() => {
    const onAsk = () => {
      if (chatPref !== "open") setChatPref("open");
      setChatFocus((n) => n + 1);
    };
    window.addEventListener("cortex:ask", onAsk);
    return () => window.removeEventListener("cortex:ask", onAsk);
  }, [chatPref, setChatPref]);

  // Global keyboard: Cmd+K palette, Cmd+N new note, Cmd+S save now, Cmd+/ toggle chat.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod || e.altKey || e.isComposing) return;
      const k = e.key.toLowerCase();
      if (k === "k") {
        e.preventDefault();
        setPalette(true);
      } else if (k === "n" && e.shiftKey) {
        e.preventDefault();
        emitCommand("new-chat");
      } else if (k === "n" && !e.shiftKey) {
        // Note: browsers may reserve Cmd+N; Ctrl+N and the palette path always work.
        e.preventDefault();
        setPrompt("note");
      } else if (k === "s") {
        e.preventDefault();
        emitCommand("save");
      } else if (k === "/") {
        e.preventDefault();
        toggleChat();
      }
    };
    // Capture phase so editors and embedded frames can't swallow the shortcut first.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [toggleChat]);

  const createNote = async (title: string) => {
    const note = await api.notes.create({ title, kind: newKind });
    setPrompt(null);
    emitCommand("vault-changed");
    navigate({ kind: "note", slug: note.slug });
  };
  const createSpace = async (title: string) => {
    try {
      const p = await api.projects.create({ title, status: "active", type: "research" });
      setPrompt(null);
      emitCommand("vault-changed");
      setSpacePref(p.slug);
      navigate({ kind: "project", slug: p.slug });
    } catch (e) {
      toast(errorMessage(e), "error");
      throw e;
    }
  };

  let view: JSX.Element;
  switch (route.kind) {
    case "home":
      view = spaceProject ? <ProjectView slug={spaceProject.slug} refresh={vaultTick} /> : <EmptyState title="Pick a paper on the left, or add one." />;
      break;
    case "daily":
      view = <DailyView topics={topics.data} refresh={vaultTick} />;
      break;
    case "note":
      view = <NoteView slug={route.slug} topics={topics.data} refresh={vaultTick} />;
      break;
    case "paper":
      view = <PaperView id={route.id} projects={projects.data} refresh={vaultTick} />;
      break;
    case "project":
      view = <ProjectView slug={route.slug} refresh={vaultTick} />;
      break;
    case "notes":
      view = <NotesList noteKind={route.noteKind} onNewNote={openNewNote} refresh={vaultTick} />;
      break;
    case "topics":
      view = <TopicsList topics={topics.data} loading={topics.loading} error={topics.error} reload={topics.reload} />;
      break;
    case "topic":
      view = <TopicView slug={route.slug} topics={topics.data} refresh={vaultTick} />;
      break;
    case "lab":
      view = <LabView station={route.station} runId={route.run} plan={route.plan} terminal={route.terminal} refresh={vaultTick} />;
      break;
  }

  return (
    <div
      className={`app ${chatOpen ? "" : "chat-closed"}${dropping ? " dropping" : ""}`}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={(e) => void onDrop(e)}
    >
      {dropping && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-card">{space === "all" ? "Drop PDFs to file them" : `Drop PDFs to file them in ${spaceName}`}</div>
        </div>
      )}
      <Sidebar
        route={route}
        space={space}
        projects={projects.data}
        onSpace={setSpacePref}
        onNewSpace={openNewSpace}
        onFiled={(metas) => void onFiled(metas)}
        refresh={vaultTick}
        chatOpen={chatOpen}
        onToggleChat={toggleChat}
        onOpenPalette={() => setPalette(true)}
        themePref={themePref}
        onTheme={setThemePref}
        agentReady={agentReady}
        agentToolCount={webmcpTools.length}
      />
      <main className="center" aria-live="polite">
        <Resizer cssVar="--rail-w" storageKey="cortex.w.rail" defaultPx={260} min={200} max={480} grows="right" label="Resize papers" className="at-left" />
        {view}
        {chatOpen && (
          <Resizer cssVar="--chat-w" storageKey="cortex.w.chat" defaultPx={380} min={300} max={760} grows="left" label="Resize chat" className="at-right" />
        )}
      </main>
      <aside className="chat" aria-label="Chat" hidden={!chatOpen}>
        <ChatPanel space={space} spaceName={spaceName} focusSignal={chatFocus} onClose={toggleChat} agentCalls={agentCalls} agentReady={agentReady} />
      </aside>

      <SearchPalette open={palette} onClose={() => setPalette(false)} onNewNote={openNewNote} onNewSpace={openNewSpace} projects={projects.data} space={space} />
      {prompt === "note" && (
        <PromptDialog
          title="New note"
          label="Title"
          placeholder="What is this note about?"
          onSubmit={createNote}
          onClose={() => setPrompt(null)}
          extra={
            <div className="field">
              <label htmlFor="new-kind">Kind</label>
              <select id="new-kind" className="select sm" value={newKind} onChange={(e) => setNewKind(e.target.value as NoteKind)}>
                {NOTE_KINDS.filter((k) => k !== "daily").map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
          }
        />
      )}
      {prompt === "space" && <PromptDialog title="New space" label="Name" placeholder="Space name" onSubmit={createSpace} onClose={() => setPrompt(null)} />}
    </div>
  );
}
