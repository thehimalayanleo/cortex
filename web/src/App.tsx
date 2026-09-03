import { useCallback, useEffect, useState } from "react";
import { api, errorMessage } from "./api";
import { NOTE_KINDS } from "./types";
import type { NoteKind } from "./types";
import { useRoute } from "./lib/router";
import { navigate } from "./lib/router";
import { useAsync, useLocalStorage } from "./lib/hooks";
import { emitCommand, onCommand } from "./lib/events";
import { useTheme } from "./lib/theme";
import { installWebMCP, onWebMCPCall } from "./lib/webmcp";
import type { WebMCPCall } from "./lib/webmcp";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { SearchPalette } from "./components/SearchPalette";
import { PromptDialog } from "./components/Dialog";
import { ToastProvider, useToast } from "./components/Toast";
import { NoteView } from "./views/NoteView";
import { DailyView } from "./views/DailyView";
import { PaperView } from "./views/PaperView";
import { ProjectView } from "./views/ProjectView";
import { NotesList } from "./views/NotesList";
import { LibraryList } from "./views/LibraryList";
import { ProjectsList } from "./views/ProjectsList";
import { TopicsList, TopicView } from "./views/TopicsView";
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
  const [palette, setPalette] = useState(false);
  const [prompt, setPrompt] = useState<null | "note" | "project">(null);
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

  const toggleChat = useCallback(() => {
    setChatPref(chatOpen ? "closed" : "open");
    if (!chatOpen) setChatFocus((n) => n + 1);
  }, [chatOpen, setChatPref]);

  const openNewNote = useCallback(() => setPrompt("note"), []);
  const openNewProject = useCallback(() => setPrompt("project"), []);

  // Global keyboard: Cmd+K palette, Cmd+N new note, Cmd+S save now, Cmd+/ toggle chat.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod || e.altKey || e.isComposing) return;
      const k = e.key.toLowerCase();
      if (k === "k") {
        e.preventDefault();
        setPalette(true);
      } else if (k === "n" && !e.shiftKey) {
        // Note: browsers may reserve Cmd+N; Ctrl+N and the palette/button paths always work.
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
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleChat]);

  const createNote = async (title: string) => {
    const note = await api.notes.create({ title, kind: newKind });
    setPrompt(null);
    emitCommand("vault-changed");
    navigate({ kind: "note", slug: note.slug });
  };
  const createProject = async (title: string) => {
    try {
      const p = await api.projects.create({ title, status: "active", type: "research" });
      setPrompt(null);
      emitCommand("vault-changed");
      navigate({ kind: "project", slug: p.slug });
    } catch (e) {
      toast(errorMessage(e), "error");
      throw e;
    }
  };

  let view: JSX.Element;
  switch (route.kind) {
    case "daily":
      view = <DailyView topics={topics.data} refresh={vaultTick} />;
      break;
    case "note":
      view = <NoteView slug={route.slug} topics={topics.data} refresh={vaultTick} />;
      break;
    case "paper":
      view = <PaperView id={route.id} topics={topics.data} refresh={vaultTick} />;
      break;
    case "project":
      view = <ProjectView slug={route.slug} topics={topics.data} refresh={vaultTick} />;
      break;
    case "notes":
      view = <NotesList noteKind={route.noteKind} onNewNote={openNewNote} refresh={vaultTick} />;
      break;
    case "library":
      view = <LibraryList status={route.status} topic={route.topic} refresh={vaultTick} />;
      break;
    case "projects":
      view = <ProjectsList status={route.status} onNewProject={openNewProject} refresh={vaultTick} />;
      break;
    case "topics":
      view = <TopicsList topics={topics.data} loading={topics.loading} error={topics.error} reload={topics.reload} />;
      break;
    case "topic":
      view = <TopicView slug={route.slug} topics={topics.data} refresh={vaultTick} />;
      break;
  }

  return (
    <div className={`app ${chatOpen ? "" : "chat-closed"}`}>
      <Sidebar
        route={route}
        chatOpen={chatOpen}
        onToggleChat={toggleChat}
        onOpenPalette={() => setPalette(true)}
        onNewNote={openNewNote}
        themePref={themePref}
        onTheme={setThemePref}
        agentReady={agentReady}
        agentToolCount={10}
      />
      <main className="center" aria-live="polite">
        <Resizer cssVar="--rail-w" storageKey="cortex.w.rail" defaultPx={250} min={180} max={480} grows="right" label="Resize sidebar" className="at-left" />
        {view}
        {chatOpen && (
          <Resizer cssVar="--chat-w" storageKey="cortex.w.chat" defaultPx={380} min={300} max={760} grows="left" label="Resize chat" className="at-right" />
        )}
      </main>
      <aside className="chat" aria-label="Chat" hidden={!chatOpen}>
        <ChatPanel focusSignal={chatFocus} onClose={toggleChat} agentCalls={agentCalls} agentReady={agentReady} />
      </aside>

      <SearchPalette open={palette} onClose={() => setPalette(false)} onNewNote={openNewNote} />
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
      {prompt === "project" && <PromptDialog title="New project" label="Title" placeholder="Project name" onSubmit={createProject} onClose={() => setPrompt(null)} />}
    </div>
  );
}
