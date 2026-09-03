import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, MouseEvent } from "react";
import { api, errorMessage } from "../api";
import type { AgentId, AgentInfo, Channel, ChatMessage, ModelInfo, ToolTrace } from "../types";
import { useAsync, useLocalStorage } from "../lib/hooks";
import { emitCommand } from "../lib/events";
import { navigate, parseCortexLink } from "../lib/router";
import { clock, parseDate } from "../lib/format";
import { MarkdownPreview, handleCortexClick } from "./MarkdownPreview";
import { ErrorState, Loading } from "./States";
import { useToast } from "./Toast";
import type { WebMCPCall } from "../lib/webmcp";

// Spec: channels are general, papers, projects, ideas, daily. Used as a fallback only when
// GET /api/chat/channels fails, so the panel still works against a partially built server.
const FALLBACK_CHANNELS: Channel[] = [
  { id: "general", name: "general", desc: "Anything", count: 0 },
  { id: "papers", name: "papers", desc: "Library and reading", count: 0 },
  { id: "projects", name: "projects", desc: "Project verdicts and next actions", count: 0 },
  { id: "ideas", name: "ideas", desc: "Half-formed thoughts", count: 0 },
  { id: "daily", name: "daily", desc: "Today", count: 0 },
];
const FALLBACK_MODELS: ModelInfo[] = [
  { id: "glm-5.3", name: "glm-5.3" },
  { id: "glm-5.3-flash", name: "glm-5.3-flash (quick)" },
];

interface MsgItem {
  kind: "msg";
  msg: ChatMessage;
  pending?: boolean;
  error?: string;
  stopped?: boolean;
}
interface AgentRun {
  id: string;
  agent: AgentId;
  task: string;
  editing: boolean;
  lines: string[];
  code: number | null;
  running: boolean;
  error?: string;
  stopped?: boolean;
}
interface AgentItem {
  kind: "agent";
  run: AgentRun;
}
type Item = MsgItem | AgentItem;

interface Props {
  focusSignal: number;
  onClose: () => void;
  agentCalls: WebMCPCall[];
  agentReady: boolean;
}

export function ChatPanel({ focusSignal, onClose, agentCalls, agentReady }: Props) {
  const { toast } = useToast();
  const [panelTab, setPanelTab] = useLocalStorage<"chat" | "agent">("cortex.chat.tab", "chat");
  const channels = useAsync(() => api.chat.channels(), []);
  const models = useAsync(() => api.models(), []);
  const agents = useAsync(() => api.agents.list(), []);
  const [channel, setChannel] = useLocalStorage<string>("cortex.chat.channel", "general");
  const [model, setModel] = useLocalStorage<string>("cortex.chat.model", "glm-5.3");
  const [items, setItems] = useState<Item[]>([]);
  const [loadState, setLoadState] = useState<{ loading: boolean; error: string | null }>({ loading: true, error: null });
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState<{ id: string; ac: AbortController } | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const stickToBottom = useRef(true);
  const runAborts = useRef(new Map<string, AbortController>());
  const seq = useRef(0);

  const channelList = channels.data && channels.data.length > 0 ? channels.data : channels.error ? FALLBACK_CHANNELS : channels.data ?? [];
  const modelList = models.data && models.data.length > 0 ? models.data : FALLBACK_MODELS;
  const activeChannel = channelList.find((c) => c.id === channel) ?? channelList[0];

  // Load history when the channel changes.
  const loadMessages = useCallback(
    (ch: string) => {
      setLoadState({ loading: true, error: null });
      let alive = true;
      api.chat
        .messages(ch)
        .then((ms) => {
          if (!alive) return;
          setItems((Array.isArray(ms) ? ms : []).map((m) => ({ kind: "msg", msg: m })));
          setLoadState({ loading: false, error: null });
          stickToBottom.current = true;
        })
        .catch((e) => {
          if (!alive) return;
          setItems([]);
          setLoadState({ loading: false, error: errorMessage(e) });
        });
      return () => {
        alive = false;
      };
    },
    [],
  );
  useEffect(() => loadMessages(channel), [channel, loadMessages]);

  useEffect(() => {
    if (focusSignal > 0) composerRef.current?.focus();
  }, [focusSignal]);

  // Auto-scroll when the user is near the bottom.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    if (stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [items]);
  const onScroll = () => {
    const el = logRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const patchMsg = useCallback((id: string, fn: (m: MsgItem) => MsgItem) => {
    setItems((xs) => xs.map((it) => (it.kind === "msg" && it.msg.id === id ? fn(it) : it)));
  }, []);
  const patchRun = useCallback((id: string, fn: (r: AgentRun) => AgentRun) => {
    setItems((xs) => xs.map((it) => (it.kind === "agent" && it.run.id === id ? { kind: "agent", run: fn(it.run) } : it)));
  }, []);

  const send = async () => {
    const content = draft.trim();
    if (!content || streaming || !activeChannel) return;
    setDraft("");
    const now = new Date().toISOString();
    const userId = `local-u-${++seq.current}`;
    const asstId = `local-a-${++seq.current}`;
    stickToBottom.current = true;
    setItems((xs) => [
      ...xs,
      { kind: "msg", msg: { id: userId, role: "user", content, ts: now } },
      { kind: "msg", msg: { id: asstId, role: "assistant", content: "", ts: now, trace: [] }, pending: true },
    ]);
    const ac = new AbortController();
    setStreaming({ id: asstId, ac });
    let touchedVault = false;
    try {
      await api.chat.send(
        activeChannel.id,
        { content, model },
        (ev) => {
          if (ev.type === "text") {
            patchMsg(asstId, (m) => ({ ...m, msg: { ...m.msg, content: m.msg.content + (ev.delta ?? "") } }));
          } else if (ev.type === "tool") {
            if (/write|append|file|set|update|run/.test(ev.name)) touchedVault = true;
            patchMsg(asstId, (m) => {
              const trace = [...(m.msg.trace ?? [])];
              const idx = trace.findIndex((t) => t.id === ev.id);
              const row: ToolTrace = { id: ev.id, name: ev.name, input: ev.input, status: ev.status, summary: ev.summary, link: ev.link };
              if (idx === -1) trace.push(row);
              else trace[idx] = { ...trace[idx], ...row };
              return { ...m, msg: { ...m.msg, trace } };
            });
          } else if (ev.type === "done") {
            patchMsg(asstId, (m) => ({ ...m, pending: false, msg: { ...ev.message, id: ev.message?.id ?? asstId, trace: ev.message?.trace ?? m.msg.trace } }));
          } else if (ev.type === "error") {
            patchMsg(asstId, (m) => ({ ...m, pending: false, error: `${ev.code ? `${ev.code}: ` : ""}${ev.message}` }));
          }
        },
        ac.signal,
      );
    } catch (e) {
      patchMsg(asstId, (m) => ({ ...m, pending: false, error: errorMessage(e) }));
    } finally {
      const aborted = ac.signal.aborted;
      patchMsg(asstId, (m) => ({ ...m, pending: false, stopped: aborted || m.stopped }));
      setStreaming(null);
      channels.reload();
      if (touchedVault) emitCommand("vault-changed");
    }
  };

  const stop = () => {
    streaming?.ac.abort();
  };

  const clearChannel = async () => {
    if (!activeChannel) return;
    if (!window.confirm(`Clear all messages in #${activeChannel.name}?`)) return;
    try {
      await api.chat.clear(activeChannel.id);
      setItems([]);
      channels.reload();
    } catch (e) {
      toast(`Clear failed: ${errorMessage(e)}`, "error");
    }
  };

  // ----- agents -----
  const lastAssistant = useMemo(() => {
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      if (it.kind === "msg" && it.msg.role === "assistant" && it.msg.content.trim()) return it.msg.content;
    }
    return "";
  }, [items]);

  const handTo = (agent: AgentId) => {
    const task = draft.trim() || lastAssistant;
    const id = `run-${++seq.current}`;
    stickToBottom.current = true;
    setItems((xs) => [...xs, { kind: "agent", run: { id, agent, task, editing: true, lines: [], code: null, running: false } }]);
  };

  const startRun = async (id: string, task: string) => {
    const t = task.trim();
    if (!t) return;
    const ac = new AbortController();
    runAborts.current.set(id, ac);
    let agent: AgentId = "codex";
    setItems((xs) =>
      xs.map((it) => {
        if (it.kind === "agent" && it.run.id === id) {
          agent = it.run.agent;
          return { kind: "agent", run: { ...it.run, task: t, editing: false, running: true, lines: [], code: null, error: undefined } };
        }
        return it;
      }),
    );
    try {
      await api.agents.run(
        { agent, task: t },
        (ev) => {
          if (ev.type === "log") patchRun(id, (r) => ({ ...r, lines: [...r.lines, ev.line] }));
          else if (ev.type === "done") patchRun(id, (r) => ({ ...r, code: ev.code, running: false }));
        },
        ac.signal,
      );
    } catch (e) {
      patchRun(id, (r) => ({ ...r, error: errorMessage(e), running: false }));
    } finally {
      runAborts.current.delete(id);
      patchRun(id, (r) => ({ ...r, running: false, stopped: ac.signal.aborted }));
      emitCommand("vault-changed");
    }
  };
  const stopRun = (id: string) => runAborts.current.get(id)?.abort();
  const dismissRun = (id: string) => setItems((xs) => xs.filter((it) => !(it.kind === "agent" && it.run.id === id)));

  const agentInfo = (id: AgentId): AgentInfo | undefined => agents.data?.find((a) => a.id === id);
  const agentTitle = (id: AgentId) => {
    const a = agentInfo(id);
    if (agents.error) return `Agent status unknown: ${agents.error}`;
    if (!a) return agents.loading ? "Checking agents…" : `${id} not reported by the server`;
    return a.available ? `Hand the task to ${id}${a.version ? ` (${a.version})` : ""}` : `${id} is not installed`;
  };
  const agentDisabled = (id: AgentId) => Boolean(agents.data) && !agentInfo(id)?.available;

  const onComposerKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  };

  return (
    <>
      <div className="chat-head">
        <div className="top">
          <div className="panel-tabs" role="tablist" aria-label="Panel">
            <button role="tab" aria-selected={panelTab === "chat"} onClick={() => setPanelTab("chat")}>
              Chat
            </button>
            <button role="tab" aria-selected={panelTab === "agent"} onClick={() => setPanelTab("agent")}>
              Agent ledger{agentCalls.length > 0 && <span className="n">{agentCalls.length}</span>}
            </button>
          </div>
          <span className="spacer" style={{ flex: 1 }} />
          {panelTab === "chat" && (
            <button className="btn ghost sm" onClick={() => void clearChannel()} disabled={!activeChannel || items.length === 0} title="Delete this channel's history">
              Clear
            </button>
          )}
          <button className="icon-btn" onClick={onClose} title="Hide chat (Cmd+/)" aria-label="Hide chat">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
        {panelTab === "agent" && (
          <div className="chat-desc">
            {agentReady ? "Tools the browser's agent called on this page." : "WebMCP not detected: open in Chrome 149+ with WebMCP or ChatGPT's browser."}
          </div>
        )}
        {panelTab === "chat" && (
        <div className="channel-tabs" role="tablist" aria-label="Channels">
          {channels.loading && !channels.data && <span className="chat-desc">loading channels…</span>}
          {channelList.map((c) => (
            <button key={c.id} role="tab" aria-selected={activeChannel?.id === c.id} onClick={() => setChannel(c.id)} title={c.desc}>
              {c.name}
              {c.count > 0 && <span className="n">{c.count}</span>}
            </button>
          ))}
        </div>
        )}
        {panelTab === "chat" && activeChannel?.desc && (
          <div className="chat-desc">
            {activeChannel.desc}
            {channels.error && <span title={channels.error}> · channel list unavailable, using defaults</span>}
          </div>
        )}
      </div>

      {panelTab === "agent" && <AgentLedger calls={agentCalls} />}
      {panelTab === "chat" && (
      <div className="chat-log" ref={logRef} onScroll={onScroll}>
        {loadState.loading && <Loading label="Loading history" />}
        {loadState.error && <ErrorState title="History unavailable" error={loadState.error} onRetry={() => loadMessages(channel)} compact />}
        {!loadState.loading && !loadState.error && items.length === 0 && (
          <div className="state compact">
            <h2 style={{ fontSize: "var(--fs-md)" }}>#{activeChannel?.name ?? channel}</h2>
            <p>Ask about the vault. Replies cite notes and papers with links that open in the center pane.</p>
          </div>
        )}
        {items.map((it) =>
          it.kind === "msg" ? (
            <Message key={it.msg.id} item={it} />
          ) : (
            <AgentRunBlock key={it.run.id} run={it.run} onStart={(t) => void startRun(it.run.id, t)} onStop={() => stopRun(it.run.id)} onDismiss={() => dismissRun(it.run.id)} />
          ),
        )}
      </div>
      )}

      {panelTab === "chat" && (
      <div className="composer">
        <textarea
          ref={composerRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onComposerKey}
          placeholder={activeChannel ? `Message #${activeChannel.name}` : "Message"}
          aria-label="Message"
          rows={3}
          disabled={!activeChannel}
        />
        <div className="bar">
          <select className="select sm" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model" title={models.error ? `Model list unavailable: ${models.error}` : "Model"}>
            {!modelList.some((m) => m.id === model) && <option value={model}>{model}</option>}
            {modelList.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.id}
              </option>
            ))}
          </select>
          <button className="btn sm" onClick={() => handTo("codex")} disabled={agentDisabled("codex") || (!draft.trim() && !lastAssistant)} title={agentTitle("codex")}>
            Hand to Codex
          </button>
          <button className="btn sm" onClick={() => handTo("opencode")} disabled={agentDisabled("opencode") || (!draft.trim() && !lastAssistant)} title={agentTitle("opencode")}>
            Hand to OpenCode
          </button>
          <span className="grow" />
          {streaming ? (
            <button className="btn sm danger" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="btn sm primary" onClick={() => void send()} disabled={!draft.trim() || !activeChannel}>
              Send
            </button>
          )}
        </div>
        <span className="hint">Enter sends · Shift+Enter newline · agent buttons use the draft or the last reply as the task</span>
      </div>
      )}
    </>
  );
}

/** Ledger of WebMCP tool calls made by the browser's agent (same styling as the chat tool ledger). */
function AgentLedger({ calls }: { calls: WebMCPCall[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [calls.length]);
  return (
    <div className="chat-log" ref={ref}>
      {calls.length === 0 ? (
        <div className="state compact">
          <h2 style={{ fontSize: "var(--fs-md)" }}>No agent calls yet</h2>
          <p>When a browser agent uses this page's tools (search, read, write, file paper, update project), each call is listed here.</p>
        </div>
      ) : (
        <div className="ledger" aria-label="Agent tool calls">
          {calls.map((c, i) => (
            <div className="ledger-row" key={`${c.ts}-${i}`} data-status={c.ok ? "ok" : "error"} title={JSON.stringify(c.input ?? {}, null, 1)}>
              <span className="st" aria-hidden="true" />
              <span className="what">
                <b>{c.tool}</b> <span className="arg">{c.summary}</span>
                {!c.ok && <span style={{ color: "var(--danger)" }}> · failed</span>}
              </span>
              <span className="lnk" style={{ color: "var(--text-3)" }}>
                {clock(new Date(c.ts))}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ item }: { item: MsgItem }) {
  const { msg } = item;
  const d = parseDate(msg.ts);
  return (
    <div className={`msg ${msg.role}`}>
      <div className="who">
        <span>{msg.role === "assistant" ? "cortex" : msg.role}</span>
        {d && <span className="ts">{clock(d)}</span>}
      </div>
      <div className="body">
        {msg.role === "user" ? (
          msg.content
        ) : (
          <>
            {msg.content ? <MarkdownPreview source={msg.content} debounce={item.pending ? 60 : 0} /> : item.pending ? null : <span className="faint">(empty reply)</span>}
            {item.pending && <span className="cursor" aria-label="streaming" />}
          </>
        )}
      </div>
      {msg.trace && msg.trace.length > 0 && <Ledger trace={msg.trace} />}
      {item.error && <div className="err">{item.error}</div>}
      {item.stopped && !item.error && <div className="stopped">stopped</div>}
    </div>
  );
}

const VERBS: Record<string, string> = {
  search_vault: "search",
  read_note: "read note",
  write_note: "write note",
  append_daily: "append to daily",
  read_paper: "read paper",
  file_paper: "file paper",
  set_paper: "set paper",
  list_projects: "list projects",
  update_project: "update project",
  run_agent: "run agent",
};

function describeInput(input: unknown): string {
  if (input == null) return "";
  if (typeof input === "string") return input;
  if (typeof input !== "object") return String(input);
  const o = input as Record<string, unknown>;
  const first = ["query", "slug", "id", "title", "arxiv", "url", "path", "task", "agent", "status", "scope", "text"].find((k) => typeof o[k] === "string" && (o[k] as string).trim());
  if (first) {
    const v = String(o[first]);
    return v.length > 80 ? `${v.slice(0, 79)}…` : v;
  }
  const s = JSON.stringify(o);
  return s.length > 80 ? `${s.slice(0, 79)}…` : s;
}

function Ledger({ trace }: { trace: ToolTrace[] }) {
  const onLink = (e: MouseEvent<HTMLElement>) => {
    handleCortexClick(e);
  };
  return (
    <div className="ledger" aria-label="Tool calls">
      {trace.map((t) => {
        const link = t.link ?? "";
        const isCortex = Boolean(parseCortexLink(link));
        return (
          <div className="ledger-row" key={t.id} data-status={t.status} title={typeof t.input === "string" ? t.input : JSON.stringify(t.input ?? {}, null, 1)}>
            <span className="st" aria-hidden="true" />
            <span className="what">
              <b>{VERBS[t.name] ?? t.name.replace(/_/g, " ")}</b> <span className="arg">{t.summary || describeInput(t.input)}</span>
              {t.status === "error" && <span style={{ color: "var(--danger)" }}> · failed</span>}
            </span>
            {link ? (
              isCortex ? (
                <button className="lnk" onClick={() => { const r = parseCortexLink(link); if (r) navigate(r); }}>
                  open
                </button>
              ) : (
                <a href={link} target="_blank" rel="noopener noreferrer" onClick={onLink}>
                  open
                </a>
              )
            ) : (
              <span />
            )}
          </div>
        );
      })}
    </div>
  );
}

function AgentRunBlock({ run, onStart, onStop, onDismiss }: { run: AgentRun; onStart: (task: string) => void; onStop: () => void; onDismiss: () => void }) {
  const [task, setTask] = useState(run.task);
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run.lines.length]);
  const label = run.agent === "codex" ? "Codex" : run.agent === "opencode" ? "OpenCode" : "Claude";
  return (
    <div className="agent-run">
      <div className="head">
        <b>{label}</b>
        <span>{run.editing ? "task" : run.running ? "running" : run.error ? "failed" : run.stopped ? "stopped" : `exit ${run.code ?? "?"}`}</span>
        <span className="grow" />
        {run.running ? (
          <button className="btn sm danger" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="btn ghost sm" onClick={onDismiss} title="Remove this block">
            Dismiss
          </button>
        )}
      </div>
      {run.editing ? (
        <div className="task">
          <textarea value={task} onChange={(e) => setTask(e.target.value)} aria-label={`Task for ${label}`} placeholder="Describe the task" />
          <div className="actions">
            <button className="btn sm" onClick={onDismiss}>
              Cancel
            </button>
            <button className="btn sm primary" onClick={() => onStart(task)} disabled={!task.trim()}>
              Run {label}
            </button>
          </div>
        </div>
      ) : (
        <div className="task">{run.task}</div>
      )}
      {!run.editing && (
        <pre className="agent-log" ref={logRef} aria-live="polite">
          {run.lines.length === 0 && run.running ? "waiting for output…" : null}
          {run.lines.map((l, i) => (
            <span key={i}>
              <span className="ln">{i + 1}</span>
              {l}
              {"\n"}
            </span>
          ))}
        </pre>
      )}
      {run.error && <div className="foot bad">{run.error}</div>}
      {!run.editing && !run.running && !run.error && run.code !== null && (
        <div className={`foot ${run.code === 0 ? "ok" : "bad"}`}>exited with code {run.code}</div>
      )}
    </div>
  );
}
