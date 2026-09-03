/**
 * WebMCP layer: expose the brain's tools to the browser's agent via navigator.modelContext.
 *
 * Every tool is the same operation a person does in the UI, so an agent driving the page and the
 * person watching it see the same thing: tools navigate the center pane and fire `vault-changed`.
 * Feature-detected; a no-op outside Chrome 149+ (flag) or ChatGPT's browser. Agent input is untrusted: coerce it.
 */
import { api } from "../api";
import { emitCommand } from "./events";
import { navigate } from "./router";
import type { NoteKind } from "../types";

type ToolResult = { content: { type: "text"; text: string }[] };

export interface ModelContextTool {
  name: string;
  description: string;
  inputSchema?: Record<string, unknown>;
  annotations?: { readOnlyHint?: boolean };
  execute: (input: Record<string, unknown>, client?: unknown) => Promise<ToolResult>;
}

interface ModelContext {
  registerTool: (tool: ModelContextTool) => void;
  unregisterTool: (name: string) => void;
}

declare global {
  interface Navigator {
    modelContext?: ModelContext;
  }
}

const text = (v: unknown): ToolResult => ({ content: [{ type: "text", text: typeof v === "string" ? v : JSON.stringify(v, null, 1) }] });
const s = (v: unknown, max = 4000) => String(v ?? "").slice(0, max);
const n = (v: unknown, lo: number, hi: number) => Math.min(hi, Math.max(lo, Number(v) || lo));
const KINDS: NoteKind[] = ["fleeting", "literature", "permanent", "meeting", "daily"];
const changed = () => emitCommand("vault-changed");

/** Fired after every agent tool call so the UI can show a ledger of what the agent did. */
export type WebMCPCall = { tool: string; input: Record<string, unknown>; ok: boolean; summary: string; ts: number };
const listeners = new Set<(c: WebMCPCall) => void>();
export function onWebMCPCall(fn: (c: WebMCPCall) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
function record(tool: string, input: Record<string, unknown>, ok: boolean, summary: string) {
  const c = { tool, input, ok, summary, ts: Date.now() };
  for (const l of Array.from(listeners)) l(c);
}

export const webmcpTools: ModelContextTool[] = [
  {
    name: "search_brain",
    description: "Full-text search across the user's second brain: notes, projects, and the PDF library. Returns up to 20 hits with type, id, title, snippet. Use before answering questions about what the user knows, read, or is working on.",
    inputSchema: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer", minimum: 1, maximum: 20 } }, required: ["query"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => { const hits = await api.search(s(i.query, 300), n(i.limit, 1, 20)); record("search_brain", i, true, `${s(i.query, 60)} · ${hits.length} hits`); return text(hits); },
  },
  {
    name: "open_item",
    description: "Open a note, paper, or project in the app so the user sees it. kind: note|paper|project; id is the slug (note/project) or the paper id from search_brain.",
    inputSchema: { type: "object", properties: { kind: { type: "string", enum: ["note", "paper", "project"] }, id: { type: "string" } }, required: ["kind", "id"] },
    execute: async (i) => {
      const id = s(i.id, 200); const k = s(i.kind, 10);
      if (k === "note") navigate({ kind: "note", slug: id }); else if (k === "paper") navigate({ kind: "paper", id }); else if (k === "project") navigate({ kind: "project", slug: id });
      else throw new Error("kind must be note, paper, or project");
      record("open_item", i, true, `${k}/${id}`); return text({ opened: `cortex://${k}/${id}` });
    },
  },
  {
    name: "read_note",
    description: "Read a note or daily page in full: frontmatter (title, kind, topics) and the markdown body, which may contain LaTeX.",
    inputSchema: { type: "object", properties: { slug: { type: "string" } }, required: ["slug"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => { const note = await api.notes.get(s(i.slug, 200)); record("read_note", i, true, String(note.frontmatter.title ?? note.slug)); return text(note); },
  },
  {
    name: "write_note",
    description: "Create a note in the brain and open it for the user. kind: fleeting (default), permanent, literature, meeting. body is markdown; $...$ and $$...$$ render as LaTeX. Only when the user asks to save, note, or remember something.",
    inputSchema: { type: "object", properties: { title: { type: "string" }, kind: { type: "string", enum: ["fleeting", "permanent", "literature", "meeting"] }, body: { type: "string" }, topics: { type: "array", items: { type: "string" } } }, required: ["title", "body"] },
    execute: async (i) => {
      const kind = (KINDS.includes(s(i.kind) as NoteKind) ? s(i.kind) : "fleeting") as NoteKind;
      const note = await api.notes.create({ title: s(i.title, 200), kind, body: s(i.body, 60000), topics: Array.isArray(i.topics) ? i.topics.map((t) => s(t, 40)) : [] });
      changed(); navigate({ kind: "note", slug: note.slug }); record("write_note", i, true, s(i.title, 60));
      return text({ slug: note.slug, link: `cortex://note/${note.slug}` });
    },
  },
  {
    name: "append_today",
    description: "Append a line or paragraph to today's daily page and show it.",
    inputSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: async (i) => {
      const today = await api.daily.today();
      const body = (today.body.trimEnd() + "\n\n" + s(i.text, 8000).trim() + "\n").replace(/^\n+/, "");
      await api.notes.update(today.slug, { body });
      changed(); navigate({ kind: "daily" }); record("append_today", i, true, s(i.text, 60));
      return text({ slug: today.slug });
    },
  },
  {
    name: "read_paper",
    description: "Read a paper from the library: metadata (title, authors, year, status, rating, takeaway), the user's reading notes, and a preview of the extracted text.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => { const p = await api.library.get(s(i.id, 200)); record("read_paper", i, true, String(p.meta.title ?? p.meta.id)); return text(p); },
  },
  {
    name: "file_paper",
    description: "Add a paper to the library from an arXiv id or arXiv URL: the PDF is downloaded, its text indexed, and the paper opened for the user.",
    inputSchema: { type: "object", properties: { arxiv: { type: "string" } }, required: ["arxiv"] },
    execute: async (i) => {
      const meta = await api.library.ingest({ arxiv: s(i.arxiv, 80) });
      changed(); navigate({ kind: "paper", id: meta.id }); record("file_paper", i, true, String(meta.title));
      return text({ id: meta.id, title: meta.title, link: `cortex://paper/${meta.id}` });
    },
  },
  {
    name: "set_paper",
    description: "Update a paper's reading status (inbox|reading|read|reference), rating (1-5), or one-line takeaway.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, status: { type: "string", enum: ["inbox", "reading", "read", "reference"] }, rating: { type: "integer", minimum: 1, maximum: 5 }, takeaway: { type: "string" } }, required: ["id"] },
    execute: async (i) => {
      const patch: Record<string, unknown> = {};
      if (i.status) patch.status = s(i.status, 20);
      if (i.rating != null) patch.rating = n(i.rating, 1, 5);
      if (i.takeaway) patch.takeaway = s(i.takeaway, 1000);
      const r = await api.library.update(s(i.id, 200), patch);
      changed(); record("set_paper", i, true, `${s(i.id, 30)} · ${Object.keys(patch).join(", ")}`);
      return text("meta" in r ? r.meta : r);
    },
  },
  {
    name: "list_projects",
    description: "List the user's projects with status, one-line verdict, and next action. status: active|paused|done|banked|refuted|all (default active).",
    inputSchema: { type: "object", properties: { status: { type: "string" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const st = s(i.status || "active", 20);
      const ps = await api.projects.list(st === "all" ? undefined : st);
      record("list_projects", i, true, `${st} · ${ps.length}`);
      return text(ps.map((p) => ({ slug: p.slug, title: p.frontmatter.title, status: p.frontmatter.status, verdict: p.frontmatter.verdict, next_action: p.frontmatter.next_action })));
    },
  },
  {
    name: "update_project",
    description: "Update a project's verdict, next action, or status (active|paused|done|banked|refuted), then show it to the user.",
    inputSchema: { type: "object", properties: { slug: { type: "string" }, verdict: { type: "string" }, next_action: { type: "string" }, status: { type: "string" } }, required: ["slug"] },
    execute: async (i) => {
      const fm: Record<string, unknown> = {};
      for (const k of ["verdict", "next_action", "status"] as const) if (i[k]) fm[k] = s(i[k], 1000);
      const p = await api.projects.update(s(i.slug, 200), { frontmatter: fm });
      changed(); navigate({ kind: "project", slug: p.slug }); record("update_project", i, true, `${String(p.frontmatter.title)} · ${Object.keys(fm).join(", ")}`);
      return text(p.frontmatter);
    },
  },
];

/** Wrap execute so failures are reported to the agent as text and to the UI ledger, never as an unhandled rejection. */
function guarded(t: ModelContextTool): ModelContextTool {
  return { ...t, execute: async (input, client) => {
    try { return await t.execute(input ?? {}, client); }
    catch (e) { const msg = e instanceof Error ? e.message : String(e); record(t.name, input ?? {}, false, msg); return text({ error: msg }); }
  } };
}

/** Register the brain's tools once. Returns true when the browser exposes WebMCP. */
export function installWebMCP(): boolean {
  // Always expose the tool list for DevTools and demos: window.cortex.call("search_brain", {query: "probes"})
  const w = window as unknown as { cortex?: { tools: ModelContextTool[]; call: (name: string, input?: Record<string, unknown>) => Promise<ToolResult> } };
  w.cortex = {
    tools: webmcpTools,
    call: (name, input = {}) => {
      const t = webmcpTools.find((x) => x.name === name);
      if (!t) return Promise.resolve(text({ error: `no tool ${name}` }));
      return guarded(t).execute(input);
    },
  };
  const mc = typeof navigator !== "undefined" ? navigator.modelContext : undefined;
  if (!mc) return false;
  for (const t of webmcpTools) {
    try { mc.registerTool(guarded(t)); } catch (e) { console.warn("webmcp: could not register", t.name, e); }
  }
  return true;
}

export const webmcpAvailable = () => typeof navigator !== "undefined" && !!navigator.modelContext;
