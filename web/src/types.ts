// Shared types mirroring cortex/SPEC.md. Keep in sync with the server.

export type NoteKind = "fleeting" | "literature" | "permanent" | "meeting" | "daily";
export const NOTE_KINDS: NoteKind[] = ["fleeting", "literature", "permanent", "meeting", "daily"];

export interface NoteSummary {
  slug: string;
  title: string;
  kind: NoteKind;
  topics: string[];
  updated: string;
  preview: string;
}

export interface NoteFrontmatter {
  title?: string;
  kind?: NoteKind;
  topics?: string[];
  sources?: string[];
  projects?: string[];
  created?: string;
  updated?: string;
  [key: string]: unknown;
}

export interface Note {
  slug: string;
  frontmatter: NoteFrontmatter;
  body: string;
}

export type ProjectStatus = "active" | "paused" | "done" | "banked" | "refuted";
export const PROJECT_STATUSES: ProjectStatus[] = ["active", "paused", "done", "banked", "refuted"];
export type ProjectType = "research" | "competition" | "build" | "career" | "life";
export const PROJECT_TYPES: ProjectType[] = ["research", "competition", "build", "career", "life"];

export interface ProjectFrontmatter {
  title?: string;
  status?: ProjectStatus;
  type?: ProjectType;
  verdict?: string;
  next_action?: string;
  deadline?: string;
  repo?: string;
  topics?: string[];
  created?: string;
  updated?: string;
  [key: string]: unknown;
}

export interface Project {
  slug: string;
  frontmatter: ProjectFrontmatter;
  body: string;
}

// TODO(spec): the spec lists the library sidebar buckets as Inbox / Reading / Read / Reference
// but does not fix the stored `status` strings. We assume lowercase; adjust if the server differs.
export type PaperStatus = "inbox" | "reading" | "read" | "reference";
export const PAPER_STATUSES: PaperStatus[] = ["inbox", "reading", "read", "reference"];

export interface PaperMeta {
  has_pdf?: boolean; // set on list rows by the server
  id: string;
  title: string;
  authors: string | string[]; // server stores a comma-separated string; older data may be an array
  year?: number | null;
  arxiv?: string | null;
  link?: string | null;
  type?: string | null;
  status: PaperStatus;
  rating?: number | null;
  takeaway?: string | null;
  topics: string[];
  added?: string | null;
  pages?: number | null;
  source_path?: string | null;
}

export interface PaperDetail {
  meta: PaperMeta;
  notes: string;
  text_preview: string;
}

export interface Topic {
  slug: string;
  name: string;
  kind?: string;
  one_liner?: string;
}

export type SearchType = "note" | "project" | "paper";
export interface SearchHit {
  type: SearchType;
  id: string;
  title: string;
  snippet: string;
  score: number;
}

export interface Channel {
  id: string;
  name: string;
  desc: string;
  count: number;
}

export type ToolStatus = "running" | "ok" | "error";
export interface ToolTrace {
  id: string;
  name: string;
  input: unknown;
  status: ToolStatus;
  summary?: string;
  link?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  ts: string;
  trace?: ToolTrace[];
}

export type ChatEvent =
  | { type: "text"; delta: string }
  | { type: "tool"; id: string; name: string; input: unknown; status: ToolStatus; summary?: string; link?: string }
  | { type: "done"; message: ChatMessage }
  | { type: "error"; code: string; message: string };

export interface ModelInfo {
  id: string;
  name: string;
}

export type AgentId = "codex" | "opencode" | "claude";
export interface AgentInfo {
  id: AgentId;
  available: boolean;
  version?: string | null;
}

export type AgentEvent = { type: "log"; line: string } | { type: "done"; code: number };

export interface Health {
  ok: boolean;
  vault: string;
  counts: Record<string, number>;
}
