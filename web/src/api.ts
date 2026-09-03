// Typed client for every endpoint in cortex/SPEC.md.
import { readSSE } from "./lib/sse";
import type {
  AgentEvent,
  AgentId,
  AgentInfo,
  Channel,
  ChatEvent,
  ChatMessage,
  Health,
  Highlights,
  ModelInfo,
  Note,
  NoteFrontmatter,
  NoteKind,
  NoteSummary,
  PaperDetail,
  PaperMeta,
  Project,
  ProjectFrontmatter,
  SearchHit,
  Topic,
} from "./types";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

const BASE = "/api";

function qs(params: Record<string, string | number | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

async function toError(res: Response): Promise<ApiError> {
  let body: unknown = undefined;
  let message = `${res.status} ${res.statusText}`;
  try {
    const text = await res.text();
    try {
      body = JSON.parse(text);
      const detail = (body as { detail?: unknown; message?: unknown })?.detail ?? (body as { message?: unknown })?.message;
      if (typeof detail === "string") message = detail;
      else if (detail) message = JSON.stringify(detail);
    } catch {
      if (text) message = text.slice(0, 300);
    }
  } catch {
    /* ignore */
  }
  return new ApiError(res.status, message, body);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", ...(init.headers as Record<string, string>) };
  if (init.body && !(init.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch (e) {
    throw new ApiError(0, "Cannot reach the Cortex server. Is it running on port 8788?", e);
  }
  if (!res.ok) throw await toError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

async function stream<E>(
  path: string,
  body: unknown,
  onEvent: (e: E) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error)?.name === "AbortError") return;
    throw new ApiError(0, "Cannot reach the Cortex server. Is it running on port 8788?", e);
  }
  if (!res.ok) throw await toError(res);
  try {
    await readSSE<E>(res, onEvent, signal);
  } catch (e) {
    if ((e as Error)?.name === "AbortError") return;
    throw e;
  }
}

export const api = {
  health: () => request<Health>("/health"),
  topics: () => request<Topic[]>("/topics"),

  notes: {
    list: (p: { kind?: NoteKind | ""; q?: string; limit?: number } = {}) =>
      request<NoteSummary[]>(`/notes${qs({ kind: p.kind, q: p.q, limit: p.limit })}`),
    create: (data: { title: string; kind?: NoteKind; body?: string; topics?: string[] }) =>
      request<Note>("/notes", { method: "POST", body: JSON.stringify(data) }),
    get: (slug: string) => request<Note>(`/notes/${encodeURIComponent(slug)}`),
    update: (slug: string, patch: { frontmatter?: NoteFrontmatter; body?: string }) =>
      request<Note>(`/notes/${encodeURIComponent(slug)}`, { method: "PUT", body: JSON.stringify(patch) }),
    remove: (slug: string) => request<void>(`/notes/${encodeURIComponent(slug)}`, { method: "DELETE" }),
    /** Store a pasted/dropped file or image for a note; returns the markdown that embeds it. */
    attach: (slug: string, file: File) => {
      const fd = new FormData();
      fd.append("file", file, file.name || "pasted.png");
      return request<{ url: string; name: string; kind: "image" | "file"; markdown: string; bytes: number }>(
        `/notes/${encodeURIComponent(slug)}/attach`,
        { method: "POST", body: fd },
      );
    },
  },

  daily: {
    today: () => request<Note>("/daily/today"),
  },

  projects: {
    list: (status?: string) => request<Project[]>(`/projects${qs({ status })}`),
    get: (slug: string) => request<Project>(`/projects/${encodeURIComponent(slug)}`),
    update: (slug: string, patch: { frontmatter?: ProjectFrontmatter; body?: string }) =>
      request<Project>(`/projects/${encodeURIComponent(slug)}`, { method: "PUT", body: JSON.stringify(patch) }),
    create: (data: { title: string } & ProjectFrontmatter) =>
      request<Project>("/projects", { method: "POST", body: JSON.stringify(data) }),
  },

  library: {
    list: (p: { status?: string; topic?: string; q?: string; project?: string } = {}) =>
      request<PaperMeta[]>(`/library${qs({ status: p.status, topic: p.topic, q: p.q, project: p.project })}`),
    get: (id: string) => request<PaperDetail>(`/library/${encodeURIComponent(id)}`),
    update: (id: string, patch: Partial<PaperMeta> & { notes?: string }) =>
      request<PaperDetail | PaperMeta>(`/library/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: JSON.stringify(patch),
      }),
    pdfUrl: (id: string) => `${BASE}/library/${encodeURIComponent(id)}/pdf`,
    highlights: (id: string) => request<Highlights>(`/library/${encodeURIComponent(id)}/highlights`),
    makeHighlights: (id: string, refresh = false) =>
      request<Highlights>(`/library/${encodeURIComponent(id)}/highlights`, { method: "POST", body: JSON.stringify({ refresh }) }),
    ingest: (src: { path: string } | { arxiv: string } | { url: string }) =>
      request<PaperMeta>("/library/ingest", { method: "POST", body: JSON.stringify(src) }),
    upload: (file: File) => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      return request<PaperMeta>("/library/upload", { method: "POST", body: fd });
    },
  },

  search: (q: string, limit = 30) => request<SearchHit[]>(`/search${qs({ q, limit })}`),

  chat: {
    channels: () => request<Channel[]>("/chat/channels"),
    messages: (channel: string) => request<ChatMessage[]>(`/chat/${encodeURIComponent(channel)}`),
    clear: (channel: string) => request<void>(`/chat/${encodeURIComponent(channel)}`, { method: "DELETE" }),
    send: (
      channel: string,
      data: { content: string; model?: string; context?: { kind?: string; id?: string; title?: string; space?: string } },
      onEvent: (e: ChatEvent) => void,
      signal?: AbortSignal,
    ) =>
      stream<ChatEvent>(`/chat/${encodeURIComponent(channel)}`, data, onEvent, signal),
  },

  models: () => request<ModelInfo[]>("/models"),

  agents: {
    list: () => request<AgentInfo[]>("/agents"),
    run: (data: { agent: AgentId; task: string }, onEvent: (e: AgentEvent) => void, signal?: AbortSignal) =>
      stream<AgentEvent>("/agents/run", data, onEvent, signal),
  },
};

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
