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

export interface LabExecutors {
  local: { available: boolean; note: string };
  ssh: { available: boolean; host: string | null; note: string };
  modal: { available: boolean; note: string };
  demo: boolean;
}
export interface LabRecipe { name: string; file: string; doc: string }
export type PlanCol = "todo" | "doing" | "done";
export interface PlanCard { id: string; chapter: number; kind: "read" | "station" | "build" | "recipe" | "quiz" | "custom"; title: string; col: PlanCol; note?: string; station?: string; recipe?: string; done_at?: string; comment?: string; custom?: boolean }
export interface LabPlan { columns: PlanCol[]; cards: PlanCard[]; done: number; total: number; xp: number; xp_total: number; level: number; level_name: string; next_level_xp: number; streak: number; done_today: number; xp_by_kind: Record<string, number> }
export interface GpuStatus {
  host: string | null; reachable: boolean; ready: boolean; message?: string; busy?: boolean; torch?: string; cuda?: boolean; python?: string;
  gpu?: { name: string; memory_total: string; memory_used: string; utilization: string };
  ssh_round_trip_ms?: number;
  tailscale?: { ip: string | null; os?: string | null; link?: string; state?: string } | null;
}
export type GpuSetupEvent = { type: "log"; lines: string[] } | { type: "status"; status: "done" | "failed"; exit: number; gpu: GpuStatus | null } | { type: "error"; message: string };
export interface LabChapter { slug: string; file: string; title?: string; chapter?: number; station?: string; recipe?: string; reading_time?: string }
export interface LabRun {
  id: string; recipe: string; args: string; executor: string;
  status: "queued" | "running" | "stopping" | "done" | "failed" | "stopped";
  started: string; ended: string | null; exit: number | null; error?: string;
  last?: Record<string, number> | null;
  script?: string; code_preview?: string; cmd?: string; origin?: string;
}
export interface LabRunDetail extends LabRun { log: string[]; log_lines: number; metrics: Record<string, number>[]; rollouts: Record<string, unknown>[]; result: Record<string, unknown> | null }

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
    makeHighlights: (id: string, refresh = false, pages?: string) =>
      request<Highlights>(`/library/${encodeURIComponent(id)}/highlights`, { method: "POST", body: JSON.stringify({ refresh, pages: pages || null }) }),
    ingest: (src: { path: string } | { arxiv: string } | { url: string }) =>
      request<PaperMeta>("/library/ingest", { method: "POST", body: JSON.stringify(src) }),
    upload: (file: File) => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      return request<PaperMeta>("/library/upload", { method: "POST", body: fd });
    },
  },

  search: (q: string, limit = 30) => request<SearchHit[]>(`/search${qs({ q, limit })}`),
  arxiv: (q: string, n = 5) => request<{ arxiv: string; title: string; authors: string; year: string; summary: string; in_library: boolean }[]>(`/arxiv${qs({ q, n })}`),

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

  chatToolResult: (id: string, result: unknown) => request<{ ok: boolean }>(`/chat/tool_result/${encodeURIComponent(id)}`, { method: "POST", body: JSON.stringify({ result }) }),

  models: () => request<ModelInfo[]>("/models"),

  lab: {
    executors: () => request<LabExecutors>("/lab/executors"),
    recipes: () => request<LabRecipe[]>("/lab/recipes"),
    chapters: () => request<LabChapter[]>("/lab/chapters"),
    runs: (limit = 50) => request<LabRun[]>(`/lab/runs${qs({ limit })}`),
    run: (id: string, tail = 200) => request<LabRunDetail>(`/lab/runs/${encodeURIComponent(id)}${qs({ tail })}`),
    start: (data: { recipe: string; args?: string; executor?: string; code?: string; cmd?: string; origin?: string }) => request<LabRun>("/lab/runs", { method: "POST", body: JSON.stringify(data) }),
    script: (id: string) => request<{ code: string }>(`/lab/runs/${encodeURIComponent(id)}/script`),
    stop: (id: string) => request<{ ok: boolean }>(`/lab/runs/${encodeURIComponent(id)}/stop`, { method: "POST" }),
    remove: (id: string) => request<{ ok: boolean }>(`/lab/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
    eventsUrl: (id: string) => `${BASE}/lab/runs/${encodeURIComponent(id)}/events`,
    gpu: () => request<GpuStatus>("/lab/gpu"),
    plan: () => request<LabPlan>("/lab/plan"),
    planMove: (id: string, col: PlanCol, comment?: string) => request<LabPlan>("/lab/plan/move", { method: "POST", body: JSON.stringify({ id, col, comment }) }),
    planAdd: (data: { title: string; kind?: string; note?: string; station?: string; recipe?: string; chapter?: number }) => request<LabPlan & { added?: string }>("/lab/plan/cards", { method: "POST", body: JSON.stringify(data) }),
    planRemove: (id: string) => request<LabPlan>(`/lab/plan/cards/${encodeURIComponent(id)}`, { method: "DELETE" }),
    gpuSetup: (onEvent: (e: GpuSetupEvent) => void, signal?: AbortSignal) => stream<GpuSetupEvent>("/lab/gpu/setup", {}, onEvent, signal),
  },

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
