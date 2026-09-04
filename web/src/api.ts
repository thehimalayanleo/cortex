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

export interface Take { id: string; status: string; started?: string; ended?: string | null; executor?: string; origin?: string; verdict?: string | null; identity_mean?: number | null; identity_min?: number | null; flicker_mean?: number | null; gen_s?: number | null; model?: string | null; contact?: string | null; clip?: string | null; fetched?: boolean }
export interface Shot { id: string; title: string; prompt: string; keyframe?: string | null; character?: string | null; frames?: number; size?: string; notes?: string; status: "planned" | "rendering" | "rendered" | "approved" | "reshoot"; takes: Take[]; created?: string; director_note?: string }
export interface CharacterBuild { id: string; status: string; stage?: string | null; started?: string; ended?: string | null; executor?: string; proto_mean?: number | null; proto_min?: number | null; p_own?: number | null; n_kept?: number | null; elapsed_s?: number | null }
export type CharacterStage = "hero" | "heroset" | "lora";
export interface Character { id: string; name: string; description: string; style?: string; negative?: string; hero?: string | null; hero_src?: string | null; workdir?: string; heroset_dir?: string | null; lora_dir?: string | null; status: "draft" | "building" | "hero" | "heroset" | "lora" | "failed"; builds: CharacterBuild[]; scores?: Record<string, number>; hero_url?: string | null; contact?: string | null; created?: string }
export interface SceneShotRow { id: string; title: string; status: string; verdict?: string | null; take?: string | null; frames?: number }
export interface Scene { id: string; title: string; kind: "filler" | "full"; set: { name: string; splat?: string | null }; characters: string[]; shots: string[]; dialogue: { who: string; line: string }[]; continuity?: string; logline?: string; status: "planned" | "rendering" | "rendered" | "assembled"; duration_s: number; shot_rows: SceneShotRow[]; video?: string | null; strip?: string | null; created?: string }
export interface StudioBoard { logline: string; shots: Shot[]; counts: Record<string, number>; assets: string[]; characters: Character[]; scenes: Scene[] }
export interface TraceIn { kind?: string; content?: string; data?: unknown; tags?: string[]; source?: string; context?: string; prompt?: string; response?: string; chosen?: string; rejected?: string; rating?: number }
export interface TraceRec extends TraceIn { id: string; ts: number; when: string; kind: string; file?: string }
export type PipelineStatus = "created" | "running" | "paused" | "done" | "failed";
export type StageStatus = "pending" | "running" | "done" | "failed";
export interface PipelineTemplate { name: string; title: string; doc: string; stages: { name: string; recipe: string; deps: string[] }[] }
export interface PipelineStage {
  name: string; recipe: string; deps: string[]; args_template: string; args: string | null; run_id: string | null; status: StageStatus;
  started: string | null; ended: string | null; error: string | null; attempts: number;
  run_status?: string | null; last?: Record<string, number> | null; result?: Record<string, unknown> | null; elapsed_s?: number | null;
}
export interface PipelineSummary { id: string; template: string; title: string; executor: string; smoke: boolean; status: PipelineStatus; created: string; updated: string; error: string | null; progress: { done: number; total: number }; current: string | null }
export interface Pipeline extends PipelineSummary { out: string; stages: PipelineStage[]; data: { sources: Record<string, number>; total_tokens: number; tokenizer: string } | null; final: Record<string, unknown> | null }
export interface TraceStats { total: number; by_kind: Record<string, number>; by_month: Record<string, number>; since: string | null; sft_pairs: number; dpo_pairs: number; root: string }

export interface GalaxyPaper { id: string; title: string; year?: number | null; status?: string; topics?: string[]; authors?: string; x: number; y: number; x3: number; y3: number; z3: number; cluster: number; universe: number; near?: [string, number][] }
export interface GalaxyCluster { id: number; label: string; size: number; cx: number; cy: number; universe: number }
export interface GalaxyUniverse { id: number; label: string; clusters: number[]; size: number }
export interface Galaxy { generated: string | null; model?: string; n: number; papers: GalaxyPaper[]; clusters: GalaxyCluster[]; universes: GalaxyUniverse[]; building?: string | null; stale?: boolean; error?: string }
export interface GalaxySummary { generated: string | null; n: number; model?: string; stale?: boolean; building?: string | null; universes: { id: number; label: string; size: number }[]; solar_systems: { id: number; label: string; size: number; universe: number }[] }

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

  studio: {
    board: () => request<StudioBoard>("/studio"),
    plan: (logline: string, n = 4) => request<{ shots: Shot[]; board: StudioBoard }>("/studio/plan", { method: "POST", body: JSON.stringify({ logline, n }) }),
    logline: (logline: string) => request<StudioBoard>("/studio/logline", { method: "POST", body: JSON.stringify({ logline }) }),
    add: (data: { title: string; prompt: string; keyframe?: string; frames?: number; size?: string; notes?: string }) => request<Shot>("/studio/shots", { method: "POST", body: JSON.stringify(data) }),
    update: (id: string, patch: Partial<Shot> & { director_note?: string }) => request<Shot>(`/studio/shots/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(patch) }),
    remove: (id: string) => request<{ ok: boolean }>(`/studio/shots/${encodeURIComponent(id)}`, { method: "DELETE" }),
    render: (id: string, opts: { executor?: string; smoke?: boolean } = {}) => request<LabRun>(`/studio/shots/${encodeURIComponent(id)}/render`, { method: "POST", body: JSON.stringify(opts) }),
    refresh: (id: string) => request<Shot>(`/studio/shots/${encodeURIComponent(id)}/refresh`, { method: "POST" }),
    upload: (file: File) => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      return request<{ name: string; url: string }>("/studio/assets", { method: "POST", body: fd });
    },
    assetUrl: (name: string) => `${BASE}/studio/assets/${encodeURIComponent(name)}`,
    // characters: the bible; hero, hero set and LoRA built by the cinema_character recipe
    characters: () => request<Character[]>("/studio/characters"),
    addCharacter: (data: { name: string; description: string; style?: string; negative?: string; hero?: string; hero_src?: string; heroset_dir?: string; lora_dir?: string; workdir?: string }) => request<Character>("/studio/characters", { method: "POST", body: JSON.stringify(data) }),
    updateCharacter: (id: string, patch: Partial<Omit<Character, "builds" | "scores">>) => request<Character>(`/studio/characters/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(patch) }),
    removeCharacter: (id: string) => request<{ ok: boolean }>(`/studio/characters/${encodeURIComponent(id)}`, { method: "DELETE" }),
    buildCharacter: (id: string, opts: { stage: CharacterStage; executor?: string; smoke?: boolean; force?: boolean }) => request<LabRun>(`/studio/characters/${encodeURIComponent(id)}/build`, { method: "POST", body: JSON.stringify(opts) }),
    refreshCharacter: (id: string) => request<Character>(`/studio/characters/${encodeURIComponent(id)}/refresh`, { method: "POST" }),
    // scenes: ordered shots; filler b-roll or a full scene; assembled here with ffmpeg
    scenes: () => request<Scene[]>("/studio/scenes"),
    addScene: (data: { title: string; kind?: "filler" | "full"; set_name?: string; splat?: string; characters?: string[]; shots?: string[]; dialogue?: { who: string; line: string }[]; continuity?: string; logline?: string }) => request<Scene>("/studio/scenes", { method: "POST", body: JSON.stringify(data) }),
    planScene: (data: { logline: string; kind?: "filler" | "full"; n?: number; characters?: string[]; set_name?: string }) => request<Scene>("/studio/scenes/plan", { method: "POST", body: JSON.stringify(data) }),
    updateScene: (id: string, patch: Partial<Pick<Scene, "title" | "kind" | "set" | "characters" | "shots" | "dialogue" | "continuity" | "status" | "logline">>) => request<Scene>(`/studio/scenes/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(patch) }),
    removeScene: (id: string, withShots = false) => request<{ ok: boolean }>(`/studio/scenes/${encodeURIComponent(id)}?with_shots=${withShots}`, { method: "DELETE" }),
    renderScene: (id: string, opts: { executor?: string; smoke?: boolean; only_missing?: boolean } = {}) => request<Scene>(`/studio/scenes/${encodeURIComponent(id)}/render`, { method: "POST", body: JSON.stringify(opts) }),
    assembleScene: (id: string) => request<Scene & { clips: number; missing: string[]; path: string }>(`/studio/scenes/${encodeURIComponent(id)}/assemble`, { method: "POST" }),
  },

  traces: {
    add: (t: TraceIn) => request<TraceRec>("/traces", { method: "POST", body: JSON.stringify(t) }),
    upload: (file: File, kind = "file", tags = "", context = "") => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      return request<TraceRec>(`/traces/file${qs({ kind, tags, context })}`, { method: "POST", body: fd });
    },
    list: (p: { limit?: number; kind?: string; q?: string } = {}) => request<TraceRec[]>(`/traces${qs({ limit: p.limit, kind: p.kind, q: p.q })}`),
    stats: () => request<TraceStats>("/traces/stats"),
    exportUrl: (fmt: string) => `${BASE}/traces/export${qs({ fmt })}`,
  },

  pipelines: {
    templates: () => request<PipelineTemplate[]>("/pipelines/templates"),
    list: (limit = 50) => request<PipelineSummary[]>(`/pipelines${qs({ limit })}`),
    get: (id: string) => request<Pipeline>(`/pipelines/${encodeURIComponent(id)}`),
    create: (data: { template: string; executor?: string; smoke?: boolean; overrides?: Record<string, unknown>; start?: boolean }) => request<Pipeline>("/pipelines", { method: "POST", body: JSON.stringify(data) }),
    start: (id: string) => request<Pipeline>(`/pipelines/${encodeURIComponent(id)}/start`, { method: "POST" }),
    pause: (id: string) => request<Pipeline>(`/pipelines/${encodeURIComponent(id)}/pause`, { method: "POST" }),
    retry: (id: string, stage: string) => request<Pipeline>(`/pipelines/${encodeURIComponent(id)}/retry/${encodeURIComponent(stage)}`, { method: "POST" }),
    remove: (id: string, runs = false) => request<{ ok: boolean }>(`/pipelines/${encodeURIComponent(id)}${qs({ runs: runs ? "true" : "" })}`, { method: "DELETE" }),
  },

  galaxy: {
    get: () => request<Galaxy>("/galaxy"),
    summary: () => request<GalaxySummary>("/galaxy/summary"),
    rebuild: (smoke = false) => request<LabRun>(`/galaxy/rebuild${qs({ smoke: smoke ? "true" : undefined })}`, { method: "POST" }),
  },

  telemetry: () => request<{ metrics: boolean; logs: boolean; mcp: boolean; grafana_url: string | null; queued: number; metrics_sent: number; logs_sent: number; errors: number; last_error: string | null }>("/telemetry"),

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
