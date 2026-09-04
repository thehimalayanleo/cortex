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
import { labMessage } from "../views/LabView";
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
  interface Document {
    modelContext?: ModelContext;
  }
}

/** The spec moved the entry point from navigator to document; ChatGPT's browser and Chrome differ by build. Accept either. */
function modelContext(): ModelContext | undefined {
  if (typeof document !== "undefined" && document.modelContext) return document.modelContext;
  if (typeof navigator !== "undefined" && navigator.modelContext) return navigator.modelContext;
  const w = window as unknown as { modelContext?: ModelContext };
  return w.modelContext;
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
    name: "key_passages",
    description: "The most important passages of a paper, quoted verbatim with page numbers: theorems, main results, the central claim, the method, the stated limitation. Cached per paper; refresh: true re-extracts. pages: an optional range such as '1-5' to read only part of a long paper or book (omit for the whole paper; very long papers automatically use the first 12 and last 3 pages).",
    inputSchema: { type: "object", properties: { id: { type: "string" }, refresh: { type: "boolean" }, pages: { type: "string" } }, required: ["id"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const pid = s(i.id, 200);
      const pages = i.pages ? s(i.pages, 20) : undefined;
      let h = i.refresh || pages ? null : await api.library.highlights(pid);
      if (!h || !h.items) h = await api.library.makeHighlights(pid, !!i.refresh, pages);
      record("key_passages", i, true, `${pid} · ${h.items?.length ?? 0} passages`);
      return text(h.items ?? []);
    },
  },
  {
    name: "galaxy",
    description: "The library as a map: universes (broad areas) containing solar systems (tight clusters of papers), labelled and sized; opens the Galaxy view. Give system to list that solar system's papers.",
    inputSchema: { type: "object", properties: { system: { type: "integer" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      navigate({ kind: "galaxy" });
      const g = await api.galaxy.get();
      if (typeof i.system === "number") {
        const rows = g.papers.filter((p) => p.cluster === i.system).map((p) => ({ id: p.id, title: p.title, year: p.year, status: p.status }));
        record("galaxy", i, true, `system ${i.system} · ${rows.length}`);
        return text(rows);
      }
      record("galaxy", i, true, `${g.clusters.length} systems`);
      return text({ n: g.n, model: g.model, universes: g.universes.map((u) => ({ id: u.id, label: u.label, size: u.size })), solar_systems: g.clusters.filter((c) => c.id >= 0).map((c) => ({ id: c.id, label: c.label, size: c.size, universe: c.universe })) });
    },
  },
  {
    name: "search_arxiv",
    description: "Search arXiv on the web by topic when the brain has nothing: returns up to n results with arxiv id, title, authors, year, summary, and whether each is already in the library. Follow with file_paper to import one.",
    inputSchema: { type: "object", properties: { query: { type: "string" }, n: { type: "integer" } }, required: ["query"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const rows = await api.arxiv(s(i.query, 200), n(i.n ?? 5, 1, 15));
      record("search_arxiv", i, true, `${s(i.query, 40)} · ${rows.length}`);
      return text(rows);
    },
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
  // ---- the Training Lab: the agent gets the same buttons the person has (train in the browser, run on a GPU)
  {
    name: "open_lab",
    description: "Open the Training Lab and show a station: overview|data|pretrain|midtrain|posttrain|encoder|cluster|paint|speculative|moe. The lab trains a tiny transformer in the browser so the user can watch each stage of an LLM pipeline.",
    inputSchema: { type: "object", properties: { station: { type: "string" } } },
    execute: async (i) => {
      const st = s(i.station || "overview", 20);
      navigate({ kind: "lab", station: st });
      record("open_lab", i, true, st);
      return text({ ok: true, station: st });
    },
  },
  {
    name: "lab_train",
    description: "Train in the user's browser at a lab station and watch it live: station pretrain|midtrain|sft|dpo|encoder|contrastive|cluster|paint|speculative|moe (speculative and moe need pretrain first), steps (default the station's setting). Returns immediately; call lab_status to read the loss, samples, and metrics as it runs.",
    inputSchema: { type: "object", properties: { station: { type: "string" }, steps: { type: "integer" } }, required: ["station"] },
    execute: async (i) => {
      const st = s(i.station, 20);
      const target = ({ sft: "posttrain", dpo: "posttrain", contrastive: "encoder" } as Record<string, string>)[st] ?? st;
      navigate({ kind: "lab", station: target });
      await new Promise((r) => setTimeout(r, 700)); // let the frame mount
      const r = await labMessage({ type: "lab:run", station: st, steps: i.steps ? n(i.steps, 1, 5000) : undefined }, 6000);
      record("lab_train", i, r.ok !== false, `${st} · ${String(r.steps ?? "")} steps`);
      return text(r);
    },
  },
  {
    name: "lab_status",
    description: "Read the live state of the in-browser lab: backend, current station, and per-station step, loss, samples, purity.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const r = await labMessage({ type: "lab:status" });
      record("lab_status", i, true, "ok");
      return text(r.status);
    },
  },
  {
    name: "list_lab_chapters",
    description: "List the lab's teaching chapters (15: data, pretraining, mid-training, SFT, RL, tool use, embeddings, clustering, evals, red-teaming, architecture, optimizers, GPU and KV cache, Lean, paint with code). Open one with open_item kind note and its slug.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const ch = await api.lab.chapters();
      record("list_lab_chapters", i, true, `${ch.length}`);
      return text(ch);
    },
  },
  {
    name: "gpu_status",
    description: "Check the user's GPU box (the home RTX 5090 over Tailscale): reachable, Tailscale IP and whether the link is direct, SSH round trip, GPU name and memory, whether PyTorch is ready, whether a run is in progress. Call before start_run with executor ssh.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      navigate({ kind: "lab", run: "" });
      const g = await api.lab.gpu();
      record("gpu_status", i, g.reachable, g.message ?? "");
      return text(g);
    },
  },
  {
    name: "gpu_benchmark",
    description: "Prove the GPU link end to end: run the kernel_bench recipe on the user's RTX 5090 over Tailscale SSH (matmul and attention throughput, KV-cache bytes) and open the run so the person watches the numbers stream in.",
    inputSchema: { type: "object", properties: {} },
    execute: async (i) => {
      const r = await api.lab.start({ recipe: "kernel_bench", args: "", executor: "ssh", origin: "webmcp:gpu_benchmark" });
      changed(); navigate({ kind: "lab", run: r.id }); record("gpu_benchmark", i, true, `kernel_bench · ${r.id}`);
      return text(r);
    },
  },
  {
    name: "gpu_setup",
    description: "Prepare the user's GPU box for runs: installs uv, a Python 3.11 venv, CUDA 12.8 PyTorch and the training libraries over SSH. Idempotent; minutes the first time. The person sees the log stream in the Lab.",
    inputSchema: { type: "object", properties: {} },
    execute: async (i) => {
      navigate({ kind: "lab", run: "" });
      const lines: string[] = [];
      let final: unknown = null;
      await api.lab.gpuSetup((ev) => {
        if (ev.type === "log") lines.push(...ev.lines);
        else final = ev;
      });
      record("gpu_setup", i, true, "done");
      return text({ log: lines.slice(-30), final });
    },
  },
  {
    name: "lab_plan",
    description: "The user's learning plan: a kanban of cards (read a chapter, train the station, run the snippet, run the recipe on the GPU, pass the self-test) in columns todo|doing|done. Opens the board. Use it to suggest what to do next.",
    inputSchema: { type: "object", properties: { col: { type: "string" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      navigate({ kind: "lab", plan: true });
      const p = await api.lab.plan();
      const col = s(i.col || "all", 8);
      record("lab_plan", i, true, `${p.done}/${p.total} done`);
      return text({ done: p.done, total: p.total, cards: p.cards.filter((c) => col === "all" || c.col === col) });
    },
  },
  {
    name: "lab_plan_add",
    description: "Add a learning card to the user's plan on the fly (a topic, a paper to work through, a snippet to build, a run, a quiz). kind: custom|read|build|recipe|quiz|station; note: a note slug to open. Use when the user says what they want to learn next.",
    inputSchema: { type: "object", properties: { title: { type: "string" }, kind: { type: "string" }, note: { type: "string" }, station: { type: "string" }, recipe: { type: "string" } }, required: ["title"] },
    execute: async (i) => {
      const p = await api.lab.planAdd({ title: s(i.title, 200), kind: s(i.kind || "custom", 10), note: i.note ? s(i.note, 200) : undefined, station: i.station ? s(i.station, 20) : undefined, recipe: i.recipe ? s(i.recipe, 40) : undefined });
      changed(); navigate({ kind: "lab", plan: true }); record("lab_plan_add", i, true, s(i.title, 60));
      return text({ added: p.added, done: p.done, total: p.total });
    },
  },
  {
    name: "lab_plan_remove",
    description: "Delete a custom learning card (id starts with custom-). Built-in chapter cards cannot be deleted; move them to done instead.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async (i) => {
      const p = await api.lab.planRemove(s(i.id, 40));
      changed(); navigate({ kind: "lab", plan: true }); record("lab_plan_remove", i, true, s(i.id, 40));
      return text({ done: p.done, total: p.total });
    },
  },
  {
    name: "lab_plan_move",
    description: "Move a learning card to todo|doing|done with an optional comment (for example after the user passes a quiz you gave them, or a run finishes). The board updates in front of the user.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, col: { type: "string" }, comment: { type: "string" } }, required: ["id", "col"] },
    execute: async (i) => {
      const p = await api.lab.planMove(s(i.id, 40), s(i.col, 8) as "todo" | "doing" | "done", i.comment ? s(i.comment, 500) : undefined);
      changed(); navigate({ kind: "lab", plan: true }); record("lab_plan_move", i, true, `${s(i.id, 40)} → ${s(i.col, 8)}`);
      return text({ done: p.done, total: p.total });
    },
  },
  {
    name: "list_runs",
    description: "List GPU training runs launched from the lab (recipe, executor, status, last metric).",
    inputSchema: { type: "object", properties: { limit: { type: "integer" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const rs = await api.lab.runs(n(i.limit ?? 20, 1, 100));
      record("list_runs", i, true, `${rs.length}`);
      return text(rs.map((r) => ({ id: r.id, recipe: r.recipe, args: r.args, executor: r.executor, status: r.status, last: r.last })));
    },
  },
  {
    name: "start_run",
    description: "Launch a lab recipe on a GPU (executor ssh = the user's box, modal = rented, local = this machine) and open it so the user watches the metrics. recipe: pretrain_nano|midtrain|sft_lora|dpo|grpo_tool|embed_contrastive|eval_suite|redteam_suite|kernel_bench|optim_bench|paint_grpo|lean_eval; args is the script's command line, e.g. '--smoke --steps 200'. Only when the user asks to train, run, or benchmark.",
    inputSchema: { type: "object", properties: { recipe: { type: "string" }, args: { type: "string" }, executor: { type: "string" } }, required: ["recipe"] },
    execute: async (i) => {
      const r = await api.lab.start({ recipe: s(i.recipe, 60), args: s(i.args || "", 500), executor: s(i.executor || "local", 10), origin: "webmcp:start_run" });
      changed(); navigate({ kind: "lab", run: r.id }); record("start_run", i, true, `${r.recipe} on ${r.executor} · ${r.id}`);
      return text(r);
    },
  },
  {
    name: "run_code",
    description: "Run a Python snippet (for example a chapter's 'Build it small' block, edited) as a one-off job on the user's GPU box (executor ssh, default), Modal, or locally, and open the run so the person watches the output. Print METRIC {\"step\":..} lines to get charts. Only when the user asks to run code.",
    inputSchema: { type: "object", properties: { code: { type: "string" }, executor: { type: "string" }, args: { type: "string" } }, required: ["code"] },
    execute: async (i) => {
      const r = await api.lab.start({ recipe: "scratch", code: s(i.code, 60000), executor: s(i.executor || "ssh", 10), args: s(i.args || "", 500), origin: "webmcp:run_code" });
      changed(); navigate({ kind: "lab", run: r.id }); record("run_code", i, true, `code on ${r.executor} · ${r.id}`);
      return text(r);
    },
  },
  {
    name: "shell",
    description: "Run one shell command in the Lab terminal on the user's GPU box (executor ssh, default; cwd ~/cortex-lab) or this machine (local); the person watches the output stream. Never destructive commands unless explicitly asked.",
    inputSchema: { type: "object", properties: { cmd: { type: "string" }, executor: { type: "string" } }, required: ["cmd"] },
    execute: async (i) => {
      const r = await api.lab.start({ recipe: "shell", cmd: s(i.cmd, 4000), executor: s(i.executor || "ssh", 10), origin: "webmcp:shell" });
      changed(); navigate({ kind: "lab", run: r.id }); record("shell", i, true, `$ ${s(i.cmd, 50)} · ${r.executor}`);
      return text(r);
    },
  },
  // ---- the Studio (agentic cinema) and the collector
  {
    name: "studio_board",
    description: "The Studio: logline, shot list with status, each shot's takes with critic scores (identity, flicker) and verdicts, keyframe assets, the character bible, and scenes. Opens the board.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      navigate({ kind: "lab", studio: true });
      const b = await api.studio.board();
      record("studio_board", i, true, `${b.shots.length} shots`);
      return text(b);
    },
  },
  {
    name: "plan_shots",
    description: "Director: turn a logline into n planned shots (title, image-to-video prompt, notes) on the Studio board, in front of the user.",
    inputSchema: { type: "object", properties: { logline: { type: "string" }, n: { type: "integer" } }, required: ["logline"] },
    execute: async (i) => {
      navigate({ kind: "lab", studio: true });
      const r = await api.studio.plan(s(i.logline, 2000), n(i.n ?? 4, 1, 12));
      changed(); record("plan_shots", i, true, `${r.shots.length} shots`);
      return text(r.shots);
    },
  },
  {
    name: "render_shot",
    description: "Render a take of a shot on the user's GPU box (Wan 2.2 image-to-video from its keyframe) and open the run; smoke: true runs the offline test brick. Then refresh_shot for the verdict.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, executor: { type: "string" }, smoke: { type: "boolean" } }, required: ["id"] },
    execute: async (i) => {
      const r = await api.studio.render(s(i.id, 80), { executor: i.executor ? s(i.executor, 10) : undefined, smoke: typeof i.smoke === "boolean" ? i.smoke : undefined });
      changed(); navigate({ kind: "lab", run: r.id }); record("render_shot", i, true, `${s(i.id, 40)} · ${r.id}`);
      return text(r);
    },
  },
  {
    name: "refresh_shot",
    description: "After a take finishes: fetch its clip, contact sheet and scores into the vault and set the shot's status from the critics' verdict.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async (i) => {
      const sh = await api.studio.refresh(s(i.id, 80));
      changed(); navigate({ kind: "lab", studio: true }); record("refresh_shot", i, true, `${sh.id} · ${sh.status}`);
      return text(sh);
    },
  },
  {
    name: "list_characters",
    description: "The Studio's character bible: id, name, description (identity text), hero image, hero set and LoRA folders on the GPU box, status and prototype scores.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const cs = await api.studio.characters();
      record("list_characters", i, true, `${cs.length} characters`);
      return text(cs);
    },
  },
  {
    name: "build_character",
    description: "Build a character on the user's GPU box: stage hero (image from its description), heroset (framings; the identity prototype) or lora (character LoRA). add: true with name and description creates it first. Opens the run; smoke: true runs the offline brick.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, stage: { type: "string", enum: ["hero", "heroset", "lora"] }, add: { type: "boolean" }, name: { type: "string" }, description: { type: "string" }, style: { type: "string" }, executor: { type: "string" }, smoke: { type: "boolean" } } },
    execute: async (i) => {
      let id = s(i.id, 80);
      if (i.add || !id) {
        const c = await api.studio.addCharacter({ name: s(i.name || "Character", 80), description: s(i.description, 1500), style: s(i.style, 300) });
        id = c.id;
      }
      const stage = (["hero", "heroset", "lora"].includes(String(i.stage)) ? String(i.stage) : "heroset") as "hero" | "heroset" | "lora";
      const r = await api.studio.buildCharacter(id, { stage, executor: i.executor ? s(i.executor, 10) : undefined, smoke: typeof i.smoke === "boolean" ? i.smoke : undefined });
      changed(); navigate({ kind: "lab", run: r.id }); record("build_character", i, true, `${stage} of ${id} · ${r.id}`);
      return text(r);
    },
  },
  {
    name: "plan_scene",
    description: "Director: plan a scene from a logline and add its shots to the board. kind filler = 2 to 4 short b-roll shots (17 to 33 frames) for post-production cutaways; full = n shots with dialogue and continuity notes. characters: ids to cast.",
    inputSchema: { type: "object", properties: { logline: { type: "string" }, kind: { type: "string", enum: ["filler", "full"] }, n: { type: "integer" }, characters: { type: "array", items: { type: "string" } }, set_name: { type: "string" } }, required: ["logline"] },
    execute: async (i) => {
      navigate({ kind: "lab", studio: true });
      const kind = i.kind === "full" ? "full" : "filler";
      const sc = await api.studio.planScene({ logline: s(i.logline, 2000), kind, n: i.n != null ? n(i.n, 2, 12) : undefined, characters: Array.isArray(i.characters) ? i.characters.map((x) => s(x, 80)) : undefined, set_name: i.set_name ? s(i.set_name, 120) : undefined });
      changed(); record("plan_scene", i, true, `${sc.kind} scene ${sc.id} · ${sc.shots.length} shots`);
      return text(sc);
    },
  },
  {
    name: "render_scene",
    description: "Render every shot of a scene in order on the GPU box, one take at a time, then mark it rendered; only_missing skips shots already rendered. smoke: true uses the offline brick. Then assemble_scene.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, smoke: { type: "boolean" }, only_missing: { type: "boolean" } }, required: ["id"] },
    execute: async (i) => {
      navigate({ kind: "lab", studio: true });
      const sc = await api.studio.renderScene(s(i.id, 80), { smoke: typeof i.smoke === "boolean" ? i.smoke : undefined, only_missing: !!i.only_missing });
      changed(); record("render_scene", i, true, `rendering ${sc.id} · ${sc.shots.length} shots`);
      return text(sc);
    },
  },
  {
    name: "assemble_scene",
    description: "Concatenate the latest kept take of each shot of a scene into scene.mp4 (ffmpeg, on this machine) with a contact strip; returns the video path and any shots still missing a clip.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    execute: async (i) => {
      navigate({ kind: "lab", studio: true });
      const sc = await api.studio.assembleScene(s(i.id, 80));
      changed(); record("assemble_scene", i, true, `${sc.id} · ${sc.clips} clips${sc.missing.length ? ` · ${sc.missing.length} missing` : ""}`);
      return text(sc);
    },
  },
  {
    name: "collect",
    description: "Save a trace to the user's long-term data collector for future post-training: kind note|preference|pair|rating|decision|taste|feedback|idea|link, content, tags; a preference gives chosen and rejected (and prompt); a pair gives prompt and response.",
    inputSchema: { type: "object", properties: { kind: { type: "string" }, content: { type: "string" }, tags: { type: "array", items: { type: "string" } }, context: { type: "string" }, prompt: { type: "string" }, response: { type: "string" }, chosen: { type: "string" }, rejected: { type: "string" }, rating: { type: "number" } } },
    execute: async (i) => {
      const rec = await api.traces.add({ kind: s(i.kind || "note", 30), content: i.content ? s(i.content, 20000) : undefined, tags: Array.isArray(i.tags) ? (i.tags as unknown[]).map((t) => s(t, 40)) : undefined, context: i.context ? s(i.context, 500) : undefined, prompt: i.prompt ? s(i.prompt, 20000) : undefined, response: i.response ? s(i.response, 20000) : undefined, chosen: i.chosen ? s(i.chosen, 20000) : undefined, rejected: i.rejected ? s(i.rejected, 20000) : undefined, rating: typeof i.rating === "number" ? i.rating : undefined, source: "webmcp" });
      changed(); record("collect", i, true, `${rec.kind} · ${rec.id}`);
      return text({ id: rec.id, kind: rec.kind, when: rec.when });
    },
  },
  {
    name: "list_traces",
    description: "Read the collector: recent traces (by kind or text query) and its stats.",
    inputSchema: { type: "object", properties: { limit: { type: "integer" }, kind: { type: "string" }, q: { type: "string" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const [rows, st] = await Promise.all([api.traces.list({ limit: n(i.limit ?? 20, 1, 200), kind: i.kind ? s(i.kind, 30) : undefined, q: i.q ? s(i.q, 200) : undefined }), api.traces.stats()]);
      navigate({ kind: "lab", traces: true }); record("list_traces", i, true, `${rows.length} · total ${st.total}`);
      return text({ stats: st, traces: rows });
    },
  },
  // ---- the training pie: pipelines of recipe runs
  {
    name: "list_pipelines",
    description: "List training pipelines (data -> pretrain -> midtrain -> sft -> rl -> eval as one DAG of runs) with status, progress and the running stage, plus the available templates (reasoning-nano, embed-mine). Opens the Pipeline tab.",
    inputSchema: { type: "object", properties: { limit: { type: "integer" } } },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      navigate({ kind: "lab", pipeline: true });
      const [rows, ts] = await Promise.all([api.pipelines.list(n(i.limit ?? 20, 1, 100)), api.pipelines.templates()]);
      record("list_pipelines", i, true, `${rows.length}`);
      return text({ pipelines: rows, templates: ts });
    },
  },
  {
    name: "start_pipeline",
    description: "Create and start a training pipeline from a template (reasoning-nano: a small reasoning model end to end from the user's Traces plus a synthetic verifiable reasoning set; embed-mine: embed the vault then contrastive fine-tuning) on executor local|ssh|modal; smoke true runs the CPU-sized version in minutes. Opens it so the person watches the flow. Only when the user asks to train or run a pipeline.",
    inputSchema: { type: "object", properties: { template: { type: "string", enum: ["reasoning-nano", "embed-mine"] }, executor: { type: "string" }, smoke: { type: "boolean" } }, required: ["template"] },
    execute: async (i) => {
      const p = await api.pipelines.create({ template: s(i.template, 40), executor: s(i.executor || "local", 10), smoke: typeof i.smoke === "boolean" ? i.smoke : true, start: true });
      changed(); navigate({ kind: "lab", pipeline: true, pipelineId: p.id }); record("start_pipeline", i, true, `${p.template} on ${p.executor} · ${p.id}`);
      return text({ id: p.id, template: p.template, executor: p.executor, smoke: p.smoke, status: p.status, stages: p.stages.map((st) => ({ name: st.name, recipe: st.recipe, status: st.status })) });
    },
  },
  {
    name: "read_pipeline",
    description: "Read a pipeline: every stage with status, run id, last metric, elapsed time and RESULT; the corpus composition (tokens per source); the final eval report when done. Opens it.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const p = await api.pipelines.get(s(i.id, 80));
      navigate({ kind: "lab", pipeline: true, pipelineId: p.id });
      record("read_pipeline", i, true, `${p.id} · ${p.status} ${p.progress.done}/${p.progress.total}`);
      return text({ id: p.id, template: p.template, status: p.status, error: p.error, progress: p.progress, data: p.data, final: p.final, stages: p.stages.map((st) => ({ name: st.name, recipe: st.recipe, status: st.status, run_id: st.run_id, args: st.args, last: st.last, elapsed_s: st.elapsed_s, error: st.error, result: st.result })) });
    },
  },
  {
    name: "retry_stage",
    description: "Re-queue a failed pipeline stage (and everything downstream) and resume the pipeline in front of the user.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, stage: { type: "string" } }, required: ["id", "stage"] },
    execute: async (i) => {
      const p = await api.pipelines.retry(s(i.id, 80), s(i.stage, 40));
      changed(); navigate({ kind: "lab", pipeline: true, pipelineId: p.id }); record("retry_stage", i, true, `${s(i.stage, 40)} · ${p.id}`);
      return text({ id: p.id, status: p.status, stages: p.stages.map((st) => ({ name: st.name, status: st.status, run_id: st.run_id })) });
    },
  },
  {
    name: "read_run",
    description: "Read a run: status, parsed metrics (thinned), final result, and the last log lines.",
    inputSchema: { type: "object", properties: { id: { type: "string" }, tail: { type: "integer" } }, required: ["id"] },
    annotations: { readOnlyHint: true },
    execute: async (i) => {
      const r = await api.lab.run(s(i.id, 80), n(i.tail ?? 60, 1, 1000));
      const m = r.metrics;
      const thin = m.length <= 60 ? m : [...Array.from({ length: 60 }, (_, k) => m[Math.floor((k * m.length) / 60)]), m[m.length - 1]];
      navigate({ kind: "lab", run: r.id });
      record("read_run", i, true, `${r.id} · ${r.status}`);
      return text({ id: r.id, recipe: r.recipe, args: r.args, executor: r.executor, status: r.status, result: r.result, metrics: thin, log: r.log });
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
  const mc = modelContext();
  if (!mc) return false;
  for (const t of webmcpTools) {
    try { mc.registerTool(guarded(t)); } catch (e) { console.warn("webmcp: could not register", t.name, e); }
  }
  return true;
}

export const webmcpAvailable = () => !!modelContext();
