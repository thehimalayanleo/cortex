// markdown-it + KaTeX renderer with cortex:// link support.
import MarkdownIt from "markdown-it";
import type StateInline from "markdown-it/lib/rules_inline/state_inline.mjs";
import type StateBlock from "markdown-it/lib/rules_block/state_block.mjs";
import katex from "katex";

const md = new MarkdownIt({
  html: true, // only a small whitelist survives: see sanitizeHtml below

  linkify: true,
  typographer: false,
  breaks: false,
});

// Raw HTML in notes is reduced to a harmless whitelist (collapsible answers in the lab chapters, sub/sup, kbd, mark);
// everything else is shown as escaped text, so no script, style, iframe, or event handler ever reaches the DOM.
const ALLOWED = new Set(["details", "summary", "sub", "sup", "kbd", "mark", "br", "hr", "small", "abbr"]);
function sanitizeHtml(raw: string): string {
  return raw.replace(/<\/?([A-Za-z][A-Za-z0-9-]*)(\s[^<>]*)?>/g, (m, tag: string, attrs: string | undefined) => {
    const t = tag.toLowerCase();
    if (!ALLOWED.has(t)) return m.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const closing = m.startsWith("</");
    const open = t === "details" && !closing && /\bopen\b/i.test(attrs ?? "") ? " open" : "";
    return closing ? `</${t}>` : `<${t}${open}>`;
  });
}
md.renderer.rules.html_block = (tokens, idx) => sanitizeHtml(tokens[idx].content);
md.renderer.rules.html_inline = (tokens, idx) => sanitizeHtml(tokens[idx].content);

// Bare cortex://note/x links in text (e.g. chat replies) become anchors.
md.linkify.add("cortex:", {
  validate: /^\/\/(note|paper|project|topic|daily)(\/[A-Za-z0-9._~\-/%]*)?/,
  normalize(match) {
    match.url = match.raw;
  },
});

// ---------- math: $...$ inline, $$...$$ display ----------

function isValidInlineDelim(state: StateInline, pos: number): { canOpen: boolean; canClose: boolean } {
  const max = state.posMax;
  const prev = pos > 0 ? state.src.charCodeAt(pos - 1) : -1;
  const next = pos + 1 <= max ? state.src.charCodeAt(pos + 1) : -1;
  let canOpen = true;
  let canClose = true;
  if (prev === 0x20 || prev === 0x09 || (next >= 0x30 && next <= 0x39)) canClose = false;
  if (next === 0x20 || next === 0x09) canOpen = false;
  return { canOpen, canClose };
}

function mathInline(state: StateInline, silent: boolean): boolean {
  if (state.src[state.pos] !== "$") return false;
  let res = isValidInlineDelim(state, state.pos);
  if (!res.canOpen) {
    if (!silent) state.pending += "$";
    state.pos += 1;
    return true;
  }
  const start = state.pos + 1;
  let match = start;
  let pos: number;
  while ((match = state.src.indexOf("$", match)) !== -1) {
    pos = match - 1;
    while (state.src[pos] === "\\") pos -= 1;
    if ((match - pos) % 2 === 1) break;
    match += 1;
  }
  if (match === -1) {
    if (!silent) state.pending += "$";
    state.pos = start;
    return true;
  }
  if (match - start === 0) {
    if (!silent) state.pending += "$$";
    state.pos = start + 1;
    return true;
  }
  res = isValidInlineDelim(state, match);
  if (!res.canClose) {
    if (!silent) state.pending += "$";
    state.pos = start;
    return true;
  }
  if (!silent) {
    const token = state.push("math_inline", "math", 0);
    token.markup = "$";
    token.content = state.src.slice(start, match);
  }
  state.pos = match + 1;
  return true;
}

function mathBlock(state: StateBlock, start: number, end: number, silent: boolean): boolean {
  let pos = state.bMarks[start] + state.tShift[start];
  let max = state.eMarks[start];
  if (pos + 2 > max) return false;
  if (state.src.slice(pos, pos + 2) !== "$$") return false;
  pos += 2;
  let firstLine = state.src.slice(pos, max);
  if (silent) return true;
  let found = false;
  let lastLine = "";
  if (firstLine.trim().endsWith("$$")) {
    firstLine = firstLine.trim().slice(0, -2);
    found = true;
  }
  let next = start;
  for (; !found; ) {
    next += 1;
    if (next >= end) break;
    pos = state.bMarks[next] + state.tShift[next];
    max = state.eMarks[next];
    if (pos < max && state.tShift[next] < state.blkIndent) break;
    const line = state.src.slice(pos, max);
    if (line.trim().endsWith("$$")) {
      const lastPos = line.trim().lastIndexOf("$$");
      lastLine = line.trim().slice(0, lastPos);
      found = true;
    }
  }
  state.line = next + 1;
  const token = state.push("math_block", "math", 0);
  token.block = true;
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : "") +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : "");
  token.map = [start, state.line];
  token.markup = "$$";
  return true;
}

function renderKatex(tex: string, display: boolean): string {
  try {
    return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
  } catch (e) {
    const msg = (e as Error).message || "KaTeX error";
    return `<span class="math-error" title="${md.utils.escapeHtml(msg)}">${md.utils.escapeHtml(tex)}</span>`;
  }
}

md.inline.ruler.after("escape", "math_inline", mathInline);
md.block.ruler.after("blockquote", "math_block", mathBlock, { alt: ["paragraph", "reference", "blockquote", "list"] });
md.renderer.rules.math_inline = (tokens, idx) => renderKatex(tokens[idx].content, false);
md.renderer.rules.math_block = (tokens, idx) => `<div class="math-block">${renderKatex(tokens[idx].content, true)}</div>\n`;

// ---------- links: cortex:// stays in-app, everything else opens a new tab ----------

const defaultLinkOpen =
  md.renderer.rules.link_open ||
  ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  const href = token.attrGet("href") || "";
  if (/^cortex:\/\//i.test(href)) {
    token.attrSet("data-cortex", href);
    token.attrJoin("class", "cortex-link");
  } else if (/^[a-z]+:/i.test(href)) {
    token.attrSet("target", "_blank");
    token.attrSet("rel", "noopener noreferrer");
  }
  return defaultLinkOpen(tokens, idx, options, env, self);
};

// Task-list checkboxes (rendered read-only).
md.core.ruler.push("task_lists", (state) => {
  const tokens = state.tokens;
  for (let i = 2; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type !== "inline" || tokens[i - 1].type !== "paragraph_open" || tokens[i - 2].type !== "list_item_open") continue;
    const m = /^\[( |x|X)\]\s+/.exec(t.content);
    if (!m) continue;
    const checked = m[1] !== " ";
    t.content = t.content.slice(m[0].length);
    if (t.children && t.children.length > 0 && t.children[0].type === "text") {
      t.children[0].content = t.children[0].content.slice(m[0].length);
    }
    const box = new state.Token("html_inline", "", 0);
    box.content = `<input type="checkbox" class="task-box" disabled${checked ? " checked" : ""}> `;
    t.children = [box, ...(t.children || [])];
    tokens[i - 2].attrJoin("class", "task-item");
  }
});

export function renderMarkdown(src: string): string {
  return md.render(src || "");
}

export function renderInline(src: string): string {
  return md.renderInline(src || "");
}
