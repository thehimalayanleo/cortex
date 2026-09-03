// Defensive frontmatter handling. The API already splits {frontmatter, body}, but a body
// that still carries a leading `---` block (e.g. written by an external agent) is stripped
// here and merged so the editor never shows raw YAML.

export interface Split {
  frontmatter: Record<string, unknown> | null;
  body: string;
}

export function splitFrontmatter(text: string): Split {
  if (!text.startsWith("---")) return { frontmatter: null, body: text };
  const m = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/.exec(text);
  if (!m) return { frontmatter: null, body: text };
  return { frontmatter: parseMiniYaml(m[1]), body: text.slice(m[0].length) };
}

/** Very small YAML subset: `key: scalar`, `key: [a, b]`, and `key:` followed by `- item` lines. */
export function parseMiniYaml(src: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const lines = src.split(/\r?\n/);
  let listKey: string | null = null;
  for (const line of lines) {
    if (!line.trim() || line.trim().startsWith("#")) continue;
    const item = /^\s+-\s*(.*)$/.exec(line);
    if (item && listKey) {
      (out[listKey] as unknown[]).push(scalar(item[1]));
      continue;
    }
    const kv = /^([A-Za-z0-9_.-]+):\s*(.*)$/.exec(line);
    if (!kv) continue;
    const [, key, rawVal] = kv;
    const val = rawVal.trim();
    if (val === "") {
      out[key] = [];
      listKey = key;
      continue;
    }
    listKey = null;
    if (val.startsWith("[") && val.endsWith("]")) {
      const inner = val.slice(1, -1).trim();
      out[key] = inner ? inner.split(",").map((s) => scalar(s.trim())) : [];
    } else {
      out[key] = scalar(val);
    }
  }
  return out;
}

function scalar(v: string): unknown {
  const s = v.trim();
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  if (s === "true") return true;
  if (s === "false") return false;
  if (s === "null" || s === "~") return null;
  if (/^-?\d+(\.\d+)?$/.test(s)) return Number(s);
  return s;
}

export function asStringArray(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((x) => String(x)).filter(Boolean);
  if (typeof v === "string" && v.trim()) return v.split(",").map((s) => s.trim()).filter(Boolean);
  return [];
}
