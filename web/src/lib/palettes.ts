/**
 * Color palettes, VS Code style. Each one overrides the design tokens on <html> and pins the
 * light/dark scheme it belongs to. Picked from the Cmd+K palette ("Theme: …"); remembered per browser.
 * "Verdigris" is the built-in default (no overrides).
 */
export interface Palette {
  id: string;
  name: string;
  dark: boolean;
  bg: string;
  surface: string;
  surface2: string;
  surface3: string;
  border: string;
  text: string;
  text2: string;
  text3: string;
  accent: string;
}

export const PALETTES: Palette[] = [
  { id: "verdigris", name: "Verdigris (default)", dark: true, bg: "", surface: "", surface2: "", surface3: "", border: "", text: "", text2: "", text3: "", accent: "" },
  { id: "one-dark", name: "One Dark Pro", dark: true, bg: "#282c34", surface: "#21252b", surface2: "#2c313a", surface3: "#333842", border: "#3a3f4b", text: "#abb2bf", text2: "#8b93a1", text3: "#6b7280", accent: "#61afef" },
  { id: "dracula", name: "Dracula", dark: true, bg: "#282a36", surface: "#21222c", surface2: "#343746", surface3: "#44475a", border: "#44475a", text: "#f8f8f2", text2: "#bfc2d0", text3: "#6272a4", accent: "#bd93f9" },
  { id: "nord", name: "Nord", dark: true, bg: "#2e3440", surface: "#3b4252", surface2: "#434c5e", surface3: "#4c566a", border: "#4c566a", text: "#eceff4", text2: "#d8dee9", text3: "#9aa5b8", accent: "#88c0d0" },
  { id: "tokyo-night", name: "Tokyo Night", dark: true, bg: "#1a1b26", surface: "#16161e", surface2: "#1f2335", surface3: "#292e42", border: "#3b4261", text: "#c0caf5", text2: "#a9b1d6", text3: "#565f89", accent: "#7aa2f7" },
  { id: "catppuccin-mocha", name: "Catppuccin Mocha", dark: true, bg: "#1e1e2e", surface: "#181825", surface2: "#313244", surface3: "#45475a", border: "#45475a", text: "#cdd6f4", text2: "#bac2de", text3: "#7f849c", accent: "#89b4fa" },
  { id: "gruvbox-dark", name: "Gruvbox Dark", dark: true, bg: "#282828", surface: "#1d2021", surface2: "#32302f", surface3: "#3c3836", border: "#504945", text: "#ebdbb2", text2: "#bdae93", text3: "#928374", accent: "#8ec07c" },
  { id: "monokai", name: "Monokai", dark: true, bg: "#272822", surface: "#1e1f1c", surface2: "#2d2e27", surface3: "#3e3d32", border: "#49483e", text: "#f8f8f2", text2: "#c5c5b9", text3: "#75715e", accent: "#a6e22e" },
  { id: "rose-pine", name: "Rosé Pine", dark: true, bg: "#191724", surface: "#1f1d2e", surface2: "#26233a", surface3: "#2a273f", border: "#403d52", text: "#e0def4", text2: "#908caa", text3: "#6e6a86", accent: "#ebbcba" },
  { id: "solarized-dark", name: "Solarized Dark", dark: true, bg: "#002b36", surface: "#073642", surface2: "#0a3f4d", surface3: "#10505f", border: "#164e5c", text: "#eee8d5", text2: "#93a1a1", text3: "#657b83", accent: "#268bd2" },
  { id: "github-dark", name: "GitHub Dark", dark: true, bg: "#0d1117", surface: "#161b22", surface2: "#21262d", surface3: "#30363d", border: "#30363d", text: "#e6edf3", text2: "#9198a1", text3: "#6e7681", accent: "#2f81f7" },
  { id: "solarized-light", name: "Solarized Light", dark: false, bg: "#fdf6e3", surface: "#fffbf0", surface2: "#eee8d5", surface3: "#e4dcc4", border: "#d9d2bc", text: "#073642", text2: "#586e75", text3: "#93a1a1", accent: "#268bd2" },
  { id: "github-light", name: "GitHub Light", dark: false, bg: "#f6f8fa", surface: "#ffffff", surface2: "#eaeef2", surface3: "#d0d7de", border: "#d0d7de", text: "#1f2328", text2: "#59636e", text3: "#8c959f", accent: "#0969da" },
  { id: "ayu-light", name: "Ayu Light", dark: false, bg: "#fafafa", surface: "#ffffff", surface2: "#f0f0f0", surface3: "#e6e6e6", border: "#d9d9d9", text: "#5c6166", text2: "#787b80", text3: "#a0a6ac", accent: "#ff9940" },
  // House palettes
  { id: "matcha", name: "Matcha", dark: false, bg: "#f3f1e6", surface: "#faf8ef", surface2: "#e9e6d6", surface3: "#dcd8c3", border: "#d3cfb8", text: "#2b3325", text2: "#5a6650", text3: "#8d977f", accent: "#5f8f3e" },
  { id: "y2k-lightning", name: "Y2K Lightning", dark: true, bg: "#0b0c1a", surface: "#11132a", surface2: "#1a1d3a", surface3: "#252a4d", border: "#2e3460", text: "#e8ecff", text2: "#aab3e6", text3: "#6c74a8", accent: "#7df9ff" },
  { id: "ultraviolet", name: "Ultraviolet", dark: true, bg: "#120a1f", surface: "#1a1030", surface2: "#241742", surface3: "#2f1f55", border: "#3a2a66", text: "#efe6ff", text2: "#c3b3e6", text3: "#8574ad", accent: "#c77dff" },
  { id: "tangerine", name: "Tangerine Dream", dark: true, bg: "#1c1410", surface: "#241a14", surface2: "#31241c", surface3: "#3f2f24", border: "#4a382b", text: "#fbeee2", text2: "#d7bfa9", text3: "#9c8470", accent: "#ff8c42" },
  { id: "sakura", name: "Sakura", dark: false, bg: "#fbf3f5", surface: "#fffafb", surface2: "#f4e6ea", surface3: "#ead6dc", border: "#e3ccd3", text: "#3a2830", text2: "#6f5560", text3: "#a58d97", accent: "#d64f7a" },
  { id: "chrome", name: "Chrome", dark: true, bg: "#1b1d20", surface: "#212428", surface2: "#2b2f34", surface3: "#363b41", border: "#42484f", text: "#eef1f4", text2: "#b8c0c8", text3: "#7f8891", accent: "#c8d3dc" },
  { id: "paper", name: "Paper", dark: false, bg: "#f7f4ee", surface: "#fffdf8", surface2: "#efeae0", surface3: "#e3ddcf", border: "#d9d2c2", text: "#2a2620", text2: "#5f584d", text3: "#948b7c", accent: "#b4552d" },
];

const KEY = "cortex.palette";
const VARS = ["--bg", "--surface", "--surface-2", "--surface-3", "--border", "--border-strong", "--text", "--text-2", "--text-3", "--accent", "--accent-text", "--accent-soft", "--accent-soft-2", "--on-accent", "--selection", "--code-bg", "--ok"];

function hex(c: string): [number, number, number] {
  const n = parseInt(c.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
/** Mix `a` toward `b` by t in [0,1]; returns hex. */
function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hex(a), [r2, g2, b2] = hex(b);
  const ch = (x: number, y: number) => Math.round(x + (y - x) * t).toString(16).padStart(2, "0");
  return `#${ch(r1, r2)}${ch(g1, g2)}${ch(b1, b2)}`;
}
function luminance(c: string): number {
  const [r, g, b] = hex(c).map((v) => { const s = v / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function readPaletteId(): string {
  try { return localStorage.getItem(KEY) || "verdigris"; } catch { return "verdigris"; }
}

export function clearPalette(persist = true) {
  const root = document.documentElement;
  for (const v of VARS) root.style.removeProperty(v);
  delete root.dataset.palette;
  if (persist) { try { localStorage.removeItem(KEY); } catch { /* ignore */ } }
}

/** Apply a palette by id. Returns the palette (or the default when unknown). */
export function applyPalette(id: string, persist = true): Palette {
  const p = PALETTES.find((x) => x.id === id) ?? PALETTES[0];
  const root = document.documentElement;
  if (p.id === "verdigris") { clearPalette(persist); return p; }
  const s = root.style;
  s.setProperty("--bg", p.bg); s.setProperty("--surface", p.surface); s.setProperty("--surface-2", p.surface2); s.setProperty("--surface-3", p.surface3);
  s.setProperty("--border", p.border); s.setProperty("--border-strong", mix(p.border, p.text, 0.25));
  s.setProperty("--text", p.text); s.setProperty("--text-2", p.text2); s.setProperty("--text-3", p.text3);
  s.setProperty("--accent", p.accent); s.setProperty("--ok", p.accent);
  s.setProperty("--accent-text", p.dark ? mix(p.accent, "#ffffff", 0.15) : mix(p.accent, "#000000", 0.2));
  s.setProperty("--accent-soft", mix(p.bg, p.accent, p.dark ? 0.22 : 0.16));
  s.setProperty("--accent-soft-2", mix(p.bg, p.accent, p.dark ? 0.36 : 0.3));
  s.setProperty("--on-accent", luminance(p.accent) > 0.45 ? "#111111" : "#ffffff");
  const [r, g, b] = hex(p.accent);
  s.setProperty("--selection", `rgba(${r}, ${g}, ${b}, 0.28)`);
  s.setProperty("--code-bg", p.surface2);
  root.dataset.palette = p.id;
  root.dataset.theme = p.dark ? "dark" : "light"; // pin the scheme the palette was designed for
  if (persist) { try { localStorage.setItem(KEY, p.id); localStorage.setItem("cortex.theme", p.dark ? "dark" : "light"); } catch { /* ignore */ } }
  return p;
}

/** Try a palette on without saving it (the Cmd+K list previews as you arrow through). */
export function previewPalette(id: string) { applyPalette(id, false); }
/** Put the saved palette back after a preview. */
export function endPreview() {
  const id = readPaletteId();
  if (id === "verdigris") clearPalette(false); else applyPalette(id, false);
  try { const t = localStorage.getItem("cortex.theme"); if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; } catch { /* ignore */ }
}

/** Re-apply the saved palette at boot (call before first paint). */
export function restorePalette() {
  const id = readPaletteId();
  if (id !== "verdigris") applyPalette(id);
}
