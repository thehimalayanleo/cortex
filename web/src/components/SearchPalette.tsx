import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { api, errorMessage } from "../api";
import type { Project, SearchHit, SearchType } from "../types";
import { navigate } from "../lib/router";
import type { Route } from "../lib/router";
import { useDebouncedValue } from "../lib/hooks";
import { PALETTES, applyPalette, previewPalette, endPreview, readPaletteId } from "../lib/palettes";

interface Props {
  open: boolean;
  onClose: () => void;
  onNewNote: () => void;
  onNewSpace: () => void;
  projects: Project[] | null;
  space: string;
}

interface Item {
  key: string;
  title: string;
  snippet?: string;
  group: string;
  hint?: string;
  run: () => void;
}

const GROUP_LABEL: Record<SearchType, string> = { note: "Notes", paper: "Papers", project: "Spaces" };
const GROUP_ORDER: SearchType[] = ["note", "paper", "project"];

function hitRoute(h: SearchHit): Route {
  if (h.type === "paper") return { kind: "paper", id: h.id };
  if (h.type === "project") return { kind: "project", slug: h.id };
  return { kind: "note", slug: h.id };
}

// TODO(spec): the snippet's highlight markup is unspecified. We escape everything and
// re-allow only <mark>/<b> so either plain or FTS-highlighted snippets render safely.
function snippetHtml(s: string): string {
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/&lt;(\/?)(mark|b)&gt;/gi, "<$1mark>");
}

const LAB_STATION_ITEMS = [
  { id: "data", title: "Data", hint: "tokens and windows" },
  { id: "pretrain", title: "Pretrain", hint: "next-token loss, attention map" },
  { id: "midtrain", title: "Mid-train", hint: "mixture and cooldown" },
  { id: "posttrain", title: "Post-train", hint: "SFT, then DPO" },
  { id: "encoder", title: "Encoder", hint: "masked LM to embeddings" },
  { id: "cluster", title: "Cluster", hint: "k-means on embeddings" },
  { id: "paint", title: "Paint with code", hint: "GRPO with a rendered reward" },
  { id: "speculative", title: "Speculative decoding", hint: "draft, then verify" },
  { id: "moe", title: "Mixture of experts", hint: "router and experts" },
  { id: "arch", title: "Architecture", hint: "calculator and model probe" },
];

export function SearchPalette({ open, onClose, onNewNote, onNewSpace, projects, space }: Props) {
  const [q, setQ] = useState("");
  const dq = useDebouncedValue(q.trim(), 150);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setHits([]);
      setError(null);
      setIndex(0);
      window.setTimeout(() => input.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    if (!dq) {
      setHits([]);
      setLoading(false);
      setError(null);
      return;
    }
    let alive = true;
    setLoading(true);
    api
      .search(dq, 30)
      .then((r) => {
        if (!alive) return;
        setHits(Array.isArray(r) ? r : []);
        setError(null);
        setIndex(0);
      })
      .catch((e) => alive && setError(errorMessage(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [dq, open]);

  const items = useMemo<Item[]>(() => {
    const close = onClose;
    const current = readPaletteId();
    const themeItems: Item[] = PALETTES.map((p) => ({
      key: `theme:${p.id}`,
      title: `Theme: ${p.name}`,
      group: "Theme",
      hint: p.id === current ? "current" : p.dark ? "dark" : "light",
      run: () => {
        applyPalette(p.id);
        close();
      },
    }));
    if (!dq) {
      const go = (r: Route) => () => {
        close();
        navigate(r);
      };
      const spaceItems: Item[] = (projects ?? []).map((p) => ({
        key: `space:${p.slug}`,
        title: String(p.frontmatter.title ?? p.slug),
        group: "Spaces",
        hint: p.slug === space ? "active" : String(p.frontmatter.status ?? ""),
        run: go({ kind: "project", slug: p.slug }),
      }));
      return [
        { key: "today", title: "Today", group: "Go to", hint: "daily note", run: go({ kind: "daily" }) },
        {
          key: "new",
          title: "New note",
          group: "Go to",
          hint: "⌘N",
          run: () => {
            close();
            onNewNote();
          },
        },
        { key: "notes", title: "All notes", group: "Go to", run: go({ kind: "notes" }) },
        {
          key: "new-space",
          title: "New space",
          group: "Go to",
          run: () => {
            close();
            onNewSpace();
          },
        },
        { key: "topics", title: "Topics", group: "Go to", run: go({ kind: "topics" }) },
        { key: "lab", title: "Training Lab", group: "Go to", hint: "train in the browser or on a GPU", run: go({ kind: "lab" }) },
        { key: "lab-runs", title: "Lab: GPU runs", group: "Go to", hint: "5090 over SSH, Modal, or local", run: go({ kind: "lab", run: "" }) },
        { key: "lab-plan", title: "Lab: My plan", group: "Go to", hint: "learning kanban", run: go({ kind: "lab", plan: true }) },
        { key: "lab-terminal", title: "Lab: Terminal", group: "Go to", hint: "commands on the 5090", run: go({ kind: "lab", terminal: true }) },
        ...LAB_STATION_ITEMS.map((st) => ({ key: `lab-st:${st.id}`, title: `Lab: ${st.title}`, group: "Lab stations", hint: st.hint, run: go({ kind: "lab", station: st.id }) })),
        ...spaceItems,
        ...themeItems,
      ];
    }
    const out: Item[] = [];
    // Typed queries: theme commands match on "theme" or the palette name (like VS Code's "Preferences: Color Theme").
    const q = dq.toLowerCase();
    for (const st of LAB_STATION_ITEMS) {
      if (`lab ${st.title} ${st.hint}`.toLowerCase().includes(q)) out.push({ key: `lab-st:${st.id}`, title: `Lab: ${st.title}`, group: "Lab stations", hint: st.hint, run: () => { close(); navigate({ kind: "lab", station: st.id }); } });
    }
    for (const t of themeItems) {
      if (q.startsWith("theme") || t.title.toLowerCase().includes(q)) out.push(t);
    }
    for (const t of GROUP_ORDER) {
      for (const h of hits.filter((x) => x.type === t)) {
        out.push({
          key: `${h.type}:${h.id}`,
          title: h.title || h.id,
          snippet: h.snippet,
          group: GROUP_LABEL[t],
          hint: h.id,
          run: () => {
            close();
            navigate(hitRoute(h));
          },
        });
      }
    }
    for (const h of hits.filter((x) => !GROUP_ORDER.includes(x.type))) {
      out.push({ key: `${h.type}:${h.id}`, title: h.title || h.id, snippet: h.snippet, group: String(h.type), run: () => { close(); navigate(hitRoute(h)); } });
    }
    return out;
  }, [dq, hits, onClose, onNewNote, onNewSpace, projects, space]);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    el?.scrollIntoView({ block: "nearest" });
  }, [index]);

  // Live theme preview: arrowing over a "Theme: …" row tries it on; leaving the rows (or closing) puts the saved one back.
  useEffect(() => {
    if (!open) return;
    const it = items[index];
    if (it && it.key.startsWith("theme:")) previewPalette(it.key.slice(6));
    else endPreview();
  }, [open, index, items]);
  useEffect(() => {
    if (!open) endPreview();
  }, [open]);

  if (!open) return null;

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIndex((i) => Math.min(items.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      items[index]?.run();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  let lastGroup = "";
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search">
        <input
          ref={input}
          className="palette-input"
          placeholder="Search papers, notes, spaces…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onKey}
          role="combobox"
          aria-expanded="true"
          aria-controls="palette-list"
          aria-activedescendant={items[index] ? `pi-${items[index].key}` : undefined}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="palette-results" id="palette-list" role="listbox" ref={listRef}>
          {error && (
            <div className="state error compact" style={{ padding: "10px 18px" }}>
              <p className="mono">{error}</p>
            </div>
          )}
          {!error && dq && !loading && items.length === 0 && (
            <div className="state compact" style={{ padding: "12px 18px" }}>
              <p>No results for “{dq}”.</p>
            </div>
          )}
          {items.map((it, i) => {
            const showGroup = it.group !== lastGroup;
            lastGroup = it.group;
            return (
              <div key={it.key}>
                {showGroup && <div className="palette-group">{it.group}</div>}
                <button
                  id={`pi-${it.key}`}
                  className="palette-item"
                  role="option"
                  aria-selected={i === index}
                  onMouseEnter={() => setIndex(i)}
                  onClick={it.run}
                >
                  <span className="t">{it.title}</span>
                  {it.snippet && <span className="s" dangerouslySetInnerHTML={{ __html: snippetHtml(it.snippet) }} />}
                  {it.hint && <span className="k">{it.hint}</span>}
                </button>
              </div>
            );
          })}
        </div>
        <div className="palette-foot">
          <span>
            <kbd>↑</kbd>
            <kbd>↓</kbd> move
          </span>
          <span>
            <kbd>↵</kbd> open
          </span>
          <span>
            <kbd>esc</kbd> close
          </span>
          {loading && <span style={{ marginLeft: "auto" }}>searching…</span>}
        </div>
      </div>
    </div>
  );
}
