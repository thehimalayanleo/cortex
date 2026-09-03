import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { api, errorMessage } from "../api";
import type { SearchHit, SearchType } from "../types";
import { navigate } from "../lib/router";
import type { Route } from "../lib/router";
import { useDebouncedValue } from "../lib/hooks";

interface Props {
  open: boolean;
  onClose: () => void;
  onNewNote: () => void;
}

interface Item {
  key: string;
  title: string;
  snippet?: string;
  group: string;
  hint?: string;
  run: () => void;
}

const GROUP_LABEL: Record<SearchType, string> = { note: "Notes", paper: "Library", project: "Projects" };
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

export function SearchPalette({ open, onClose, onNewNote }: Props) {
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
    if (!dq) {
      const go = (r: Route) => () => {
        close();
        navigate(r);
      };
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
        { key: "library", title: "Library", group: "Go to", run: go({ kind: "library" }) },
        { key: "projects", title: "Projects", group: "Go to", run: go({ kind: "projects" }) },
        { key: "topics", title: "Topics", group: "Go to", run: go({ kind: "topics" }) },
      ];
    }
    const out: Item[] = [];
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
  }, [dq, hits, onClose, onNewNote]);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    el?.scrollIntoView({ block: "nearest" });
  }, [index]);

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
          placeholder="Search notes, papers, projects…"
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
