import { useCallback, useMemo } from "react";
import type { MouseEvent } from "react";
import { renderMarkdown } from "../lib/markdown";
import { navigate, parseCortexLink } from "../lib/router";
import { useDebouncedValue } from "../lib/hooks";

interface Props {
  source: string;
  className?: string;
  emptyText?: string;
  /** Debounce re-render while typing (ms). 0 = immediate. */
  debounce?: number;
}

/** Intercepts clicks on cortex:// links anywhere inside rendered markdown. */
export function handleCortexClick(e: MouseEvent<HTMLElement>): boolean {
  const target = e.target as HTMLElement | null;
  const a = target?.closest?.("a");
  if (!a) return false;
  const href = a.getAttribute("data-cortex") || a.getAttribute("href") || "";
  const route = parseCortexLink(href);
  if (!route) return false;
  e.preventDefault();
  navigate(route);
  return true;
}

export function MarkdownPreview({ source, className = "", emptyText = "Nothing here yet.", debounce = 120 }: Props) {
  const debounced = useDebouncedValue(source, debounce);
  const html = useMemo(() => renderMarkdown(debounced), [debounced]);
  const onClick = useCallback((e: MouseEvent<HTMLElement>) => {
    handleCortexClick(e);
  }, []);
  if (!debounced || !debounced.trim()) {
    return <div className={`md empty ${className}`}>{emptyText}</div>;
  }
  return <div className={`md ${className}`} onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />;
}
