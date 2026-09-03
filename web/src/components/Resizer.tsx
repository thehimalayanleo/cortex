import { useEffect, useRef } from "react";

/**
 * A drag handle that resizes a pane by writing a CSS custom property (e.g. --rail-w) on <html>.
 * `grows` says which way dragging right changes the width: "right" for a left-anchored pane
 * (sidebar), "left" for a right-anchored pane (chat, metadata strip). Width persists per browser;
 * double-click or Home resets to the default; arrow keys nudge for keyboard users.
 */
export function Resizer({
  cssVar,
  storageKey,
  defaultPx,
  min,
  max,
  grows,
  label,
  className = "",
}: {
  cssVar: string;
  storageKey: string;
  defaultPx: number;
  min: number;
  max: number;
  grows: "left" | "right";
  label: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const clamp = (px: number) => Math.round(Math.min(max, Math.max(min, px)));
  const apply = (px: number) => document.documentElement.style.setProperty(cssVar, `${clamp(px)}px`);
  const current = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue(cssVar)) || defaultPx;
  const persist = (px: number | null) => {
    try {
      if (px == null) localStorage.removeItem(storageKey);
      else localStorage.setItem(storageKey, String(clamp(px)));
    } catch {
      /* storage unavailable: width lives for this page only */
    }
  };

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) apply(Number(saved));
    } catch {
      /* ignore */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const startX = e.clientX;
    const startW = current();
    const el = ref.current;
    el?.setPointerCapture(e.pointerId);
    el?.classList.add("dragging");
    document.body.classList.add("resizing");
    let last = startW;
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      last = clamp(grows === "right" ? startW + dx : startW - dx);
      apply(last);
    };
    const up = () => {
      el?.classList.remove("dragging");
      document.body.classList.remove("resizing");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      persist(last);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    e.preventDefault();
  };

  const reset = () => {
    document.documentElement.style.removeProperty(cssVar);
    persist(null);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 64 : 16;
    let next: number | null = null;
    if (e.key === "ArrowLeft") next = current() + (grows === "right" ? -step : step);
    else if (e.key === "ArrowRight") next = current() + (grows === "right" ? step : -step);
    else if (e.key === "Home") { reset(); e.preventDefault(); return; }
    if (next != null) { apply(next); persist(next); e.preventDefault(); }
  };

  return (
    <div
      ref={ref}
      className={`resizer ${className}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title={`Drag to resize · double-click to reset`}
      onPointerDown={onPointerDown}
      onDoubleClick={reset}
      onKeyDown={onKeyDown}
    />
  );
}
