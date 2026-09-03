import { useCallback, useEffect, useState } from "react";
import { PALETTES, clearPalette } from "./palettes";

export type ThemePref = "system" | "light" | "dark";
const KEY = "cortex.theme";

export function readThemePref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore */
  }
  return "system";
}

export function applyTheme(pref: ThemePref) {
  const root = document.documentElement;
  // A color palette pins its own scheme; switching to the other scheme (or system) drops the palette
  // so its tokens never render on the wrong ground.
  const active = root.dataset.palette ? PALETTES.find((p) => p.id === root.dataset.palette) : undefined;
  if (active && (pref === "system" ? resolvedTheme("system") : pref) !== (active.dark ? "dark" : "light")) clearPalette();
  if (pref === "system") delete root.dataset.theme;
  else root.dataset.theme = pref;
  try {
    if (pref === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, pref);
  } catch {
    /* ignore */
  }
}

export function resolvedTheme(pref: ThemePref): "light" | "dark" {
  if (pref !== "system") return pref;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useTheme(): [ThemePref, (p: ThemePref) => void, "light" | "dark"] {
  const [pref, setPrefState] = useState<ThemePref>(readThemePref);
  const [, bump] = useState(0);
  useEffect(() => {
    applyTheme(pref);
  }, [pref]);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const onChange = () => bump((n) => n + 1);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  const setPref = useCallback((p: ThemePref) => setPrefState(p), []);
  return [pref, setPref, resolvedTheme(pref)];
}
