import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
  setData: (updater: T | ((prev: T | null) => T | null)) => void;
}

/** Fetch-on-mount helper with reload; ignores stale responses when deps change. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], soft: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fnRef
      .current()
      .then((d) => {
        if (!alive) return;
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        if (!alive) return;
        setError(errorMessage(e));
        setLoading(false);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  // Soft deps (e.g. a vault-changed tick) refetch in the background: no loading state, errors ignored.
  const softFirst = useRef(true);
  useEffect(() => {
    if (softFirst.current) {
      softFirst.current = false;
      return;
    }
    let alive = true;
    fnRef
      .current()
      .then((d) => alive && setData(d))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, soft);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const set = useCallback((updater: T | ((prev: T | null) => T | null)) => {
    setData((prev) => (typeof updater === "function" ? (updater as (p: T | null) => T | null)(prev) : updater));
  }, []);
  return { data, error, loading, reload, setData: set };
}

export type SaveStatus = "idle" | "dirty" | "saving" | "saved" | "error";

export interface Autosave<D> {
  status: SaveStatus;
  error: string | null;
  savedAt: Date | null;
  schedule: (draft: D) => void;
  flush: () => Promise<void>;
  reset: () => void;
}

/**
 * Debounced autosave. Call `schedule(draft)` on every edit; the latest draft is written
 * `delay` ms after the last edit, or immediately on `flush()` (Cmd+S). Pending drafts are
 * flushed on unmount and before the window unloads.
 */
export function useAutosave<D>(save: (draft: D) => Promise<void>, delay = 800): Autosave<D> {
  const [status, setStatus] = useState<SaveStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const pending = useRef<D | null>(null);
  const timer = useRef<number | null>(null);
  const saveRef = useRef(save);
  saveRef.current = save;
  const inflight = useRef<Promise<void> | null>(null);

  const run = useCallback(async () => {
    if (timer.current) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    if (inflight.current) await inflight.current;
    const draft = pending.current;
    if (draft === null) return;
    pending.current = null;
    setStatus("saving");
    const p = saveRef
      .current(draft)
      .then(() => {
        setError(null);
        setSavedAt(new Date());
        setStatus(pending.current !== null ? "dirty" : "saved");
      })
      .catch((e) => {
        // Keep the draft so the next attempt retries it.
        if (pending.current === null) pending.current = draft;
        setError(errorMessage(e));
        setStatus("error");
      })
      .finally(() => {
        inflight.current = null;
      });
    inflight.current = p;
    await p;
  }, []);

  const schedule = useCallback(
    (draft: D) => {
      pending.current = draft;
      setStatus("dirty");
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => void run(), delay);
    },
    [delay, run],
  );

  const flush = useCallback(async () => {
    await run();
  }, [run]);

  const reset = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = null;
    pending.current = null;
    setStatus("idle");
    setError(null);
  }, []);

  useEffect(() => {
    const onUnload = (e: BeforeUnloadEvent) => {
      if (pending.current !== null) {
        void run();
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", onUnload);
    return () => {
      window.removeEventListener("beforeunload", onUnload);
      if (pending.current !== null) void run();
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [run]);

  return { status, error, savedAt, schedule, flush, reset };
}

export function useLocalStorage<T extends string>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return (localStorage.getItem(key) as T) || initial;
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (v: T) => {
      setValue(v);
      try {
        localStorage.setItem(key, v);
      } catch {
        /* ignore */
      }
    },
    [key],
  );
  return [value, set];
}

export function useDebouncedValue<T>(value: T, delay: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), delay);
    return () => window.clearTimeout(t);
  }, [value, delay]);
  return v;
}
