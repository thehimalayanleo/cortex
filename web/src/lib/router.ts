// Tiny hash router. Routes are the things the center pane can show.
import { useCallback, useEffect, useState } from "react";
import type { NoteKind } from "../types";

export type Route =
  | { kind: "home" }
  | { kind: "daily" }
  | { kind: "note"; slug: string }
  | { kind: "paper"; id: string }
  | { kind: "project"; slug: string }
  | { kind: "notes"; noteKind?: NoteKind }
  | { kind: "topics" }
  | { kind: "topic"; slug: string }
  | { kind: "lab"; station?: string; run?: string; plan?: boolean };

export function routeToHash(r: Route): string {
  switch (r.kind) {
    case "home":
      return "#/";
    case "daily":
      return "#/daily";
    case "note":
      return `#/note/${encodeURIComponent(r.slug)}`;
    case "paper":
      return `#/paper/${encodeURIComponent(r.id)}`;
    case "project":
      return `#/project/${encodeURIComponent(r.slug)}`;
    case "notes":
      return r.noteKind ? `#/notes?kind=${encodeURIComponent(r.noteKind)}` : "#/notes";
    case "topics":
      return "#/topics";
    case "topic":
      return `#/topic/${encodeURIComponent(r.slug)}`;
    case "lab":
      return r.plan ? "#/lab/plan" : r.run ? `#/lab/run/${encodeURIComponent(r.run)}` : r.station ? `#/lab/${encodeURIComponent(r.station)}` : "#/lab";
  }
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const [pathPart, query = ""] = raw.split("?");
  const segs = pathPart.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query);
  const head = segs[0] ?? "";
  const rest = segs.slice(1).join("/");
  switch (head) {
    case "daily":
      return { kind: "daily" };
    case "note":
      return rest ? { kind: "note", slug: rest } : { kind: "notes" };
    case "paper":
      return rest ? { kind: "paper", id: rest } : { kind: "home" };
    case "project":
      return rest ? { kind: "project", slug: rest } : { kind: "home" };
    case "notes":
      return { kind: "notes", noteKind: (params.get("kind") as NoteKind) || undefined };
    case "topics":
      return { kind: "topics" };
    case "topic":
      return rest ? { kind: "topic", slug: rest } : { kind: "topics" };
    case "lab":
      if (segs[1] === "run" && segs[2]) return { kind: "lab", run: segs[2] };
      if (segs[1] === "run") return { kind: "lab", run: "" };
      if (segs[1] === "plan") return { kind: "lab", plan: true };
      return { kind: "lab", station: segs[1] || undefined };
    default:
      // Old #/library and #/projects links land on the home view (the rail is the library now).
      return { kind: "home" };
  }
}

/** cortex://note/<slug>, cortex://paper/<id>, cortex://project/<slug> -> Route */
export function parseCortexLink(href: string): Route | null {
  const m = /^cortex:\/\/(note|paper|project|topic|daily|lab)(?:\/(.*))?$/i.exec(href.trim());
  if (!m) return null;
  const type = m[1].toLowerCase();
  const id = m[2] ? decodeURIComponent(m[2].replace(/\/+$/, "")) : "";
  if (type === "daily") return { kind: "daily" };
  if (type === "lab") {
    const mm = /^run\/(.+)$/.exec(id);
    if (id === "plan") return { kind: "lab", plan: true };
    if (id === "runs") return { kind: "lab", run: "" };
    return mm ? { kind: "lab", run: mm[1] } : { kind: "lab", station: id || undefined };
  }
  if (!id) return null;
  if (type === "note") return { kind: "note", slug: id };
  if (type === "paper") return { kind: "paper", id };
  if (type === "project") return { kind: "project", slug: id };
  if (type === "topic") return { kind: "topic", slug: id };
  return null;
}

export function navigate(r: Route) {
  const h = routeToHash(r);
  if (window.location.hash === h) {
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = h;
  }
}

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const go = useCallback((r: Route) => navigate(r), []);
  return [route, go];
}

export function isSameRoute(a: Route, b: Route): boolean {
  return routeToHash(a) === routeToHash(b);
}
