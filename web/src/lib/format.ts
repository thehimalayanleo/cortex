const DAY = 86_400_000;

export function parseDate(value: unknown): Date | null {
  if (!value) return null;
  if (value instanceof Date) return value;
  const d = new Date(String(value));
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Short relative date for lists: "today 14:02", "yesterday", "3d ago", "Aug 12", "2025-03-01". */
export function relDate(value: unknown, now = new Date()): string {
  const d = parseDate(value);
  if (!d) return "";
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = d.getTime();
  if (t >= startToday) return `today ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (t >= startToday - DAY) return "yesterday";
  const days = Math.floor((startToday - t) / DAY);
  if (days < 14) return `${days}d ago`;
  if (d.getFullYear() === now.getFullYear()) return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return isoDate(d);
}

export function isoDate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function clock(d: Date = new Date()): string {
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function longDate(value: unknown): string {
  const d = parseDate(value);
  if (!d) return "";
  return d.toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

export function pad(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

export function truncate(s: string, n: number): string {
  if (!s) return "";
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > n ? `${t.slice(0, n - 1).trimEnd()}…` : t;
}

export function authorsList(authors: string | string[] | undefined | null): string[] {
  if (!authors) return [];
  if (Array.isArray(authors)) return authors.filter(Boolean);
  return authors.split(/,\s*/).map((a) => a.trim()).filter(Boolean);
}

export function authorsLine(authors: string | string[] | undefined | null, max = 3): string {
  const list = authorsList(authors);
  if (list.length === 0) return "";
  if (list.length <= max) return list.join(", ");
  return `${list.slice(0, max).join(", ")} +${list.length - max}`;
}

export function titleCase(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

export function slugify(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}
