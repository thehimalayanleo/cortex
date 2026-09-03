import type { ReactNode } from "react";

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading-line" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({
  title = "Could not load",
  error,
  onRetry,
  compact,
}: {
  title?: string;
  error: string;
  onRetry?: () => void;
  compact?: boolean;
}) {
  return (
    <div className={`state error ${compact ? "compact" : ""}`} role="alert">
      <h2>{title}</h2>
      <p className="mono">{error}</p>
      {onRetry && (
        <div className="actions">
          <button className="btn" onClick={onRetry}>
            Retry
          </button>
        </div>
      )}
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  children,
  compact,
  center,
}: {
  title: string;
  hint?: string;
  children?: ReactNode;
  compact?: boolean;
  center?: boolean;
}) {
  return (
    <div className={`state ${compact ? "compact" : ""} ${center ? "center" : ""}`}>
      <h2>{title}</h2>
      {hint && <p>{hint}</p>}
      {children && <div className="actions">{children}</div>}
    </div>
  );
}

export function SaveStatus({
  status,
  savedAt,
  error,
}: {
  status: "idle" | "dirty" | "saving" | "saved" | "error";
  savedAt: Date | null;
  error: string | null;
}) {
  let label = "";
  switch (status) {
    case "idle":
      label = savedAt ? `saved ${hhmm(savedAt)}` : "";
      break;
    case "dirty":
      label = "unsaved";
      break;
    case "saving":
      label = "saving";
      break;
    case "saved":
      label = savedAt ? `saved ${hhmm(savedAt)}` : "saved";
      break;
    case "error":
      label = `save failed: ${error ?? "unknown"}`;
      break;
  }
  return (
    <span className="save-status" data-status={status} title={status === "error" ? error ?? "" : "Cmd+S saves now"}>
      <span className="dot" aria-hidden="true" />
      {label}
    </span>
  );
}

function hhmm(d: Date) {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
