import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

interface PromptProps {
  title: string;
  label: string;
  placeholder?: string;
  submitLabel?: string;
  extra?: ReactNode;
  onSubmit: (value: string) => Promise<void> | void;
  onClose: () => void;
}

/** Small modal prompt (title for a new note / project). Escape closes, Enter submits. */
export function PromptDialog({ title, label, placeholder, submitLabel = "Create", extra, onSubmit, onClose }: PromptProps) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    input.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async () => {
    if (!value.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit(value.trim());
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dlg-title"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <h2 id="dlg-title">{title}</h2>
        <div className="field">
          <label htmlFor="dlg-input">{label}</label>
          <input id="dlg-input" ref={input} className="input" value={value} placeholder={placeholder} onChange={(e) => setValue(e.target.value)} />
        </div>
        {extra}
        {error && (
          <p className="mono" style={{ color: "var(--danger)" }}>
            {error}
          </p>
        )}
        <div className="actions">
          <button type="button" className="btn ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn primary" disabled={!value.trim() || busy}>
            {busy ? "Creating…" : submitLabel}
          </button>
        </div>
      </form>
    </div>
  );
}
