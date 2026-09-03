import { useEffect, useState } from "react";
import type { KeyboardEvent } from "react";

interface Props {
  value: string;
  placeholder: string;
  label: string;
  onSave: (value: string) => void;
  className?: string;
}

/** One line of text that reads as text until clicked. Enter saves, Escape cancels, blur saves. */
export function InlineEdit({ value, placeholder, label, onSave, className = "" }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  const commit = () => {
    setEditing(false);
    const v = draft.trim();
    if (v !== (value ?? "").trim()) onSave(v);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setDraft(value);
      setEditing(false);
    }
  };

  if (editing) {
    return (
      <input
        className={`inline-edit editing ${className}`}
        value={draft}
        placeholder={placeholder}
        aria-label={label}
        autoFocus
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        onBlur={commit}
      />
    );
  }
  return (
    <button type="button" className={`inline-edit ${value ? "" : "empty"} ${className}`} onClick={() => setEditing(true)} title={`${label}: click to edit`}>
      {value || placeholder}
    </button>
  );
}
