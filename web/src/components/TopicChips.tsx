import { useId, useState } from "react";
import type { KeyboardEvent } from "react";
import type { Topic } from "../types";
import { navigate } from "../lib/router";

interface Props {
  value: string[];
  onChange?: (next: string[]) => void;
  suggestions?: Topic[] | null;
  placeholder?: string;
}

export function TopicChips({ value, onChange, suggestions, placeholder = "add topic" }: Props) {
  const [draft, setDraft] = useState("");
  const listId = useId();
  const editable = Boolean(onChange);

  const add = (raw: string) => {
    const v = raw.trim().replace(/,+$/, "").trim();
    if (!v || !onChange) return;
    if (value.includes(v)) {
      setDraft("");
      return;
    }
    onChange([...value, v]);
    setDraft("");
  };
  const remove = (t: string) => onChange?.(value.filter((x) => x !== t));

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      add(draft);
    } else if (e.key === "Backspace" && draft === "" && value.length > 0) {
      remove(value[value.length - 1]);
    } else if (e.key === "Escape") {
      setDraft("");
      (e.target as HTMLInputElement).blur();
    }
  };

  const nameOf = (slug: string) => suggestions?.find((t) => t.slug === slug)?.name ?? slug;

  return (
    <div className="chips" role="list" aria-label="Topics">
      {value.map((t) => (
        <span className="chip" key={t} role="listitem">
          <button
            type="button"
            onClick={() => navigate({ kind: "topic", slug: t })}
            title={`Open topic ${nameOf(t)}`}
            style={{ width: "auto", height: "auto", borderRadius: 0 }}
          >
            {nameOf(t)}
          </button>
          {editable && (
            <button type="button" onClick={() => remove(t)} aria-label={`Remove topic ${t}`} title="Remove">
              ×
            </button>
          )}
        </span>
      ))}
      {editable && (
        <>
          <input
            className="chip-input"
            list={listId}
            value={draft}
            placeholder={placeholder}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
            onBlur={() => draft && add(draft)}
            aria-label="Add topic"
          />
          <datalist id={listId}>
            {suggestions?.filter((t) => !value.includes(t.slug)).map((t) => <option key={t.slug} value={t.slug}>{t.name}</option>)}
          </datalist>
        </>
      )}
      {!editable && value.length === 0 && <span className="faint mono">no topics</span>}
    </div>
  );
}
