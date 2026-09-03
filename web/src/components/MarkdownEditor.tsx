import { useEffect, useRef } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, placeholder as cmPlaceholder, drawSelection, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching } from "@codemirror/language";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { languages } from "@codemirror/language-data";
import { markdownHighlighting } from "../lib/editorTheme";

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
  /** Changes to this key recreate the editor (e.g. switching document). */
  docKey?: string;
  /** Pasted or dropped files. Return one markdown snippet per file; they are inserted at the cursor. */
  onFiles?: (files: File[]) => Promise<string[]>;
}

/** CodeMirror 6 markdown editor. Controlled: external `value` changes are applied without losing history for local edits. */
export function MarkdownEditor({ value, onChange, placeholder, autoFocus, docKey, onFiles }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onFilesRef = useRef(onFiles);
  onFilesRef.current = onFiles;
  const valueRef = useRef(value);
  valueRef.current = value;

  useEffect(() => {
    if (!host.current) return;
    const state = EditorState.create({
      doc: valueRef.current,
      extensions: [
        history(),
        drawSelection(),
        highlightActiveLine(),
        bracketMatching(),
        markdown({ base: markdownLanguage, codeLanguages: languages, addKeymap: true }),
        markdownHighlighting,
        EditorView.lineWrapping,
        cmPlaceholder(placeholder ?? "Write in markdown. $inline$ and $$display$$ math render in the preview."),
        keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) onChangeRef.current(u.state.doc.toString());
        }),
        // Paste or drop files/images: hand them to the owner, insert the markdown it returns at the cursor.
        EditorView.domEventHandlers({
          paste: (ev, view) => handleFiles(ev.clipboardData?.files, view, ev),
          drop: (ev, view) => handleFiles(ev.dataTransfer?.files, view, ev, ev.clientX, ev.clientY),
        }),
      ],
    });
    function handleFiles(list: FileList | undefined | null, view: EditorView, ev: Event, x?: number, y?: number): boolean {
      const files = list ? Array.from(list) : [];
      if (!files.length || !onFilesRef.current) return false;
      ev.preventDefault();
      const pos = x != null && y != null ? view.posAtCoords({ x, y }) ?? view.state.selection.main.head : view.state.selection.main.head;
      const marker = `[uploading ${files.length === 1 ? files[0].name || "image" : files.length + " files"}…]`;
      view.dispatch({ changes: { from: pos, insert: marker } });
      void onFilesRef.current(files).then(
        (mds) => {
          const doc = view.state.doc.toString();
          const at = doc.indexOf(marker);
          const text = mds.join("\n");
          if (at >= 0) view.dispatch({ changes: { from: at, to: at + marker.length, insert: text } });
          else view.dispatch({ changes: { from: view.state.selection.main.head, insert: text } });
        },
        () => {
          const doc = view.state.doc.toString();
          const at = doc.indexOf(marker);
          if (at >= 0) view.dispatch({ changes: { from: at, to: at + marker.length, insert: "" } });
        },
      );
      return true;
    }
    const view = new EditorView({ state, parent: host.current });
    viewRef.current = view;
    if (autoFocus) view.focus();
    return () => {
      view.destroy();
      viewRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docKey]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== value) {
      view.dispatch({ changes: { from: 0, to: current.length, insert: value } });
    }
  }, [value]);

  return <div className="cm-host" ref={host} />;
}
