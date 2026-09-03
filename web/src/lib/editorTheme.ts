import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

// Classes are styled in styles/editor.css so they follow the app tokens (light + dark).
export const markdownHighlight = HighlightStyle.define([
  { tag: t.heading1, class: "cm-md-heading cm-md-heading1" },
  { tag: t.heading2, class: "cm-md-heading cm-md-heading2" },
  { tag: [t.heading3, t.heading4, t.heading5, t.heading6], class: "cm-md-heading" },
  { tag: t.emphasis, class: "cm-md-emphasis" },
  { tag: t.strong, class: "cm-md-strong" },
  { tag: t.link, class: "cm-md-link" },
  { tag: t.url, class: "cm-md-url" },
  { tag: t.monospace, class: "cm-md-code" },
  { tag: [t.processingInstruction, t.meta, t.labelName, t.contentSeparator, t.comment], class: "cm-md-meta" },
  { tag: t.quote, class: "cm-md-quote" },
  { tag: t.list, class: "cm-md-list" },
  { tag: t.strikethrough, class: "cm-md-strike" },
  { tag: [t.keyword, t.atom], class: "cm-md-link" },
  { tag: [t.string, t.number], class: "cm-md-quote" },
]);

export const markdownHighlighting = syntaxHighlighting(markdownHighlight);
