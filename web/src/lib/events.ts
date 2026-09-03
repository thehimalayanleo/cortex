// Tiny command bus for app-wide keyboard commands and vault-change notifications.
export type Command = "save" | "new-note" | "palette" | "toggle-chat" | "vault-changed";

type Handler = () => void;
const handlers = new Map<Command, Set<Handler>>();

export function onCommand(cmd: Command, handler: Handler): () => void {
  let set = handlers.get(cmd);
  if (!set) {
    set = new Set();
    handlers.set(cmd, set);
  }
  set.add(handler);
  return () => {
    set?.delete(handler);
  };
}

export function emitCommand(cmd: Command) {
  const set = handlers.get(cmd);
  if (!set) return;
  for (const h of Array.from(set)) h();
}
