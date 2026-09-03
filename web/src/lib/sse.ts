// Minimal Server-Sent-Events reader over fetch + ReadableStream.
// Parses `data:` lines as JSON; blank lines delimit events; multi-line data is joined with "\n".

export async function readSSE<T>(
  response: Response,
  onEvent: (event: T) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!response.body) throw new Error("Response has no body to stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let dataLines: string[] = [];

  const flush = () => {
    if (dataLines.length === 0) return;
    const raw = dataLines.join("\n");
    dataLines = [];
    if (raw === "" || raw === "[DONE]") return;
    try {
      onEvent(JSON.parse(raw) as T);
    } catch {
      // Non-JSON payloads are surfaced as opaque text events for robustness.
      onEvent({ type: "text", delta: raw } as unknown as T);
    }
  };

  const handleLine = (line: string) => {
    if (line === "") {
      flush();
      return;
    }
    if (line.startsWith(":")) return; // comment / keepalive
    const idx = line.indexOf(":");
    const field = idx === -1 ? line : line.slice(0, idx);
    let value = idx === -1 ? "" : line.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") dataLines.push(value);
    // `event:`, `id:`, `retry:` are ignored: the spec puts the type inside the JSON.
  };

  try {
    for (;;) {
      if (signal?.aborted) break;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        let line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        handleLine(line);
      }
    }
    buffer += decoder.decode();
    if (buffer.length > 0) handleLine(buffer.replace(/\r$/, ""));
    flush();
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}
