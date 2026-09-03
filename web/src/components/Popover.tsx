import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

interface Props {
  /** Renders the trigger; call `toggle` on click. */
  render: (open: boolean, toggle: () => void) => ReactNode;
  children: ReactNode | ((close: () => void) => ReactNode);
  align?: "left" | "right";
  up?: boolean;
  className?: string;
  panelClassName?: string;
  onOpen?: () => void;
}

/** Small anchored panel. Closes on outside click, Escape, or when the window loses focus (e.g. a click into the PDF frame). */
export function Popover({ render, children, align = "left", up = false, className = "", panelClassName = "", onOpen }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    onOpen?.();
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onBlur = () => setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    window.addEventListener("blur", onBlur);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("blur", onBlur);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const close = () => setOpen(false);
  return (
    <div className={`pop ${className}`} ref={ref}>
      {render(open, () => setOpen((o) => !o))}
      {open && (
        <div className={`pop-panel ${align}${up ? " up" : ""} ${panelClassName}`} role="dialog">
          {typeof children === "function" ? children(close) : children}
        </div>
      )}
    </div>
  );
}
