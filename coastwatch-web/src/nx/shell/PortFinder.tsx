"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { savePort, usePorts, type ShellPort } from "./port";

const OPEN = "nx-open-port-finder";
/** Opens the port finder from anywhere (the home page's "Find your port", the bar's port control). */
export const openPortFinder = () => window.dispatchEvent(new Event(OPEN));

const DESTINATIONS = new Set(["/today", "/explore", "/coast", "/research"]);

/** Where choosing a port takes the reader: the same page at the new port; from `/`, Today. */
export function hrefForPort(pathname: string, key: string): string {
  const path = DESTINATIONS.has(pathname) ? pathname : "/today";
  return `${path}?place=${encodeURIComponent(key)}`;
}

/**
 * The minimal port finder (ds PortFinder, cmdk, replaces it at integration): type to filter, or pick from
 * the list, pilot ports first, then north to south. Choosing saves the port on this device and opens it.
 */
export function PortFinder() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const ports = usePorts();
  const router = useRouter();
  const pathname = usePathname();
  const input = useRef<HTMLInputElement>(null);
  const opener = useRef<Element | null>(null);
  const id = useId();
  useEffect(() => {
    const on = () => {
      opener.current = document.activeElement;
      setQ("");
      setOpen(true);
    };
    window.addEventListener(OPEN, on);
    return () => window.removeEventListener(OPEN, on);
  }, []);
  useEffect(() => {
    if (!open) return;
    input.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const close = () => {
    setOpen(false);
    (opener.current as HTMLElement | null)?.focus?.();
  };
  const list = useMemo(() => {
    const s = q.trim().toLowerCase();
    return s ? ports.filter((p) => p.name.toLowerCase().includes(s) || (p.county ?? "").toLowerCase().includes(s)) : ports;
  }, [q, ports]);
  const choose = (p: ShellPort) => {
    savePort(p.key);
    setOpen(false);
    router.push(hrefForPort(pathname, p.key));
  };
  if (!open) return null;
  return (
    <div className="n12-finder-scrim" onPointerDown={(e) => e.target === e.currentTarget && close()}>
      <div className="n12-finder" role="dialog" aria-modal="true" aria-labelledby={`${id}-t`} data-testid="nx-port-finder">
        <div className="n12-finder-head">
          <h2 id={`${id}-t`}>Find your port</h2>
          <button type="button" className="n12-text-button" onClick={close}>
            Close
          </button>
        </div>
        <label className="n12-finder-field">
          <span>Port or county</span>
          <input
            ref={input}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && list[0]) choose(list[0]);
            }}
            autoComplete="off"
            spellCheck={false}
            aria-controls={`${id}-l`}
          />
        </label>
        <ul id={`${id}-l`} className="n12-finder-list">
          {list.map((p) => (
            <li key={p.key}>
              <button type="button" onClick={() => choose(p)} data-port={p.key}>
                <span>{p.name}</span>
                {p.county ? <small>{p.county}</small> : null}
              </button>
            </li>
          ))}
          {list.length === 0 ? <li className="n12-finder-empty">{ports.length ? "No port by that name. Try the county." : "The port list could not be read just now."}</li> : null}
        </ul>
      </div>
    </div>
  );
}
