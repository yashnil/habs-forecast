"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { readPurpose } from "@/lib/purpose";

/**
 * Two presentations of the same facts.
 *
 * - "coastal": for people on or by the water (fishing, harvesting, port operations). Fewer
 *   controls, plain language, the official status and next steps first; scientific detail one
 *   tap away.
 * - "research": exact values, units, thresholds, full charts, provenance and downloads first.
 *
 * The mode changes order, defaults and how much is open, never a value, a notice, a caveat or a
 * verification status. Every page and every notice is reachable in both. It is kept on this
 * device only; a link's `mode=` parameter wins for that view without being saved.
 */
export type Mode = "coastal" | "research";
export const MODE_KEY = "cw-mode-v1";
const EVENT = "cw-mode-change";

export function parseMode(v: string | null | undefined): Mode | null {
  return v === "coastal" || v === "research" ? v : null;
}

/** The stored mode, else the one the welcome's purpose implies (research → research). */
export function readMode(): Mode {
  try {
    const v = parseMode(window.localStorage.getItem(MODE_KEY));
    if (v) return v;
  } catch {
    // storage unavailable
  }
  return readPurpose()?.purpose === "research" ? "research" : "coastal";
}

export function writeMode(m: Mode) {
  try {
    window.localStorage.setItem(MODE_KEY, m);
  } catch {
    // not remembered on this device
  }
  window.dispatchEvent(new Event(EVENT));
}

/** The mode a link asks for (`?mode=research`), if any. */
export function linkMode(): Mode | null {
  try {
    return parseMode(new URLSearchParams(window.location.search).get("mode"));
  } catch {
    return null;
  }
}

/** Mode for this view: a link's `mode=` first, then the device's choice. `ready` is false until read. */
export function useMode(): [Mode, (m: Mode) => void, boolean] {
  const [m, setM] = useState<Mode>("coastal");
  const [ready, setReady] = useState(false);
  const fromLink = useRef<Mode | null>(null);
  useEffect(() => {
    fromLink.current = linkMode();
    const on = () => setM(fromLink.current ?? readMode());
    on();
    setReady(true);
    window.addEventListener(EVENT, on);
    window.addEventListener("storage", on);
    window.addEventListener("cw-purpose-change", on);
    return () => {
      window.removeEventListener(EVENT, on);
      window.removeEventListener("storage", on);
      window.removeEventListener("cw-purpose-change", on);
    };
  }, []);
  // switching drops a link's `mode=` so the choice shows at once and is kept
  const set = useCallback((x: Mode) => {
    fromLink.current = null;
    writeMode(x);
    try {
      const u = new URL(window.location.href);
      if (u.searchParams.has("mode")) {
        u.searchParams.delete("mode");
        window.history.replaceState(window.history.state, "", u.pathname + (u.search ? u.search : "") + u.hash);
      }
    } catch {
      // nothing to tidy
    }
    setM(x);
  }, []);
  return [m, set, ready];
}

export const MODE_LABEL: Record<Mode, { label: string; hint: string }> = {
  coastal: { label: "Coastal", hint: "Your waters, official notices and next steps first" },
  research: { label: "Research", hint: "Exact values, charts, model coverage and sources first" },
};
