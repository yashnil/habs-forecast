"use client";

import { useCallback, useEffect, useState } from "react";
import { FLOW_SLOW_EVENT, flowTooSlow } from "@/components/map/flowSlow";
import { Icon } from "@/v4/ui/Icon";

/**
 * How the currents layer is drawn, kept per device: moving particles ("flow", the default) or static
 * arrows; the motion paused (WCAG 2.2.2: anything that moves for more than five seconds can be paused);
 * particles coloured by speed class. Under reduced motion the particles are one still frame and there is
 * nothing to pause. A device too slow to draw the motion is paused for this page (`slow`, never stored) and
 * says why; Play overrides it.
 */
export type CurrentsView = "flow" | "arrows";
export type FlowPrefs = { view: CurrentsView; paused: boolean; colour: boolean; slow?: boolean };
const KEY = "nx-currents-view-v1";
const DEFAULT: FlowPrefs = { view: "flow", paused: false, colour: false };

function read(): FlowPrefs {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEY) ?? "null") as Partial<FlowPrefs> | null;
    if (!v || typeof v !== "object") return DEFAULT;
    return { view: v.view === "arrows" ? "arrows" : "flow", paused: v.paused === true, colour: v.colour === true };
  } catch {
    return DEFAULT;
  }
}

/** The device's choices, and whether they have been read yet (the server and the first render have the defaults). */
export function useFlowPrefs(): [FlowPrefs, (p: Partial<FlowPrefs>) => void, boolean] {
  const [prefs, setPrefs] = useState<FlowPrefs>(DEFAULT);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    setPrefs(read());
    setReady(true);
    const slow = () => setPrefs((p) => (p.paused ? p : { ...p, paused: true, slow: true }));
    if (flowTooSlow()) slow();
    window.addEventListener(FLOW_SLOW_EVENT, slow);
    return () => window.removeEventListener(FLOW_SLOW_EVENT, slow);
  }, []);
  const set = useCallback((p: Partial<FlowPrefs>) => {
    setPrefs((was) => {
      const next = { ...was, ...p };
      if (p.paused === false) next.slow = false;
      try {
        // the slow-device pause is this page's, never the device's stored choice
        window.localStorage.setItem(KEY, JSON.stringify({ view: next.view, paused: next.slow ? false : next.paused, colour: next.colour }));
      } catch {
        // storage unavailable (private window): the choice holds for this page only
      }
      return next;
    });
  }, []);
  return [prefs, set, ready];
}

/** prefers-reduced-motion, followed live. */
export function useReducedMotion(): boolean {
  const [r, setR] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    const f = () => setR(q.matches);
    f();
    q.addEventListener("change", f);
    return () => q.removeEventListener("change", f);
  }, []);
  return r;
}

const seg = "nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 px-3 text-[13px]";

/**
 * `part` (v10): "pause" is the motion's pause, always in view beside the currents legend (WCAG 2.2.2);
 * "view" is how they are drawn (flow or arrows, colour by speed), one level in, in the layer index.
 */
export function FlowControls({ prefs, onPrefs, reduced, part = "all" }: { prefs: FlowPrefs; onPrefs: (p: Partial<FlowPrefs>) => void; reduced: boolean; part?: "all" | "pause" | "view" }) {
  const flow = prefs.view === "flow";
  const view = part !== "pause";
  const pause = part !== "view";
  if (part === "pause" && !flow) return null;
  return (
    <div role="group" aria-label={part === "pause" ? "Currents motion" : "How the currents are drawn"} className="flex flex-wrap items-center gap-x-3 gap-y-1.5" data-testid={part === "pause" ? "flow-motion" : "flow-controls"}>
      {view && (
        <div className="inline-flex overflow-hidden rounded-[3px] border border-n-line-strong">
          {(["flow", "arrows"] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={prefs.view === v}
              onClick={() => onPrefs({ view: v })}
              className={`${seg} ${prefs.view === v ? "bg-n-ink text-n-bg" : "text-n-ink hover:bg-n-surface-2"}`}
              data-testid={`flow-view-${v}`}
            >
              {v === "flow" ? "Moving streaks" : "Arrows"}
            </button>
          ))}
        </div>
      )}
      {pause &&
        flow &&
        (reduced ? (
          <span className="nx-margin text-[12.5px]" data-testid="flow-still">
            Still frame: reduced motion is on
          </span>
        ) : (
          <>
            {prefs.paused && prefs.slow && (
              <span className="nx-margin text-[12.5px]" data-testid="flow-slow">
                Paused: this device draws the moving streaks too slowly
              </span>
            )}
          <button
            type="button"
            aria-pressed={prefs.paused}
            aria-label={prefs.paused ? "Play the currents motion" : "Pause the currents motion"}
            onClick={() => onPrefs({ paused: !prefs.paused })}
            className={`${seg} rounded-[3px] border border-n-line-strong text-n-ink hover:bg-n-surface-2`}
            data-testid="flow-pause"
          >
            <Icon name={prefs.paused ? "play" : "pause"} className="h-4 w-4" />
            {prefs.paused ? "Play motion" : "Pause motion"}
          </button>
          </>
        ))}
      {view && flow && (
        <label className="nx-focus inline-flex min-h-11 items-center gap-2 text-[13px] text-n-ink">
          <input type="checkbox" className="h-[18px] w-[18px] accent-[var(--nx-accent)]" checked={prefs.colour} onChange={(e) => onPrefs({ colour: e.target.checked })} data-testid="flow-colour" />
          Colour by speed
        </label>
      )}
    </div>
  );
}
