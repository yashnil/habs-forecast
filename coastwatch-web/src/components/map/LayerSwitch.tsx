"use client";

import { useRef } from "react";
import { motion } from "motion/react";
import type { Manifest } from "@/generated/schema";
import { currentsHourly } from "@/lib/currents";
import { Mark } from "@/components/ui/Mark";
import { GROUPS, type LayerGroup } from "./LayerDock";

/**
 * What the map draws, as one control at the top of the map (replaces the left rail): the seafloor
 * alone, or one data layer. The pill slides to the choice; each data layer carries its mark (cells
 * for the model, a point for observations) and kind, so a model never reads as a measurement.
 */
export function LayerSwitch({ manifest, group, onGroup, tone = "dark" }: { manifest: Manifest; group: LayerGroup | null; onGroup: (g: LayerGroup | null) => void; tone?: "dark" | "paper" }) {
  const hasCurrents = currentsHourly(manifest).length > 0;
  const opts: { id: LayerGroup | null; short: string; kind: string | null }[] = [{ id: null, short: "Seafloor", kind: null }, ...GROUPS.map((g) => ({ id: g.id, short: g.short, kind: g.kind }))];
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const dark = tone === "dark";
  const onKey = (e: React.KeyboardEvent) => {
    const i = Math.max(0, opts.findIndex((o) => o.id === group));
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    let j = i;
    do j = (j + step + opts.length) % opts.length;
    while (opts[j].id === "currents" && !hasCurrents);
    onGroup(opts[j].id);
    refs.current[j]?.focus();
  };
  return (
    <div
      role="radiogroup"
      aria-label="What the map shows"
      onKeyDown={onKey}
      data-testid="layer-switch"
      className={`relative flex h-11 items-center gap-0.5 rounded-2xl p-1 ${dark ? "bg-[rgba(11,26,43,0.86)] text-[#c9d6e3] shadow-[0_8px_28px_rgba(6,17,30,0.35)] ring-1 ring-white/10 backdrop-blur-md" : "bg-surface-3 text-ink-2"}`}
    >
      {opts.map((o, i) => {
        const on = group === o.id;
        const off = o.id === "currents" && !hasCurrents;
        return (
          <button
            key={o.short}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            disabled={off}
            title={off ? "Observed currents are not in this dataset." : o.kind ? `${o.short} (${o.kind.toLowerCase()})` : "The coast and the seafloor, with no data layer"}
            onClick={() => onGroup(o.id)}
            data-testid={`group-${o.id ?? "none"}`}
            className={`cw-press relative flex h-9 items-center gap-1.5 rounded-xl px-3 text-[14px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-35 ${on ? (dark ? "text-navy-950" : "text-ink") : dark ? "hover:text-white" : "hover:text-ink"}`}
          >
            {on && <motion.span layoutId={`layer-pill-${tone}`} className={`absolute inset-0 rounded-xl ${dark ? "bg-white" : "bg-surface shadow-[0_1px_2px_rgba(13,27,42,0.12)]"}`} transition={{ type: "spring", stiffness: 420, damping: 34 }} aria-hidden />}
            <span className="relative flex items-center gap-1.5">
              {o.kind && <Mark kind={o.kind === "Model" ? "model" : "measured"} />}
              {o.short}
              {o.kind && <span className="sr-only">, {o.kind.toLowerCase()}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}
