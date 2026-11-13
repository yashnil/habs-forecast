"use client";

import { motion } from "motion/react";
import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { panelIn } from "../motion";
import { cx } from "../util";
import s from "./synth.module.css";

export type TabItem<T extends string> = { id: T; label: ReactNode; count?: number | null; disabled?: boolean };

export interface TabsProps<T extends string> {
  /** The tablist's accessible name ("Detail"). */
  label: string;
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  /** `seg` (default): the ticket's Status / Record / Fisheries / Notices (a white segment on paper-2). `words`: Research's variable tabs. */
  look?: "seg" | "words";
  className?: string;
  /** id prefix for the panels, so `TabPanel` can name its tab */
  idBase?: string;
}

/** The tab and panel ids for a Tabs `idBase` and item id. */
export const tabIds = (base: string, id: string) => ({ tab: `${base}-tab-${id}`, panel: `${base}-panel-${id}` });

/**
 * Tabs (v14 README §3): a tablist with roving focus (Left/Right, Home/End) and counts after the words ("Fisheries 8").
 * Panels are `TabPanel`s; the chosen one fades and slides in (220 ms).
 */
export function Tabs<T extends string>({ label, items, value, onChange, look = "seg", className, idBase }: TabsProps<T>) {
  const auto = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const base = idBase ?? auto;
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const live = items.filter((i) => !i.disabled);
  const key = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const at = live.findIndex((x) => x.id === items[i].id);
    let n = -1;
    if (e.key === "ArrowRight") n = (at + 1) % live.length;
    else if (e.key === "ArrowLeft") n = (at - 1 + live.length) % live.length;
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = live.length - 1;
    if (n < 0) return;
    e.preventDefault();
    const next = live[n];
    onChange(next.id);
    refs.current[items.indexOf(next)]?.focus();
  };
  return (
    <div className={cx(look === "seg" ? s.seg : s.tabs, className)} role="tablist" aria-label={label}>
      {items.map((it, i) => {
        const on = it.id === value;
        const ids = tabIds(base, it.id);
        return (
          <button
            key={it.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={ids.tab}
            aria-selected={on}
            aria-controls={ids.panel}
            tabIndex={on ? 0 : -1}
            disabled={it.disabled}
            onClick={() => onChange(it.id)}
            onKeyDown={(e) => key(e, i)}
          >
            {it.label}
            {it.count != null ? <span className={s.n}>{it.count}</span> : null}
          </button>
        );
      })}
    </div>
  );
}

/** A tab's panel. Only the chosen one is rendered. */
export function TabPanel({ idBase, id, active, className, children }: { idBase: string; id: string; active: boolean; className?: string; children: ReactNode }) {
  if (!active) return null;
  const ids = tabIds(idBase, id);
  return (
    <motion.div role="tabpanel" id={ids.panel} aria-labelledby={ids.tab} className={cx(s.panel, className)} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={panelIn} tabIndex={0}>
      {children}
    </motion.div>
  );
}
