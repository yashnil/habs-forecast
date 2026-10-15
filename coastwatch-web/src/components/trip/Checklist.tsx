"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "motion/react";
import type { Step } from "@/lib/nextSteps";
import { Mark } from "@/components/ui/Mark";

const MARK: Record<Step["kind"], "official" | "conditions" | "gap" | "measured"> = {
  official: "official",
  conditions: "conditions",
  gap: "gap",
  hab: "measured",
  history: "measured",
};
const ext = { target: "_blank", rel: "noopener noreferrer" } as const;

function readTicks(key: string): string[] {
  try {
    return JSON.parse(window.localStorage.getItem(key) ?? "[]");
  } catch {
    return [];
  }
}

/**
 * "Before you leave": a short checklist for this trip. Each item is something to open (the agency
 * page, the phone line, the forecast) with why, for which waters or species, and what CoastWatch
 * cannot settle. Ticks are kept on this device for this trip only (waters, activity, species, day);
 * a tick records that the reader opened the source, not that anything is safe or open.
 */
export function Checklist({ shown, more, storageKey, sub }: { shown: Step[]; more: Step[]; storageKey: string; sub: string }) {
  const [ticks, setTicks] = useState<string[]>([]);
  const [all, setAll] = useState(false);
  const [why, setWhy] = useState<string | null>(null);
  useEffect(() => setTicks(readTicks(storageKey)), [storageKey]);
  const toggle = (id: string, on?: boolean) =>
    setTicks((t) => {
      const next = (on ?? !t.includes(id)) ? [...new Set([...t, id])] : t.filter((x) => x !== id);
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(next));
      } catch {
        // not kept on this device
      }
      return next;
    });
  const steps = all ? [...shown, ...more] : shown;
  const done = shown.filter((s) => ticks.includes(s.id)).length;

  return (
    <section aria-labelledby="leave-h" data-testid="checklist">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 id="leave-h" className="font-display text-[22px] font-medium text-ink sm:text-[24px]">
            Before you leave
          </h2>
          <p className="mt-0.5 text-[14px] text-ink-3">{sub}</p>
        </div>
        <p className="shrink-0 text-[13px] font-medium tabular text-ink-3" aria-live="polite" data-testid="checklist-progress">
          {done} of {shown.length} checked
        </p>
      </div>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-3" aria-hidden>
        <motion.div
          className="h-full rounded-full bg-accent"
          initial={false}
          animate={{
            width: `${shown.length ? (done / shown.length) * 100 : 0}%`,
          }}
          transition={{ type: "spring", stiffness: 200, damping: 30 }}
        />
      </div>
      <ol className="mt-3 space-y-2">
        <AnimatePresence initial={false}>
          {steps.map((s) => {
            const on = ticks.includes(s.id);
            const href = "url" in s.source ? s.source.url : `tel:${s.source.tel}`;
            const internal = href.startsWith("/");
            const linkCls = "cw-press inline-flex min-h-[44px] items-center gap-1.5 rounded-xl bg-navy-900 px-3.5 text-[14px] font-semibold text-white hover:bg-navy-800";
            const open = () => toggle(s.id, true);
            return (
              <motion.li
                key={s.id}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className={`overflow-hidden rounded-2xl border bg-surface transition-colors ${on ? "border-hairline" : "border-hairline-strong"}`}
                data-testid={`check-${s.id}`}
                data-done={on}
              >
                <div className="grid grid-cols-[44px_1fr] gap-x-1 py-2.5 pl-1.5 pr-3.5">
                  <button
                    type="button"
                    role="checkbox"
                    aria-checked={on}
                    onClick={() => toggle(s.id)}
                    aria-label={`${on ? "Checked" : "Not checked"}: ${s.title}`}
                    className="cw-press grid h-11 w-11 place-items-center rounded-xl hover:bg-surface-3"
                    data-testid={`check-${s.id}-box`}
                  >
                    <span className={`grid h-6 w-6 place-items-center rounded-md border-2 transition-colors ${on ? "border-accent bg-accent text-white" : "border-hairline-strong bg-surface"}`}>
                      <AnimatePresence>
                        {on && (
                          <motion.svg viewBox="0 0 16 16" className="h-4 w-4" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} exit={{ opacity: 0 }}>
                            <motion.path
                              d="M3.5 8.5l3 3 6-7"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth={2.2}
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 0.22 }}
                            />
                          </motion.svg>
                        )}
                      </AnimatePresence>
                    </span>
                  </button>
                  <div className="min-w-0 pt-1.5">
                    <p className={`flex items-start gap-2 text-[15.5px] font-semibold leading-snug ${on ? "text-ink-3" : "text-ink"}`}>
                      <Mark kind={MARK[s.kind]} className="mt-[5px]" />
                      <span className={on ? "line-through decoration-ink-3/40" : ""}>{s.title}</span>
                    </p>
                    <p className="mt-0.5 pl-[22px] text-[13.5px] leading-snug text-ink-2">{s.reason}</p>
                    {(s.scope || s.when) && <p className="mt-0.5 pl-[22px] text-[12.5px] text-ink-3">{[s.scope, s.when].filter(Boolean).join(" · ")}</p>}
                    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 pl-[22px]">
                      {internal ? (
                        <Link href={href} className={linkCls} onClick={open} data-testid={`check-${s.id}-go`}>
                          {s.source.label} →
                        </Link>
                      ) : (
                        <a href={href} {...("url" in s.source ? ext : {})} className={linkCls} onClick={open} data-testid={`check-${s.id}-go`}>
                          {"tel" in s.source ? `Call ${s.source.label}` : `${s.source.label} ↗`}
                        </a>
                      )}
                      {s.also && (
                        <a href={s.also.url} {...ext} className="inline-flex min-h-[44px] items-center text-[14px] font-medium text-accent hover:underline">
                          {s.also.label} ↗
                        </a>
                      )}
                      {s.caveat && (
                        <button
                          type="button"
                          onClick={() => setWhy(why === s.id ? null : s.id)}
                          aria-expanded={why === s.id}
                          className="inline-flex min-h-[44px] items-center text-[13px] font-medium text-ink-3 hover:text-ink"
                        >
                          What Neritic Signal can&apos;t settle
                        </button>
                      )}
                    </div>
                    <AnimatePresence initial={false}>
                      {why === s.id && s.caveat && (
                        <motion.p
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="overflow-hidden pl-[22px] text-[13px] leading-snug text-ink-2"
                        >
                          <span className="mt-1 block border-l-2 border-hairline-strong pl-2.5">{s.caveat}</span>
                        </motion.p>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ol>
      {more.length > 0 && (
        <button type="button" onClick={() => setAll(!all)} className="mt-2 inline-flex min-h-[44px] items-center text-[14px] font-medium text-accent hover:underline" data-testid="checklist-more">
          {all ? "Fewer" : `${more.length} more for this trip`}
        </button>
      )}
      <p className="mt-1 text-[12.5px] text-ink-3">Ticks are kept on this device for this trip only. A tick means you opened the source, not that anything is safe or open.</p>
    </section>
  );
}
