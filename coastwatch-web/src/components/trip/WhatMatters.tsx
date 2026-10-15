"use client";

import { AnimatePresence, motion } from "motion/react";
import { Mark } from "@/components/ui/Mark";
import type { Matter } from "@/lib/trip";
import type { Focus } from "@/components/coast/HabBriefing";

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;

/**
 * "What matters right now": at most five lines, each led by the mark of how it is known (official
 * line, measured point, model cells, conditions diamond, hatching for not known). An item that needs
 * action sits on an amber rule; a gap is grey and says what is not known, never a reassurance.
 * Choosing an item shows where it comes from on the map.
 */
export function WhatMatters({ items, checked, onFocus, focus, sourceNote }: { items: Matter[]; checked: string | null; onFocus: (f: Focus) => void; focus: Focus; sourceNote: React.ReactNode }) {
  return (
    <section aria-labelledby="matters-h" data-testid="what-matters">
      <h2 id="matters-h" className="font-display text-[22px] font-medium text-ink sm:text-[24px]">
        What matters right now
      </h2>
      <ol className="mt-3 space-y-2">
        <AnimatePresence initial={false} mode="popLayout">
          {items.map((m, i) => (
            <motion.li
              layout
              key={m.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{
                opacity: 1,
                y: 0,
                transition: {
                  delay: i * 0.05,
                  duration: 0.3,
                  ease: [0.2, 0, 0, 1],
                },
              }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}
              className={`relative rounded-2xl border px-4 py-3 ${
                m.tone === "alert" ? "border-official-line/70 bg-official-bg/70" : m.tone === "gap" ? "border-dashed border-hairline-strong bg-surface-2" : "border-hairline bg-surface"
              } ${focus && m.focus === focus ? "ring-2 ring-accent/50" : ""}`}
              data-testid={`matter-${m.id}`}
              data-tone={m.tone}
              data-kind={m.kind}
            >
              {m.tone === "alert" && <span aria-hidden className="absolute inset-y-3 left-0 w-[3px] rounded-r bg-official-line" />}
              <div className="grid grid-cols-[16px_1fr] gap-x-2.5">
                <Mark kind={m.kind} aged={m.aged} className="mt-[5px]" />
                <div className="min-w-0">
                  <motion.p key={m.headline} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24, ease: [0.2, 0, 0, 1] }} className={`text-[16.5px] font-semibold leading-snug ${m.tone === "gap" ? "text-ink-2" : "text-ink"}`}>{m.headline}</motion.p>
                  {m.quote && <blockquote className="mt-1 text-[16px] leading-snug text-ink">“{m.quote}”</blockquote>}
                  {m.also && (
                    <p className="mt-1.5 text-[14px] leading-snug text-ink-2" data-testid={`matter-${m.id}-also`}>
                      <span className="font-medium text-ink">{m.also.label}:</span> “{m.also.quote}”{" "}
                      {m.also.url !== m.source?.url && (
                        <a href={m.also.url} {...ext} className="font-semibold text-accent hover:underline">
                          source ↗
                        </a>
                      )}
                    </p>
                  )}
                  {m.detail && <p className="mt-0.5 text-[14px] leading-snug text-ink-2">{m.detail}</p>}
                  <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13.5px]">
                    {m.source && (
                      <a
                        href={m.source.url}
                        {...ext}
                        className="inline-flex min-h-[32px] items-center font-semibold text-accent underline-offset-2 hover:underline"
                        data-testid={`matter-${m.id}-source`}
                      >
                        {m.source.label} ↗
                      </a>
                    )}
                    {m.kind === "official" && checked && <span className="text-ink-3">Neritic&apos;s copy, checked {checked}</span>}
                    {m.focus && (
                      <button
                        type="button"
                        onClick={() => onFocus(focus === m.focus ? null : m.focus!)}
                        aria-pressed={focus === m.focus}
                        className="inline-flex min-h-[32px] items-center font-medium text-ink-3 hover:text-ink"
                        data-testid={`matter-${m.id}-map`}
                      >
                        {focus === m.focus ? "Showing on map" : "Show on map"}
                      </button>
                    )}
                  </p>
                </div>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
      <div className="mt-2.5 text-[13px] leading-snug text-ink-3">{sourceNote}</div>
    </section>
  );
}
