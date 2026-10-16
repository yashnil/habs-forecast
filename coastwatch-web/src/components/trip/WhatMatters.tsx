"use client";

import { AnimatePresence, motion } from "motion/react";
import { Mark } from "@/components/ui/Mark";
import type { Claim, EvidenceFocus } from "@/lib/harborEvidence";

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;

/**
 * "Right now": the briefing's claims, straight from the evidence graph (lib/harborEvidence), at most
 * five lines, each led by the mark of how it is known (official line, measured point, model cells,
 * conditions diamond, hatching for not known). An item that needs action sits on an amber rule; a
 * gap is grey and says what is not known, never a reassurance. Choosing an item shows where it comes
 * from on the map. Every line carries the ids of the nodes it rests on (data-supports), so the page,
 * the evidence list and the API can be checked against each other.
 */
export function WhatMatters({ items, onFocus, focus, sourceNote, placeName }: { items: Claim[]; onFocus: (f: EvidenceFocus) => void; focus: EvidenceFocus; sourceNote: React.ReactNode; placeName: string }) {
  return (
    <section aria-labelledby="matters-h" data-testid="what-matters">
      <h2 id="matters-h" className="font-display text-[24px] font-medium leading-tight text-ink sm:text-[26px]">
        Right now at {placeName}
      </h2>
      <ol className="mt-3 divide-y divide-hairline border-y border-hairline">
        <AnimatePresence initial={false} mode="popLayout">
          {items.map((m, i) => (
            <motion.li
              layout
              key={m.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0, transition: { delay: i * 0.04, duration: 0.26, ease: [0.2, 0, 0, 1] } }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}
              className={`relative py-3.5 pl-7 ${focus && m.focus === focus ? "bg-accent/[0.06]" : ""}`}
              data-testid={`matter-${m.id}`}
              data-tone={m.tone}
              data-kind={m.kind}
              data-supports={m.supports.join(" ")}
            >
              {m.tone === "alert" && <span aria-hidden className="absolute inset-y-3 left-0 w-[3px] rounded-r bg-official-line" />}
              <Mark kind={m.kind} aged={m.aged} className="absolute left-[9px] top-[21px]" />
              <p className={`text-[17px] font-semibold leading-snug ${m.tone === "gap" ? "text-ink-2" : "text-ink"}`}>{m.headline}</p>
              {m.quote && <blockquote className="mt-1 text-[16px] leading-snug text-ink">“{m.quote}”</blockquote>}
              {m.also && (
                <p className="mt-1.5 text-[15px] leading-snug text-ink-2" data-testid={`matter-${m.id}-also`}>
                  <span className="font-medium text-ink">{m.also.label}:</span> “{m.also.quote}”{" "}
                  {m.also.url !== m.source?.url && (
                    <a href={m.also.url} {...ext} className="inline-flex min-h-[44px] min-w-[44px] items-center font-semibold text-accent hover:underline">
                      source ↗
                    </a>
                  )}
                </p>
              )}
              {m.detail && <p className="mt-0.5 text-[15px] leading-snug text-ink-2">{m.detail}</p>}
              {(m.source || m.focus) && (
                <p className="mt-1 flex flex-wrap items-center gap-x-4 text-[14px]">
                  {m.source && (
                    <a href={m.source.url} {...ext} className="inline-flex min-h-[44px] min-w-[44px] items-center font-semibold text-accent underline-offset-2 hover:underline" data-testid={`matter-${m.id}-source`}>
                      {m.source.label} ↗
                    </a>
                  )}
                  {m.focus && (
                    <button
                      type="button"
                      onClick={() => onFocus(focus === m.focus ? null : m.focus!)}
                      aria-pressed={focus === m.focus}
                      className="inline-flex min-h-[44px] min-w-[44px] items-center font-medium text-ink-3 hover:text-ink"
                      data-testid={`matter-${m.id}-map`}
                    >
                      {focus === m.focus ? "Showing on map" : "Show on map"}
                    </button>
                  )}
                </p>
              )}
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
      <div className="mt-2.5 text-[14px] leading-snug text-ink-3">{sourceNote}</div>
    </section>
  );
}
