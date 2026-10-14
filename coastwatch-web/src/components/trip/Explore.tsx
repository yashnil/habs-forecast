"use client";

import { useState } from "react";
import { motion } from "motion/react";

/**
 * "Explore the evidence": the full detail behind the briefing (every station, the model's cells and
 * days, every notice, the conditions, what is not known), one subject at a time instead of one long
 * report. Nothing here is hidden from the briefing's claims; it is their working.
 */
export function Explore({ tabs, initial }: { tabs: { id: string; label: string; content: React.ReactNode }[]; initial?: string }) {
  const [tab, setTab] = useState(initial ?? tabs[0]?.id);
  const cur = tabs.find((t) => t.id === tab) ?? tabs[0];
  return (
    <section aria-labelledby="explore-h" className="scroll-mt-24" id="explore" data-testid="explore">
      <h2 id="explore-h" className="font-display text-[22px] font-medium text-ink sm:text-[24px]">
        Explore the evidence
      </h2>
      <div role="tablist" aria-label="Evidence" className="-mx-4 mt-3 flex gap-1 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:px-0">
        {tabs.map((t) => {
          const on = t.id === cur?.id;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={on}
              aria-controls={`explore-${t.id}`}
              id={`explore-tab-${t.id}`}
              onClick={() => setTab(t.id)}
              className={`cw-press relative min-h-[44px] shrink-0 rounded-xl px-3.5 text-[14.5px] font-medium transition-colors ${on ? "text-white" : "text-ink-2 hover:bg-surface-3 hover:text-ink"}`}
              data-testid={`explore-tab-${t.id}`}
            >
              {on && <motion.span layoutId="explore-pill" className="absolute inset-0 rounded-xl bg-navy-900" transition={{ type: "spring", stiffness: 420, damping: 34 }} aria-hidden />}
              <span className="relative">{t.label}</span>
            </button>
          );
        })}
      </div>
      {/* the panel fades in; it never waits for the previous one to leave (a busy panel could hold it) */}
      <motion.div
        key={cur?.id}
        role="tabpanel"
        id={`explore-${cur?.id}`}
        aria-labelledby={`explore-tab-${cur?.id}`}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0, transition: { duration: 0.22, ease: [0.2, 0, 0, 1] } }}
        className="mt-4 space-y-6"
        data-testid={`explore-panel-${cur?.id}`}
      >
        {cur?.content}
      </motion.div>
    </section>
  );
}
