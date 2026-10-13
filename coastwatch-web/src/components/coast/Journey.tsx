"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";

export type Chapter = { id: string; title: string; summary: React.ReactNode; content: React.ReactNode; cta?: string };

/**
 * A trip plan in chapters (My Coast, Coastal mode): one chapter open at a time, the others folded
 * to one line that still says the answer. A rail connects them; the open chapter's number is filled,
 * finished ones show a tick. Content opens and closes by height (Motion), never by a jump; under
 * reduced motion it simply appears.
 */
export function Journey({ chapters, active, onActive }: { chapters: Chapter[]; active: string; onActive: (id: string) => void }) {
  const refs = useRef<Record<string, HTMLLIElement | null>>({});
  const first = useRef(true);
  const ai = chapters.findIndex((c) => c.id === active);
  useEffect(() => {
    // after the first render, bring the opened chapter's heading into view
    if (first.current) {
      first.current = false;
      return;
    }
    const el = refs.current[active];
    if (!el) return;
    const t = setTimeout(() => {
      const r = el.getBoundingClientRect();
      if (r.top < 70 || r.top > window.innerHeight * 0.6) window.scrollTo({ top: window.scrollY + r.top - 110, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }, 220);
    return () => clearTimeout(t);
  }, [active]);
  return (
    <ol className="relative" data-testid="journey" aria-label="Plan your trip">
      {chapters.map((c, i) => {
        const open = c.id === active;
        const done = i < ai;
        const last = i === chapters.length - 1;
        return (
          <li
            key={c.id}
            ref={(el) => {
              refs.current[c.id] = el;
            }}
            className="relative grid grid-cols-[36px_1fr] gap-x-3"
            data-testid={`chapter-${c.id}`}
            data-open={open || undefined}
          >
            {/* the rail */}
            {!last && <span aria-hidden className={`absolute left-[17px] top-9 bottom-0 w-[2px] rounded-full transition-colors duration-500 ${done ? "bg-navy-900/70" : "bg-hairline-strong"}`} />}
            <motion.span
              aria-hidden
              className={`relative z-[1] mt-1 grid h-9 w-9 place-items-center rounded-full border-2 text-[14px] font-semibold ${open ? "border-navy-900 bg-navy-900 text-white" : done ? "border-navy-900 bg-surface text-navy-900" : "border-hairline-strong bg-surface text-ink-3"}`}
              animate={{ scale: open ? 1 : 0.9 }}
              transition={{ type: "spring", stiffness: 400, damping: 26 }}
            >
              {done ? "✓" : i + 1}
            </motion.span>
            <div className={`min-w-0 ${last ? "" : "pb-6"}`}>
              <button
                type="button"
                onClick={() => onActive(c.id)}
                aria-expanded={open}
                aria-controls={`chapter-body-${c.id}`}
                className="group flex w-full items-start justify-between gap-3 py-1.5 text-left"
                data-testid={`chapter-head-${c.id}`}
              >
                <span className="min-w-0">
                  <span className={`block font-display text-[22px] font-medium leading-snug transition-colors ${open ? "text-ink" : "text-ink-2 group-hover:text-ink"}`}>{c.title}</span>
                  {!open && <span className="mt-0.5 block text-[15px] leading-snug text-ink-2">{c.summary}</span>}
                </span>
                {!open && (
                  // "Show", never "Open": beside an official answer, "open" could read as "waters open"
                  <span aria-hidden className="mt-2 shrink-0 text-[14px] font-medium text-accent opacity-80 group-hover:opacity-100">
                    Show
                  </span>
                )}
              </button>
              <AnimatePresence initial={false}>
                {open && (
                  <motion.div
                    key="body"
                    id={`chapter-body-${c.id}`}
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ height: { type: "spring", stiffness: 260, damping: 32 }, opacity: { duration: 0.22 } }}
                    className="overflow-hidden"
                  >
                    <div className="pb-1 pt-2">
                      {c.content}
                      {!last && (
                        <motion.button
                          type="button"
                          whileTap={{ scale: 0.97 }}
                          onClick={() => onActive(chapters[i + 1].id)}
                          className="mt-4 inline-flex min-h-[44px] items-center gap-2 rounded-full bg-navy-900 px-5 text-[15px] font-medium text-white hover:bg-navy-800"
                          data-testid="journey-next"
                        >
                          {c.cta ?? `Next: ${chapters[i + 1].title.toLowerCase()}`}
                          <span aria-hidden>→</span>
                        </motion.button>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
