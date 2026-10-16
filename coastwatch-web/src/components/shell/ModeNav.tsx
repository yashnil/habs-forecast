"use client";

import Link from "next/link";
import { useRef } from "react";
import { motion } from "motion/react";
import { Glide } from "@/components/motion/Glide";
import { EXPERIENCES } from "@/content/copy";
import { MODE_LABEL, useMode, type Mode } from "@/lib/mode";

type Key = (typeof EXPERIENCES)[number]["key"];

/** Coastal mode leads with My Coast; Research with the map and the station analysis. Same pages. */
const ORDER: Record<Mode, Key[]> = {
  coastal: ["coast", "map", "bloom", "fisheries"],
  research: ["map", "bloom", "fisheries", "coast"],
};

export function PrimaryNav({ active }: { active: string }) {
  const [mode] = useMode();
  const items = ORDER[mode].map((k) => EXPERIENCES.find((e) => e.key === k)!);
  const ul = useRef<HTMLUListElement>(null);
  return (
    <nav aria-label="Primary" className="hidden h-full md:flex">
      <ul ref={ul} className="relative flex h-full gap-1">
        {/* the active underline glides from the previous page's item */}
        <Glide id={`nav-${mode}`} container={ul} activeSelector='a[aria-current="page"]' className="pointer-events-none absolute bottom-0 h-0.5 rounded-full bg-accent-on-navy" />
        {items.map((e) => {
          const current = active === e.key;
          return (
            <li key={e.key} className="h-full">
              <Link
                href={e.href}
                aria-current={current ? "page" : undefined}
                data-testid={`nav-${e.key}`}
                className={`-mb-px flex h-full items-center whitespace-nowrap border-b-2 px-3 text-[15px] font-medium ${
                  current ? "border-transparent text-white" : "border-transparent text-on-navy-2 hover:text-on-navy"
                }`}
              >
                {/* full names where the masthead has room (with the presentation switch beside them); short ones below 1536 px */}
                <span className="hidden 2xl:inline">{e.label}</span>
                <span className="2xl:hidden">{e.short}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/**
 * Switch between the two presentations. Instant, kept on this device, never changes a fact.
 * `tone`: "navy" in the masthead, "paper" on reading pages.
 */
export function ModeSwitch({ tone = "navy", className = "" }: { tone?: "navy" | "paper"; className?: string }) {
  const [mode, setMode, ready] = useMode();
  const navy = tone === "navy";
  return (
    <div
      role="group"
      aria-label="Presentation"
      className={`inline-flex shrink-0 rounded-full p-0.5 text-[13px] font-medium ${navy ? "bg-white/[0.07]" : "border border-hairline-strong bg-surface"} ${ready ? "" : "invisible"} ${className}`}
      data-testid={`mode-switch-${tone}`}
    >
      {(["coastal", "research"] as const).map((m) => {
        const on = mode === m;
        return (
          <button
            key={m}
            type="button"
            aria-pressed={on}
            title={MODE_LABEL[m].hint}
            onClick={() => setMode(m)}
            data-testid={`mode-${m}`}
            className={`cw-press relative rounded-full px-3 transition-colors duration-200 ${navy ? "min-h-[30px]" : "min-h-[44px] min-w-[44px]"} ${
              on ? (navy ? "text-navy-950" : "text-white") : navy ? "text-on-navy-2 hover:text-white" : "text-ink-2 hover:text-ink"
            }`}
          >
            {on && <motion.span layoutId={`mode-pill-${tone}`} className={`absolute inset-0 rounded-full ${navy ? "bg-white" : "bg-navy-900"}`} transition={{ type: "spring", stiffness: 420, damping: 34 }} aria-hidden />}
            <span className="relative">{MODE_LABEL[m].label}</span>
          </button>
        );
      })}
    </div>
  );
}
