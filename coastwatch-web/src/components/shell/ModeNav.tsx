"use client";

import Link from "next/link";
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
  return (
    <nav aria-label="Primary" className="hidden h-full md:flex">
      <ul className="flex h-full gap-1">
        {items.map((e) => {
          const current = active === e.key;
          return (
            <li key={e.key} className="h-full">
              <Link
                href={e.href}
                aria-current={current ? "page" : undefined}
                data-testid={`nav-${e.key}`}
                className={`-mb-px flex h-full items-center whitespace-nowrap border-b-2 px-3 text-[15px] font-medium ${
                  current ? "border-accent-on-navy text-white" : "border-transparent text-on-navy-2 hover:text-on-navy"
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
            className={`min-h-[30px] rounded-full px-3 ${
              on ? (navy ? "bg-white text-navy-950" : "bg-navy-900 text-white") : navy ? "text-on-navy-2 hover:text-white" : "text-ink-2 hover:text-ink"
            }`}
          >
            {MODE_LABEL[m].label}
          </button>
        );
      })}
    </div>
  );
}
