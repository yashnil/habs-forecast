"use client";

import { Icon } from "./Icon";
import { Pop } from "./Pop";

/**
 * The compact freshness cue (design.md §3): it names only what is old, in words ("Tests Sep 28 (old)");
 * every source and its time are one tap away. With nothing old it says "Sources" and stays quiet.
 * `items` matches the shape of the Today model's freshness rows.
 *
 * <Freshness items={[{ label: "Tests", when: "Sep 28", stale: true }, { label: "NWS", when: "2:34 AM" }]} />
 */
export type FreshnessItem = { label: string; when: string; stale?: boolean; missing?: boolean };

const line = (f: FreshnessItem) => `${f.label} ${f.when}${f.stale && !f.missing ? " (old)" : ""}`;

export function Freshness({ items, testId = "freshness", className = "" }: { items: FreshnessItem[]; testId?: string; className?: string }) {
  const old = items.filter((f) => f.stale);
  return (
    <Pop
      label="How fresh the data are"
      testId={testId}
      triggerClass={`nx-focus inline-flex min-h-11 items-center gap-1.5 text-left text-[13.5px] leading-[18px] text-n-ink-2 ${className}`}
      trigger={
        old.length ? (
          <>
            <Icon name="clock" className="h-4 w-4 text-n-stale" />
            <span>{old.map(line).join(" · ")}</span>
          </>
        ) : (
          <>
            <Icon name="clock" className="h-4 w-4 text-n-ink-3" />
            <span className="underline decoration-dotted decoration-n-line-strong underline-offset-[3px]">Sources</span>
          </>
        )
      }
    >
      <p className="text-[15px] font-semibold text-n-ink">How fresh the data are</p>
      <ul className="mt-1.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[14px] leading-[20px]">
        {items.map((f) => (
          <li key={f.label} className="contents">
            <span className="text-n-ink-2">{f.label}</span>
            <span className={f.stale ? "text-n-stale" : "text-n-ink"}>
              {f.when}
              {f.stale && !f.missing ? " (old)" : ""}
            </span>
          </li>
        ))}
      </ul>
    </Pop>
  );
}
