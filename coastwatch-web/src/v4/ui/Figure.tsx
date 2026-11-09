import type { ReactNode } from "react";

/**
 * A figure (design.md §4.2): the number that leads a section, large and light, its unit beside it at
 * 0.4 em, an optional label over it and one quiet line under it (age or source). A stale value says
 * "(old)" in words on the value itself and steps back in ink. The number never animates.
 *
 * <Figure label="Wind" value={12} unit="kt" age="NDBC 46042 · 39 min ago" />
 * <Figure label="Tests" value="5.5" unit="ppm" age="Sep 28" stale />
 */
export function Figure({
  label,
  value,
  unit,
  age,
  stale = false,
  size = "hero",
  testId,
  className = "",
}: {
  label?: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  /** one quiet line: its time, its source */
  age?: ReactNode;
  /** an old value says so in words, on the value */
  stale?: boolean;
  /** hero: .nx-figure (40-64 px); value: .nx-value (20-26 px) */
  size?: "hero" | "value";
  testId?: string;
  className?: string;
}) {
  return (
    <div className={`flex min-w-0 flex-col gap-1 ${className}`} data-testid={testId} data-stale={stale ? "" : undefined}>
      {label && <span className="nx-caps">{label}</span>}
      <span className={`${size === "hero" ? "nx-figure" : "nx-value"} whitespace-nowrap ${stale ? "!text-n-ink-2" : ""}`}>
        {value}
        {unit && <small>{unit}</small>}
        {stale && <small data-testid={testId ? `${testId}-old` : undefined}> (old)</small>}
      </span>
      {age && <span className="nx-note-t">{age}</span>}
    </div>
  );
}
