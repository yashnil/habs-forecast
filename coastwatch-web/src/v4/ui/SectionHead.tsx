import type { ReactNode } from "react";

/**
 * A section's head (design.md §4.2): a serif sentence-case title, an optional marker beside it (a Tag or
 * Unverified) and an optional quiet action at its right edge. It replaces the caps label over a section.
 *
 * <SectionHead title="Fisheries" marker={<Unverified count="7 of 7" />} action={<LinkButton kind="tertiary" href="#">Edit</LinkButton>} />
 */
export function SectionHead({ title, marker, action, as: As = "h2", id, testId, className = "" }: { title: ReactNode; marker?: ReactNode; action?: ReactNode; as?: "h1" | "h2" | "h3"; id?: string; testId?: string; className?: string }) {
  return (
    <header className={`flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 ${className}`} data-testid={testId}>
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
        <As id={id} className="nx-title">
          {title}
        </As>
        {marker}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}
