import type { ReactNode } from "react";
import type { Severity } from "@/pilot/contracts";
import { Reveal } from "@/v4/motion";

/**
 * Every statement on Today has an address: `#fact-<PortFact.id>`. Ask Neritic's citations and alerts
 * link to it (`/today?place=…#fact-status:dungeness_commercial`). A card carries its own fact's id
 * (`factProps`); facts drawn inside a shared instrument (a forecast period in the lanes, a day's
 * tides) get an invisible anchor there (`FactAnchors`). TodayView marks the card when it is reached.
 */
export const factAnchor = (factId: string) => `fact-${factId}`;
/** Spread on the element that IS the fact's card. */
export const factProps = (factId: string) => ({ id: factAnchor(factId), "data-fact-card": "" }) as const;
/** Invisible anchors for facts shown inside the surrounding card (place inside a `data-fact-card`). */
export function FactAnchors({ ids }: { ids: string[] }) {
  return (
    <>
      {ids.map((id) => (
        <span key={id} id={factAnchor(id)} className="fact-anchor" />
      ))}
    </>
  );
}

/**
 * A status shape, so state never relies on colour alone: a filled triangle (red: an emergency closure
 * or delay), a half disc (amber), a ring (none on record), a dashed ring (unknown). A fishery that is
 * simply out of season is a barred disc in ink: plainly not open, and not an alarm.
 */
export function StatusMark({ severity, seasonal = false, className = "" }: { severity: Severity; seasonal?: boolean; className?: string }) {
  const c = seasonal ? "t-season" : severity === "red" ? "t-red" : severity === "amber" ? "t-amber" : severity === "neutral" ? "t-neutral" : "t-info";
  if (seasonal)
    return (
      <svg viewBox="0 0 24 24" width="22" height="22" className={`${c} ${className} shrink-0`} aria-hidden="true" data-mark="season">
        <circle cx="12" cy="12" r="8.6" fill="currentColor" />
        <path d="M7.4 12h9.2" stroke="var(--nx-bg)" strokeWidth="2.4" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" className={`${c} ${className} shrink-0`} aria-hidden="true">
      {severity === "red" && (
        <>
          <path d="M12 2.8 22 20.5H2L12 2.8Z" fill="currentColor" />
          <path d="M12 9v5.2" stroke="var(--nx-bg)" strokeWidth="2.2" strokeLinecap="round" />
          <circle cx="12" cy="17.2" r="1.25" fill="var(--nx-bg)" />
        </>
      )}
      {severity === "amber" && (
        <>
          <circle cx="12" cy="12" r="8.6" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M12 3.4a8.6 8.6 0 0 1 0 17.2Z" fill="currentColor" />
        </>
      )}
      {severity === "neutral" && <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="2" />}
      {severity === "info" && <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3 3" />}
    </svg>
  );
}

export const sevClass = (s: Severity, seasonal = false) => (seasonal ? "t-season" : s === "red" ? "t-red" : s === "amber" ? "t-amber" : s === "neutral" ? "t-neutral" : "t-info");

/** "Not verified" in words inside a row or a sheet (the row is itself a control, so this one opens nothing; the group's header marker explains it). */
export function NotVerifiedWords() {
  return (
    <span className="nx-unverified" data-testid="not-verified">
      Not verified
    </span>
  );
}

/**
 * One chapter of the briefing: a title and one sentence, then one graphic, its detail behind a Fold or a
 * Sheet. On a wide screen the head holds to the left while the graphic reads beside it; the order in the
 * document is the order of authority.
 */
export function Chapter({ id, area, title, marker, action, say, children, testid }: { id: string; area: string; title: string; marker?: ReactNode; action?: ReactNode; say?: ReactNode; children: ReactNode; testid?: string }) {
  return (
    <Reveal as="section" className="today-ch" data-area={area} aria-labelledby={`${id}-h`} id={id} data-testid={testid ?? `today-${id}`}>
      <div className="today-ch-head">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h2 id={`${id}-h`} className="nx-title">
            {title}
          </h2>
          {action}
        </div>
        {marker && <div className="mt-1.5">{marker}</div>}
        {say && <p className="today-say">{say}</p>}
      </div>
      <div className="today-ch-body">{children}</div>
    </Reveal>
  );
}

export function SourceLink({ name, url, children }: { name: string; url: string | null; children?: ReactNode }) {
  if (!url) return <span>{children ?? name}</span>;
  const external = /^https?:/.test(url);
  return (
    <a href={url} {...(external ? { target: "_blank", rel: "noreferrer" } : {})} className="nx-focus">
      {children ?? name}
    </a>
  );
}
