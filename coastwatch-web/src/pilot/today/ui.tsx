import type { ReactNode } from "react";
import type { Severity } from "@/pilot/contracts";

/** A status shape, so state never relies on colour alone: a filled triangle (red), a half disc (amber), a ring (none), a dashed ring (unknown). */
export function StatusMark({ severity, className = "" }: { severity: Severity; className?: string }) {
  const c = severity === "red" ? "t-red" : severity === "amber" ? "t-amber" : severity === "neutral" ? "t-neutral" : "t-info";
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

export const sevClass = (s: Severity) => (s === "red" ? "t-red" : s === "amber" ? "t-amber" : s === "neutral" ? "t-neutral" : "t-info");

export function Unverified({ verified }: { verified: boolean | null }) {
  if (verified !== false) return null;
  return (
    <span className="today-unverified" title="Transcribed from the agency; no person has checked it against the source yet" data-testid="not-verified">
      Not verified
    </span>
  );
}

export function Section({ id, area, title, aside, children, testid }: { id: string; area: string; title: string; aside?: ReactNode; children: ReactNode; testid?: string }) {
  return (
    <section className="today-sec" data-area={area} aria-labelledby={`${id}-h`} id={id} data-testid={testid ?? `today-${id}`}>
      <header>
        <h2 id={`${id}-h`} className="today-h2">
          {title}
        </h2>
        {aside}
      </header>
      {children}
    </section>
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
