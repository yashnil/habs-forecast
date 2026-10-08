import { OFFICIAL_STATUS } from "@/content/copy";

/**
 * Rank 1 of the information hierarchy. Shown first and always. CoastWatch does not yet
 * ingest closures/advisories, so the card states that plainly (rule R2: unknown is not
 * "open") and routes people to the official sources and hotlines.
 */
export function OfficialStatusCard() {
  return (
    <section
      data-testid="official-status"
      aria-labelledby="official-status-h"
      className="rounded-xl border border-hairline-strong bg-surface-2 p-4"
    >
      <div className="flex items-start gap-3">
        <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden className="mt-0.5 shrink-0 text-ink">
          <path d="M10 1.8l6.5 2.6v5.1c0 4.1-2.8 7.4-6.5 8.7-3.7-1.3-6.5-4.6-6.5-8.7V4.4L10 1.8z" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M10 6v4.5M10 13.2v.1" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
        <div className="min-w-0 space-y-2">
          <h2 id="official-status-h" className="text-[13px] font-semibold text-ink">
            {OFFICIAL_STATUS.heading}
          </h2>
          <p className="text-[12.5px] leading-relaxed text-ink-2">{OFFICIAL_STATUS.notTracked}</p>
          <p className="text-[12.5px] text-ink-2">{OFFICIAL_STATUS.instruction}</p>
          <ul className="space-y-1">
            {OFFICIAL_STATUS.links.map((l) => (
              <li key={l.href}>
                <a
                  href={l.href}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[12.5px] font-medium text-accent underline-offset-2 hover:underline"
                >
                  {l.label} ↗
                </a>
              </li>
            ))}
          </ul>
          <dl className="grid gap-1 border-t border-hairline pt-2 text-[12px]">
            {OFFICIAL_STATUS.hotlines.map((h) => (
              <div key={h.tel} className="flex flex-wrap justify-between gap-x-3">
                <dt className="text-ink-3">{h.label}</dt>
                <dd>
                  <a href={`tel:${h.tel}`} className="font-medium text-ink tabular hover:underline">
                    {h.phone}
                  </a>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}
