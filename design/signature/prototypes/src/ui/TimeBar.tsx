import type { Layer } from "../data/types";
import { fmtDay, relDay, TODAY } from "../lib/time";

/**
 * Time, made honest. Two rows that are never merged:
 *   "For"     — the day each forecast describes (one button per lead day)
 *   "Issued"  — when that forecast was made: a single bracket under all four days, because one
 *               C-HARM run issues every lead. Changing the "for" day never changes "issued".
 * For the satellite, the row shows its window and that each pixel keeps its own date.
 * Keyboard: a radiogroup (←/→).
 */
export function TimeBar({ show, leads, lead, onLead, issued, freshness, satellite, loading, compact }: {
  show: "base" | "forecast" | "satellite";
  leads: { lead: number; date: string }[];
  lead: number;
  onLead: (l: number) => void;
  issued?: string;
  freshness?: string | null;
  satellite?: Layer;
  loading?: boolean;
  compact?: boolean;
}) {
  if (show === "base") return null;
  if (show === "satellite") {
    const comp = (satellite as unknown as { composite?: { newest_observed_date: string; oldest_observed_date: string; window_days: number } } | undefined)?.composite;
    return (
      <div className="timebar enter" aria-label="Satellite dates">
        <span className="timebar-k t-micro">Seen</span>
        <div className="timebar-window">
          <span className="mark mark-pixel" aria-hidden="true" />
          {comp ? <span className="t-label">Latest clear view per pixel, <b>{fmtDay(comp.oldest_observed_date)}–{fmtDay(comp.newest_observed_date)}</b>. Neighbouring pixels can be from different days; tap the water for its date.</span> : <span className="skel skel-line" style={{ width: 240 }} />}
        </div>
      </div>
    );
  }
  return (
    <div className="timebar enter" data-compact={compact || undefined} aria-label="Forecast dates">
      <span className="timebar-k t-micro">For</span>
      <div className="timebar-days" role="radiogroup" aria-label="Day the forecast describes">
        {leads.map((l, i) => (
          <button key={l.lead} role="radio" aria-checked={l.lead === lead} tabIndex={l.lead === lead ? 0 : -1} className="tday" data-today={l.date === TODAY || undefined}
            onClick={() => onLead(l.lead)}
            onKeyDown={(e) => {
              const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
              if (!d) return;
              e.preventDefault();
              const n = leads[Math.max(0, Math.min(leads.length - 1, i + d))];
              onLead(n.lead);
              (e.currentTarget.parentElement?.children[leads.indexOf(n)] as HTMLElement | undefined)?.focus();
            }}>
            <span className="tday-d">{compact ? fmtDay(l.date, { weekday: true }).replace(/,.*? (\d+)$/, " $1") : fmtDay(l.date, { weekday: true })}</span>
            <span className="tday-r t-micro">{l.lead === 0 ? "nowcast" : relDay(l.date)}</span>
          </button>
        ))}
      </div>
      <span className="timebar-k t-micro">Issued</span>
      <div className="timebar-issued">
        <span className="issued-bracket" aria-hidden="true" />
        <span className="t-meta">{loading ? "…" : <>{compact ? "Issued" : "One NOAA C-HARM run,"} {issued ? fmtDay(issued, { weekday: true }) : "—"}{freshness ? <> · <span className={`fresh fresh-${freshness.replace(" ", "-").toLowerCase()}`}>{freshness}</span></> : null}</>}</span>
      </div>
    </div>
  );
}
