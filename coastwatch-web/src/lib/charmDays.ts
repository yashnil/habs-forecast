import type { ForecastRun, LayerArtifact, SourceStatus } from "@/generated/schema";
import type { Freshness } from "@/lib/freshness";
import { formatDate, formatDateTimePT, relativeDay } from "@/lib/time";

/**
 * One vocabulary for the days of a C-HARM run, used by every control that names them (the map's
 * day chips, the inspector strip, My Coast's day buttons and outlook bars, the map stamp).
 *
 * NOAA issues one run a day holding a "nowcast" (its analysis of the previous day) and forecasts
 * for the next three days; NOAA does not publish the issue date, so the pipeline infers it as the
 * nowcast's day + 1. Earlier controls mixed two vocabularies in one row ("nowcast" beside "today",
 * "yesterday" beside "+1 day"), which read as a contradiction when the nowcast is two days old.
 * Here every day chip carries its calendar date and its distance from today ("Oct 13 · 2 days ago");
 * whether a day is the nowcast or a forecast is said once, in the run line, never on the chips.
 */
export type CharmDay = {
  lead: number;
  /** YYYY-MM-DD, or null when the run did not issue this lead */
  date: string | null;
  /** "Oct 13" */
  short: string | null;
  /** "today", "yesterday", "2 days ago", "tomorrow", "in 2 days" (relative to Pacific today); null without a date */
  relative: string | null;
  /** NOAA's name for the lead: the nowcast is its analysis of that day, the rest are forecasts */
  role: "nowcast" | "forecast";
  issued: boolean;
};

const short = (d: string) => formatDate(d).replace(/^\w+, /, "");

/** The four days of a run as chips. `layers[lead]` is any layer of that lead (null when not issued). */
export function charmDays(layers: (Pick<LayerArtifact, "time"> | null | undefined)[], now: Date | null): CharmDay[] {
  return [0, 1, 2, 3].map((lead) => {
    const date = layers[lead]?.time.valid_date ?? null;
    return {
      lead,
      date,
      short: date ? short(date) : null,
      relative: date && now ? relativeDay(date, now) : null,
      role: lead === 0 ? "nowcast" : "forecast",
      issued: !!layers[lead],
    };
  });
}

/** "nowcast" / "forecast +2 days": the lead's role, for the run line and accessible names. */
export function roleWords(lead: number): string {
  return lead === 0 ? "nowcast" : `forecast +${lead} day${lead > 1 ? "s" : ""}`;
}

/**
 * The day to open on: the issued day valid today (Pacific); failing that, the first issued day
 * after today; failing that (a run too old to reach today), its last issued day. Never a day
 * before today while a later one exists, so a fresh run opens on today, not on yesterday's nowcast.
 */
export function defaultLead(days: CharmDay[], today: string): number {
  const issued = days.filter((d) => d.issued && d.date);
  if (!issued.length) return 0;
  const on = issued.find((d) => d.date === today);
  if (on) return on.lead;
  const after = issued.find((d) => d.date! > today);
  if (after) return after.lead;
  return issued[issued.length - 1].lead;
}

/**
 * "Run of Oct 14 (date inferred): nowcast Oct 13, forecasts Oct 14–16." One sentence that explains
 * why the nowcast's date precedes the run's, said once per panel.
 */
export function runWords(run: Pick<ForecastRun, "issued_date" | "issued_date_derived"> | null, days: CharmDay[]): string {
  if (!run) return "No C-HARM run";
  const now = days.find((d) => d.lead === 0 && d.date);
  const fc = days.filter((d) => d.lead > 0 && d.date);
  const parts = [`Run of ${short(run.issued_date)}${run.issued_date_derived ? " (date inferred)" : ""}`];
  const inner: string[] = [];
  if (now) inner.push(`nowcast ${now.short}`);
  if (fc.length) inner.push(`forecast${fc.length > 1 ? "s" : ""} ${fc.length > 1 ? `${fc[0].short}–${fc[fc.length - 1].short}` : fc[0].short}`);
  return inner.length ? `${parts[0]}: ${inner.join(", ")}` : parts[0];
}

// ---------------------------------------------------------------- the run's state, in one voice

export type RunNote = {
  /** caution: ordinary latency, said plainly; problem: out of date, or the update could not be made */
  tone: "caution" | "problem";
  /** short head, e.g. "No newer run since Oct 12" */
  head: string;
  text: string;
  testid: "stale-note" | "historical-note" | "update-failed";
};

/**
 * What to say about a run that is not current, in words that match the failure: NOAA often skips a
 * day, so a run a few days old is ordinary latency (caution, not alarm); a run past the policy's
 * window is out of date (problem); an update Neritic could not make is a problem of its own, said
 * with the time it was attempted. Returns the notes in the order to show them; none when current.
 */
export function runNotes(fresh: Freshness | null, status: Pick<SourceStatus, "outcome" | "last_attempt_at"> | null | undefined, policyMaxStaleDays = 7): RunNote[] {
  const out: RunNote[] = [];
  if (fresh?.state === "stale" && fresh.basisDate) {
    const age = fresh.ageDays ?? 0;
    out.push({
      tone: "caution",
      head: `No newer run since ${short(fresh.basisDate)} (${age === 1 ? "1 day" : `${age} days`} ago)`,
      text: `NOAA's C-HARM runs are often missing a day or more. The days after ${short(fresh.basisDate)} are that run's forecasts, with their own dates.`,
      testid: "stale-note",
    });
  }
  if (fresh?.state === "historical" && fresh.basisDate) {
    out.push({
      tone: "problem",
      head: "Out of date",
      text: `The newest run Neritic has is from ${short(fresh.basisDate)}, more than ${policyMaxStaleDays} days ago. It is shown for reference only and does not describe current conditions.`,
      testid: "historical-note",
    });
  }
  if (status?.outcome === "failed") {
    out.push({
      tone: "problem",
      head: "Update failed",
      text: `Neritic could not reach NOAA at ${formatDateTimePT(status.last_attempt_at)}, so a newer run may exist. This is the last run received, with its own dates.`,
      testid: "update-failed",
    });
  }
  return out;
}

/** The one line for a briefing: "run of Oct 12, 3 days old" (caution) or "out-of-date run of Oct 5" (problem). */
export function runAgeWords(fresh: Pick<Freshness, "state" | "basisDate" | "ageDays">): string {
  const age = fresh.ageDays == null ? "" : fresh.ageDays === 1 ? "1 day old" : `${fresh.ageDays} days old`;
  const of = fresh.basisDate ? `run of ${short(fresh.basisDate)}` : "run";
  if (fresh.state === "historical") return `out-of-date ${of}${age ? `, ${age}` : ""}`;
  return `${of}${age ? `, ${age}` : ""}`;
}
