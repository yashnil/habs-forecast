/**
 * v14 Today (the approved synthesis): the pure arrangements behind the ticket and the sea dock. Like model.ts it never
 * adds a fact: it splits the record's own statement into the poster's lead words and hero date, places the record's
 * dates on the dates line, and places NWS's periods, its advisories and the buoy's one measurement on the sea bars.
 *
 *   posterParts   "Season delayed to at least Jan 1, 2027" → lede "Season delayed to at least", hero "Jan 1, 2027"
 *   datesLine     the delay on record (effective → earliest start), today, the next assessment window
 *   seaBars       NWS periods as bars (day 6 AM–6 PM, night 6 PM–6 AM, Pacific), the buoy's point, advisory spans
 *   zoneCaption   the whale-zone tag's second line in the record's words ("Season delayed · Fleet Advisory")
 */
import type { LanePeriod } from "@/pilot/today/marine";
import type { GlyphKind } from "@/nx/ds/synth/Grammar";
import type { Provenance } from "@/nx/ds/Stamp";
import type { StatusLevel } from "@/pilot/contracts";

const MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const pad = (n: number) => String(n).padStart(2, "0");

// ---------------------------------------------------------------- the poster

export type Poster = { lede: string | null; hero: string; date: string | null };

/**
 * The statement as a poster: the date the record names is the hero. Only a trailing calendar date is lifted out
 * ("… to at least Jan 1, 2027", "… until December 31"); any other statement stays whole and has no hero.
 * `year` is the year to assume when the record leaves it out (the page's own year).
 */
export function posterParts(title: string, year?: number): Poster | null {
  const m = /^(.*?)\s+((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?)\.?$/i.exec(title.trim());
  if (!m || !m[1].trim()) return null;
  const mon = MON.indexOf(m[3].toLowerCase());
  const y = m[5] ? Number(m[5]) : year;
  const d = Number(m[4]);
  return { lede: m[1].trim(), hero: m[2], date: y && d >= 1 && d <= 31 ? `${y}-${pad(mon + 1)}-${pad(d)}` : null };
}

// ---------------------------------------------------------------- Pacific time

/** ms for a Pacific wall-clock time on a day ("2026-10-13", 6 → 6 AM PDT on Oct 13). */
export function pacificMs(day: string, hour: number): number {
  const [y, m, d] = day.split("-").map(Number);
  const wall = Date.UTC(y, m - 1, d, hour);
  // the offset on that day: try PST, correct by what Pacific clocks read then
  let t = wall + 8 * 3_600_000;
  const h = Number(new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", hourCycle: "h23" }).format(new Date(t)));
  const diff = ((h - hour + 36) % 24) - 12;
  t -= diff * 3_600_000;
  return t;
}
const dayMs = (day: string) => pacificMs(day, 12);

// ---------------------------------------------------------------- the dates line

export type DatesLineModel = {
  label: string;
  domain: [number, number];
  span: { from: number; to: number; fromLabel: string; toLabel: string } | null;
  window: { from: number; to: number } | null;
  today: { t: number };
  ticks: number[];
};

const short = (day: string) => {
  const [, m, d] = day.split("-").map(Number);
  return `${MON[m - 1].charAt(0).toUpperCase()}${MON[m - 1].slice(1)} ${d}`;
};

/**
 * The dates on record for a delay: when it took effect, today, the next assessment (dashed: subject to change) and the
 * earliest start the record names. Null when there is nothing to place but today.
 */
export function datesLine(o: { today: string; effective: string | null; earliest: string | null; window: { start: string; end: string | null } | null; windowWords?: string | null }): DatesLineModel | null {
  if (!o.earliest && !o.window) return null;
  const t = dayMs(o.today);
  const eff = o.effective && o.effective <= o.today ? dayMs(o.effective) : null;
  const earliest = o.earliest ? dayMs(o.earliest) : null;
  const win = o.window ? { from: pacificMs(o.window.start, 0), to: pacificMs(o.window.end ?? o.window.start, 24) } : null;
  const lo = Math.min(t, eff ?? t, win?.from ?? t);
  const hi = Math.max(t, earliest ?? t, win?.to ?? t);
  const span = Math.max(hi - lo, 14 * 86_400_000);
  const domain: [number, number] = [lo - span * 0.03, hi + span * 0.2];
  const ticks: number[] = [];
  const start = new Date(domain[0]);
  for (let y = start.getUTCFullYear(), m = start.getUTCMonth() + 1; ; m++) {
    if (m > 11) {
      m = 0;
      y++;
    }
    const tick = dayMs(`${y}-${pad(m + 1)}-01`);
    if (tick > domain[1]) break;
    ticks.push(tick);
  }
  const words = [
    eff && o.effective ? `took effect ${short(o.effective)}` : null,
    `today ${short(o.today)}`,
    o.window ? `${o.windowWords ?? "next assessment"} ${short(o.window.start)}${o.window.end && o.window.end !== o.window.start ? ` to ${short(o.window.end)}` : ""}` : null,
    o.earliest ? `earliest start ${short(o.earliest)}` : null,
  ].filter(Boolean);
  return {
    label: `Dates on record: ${words.join("; ")}.`,
    domain,
    span: earliest ? { from: eff ?? t, to: earliest, fromLabel: eff && o.effective ? short(o.effective) : "", toLabel: "earliest start" } : null,
    window: win,
    today: { t },
    ticks,
  };
}

// ---------------------------------------------------------------- the sea bars

export type SeaBarsModel = {
  label: string;
  domain: [number, number];
  periods: { id: string; from: number; to: number; lo: number; hi: number; name: string; text: string; night: boolean }[];
  measured: { t: number; v: number; label: string; aria: string } | null;
  advisories: { id: string; from: number; to: number; label: string; level: 1 | 2 }[];
  days: { t: number; label: string; short: string }[];
  max: number;
};

const WD = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** NWS's flags: one pennant for a small craft advisory, two for a gale warning (and anything stronger). */
export const pennantsFor = (event: string): 1 | 2 => (/gale|storm|hurricane/i.test(event) ? 2 : 1);
/** The span's words over the bars: "Small craft" for a Small Craft Advisory, "Gale" for a Gale Warning. */
export const spanWords = (event: string) => {
  const w = event.replace(/\s+(advisory|warning|watch|statement)$/i, "").trim();
  return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
};

export function seaBars(o: {
  lanes: LanePeriod[];
  now: string;
  alerts: Array<{ event: string; onset: string | null; ends: string | null }>;
  buoy: { id: string; waveFt: number | null; at: string | null } | null;
  maxPeriods?: number;
}): SeaBarsModel | null {
  const lanes = o.lanes.filter((l) => l.date).slice(0, o.maxPeriods ?? 8);
  if (!lanes.length) return null;
  const slot = (l: LanePeriod) => (l.night ? { from: pacificMs(l.date!, 18), to: pacificMs(l.date!, 30) } : { from: pacificMs(l.date!, 6), to: pacificMs(l.date!, 18) });
  const all = lanes.map((l) => ({ l, ...slot(l) }));
  const domain: [number, number] = [all[0].from, all[all.length - 1].to];
  const periods = all
    .filter(({ l }) => l.seasHi != null)
    .map(({ l, from, to }) => ({ id: `${l.date}${l.night ? "n" : "d"}`, from, to, lo: l.seasLo ?? l.seasHi!, hi: l.seasHi!, name: l.name, text: l.text, night: l.night }));
  // each day's name stands under the middle of its own periods on the bars (never off their ends)
  const days = [...new Set(lanes.map((l) => l.date!))].map((d) => {
    const wd = WD[new Date(`${d}T12:00:00Z`).getUTCDay()];
    const own = all.filter((x) => x.l.date === d);
    return { t: (Math.min(...own.map((x) => x.from)) + Math.max(...own.map((x) => x.to))) / 2, label: `${wd} ${Number(d.slice(8))}`, short: wd };
  });
  const now = Date.parse(o.now);
  const b = o.buoy;
  const at = b?.at ? Date.parse(b.at) : null;
  const clock = b?.at ? new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" }).format(new Date(b.at)) : "";
  const measured =
    b && b.waveFt != null && at != null
      ? { t: Math.max(domain[0], Math.min(domain[1], at)), v: b.waveFt, label: `Buoy ${b.id} · ${clock}`, aria: `Measured at buoy ${b.id}, ${clock}: waves ${b.waveFt.toFixed(1)} ft. A measurement at one point; the bars are NWS's forecast.` }
      : null;
  const advisories = o.alerts
    .map((a, i) => {
      const from = Math.max(domain[0], a.onset ? Date.parse(a.onset) : now);
      const to = Math.min(domain[1], a.ends ? Date.parse(a.ends) : domain[1]);
      return { id: `${i}-${a.event}`, from, to, label: spanWords(a.event), level: pennantsFor(a.event) };
    })
    .filter((a) => a.to > a.from);
  const top = Math.max(0, ...periods.map((p) => p.hi), measured?.v ?? 0);
  const max = Math.max(8, Math.ceil(top * 1.15));
  const label = `Seas forecast by NWS for ${days.length} days${measured ? `, with the buoy's measured waves at ${clock}` : ""}${advisories.length ? ` and ${advisories.map((a) => a.label.toLowerCase()).join(", ")}` : ""}.`;
  return { label, domain, periods, measured, advisories, days, max };
}

// ---------------------------------------------------------------- marks and words

/** The data-state glyph for a provenance (SYSTEM.md grammar): official records are rings, measurements filled. */
export function glyphFor(p: Pick<Provenance, "kind" | "stale"> | null | undefined, coral = false): GlyphKind {
  if (!p) return "none";
  if (p.kind === "official") return coral ? "unverified-coral" : "unverified";
  if (p.kind === "measured") return p.stale ? "old" : "measured";
  return "predicted";
}

const LEVEL_WORDS: Partial<Record<StatusLevel, string>> = { delayed: "Season delayed", closed: "Closed", restricted: "Restricted" };

/** The whale-zone tag's second line: the season's state and the zone's RAMP action, each in the record's words. */
export function zoneCaption(level: StatusLevel | null, action: string | null): string | null {
  const parts = [level ? (LEVEL_WORDS[level] ?? null) : null, action?.trim() || null].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
/** Degrees the wind comes from, for its arrow; null for a word NDBC did not give. */
export const compassDeg = (dir: string | null | undefined): number | null => {
  const i = dir ? COMPASS.indexOf(dir.toUpperCase()) : -1;
  return i < 0 ? null : i * 22.5;
};

// ---------------------------------------------------------------- notices whose sources differ

export type NoticeLike = { id: string; agency: string; title: string };
export type ConflictLike = { notice: { agency: string; title: string }; record: { headline: string; source: string } } | null;

/**
 * Which listed notices may not be presented as in effect (T40): one the season record contradicts (a status row's
 * conflict names it). Each gets the words the /notices page and the Fisheries tab use: sources differ, awaiting a
 * person's review; never "lifted". (A record an agency release appears to lift, the registry's own discrepancy, never
 * reaches Today's list: src/pilot/today/load.ts leaves disputed records out, and /notices shows them as sources differ.)
 */
export function noticesDiffer(notices: NoticeLike[], conflicts: ConflictLike[]): Map<string, string> {
  const out = new Map<string, string>();
  const same = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();
  for (const n of notices) {
    const c = conflicts.find((x) => x && same(x.notice.title, n.title) && same(x.notice.agency, n.agency));
    if (c) out.set(n.id, `Sources differ: ${c.record.source}'s season record says “${c.record.headline.replace(/[.\s]+$/, "")}”. Awaiting a person's review.`);
  }
  return out;
}
