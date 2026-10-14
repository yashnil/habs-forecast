import { ageDays } from "@/lib/coast";
import { pacificToday } from "@/lib/time";
import type { Snapshot, SnapNotice } from "@/lib/briefingSnapshot";

/**
 * "Since you last checked": the differences between the snapshot a person saw last time (kept on
 * their device) and the current one, for the same waters.
 *
 * The rule that matters most: **a notice is "issued since your last check" only when the agency's own
 * date is on or after that check.** A notice CoastWatch lists for the first time, but which the agency
 * dated earlier, is "newly listed by CoastWatch" with the agency's date. It is never called new. Every
 * item names its source and the date that applies (agency date, sample date, model issue date, or
 * CoastWatch's check time), and nothing here says waters are open, closed, safe or unsafe.
 */
export type ChangeKind =
  | "agency-issued" // a notice the agency dated on or after the last check
  | "agency-lifted" // a notice the agency lifted on or after the last check
  | "agency-statement" // an agency status statement now reads differently
  | "release-pending" // an agency release appears to lift a notice; not reviewed by a person
  | "listed-late" // CoastWatch now lists (or lists as lifted) something the agency dated before the last check
  | "removed" // no longer in CoastWatch's list here; reason unknown
  | "observation-new" // a newer shore-station sample
  | "forecast-new" // a newer model run
  | "source-outage" // a source failed, or could not be read
  | "became-stale" // information crossed an age limit since the last check
  | "unavailable" // information present last time is missing now
  | "page-changed" // an agency page's content changed; not interpreted
  | "source-recovered";

export type ChangeGroup = "agency" | "coastwatch" | "observation" | "forecast" | "data";

export type Change = {
  kind: ChangeKind;
  group: ChangeGroup;
  /** concerns the chosen activity and species (false: other species or activities in these waters) */
  forYou: boolean;
  title: string;
  detail: string;
  /** the dates that apply, labelled ("CDPH dated", "sampled", "issued", "CoastWatch checked") */
  dates: { label: string; date: string }[];
  source: { label: string; url: string } | null;
};

export type DiffOptions = {
  /** formats a YYYY-MM-DD date */
  fmtDate: (d: string) => string;
  /** formats an ISO date-time */
  fmtTime: (iso: string) => string;
};

/** a result older than this is "not recent" (the same limit My Coast's next steps use for toxin samples) */
export const SAMPLE_STALE_DAYS = 30;
/** agency pages not checked for this long are stale (lib/official WATCH_STALE_DAYS) */
export const CHECK_STALE_DAYS = 2;

const ORDER: ChangeKind[] = [
  "agency-statement",
  "agency-issued",
  "agency-lifted",
  "release-pending",
  "removed",
  "listed-late",
  "source-outage",
  "unavailable",
  "became-stale",
  "observation-new",
  "forecast-new",
  "page-changed",
  "source-recovered",
];

const agencyDate = (n: SnapNotice) => n.published ?? n.effective;
const isActive = (n: SnapNotice) => n.status === "active";

/** Same waters? (Activity and species may differ: notices are compared for the place.) */
export function samePlace(a: Snapshot, b: Snapshot): boolean {
  return a.place.key === b.place.key;
}

export function diffSnapshots(prev: Snapshot, next: Snapshot, lastSeenAt: string, o: DiffOptions): Change[] {
  if (!samePlace(prev, next)) return [];
  const seenDay = pacificToday(new Date(lastSeenAt));
  const seenDate = new Date(lastSeenAt);
  const nowDate = new Date(next.builtAt);
  const out: Change[] = [];
  const add = (c: Change) => out.push(c);
  const link = (n: SnapNotice) => (n.url ? { label: `${n.agency} source`, url: n.url } : null);

  // ---------------------------------------------------------------- official notices
  if (prev.official.ok && !next.official.ok) {
    add({ kind: "unavailable", group: "data", forYou: true, title: "Official notices could not be loaded this time", detail: "Nothing below is complete. Check CDFW and CDPH directly.", dates: [], source: null });
  }
  if (prev.official.ok && next.official.ok) {
    const before = new Map(prev.official.notices.map((n) => [n.id, n]));
    const after = new Map(next.official.notices.map((n) => [n.id, n]));
    for (const n of next.official.notices) {
      const p = before.get(n.id);
      const d = agencyDate(n);
      if (!p) {
        if (isActive(n)) {
          if (d && d >= seenDay) {
            add({ kind: "agency-issued", group: "agency", forYou: n.forYou, title: `${n.agency}: ${n.title}`, detail: `${n.agency} dated it ${o.fmtDate(d)}, after your last check.${n.tier === "check" ? " Its geography does not settle whether it covers your waters." : ""}`, dates: dated(n), source: link(n) });
          } else {
            add({ kind: "listed-late", group: "coastwatch", forYou: n.forYou, title: `Newly listed by CoastWatch: ${n.title}`, detail: d ? `Not new from the agency: ${n.agency} dated it ${o.fmtDate(d)}, before your last check. CoastWatch had not listed it for these waters then.` : `${n.agency} gives no date for it. CoastWatch had not listed it for these waters at your last check.`, dates: dated(n), source: link(n) });
          }
        } else if (n.lifted && n.lifted >= seenDay) {
          add({ kind: "agency-lifted", group: "agency", forYou: n.forYou, title: `${n.agency} lifted: ${n.title}`, detail: `Lifted ${o.fmtDate(n.lifted)}.`, dates: dated(n), source: link(n) });
        }
        continue;
      }
      if (isActive(p) && !isActive(n)) {
        const when = n.lifted ?? d;
        const late = !when || when < seenDay;
        add({
          kind: late ? "listed-late" : "agency-lifted",
          group: late ? "coastwatch" : "agency",
          forYou: n.forYou,
          title: `${n.status === "superseded" ? "Replaced" : "Lifted"}: ${n.title}`,
          detail: late
            ? `CoastWatch now lists it as ${n.status === "superseded" ? "replaced" : "lifted"}. ${when ? `${n.agency} dated that ${o.fmtDate(when)}, before your last check.` : `${n.agency} gives no date for it.`}`
            : `${n.agency} dated it ${o.fmtDate(when!)}, after your last check.`,
          dates: dated(n),
          source: link(n),
        });
      }
    }
    for (const p of prev.official.notices) {
      if (after.has(p.id) || !isActive(p)) continue;
      add({ kind: "removed", group: "coastwatch", forYou: p.forYou, title: `No longer listed for these waters: ${p.title}`, detail: "CoastWatch no longer ties this notice to these waters, and does not know why. Check the agency before relying on it either way.", dates: [], source: link(p) });
    }

    // releases that appear to lift a notice here (awaiting a person's review)
    const here = new Set(next.official.notices.map((n) => n.id));
    const seenDisputes = new Set(prev.official.disputes.map((x) => `${x.recordId}|${x.release}`));
    for (const x of next.official.disputes) {
      if (!here.has(x.recordId) || seenDisputes.has(`${x.recordId}|${x.release}`)) continue;
      const n = after.get(x.recordId)!;
      add({ kind: "release-pending", group: "agency", forYou: n.forYou, title: `${x.agency} ${x.release} appears to lift: ${n.title}`, detail: `Found by CoastWatch at its check of ${next.official.checkedAt ? o.fmtTime(next.official.checkedAt) : "agency pages"}. No person has reviewed it yet; read the release.`, dates: next.official.checkedAt ? [{ label: "CoastWatch checked", date: next.official.checkedAt }] : [], source: { label: `${x.agency} ${x.release}`, url: x.url } });
    }

    // agency status statements (season status, "no toxin closures"), quoted
    const prevSt = new Map(prev.official.statements.map((s) => [s.id, s]));
    for (const s of next.official.statements) {
      const p = prevSt.get(s.id);
      if (p && p.statement === s.statement) continue;
      add({ kind: "agency-statement", group: "agency", forYou: true, title: `${s.agency} now says: “${s.statement}”`, detail: p ? `${s.topic}. At your last check it said: “${p.statement}”. The page gives no date for the change; CoastWatch saw it by ${next.official.checkedAt ? o.fmtTime(next.official.checkedAt) : "its latest check"}.` : `${s.topic}. Not in CoastWatch's list at your last check.`, dates: next.official.checkedAt ? [{ label: "CoastWatch checked", date: next.official.checkedAt }] : [], source: { label: `${s.agency} page`, url: s.url } });
    }

    // agency pages: unreadable, or changed in ways the items above may not explain
    const prevPages = new Map(prev.official.pages.map((g) => [g.id, g]));
    const explained = out.some((c) => c.group === "agency" || c.kind === "listed-late");
    for (const g of next.official.pages) {
      const p = prevPages.get(g.id);
      if (!p) continue;
      if (p.ok && !g.ok) add({ kind: "source-outage", group: "data", forYou: true, title: `CoastWatch could not read ${g.label}`, detail: `At its check of ${o.fmtTime(g.checkedAt)}. Changes there may be missing here.`, dates: [{ label: "CoastWatch checked", date: g.checkedAt }], source: { label: g.label, url: g.url } });
      else if (!p.ok && g.ok) add({ kind: "source-recovered", group: "data", forYou: true, title: `CoastWatch can read ${g.label} again`, detail: `Read at ${o.fmtTime(g.checkedAt)}.`, dates: [{ label: "CoastWatch checked", date: g.checkedAt }], source: { label: g.label, url: g.url } });
      else if (p.ok && g.ok && p.hash && g.hash && p.hash !== g.hash && !explained)
        add({ kind: "page-changed", group: "coastwatch", forYou: true, title: `${g.label} changed`, detail: `Its content differs from your last check (seen by CoastWatch at ${o.fmtTime(g.checkedAt)}). CoastWatch has not reviewed what changed; read the page.`, dates: [{ label: "CoastWatch checked", date: g.checkedAt }], source: { label: g.label, url: g.url } });
    }

    const nc = next.official.checkedAt;
    const pc = prev.official.checkedAt;
    if (nc && pc && ageDays(pc.slice(0, 10), seenDate) <= CHECK_STALE_DAYS && ageDays(nc.slice(0, 10), nowDate) > CHECK_STALE_DAYS)
      add({ kind: "became-stale", group: "data", forYou: true, title: "CoastWatch has not checked the agencies' pages recently", detail: `Last checked ${o.fmtTime(nc)}. Later agency actions will not be shown here.`, dates: [{ label: "CoastWatch checked", date: nc }], source: null });
  }

  // ---------------------------------------------------------------- shore-station samples
  const prevSt = new Map(prev.stations.map((s) => [s.id, s]));
  for (const s of next.stations) {
    const p = prevSt.get(s.id);
    if (!p) continue;
    for (const [field, what] of [["toxin", "toxin (domoic acid) result"], ["cells", "Pseudo-nitzschia cell count"]] as const) {
      const a = p[field];
      const b = s[field];
      if (b && (!a || b > a))
        add({ kind: "observation-new", group: "observation", forYou: true, title: `New ${what} at ${s.name}`, detail: `Sampled ${o.fmtDate(b)}${b < seenDay ? ", before your last check; the result was published since" : ""}. ${Math.round(s.km)} km from these waters. A seawater measurement, not a test of seafood.`, dates: [{ label: "sampled", date: b }], source: { label: s.name, url: s.url } });
    }
  }
  for (const p of prev.stations) {
    if (!next.stations.some((s) => s.id === p.id)) add({ kind: "unavailable", group: "data", forYou: true, title: `${p.name}: no longer in CoastWatch's station data`, detail: "Its samples could not be loaded this time.", dates: [], source: { label: p.name, url: p.url } });
  }
  // the newest toxin result near these waters crossing the "not recent" limit
  const newest = (snap: Snapshot) => snap.stations.map((s) => s.toxin).filter((x): x is string => !!x).sort().at(-1) ?? null;
  const tp = newest(prev);
  const tn = newest(next);
  if (tp && tn && tp === tn && ageDays(tp, seenDate) <= SAMPLE_STALE_DAYS && ageDays(tn, nowDate) > SAMPLE_STALE_DAYS)
    add({ kind: "became-stale", group: "data", forYou: true, title: `No toxin result near these waters in over ${SAMPLE_STALE_DAYS} days`, detail: `The newest was sampled ${o.fmtDate(tn)}. No recent sample is not the same as no toxin.`, dates: [{ label: "sampled", date: tn }], source: null });

  // ---------------------------------------------------------------- forecast model and source health
  if (prev.forecast?.issued && next.forecast?.issued && next.forecast.issued > prev.forecast.issued)
    add({ kind: "forecast-new", group: "forecast", forYou: true, title: "Newer C-HARM model run", detail: `Issued ${o.fmtDate(next.forecast.issued)}. A model probability of bloom conditions, not a measurement of toxin.`, dates: [{ label: "issued", date: next.forecast.issued }], source: null });
  const prevSrc = new Map(prev.sources.map((s) => [s.id, s]));
  for (const s of next.sources) {
    const p = prevSrc.get(s.id);
    if (!p) continue;
    if (p.outcome !== "failed" && s.outcome === "failed")
      add({ kind: "source-outage", group: "data", forYou: true, title: `${s.title}: latest update failed`, detail: `CoastWatch tried at ${s.lastAttempt ? o.fmtTime(s.lastAttempt) : "its latest run"}${s.lastSuccess ? `; it is showing data from ${o.fmtTime(s.lastSuccess)}` : ""}.`, dates: s.lastAttempt ? [{ label: "attempted", date: s.lastAttempt }] : [], source: null });
    else if (p.outcome === "failed" && s.outcome !== "failed")
      add({ kind: "source-recovered", group: "data", forYou: true, title: `${s.title}: updating again`, detail: s.lastSuccess ? `Updated ${o.fmtTime(s.lastSuccess)}.` : "", dates: s.lastSuccess ? [{ label: "updated", date: s.lastSuccess }] : [], source: null });
  }

  return out.sort((a, b) => Number(b.forYou) - Number(a.forYou) || ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind));
}

function dated(n: SnapNotice): { label: string; date: string }[] {
  const out: { label: string; date: string }[] = [];
  if (n.published) out.push({ label: `${n.agency} dated`, date: n.published });
  if (n.effective && n.effective !== n.published) out.push({ label: "effective", date: n.effective });
  if (n.lifted) out.push({ label: "lifted", date: n.lifted });
  return out;
}
