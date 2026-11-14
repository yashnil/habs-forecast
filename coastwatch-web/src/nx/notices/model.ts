/**
 * Every current official notice, for a port and for the rest of the coast: the v12 list (test-migration.md,
 * "Missing in v12" item 3). PURE: it reads the published registry (lib/official, lib/applicability) and never
 * decides anything the registry does not say. A record an agency release appears to lift is "sources differ"
 * (T40), never shown as lifted or in effect; an expected end date that has passed is said, never treated as an
 * end; no record is ever presented as verified (T38): the registry's own review state is quoted.
 */
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { associationsAt } from "@/lib/applicability";
import { ACTION_LABEL, criticalFlags, currentRecords, disputedRecords, DISCREPANCY_PREFIX, FISHERY_LABEL, type Dispute } from "@/lib/official";

export type NoticeItem = {
  id: string;
  agency: string;
  title: string;
  summary: string;
  /** "Health advisory", "Fishery closure" */
  action: string;
  /** "Sport harvest", "Commercial fishery" */
  fishery: string;
  species: string[];
  area: string;
  /** the official latitude band, when the notice's area is one (drawn on the chart as a bracket) */
  band: { south: number; north: number } | null;
  statewide: boolean;
  effective: string | null;
  effectiveNote: string | null;
  expectedEnd: string | null;
  expectedEndNote: string | null;
  sources: { label: string; url: string }[];
  /** always shown beside the record: an expected end that has passed */
  flags: string[];
  /** why it is listed for the port, and what is not settled about its reach */
  basis: string | null;
  caveats: string[];
  dispute: Dispute | null;
};

export type NoticesView = {
  port: { key: string; name: string } | null;
  /** notices whose official geography covers the port's waters, statewide ones included */
  here: NoticeItem[];
  /** notices whose reach at the port the registry does not settle */
  unsettled: NoticeItem[];
  /** records an agency release appears to lift: neither in effect nor lifted until a person reviews them */
  differ: NoticeItem[];
  /** every other current notice on the coast */
  elsewhere: NoticeItem[];
  /** other contradictions the registry publishes, in its own words */
  conflicts: string[];
  review: { status: string; reviewedAt: string | null; checkedAt: string | null; method: string | null };
  total: number;
};

function item(r: OfficialRecord, now: Date, extra: { basis?: string | null; caveats?: string[]; dispute?: Dispute | null } = {}): NoticeItem {
  return {
    id: r.id,
    agency: r.agency,
    title: r.title,
    summary: r.summary,
    action: ACTION_LABEL[r.action] ?? r.action,
    fishery: FISHERY_LABEL[r.fishery] ?? r.fishery,
    species: [...(r.species ?? [])],
    area: r.area.description,
    band: r.area.type === "lat_band" && r.area.lat_south != null && r.area.lat_north != null ? { south: r.area.lat_south, north: r.area.lat_north } : null,
    statewide: r.area.type === "statewide",
    effective: r.effective_date ?? null,
    effectiveNote: r.effective_date_note ?? null,
    expectedEnd: r.expected_end_date ?? null,
    expectedEndNote: r.expected_end_note ?? null,
    sources: r.sources.filter((s) => /^https:\/\//.test(s.url)).map((s) => ({ label: s.label, url: s.url })),
    flags: extra.dispute ? [] : criticalFlags(r, now),
    basis: extra.basis ?? null,
    caveats: extra.caveats ?? [],
    dispute: extra.dispute ?? null,
  };
}

/** Newest first; records without a date after those with one, then by title. */
const order = (a: NoticeItem, b: NoticeItem) => (b.effective ?? "").localeCompare(a.effective ?? "") || a.title.localeCompare(b.title);

export function buildNotices(ds: OfficialDataset | null, port: { key: string; name: string; lat: number; lon: number } | null, now: Date): NoticesView | null {
  if (!ds) return null;
  const current = currentRecords(ds);
  const assoc = port ? associationsAt(ds, port.key, port.lat, port.lon) : [];
  const hereIds = new Set<string>();
  const here: NoticeItem[] = [];
  const unsettled: NoticeItem[] = [];
  for (const a of assoc) {
    if (a.dispute || !current.some((r) => r.id === a.record.id)) continue;
    if (a.tier === "geographic") {
      here.push(item(a.record, now, { basis: a.basis, caveats: a.caveats }));
      hereIds.add(a.record.id);
    } else if (a.tier === "unresolved") {
      unsettled.push(item(a.record, now, { basis: a.basis, caveats: a.caveats }));
      hereIds.add(a.record.id);
    }
  }
  const differ = disputedRecords(ds).map(({ record, dispute }) => item(record, now, { dispute }));
  const elsewhere = current.filter((r) => !hereIds.has(r.id)).map((r) => item(r, now));
  const review = ds.registry.review as { status: string; reviewed_at?: string | null; method?: string | null };
  return {
    port: port ? { key: port.key, name: port.name } : null,
    here: here.sort(order),
    unsettled: unsettled.sort(order),
    differ: differ.sort(order),
    elsewhere: elsewhere.sort(order),
    conflicts: ds.conflicts.filter((c) => !c.startsWith(DISCREPANCY_PREFIX)),
    review: { status: review.status, reviewedAt: review.reviewed_at ?? null, checkedAt: ds.watch.map((w) => w.checked_at).sort().at(-1) ?? null, method: review.method ?? null },
    // the coast's current notices, by the same rule as every other count (lib/official currentRecords): a record an
    // agency release appears to lift is listed under `differ`, never counted as in effect
    total: current.length,
  };
}

/** Today's one-line count for the notices row: "3 here, 8 elsewhere on the coast". */
export function noticesLine(v: NoticesView): string {
  const here = v.here.length + v.unsettled.length;
  const parts = [v.port ? `${here} at ${v.port.name}` : `${v.total} on the coast`];
  if (v.port) parts.push(`${v.elsewhere.length} elsewhere`);
  if (v.differ.length) parts.push(`${v.differ.length} where sources differ`);
  return parts.join(", ");
}
