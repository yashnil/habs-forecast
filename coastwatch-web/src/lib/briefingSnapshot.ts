import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { AUDIENCE, associationsAt } from "@/lib/applicability";
import { resolvePlace, type Audience } from "@/lib/coast";
import { distanceKm } from "@/lib/conditions";
import type { StationLite } from "@/lib/data";
import { disputes } from "@/lib/official";
import { recordNamesSpecies, speciesById } from "@/content/species";

/**
 * A briefing snapshot: what Neritic's published data say about one place, activity and species,
 * with every date the data carry. It is the unit "Since you last checked" compares and the content of
 * a kept briefing. Built only from CoastWatch's own published artifacts (no third-party request), and
 * small enough to keep on the device.
 *
 * Dates kept apart, never merged:
 * - agency dates (a release's published date, a notice's effective or lifted date);
 * - sample dates (when the water was sampled, not when the result was published);
 * - model issue dates;
 * - CoastWatch's own times (when it last checked agency pages, when its data were generated).
 */
export const SNAPSHOT_VERSION = 1;
/** stations whose samples are listed with the waters (the HAB briefing's evidence radius) */
export const SNAPSHOT_STATION_KM = 60;
export const SNAPSHOT_MAX_STATIONS = 4;
/** the sources whose health changes how the briefing should be read */
export const SNAPSHOT_SOURCES = ["official", "calhabmap", "charm", "satellite_chl"] as const;

export type NoticeTier = "applies" | "check";
export type SnapNotice = {
  id: string;
  agency: string;
  title: string;
  status: OfficialRecord["status"];
  /** the agency's own dates (YYYY-MM-DD); null when the agency gives none */
  published: string | null;
  effective: string | null;
  lifted: string | null;
  url: string | null;
  tier: NoticeTier;
  /** names the chosen species and suits the chosen activity */
  forYou: boolean;
};
export type SnapStatement = { id: string; agency: string; topic: string; statement: string; url: string };
export type SnapDispute = { recordId: string; agency: string; release: string; title: string; url: string };
export type SnapPage = { id: string; label: string; url: string; hash: string | null; checkedAt: string; ok: boolean };
export type SnapStation = { id: string; name: string; km: number; url: string; toxin: string | null; cells: string | null };
export type SnapSource = { id: string; title: string; outcome: string; lastSuccess: string | null; lastAttempt: string | null; latestValid: string | null; latestIssued: string | null };

export type Snapshot = {
  v: typeof SNAPSHOT_VERSION;
  /** when this snapshot was built (server time) */
  builtAt: string;
  /** when Neritic's data were generated (manifest) */
  dataGeneratedAt: string;
  place: { key: string; name: string; county: string | null; lat: number; lon: number };
  audience: Audience;
  species: { id: string; label: string };
  official: {
    /** false when the notice list could not be loaded: nothing below is complete */
    ok: boolean;
    /** when Neritic last checked the agencies' pages */
    checkedAt: string | null;
    reviewStatus: string | null;
    reviewedAt: string | null;
    notices: SnapNotice[];
    statements: SnapStatement[];
    disputes: SnapDispute[];
    pages: SnapPage[];
    hotlines: { label: string; phone: string }[];
  };
  stations: SnapStation[];
  forecast: { issued: string | null; outcome: string | null } | null;
  sources: SnapSource[];
};

export type SnapshotInput = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
};

const TOXIN = ["pDA"];
const CELLS = ["pn_seriata", "pn_delicatissima"];

function latestOf(st: StationLite, vars: string[]): string | null {
  const dates = st.latest.filter((l) => vars.includes(l.variable)).map((l) => l.date.slice(0, 10));
  return dates.length ? dates.sort().at(-1)! : null;
}

function sourceUrl(r: OfficialRecord): string | null {
  return r.sources.find((s) => s.url.startsWith("https://"))?.url ?? null;
}

function publishedDate(r: OfficialRecord): string | null {
  const d = r.sources.map((s) => s.published_date).filter((x): x is string => !!x).sort();
  return d.at(-1) ?? null;
}

/**
 * Notices at a place, including lifted and superseded ones, so a later snapshot can say a notice was
 * lifted instead of only "no longer listed". Geography uses the same applicability model as My Coast.
 */
function noticesAt(ds: OfficialDataset, key: string, lat: number, lon: number, audience: Audience, speciesId: string): SnapNotice[] {
  const species = speciesById(speciesId);
  const status = new Map(ds.registry.records.map((r) => [r.id, r.status]));
  const all: OfficialDataset = { ...ds, registry: { ...ds.registry, records: ds.registry.records.map((r) => ({ ...r, status: "active" as const })) } };
  return associationsAt(all, key, lat, lon)
    .filter((a) => a.tier !== "outside")
    .map((a) => {
      const r = a.record;
      const fits = audience === "all" || AUDIENCE[r.fishery].groups.includes(audience);
      return {
        id: r.id,
        agency: r.agency,
        title: r.title,
        status: status.get(r.id) ?? r.status,
        published: publishedDate(r),
        effective: r.effective_date ?? null,
        lifted: r.lifted_date ?? null,
        url: sourceUrl(r),
        tier: a.tier === "geographic" ? ("applies" as const) : ("check" as const),
        forYou: fits && recordNamesSpecies(r.species, species),
      };
    });
}

function statementsFor(ds: OfficialDataset, audience: Audience, speciesId: string): SnapStatement[] {
  const topic = speciesById(speciesId).statementTopic;
  if (!topic) return [];
  const forActivity = (t: string) => audience === "all" || audience === "shellfish" || !/\b(commercial|recreational)\b/i.test(t) || t.toLowerCase().includes(audience);
  return ds.registry.statements
    .filter((s) => s.topic.toLowerCase().includes(topic) && forActivity(s.topic))
    .map((s) => ({ id: s.id, agency: s.agency, topic: s.topic, statement: s.statement, url: s.source.url }));
}

/** Build the snapshot for a place key, activity and species; null when the place is unknown. */
export function buildSnapshot(input: SnapshotInput, placeKey: string, audience: Audience, speciesId: string, now: Date): Snapshot | null {
  const place = resolvePlace(placeKey, input.ports);
  if (!place) return null;
  const ds = input.official;
  const species = speciesById(speciesId);
  const stations = (input.stations ?? [])
    .map((s) => ({ s, km: distanceKm(place.lat, place.lon, s.lat, s.lon) }))
    .filter((x) => x.km <= SNAPSHOT_STATION_KM)
    .sort((a, b) => a.km - b.km)
    .slice(0, SNAPSHOT_MAX_STATIONS)
    .map(({ s, km }) => ({ id: s.station_id, name: s.name, km: Math.round(km * 10) / 10, url: s.source_url, toxin: latestOf(s, TOXIN), cells: latestOf(s, CELLS) }));
  const run = input.manifest.forecast_runs.find((r) => r.source_id === "charm") ?? null;
  const charm = input.manifest.sources.find((s) => s.source_id === "charm") ?? null;
  return {
    v: SNAPSHOT_VERSION,
    builtAt: now.toISOString(),
    dataGeneratedAt: input.manifest.generated_at,
    place: { key: place.key, name: place.name, county: place.county, lat: place.lat, lon: place.lon },
    audience,
    species: { id: species.id, label: species.label },
    official: ds
      ? {
          ok: true,
          checkedAt: ds.watch.map((w) => w.checked_at).sort().at(-1) ?? null,
          reviewStatus: ds.registry.review.status,
          reviewedAt: ds.registry.review.reviewed_at,
          notices: noticesAt(ds, place.key, place.lat, place.lon, audience, species.id),
          statements: statementsFor(ds, audience, species.id),
          disputes: disputes(ds).map((d) => ({ recordId: d.recordId, agency: d.agency, release: d.release, title: d.title, url: d.url })),
          pages: ds.watch.map((w) => ({ id: w.source_id, label: ds.registry.watched_sources.find((s) => s.id === w.source_id)?.label ?? w.source_id, url: w.url, hash: w.content_hash ?? null, checkedAt: w.checked_at, ok: w.ok })),
          hotlines: hotlines(ds),
        }
      : { ok: false, checkedAt: null, reviewStatus: null, reviewedAt: null, notices: [], statements: [], disputes: [], pages: [], hotlines: [] },
    stations,
    forecast: run || charm ? { issued: run?.issued_date ?? null, outcome: charm?.outcome ?? null } : null,
    sources: SNAPSHOT_SOURCES.flatMap((id) => {
      const s = input.manifest.sources.find((x) => x.source_id === id);
      return s ? [{ id, title: s.title, outcome: s.outcome, lastSuccess: s.last_success_at ?? null, lastAttempt: s.last_attempt_at ?? null, latestValid: s.latest_valid_date ?? null, latestIssued: s.latest_issued_date ?? null }] : [];
    }),
  };
}

function hotlines(ds: OfficialDataset): { label: string; phone: string }[] {
  return (ds.registry.hotlines ?? []).flatMap((h) => (h.label && h.phone ? [{ label: h.label, phone: h.phone }] : []));
}
