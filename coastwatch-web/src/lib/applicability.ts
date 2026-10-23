import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { APPLICABILITY, APPLICABILITY_KM, CDFW_REGIONS, type AssociationReview, type NoticePart } from "@/content/coast";
import { activeRecords, disputeOf, pointInGeometry, type Dispute } from "@/lib/official";

/**
 * Which official notices relate to a location, and on what evidence (docs/coastwatch/28 §3).
 *
 * Only the notices' own published geography is used: statewide scope, official latitude limits,
 * and the agencies' area outlines. No place name or county name is ever matched as text.
 *
 * Three tiers, so that being near a notice's area is never presented as the notice applying:
 *
 * - "geographic": the notice's official geography includes the location (statewide; inside its
 *   official latitudes; inside, or on the shore of, the agency's outline of its area).
 * - "unresolved": the notice may apply, but its official geography does not settle it: a point on
 *   the water off a county- or area-defined notice (no notice states how far offshore it reaches),
 *   a shore location within a few km of a generalized outline's edge, or a part of a notice whose
 *   area has no official outline (borrowed and approximate, see `noticeParts`).
 * - "outside": the documented boundary excludes the location: beyond the notice's official
 *   latitude limit, or a shore location clearly outside the agency's outline (another county, the
 *   islands seen from the mainland). Listed only because it is nearby, for people who also fish or
 *   harvest there; never shown as applying here.
 *
 * Shore locations (ports, harvest sites) and points on the water are treated differently because
 * county and area notices are drawn on land and say nothing about their offshore reach.
 *
 * Every association is "not reviewed" unless a person's review is recorded in content/coast.ts.
 */

export type Tier = "geographic" | "unresolved" | "outside";
/** a port or a harvest site is on the shore; a point chosen on the map is on the water */
export type LocationKind = "shore" | "water";
export type Review = { state: "not_reviewed" } | { state: "confirmed" | "rejected"; reviewer: string; reviewedAt: string; note: string };

export type Association = {
  record: OfficialRecord;
  tier: Tier;
  /** one sentence: why this notice is listed */
  basis: string;
  /** extra caution, e.g. a limit close by or an unstated offshore reach */
  caveats: string[];
  /** distance from the location to the notice's geography, when it is outside it */
  distanceKm: number | null;
  review: Review;
  /** an agency release appears to lift this notice (awaiting review): never shown as applying */
  dispute?: Dispute | null;
};

type Ring = [number, number][];
type Feature = { geometry: { type: string; coordinates: unknown }; properties: { record_ids: string[]; kind: string; basis?: string } };

const KM_PER_DEG = 111.32;
const toKm = (lat: number) => ({ x: KM_PER_DEG * Math.cos((lat * Math.PI) / 180), y: KM_PER_DEG });

function features(ds: OfficialDataset): Feature[] {
  return ((ds.geometry as { features?: Feature[] }).features ?? []).filter((f) => f && f.geometry && f.properties);
}

/** Distance (km) from a point to a polygon's boundary; 0 when inside. Local equirectangular projection. */
export function distanceToPolygonKm(lon: number, lat: number, g: { type: string; coordinates: unknown }): number {
  if (pointInGeometry(lon, lat, g)) return 0;
  const polys = g.type === "Polygon" ? [g.coordinates as Ring[]] : g.type === "MultiPolygon" ? (g.coordinates as Ring[][]) : [];
  const k = toKm(lat);
  let best = Infinity;
  for (const poly of polys)
    for (const ring of poly)
      for (let i = 1; i < ring.length; i++) {
        const ax = (ring[i - 1][0] - lon) * k.x, ay = (ring[i - 1][1] - lat) * k.y;
        const bx = (ring[i][0] - lon) * k.x, by = (ring[i][1] - lat) * k.y;
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy;
        const t = len2 > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2)) : 0;
        best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
      }
  return best;
}

const fmtKm = (km: number) => (km < 10 ? km.toFixed(1) : km.toFixed(0));
const latLabel = (lat: number) => `${Math.floor(lat)}°${((lat - Math.floor(lat)) * 60).toFixed(2).padStart(5, "0")}′ N`;

function reviewOf(location: string, noticeId: string, reviews: AssociationReview[]): Review {
  const r = reviews.find((x) => x.location === location && x.noticeId === noticeId);
  return r ? { state: r.decision, reviewer: r.reviewer, reviewedAt: r.reviewedAt, note: r.note } : { state: "not_reviewed" };
}

function offshoreUnstated(r: OfficialRecord): boolean {
  return (r.uncertainties ?? []).some((u) => /offshore/i.test(u)) || r.area.type === "county" || r.area.type === "named_area";
}

/** The outline(s) a notice is drawn with, from the published dataset (never from text). */
function outlinesFor(ds: OfficialDataset, recordId: string): Feature[] {
  return features(ds).filter((f) => (f.properties.kind === "county" || f.properties.kind === "named_area") && f.properties.record_ids.includes(recordId));
}

/**
 * Associations for one location. `location` is a stable id ("port-593", "site-clam-beach",
 * "point-36.950,-122.050") used only to look up recorded reviews.
 */
export function associationsAt(
  ds: OfficialDataset | null,
  location: string,
  lat: number,
  lon: number,
  model: { noticeParts: NoticePart[]; reviews: AssociationReview[] } = APPLICABILITY,
  kind: LocationKind = "shore",
): Association[] {
  if (!ds) return [];
  const out: Association[] = [];
  const kmPerDegLat = KM_PER_DEG;
  for (const r of activeRecords(ds)) {
    const review = reviewOf(location, r.id, model.reviews);
    const a = r.area;
    if (a.type === "statewide") {
      out.push({ record: r, tier: "geographic", basis: "Applies along the whole California coast.", caveats: [], distanceKm: null, review });
      continue;
    }
    if (a.type === "lat_band" && a.lat_south != null && a.lat_north != null) {
      const inside = lat >= a.lat_south && lat <= a.lat_north;
      const toS = Math.abs(lat - a.lat_south) * kmPerDegLat;
      const toN = Math.abs(lat - a.lat_north) * kmPerDegLat;
      if (inside) {
        const caveats: string[] = [];
        const near = Math.min(toS, toN);
        if (near <= APPLICABILITY_KM.nearLimit)
          caveats.push(`Its ${toS < toN ? "south" : "north"} limit (${latLabel(toS < toN ? a.lat_south : a.lat_north)}) is ${fmtKm(near)} km ${toS < toN ? "south" : "north"} of here; check which side you are on.`);
        if (offshoreUnstated(r)) caveats.push("The notice does not say how far offshore it applies.");
        out.push({ record: r, tier: "geographic", basis: `Inside the notice's official latitudes (${latLabel(a.lat_south)} to ${latLabel(a.lat_north)}).`, caveats, distanceKm: null, review });
      } else {
        const d = Math.min(toS, toN);
        if (d <= APPLICABILITY_KM.nearby)
          out.push({
            record: r,
            tier: "outside",
            basis: `Outside the notice's official latitudes: its ${lat > a.lat_north ? "north" : "south"} limit (${latLabel(lat > a.lat_north ? a.lat_north : a.lat_south)}) is ${fmtKm(d)} km ${lat > a.lat_north ? "south" : "north"} of here.`,
            caveats: [],
            distanceKm: d,
            review,
          });
      }
    }
    if (a.type === "county" || a.type === "named_area") {
      const outlines = outlinesFor(ds, r.id);
      if (!outlines.length) {
        out.push({ record: r, tier: "unresolved", basis: "Neritic has no outline for this notice's area, so whether it applies here cannot be resolved.", caveats: [], distanceKm: null, review });
        continue;
      }
      const d = Math.min(...outlines.map((f) => distanceToPolygonKm(lon, lat, f.geometry)));
      const name = a.description;
      if (d === 0 || d <= APPLICABILITY_KM.shore)
        out.push({ record: r, tier: "geographic", basis: d === 0 ? `Inside the agency's outline of ${name}.` : `On the shore of ${name} (${fmtKm(d)} km from the agency's generalized outline).`, caveats: [], distanceKm: d === 0 ? null : d, review });
      else if (kind === "water" && d <= APPLICABILITY_KM.nearby)
        out.push({ record: r, tier: "unresolved", basis: `${fmtKm(d)} km off the agency's outline of ${name}. The notice does not say how far offshore it applies, so whether it reaches these waters is unresolved.`, caveats: [], distanceKm: d, review });
      else if (kind === "shore" && d <= APPLICABILITY_KM.edge)
        out.push({ record: r, tier: "unresolved", basis: `${fmtKm(d)} km from the agency's generalized outline of ${name}; at this distance the outline cannot settle which side these waters are on.`, caveats: [], distanceKm: d, review });
      else if (d <= APPLICABILITY_KM.nearby)
        out.push({ record: r, tier: "outside", basis: `Outside the agency's outline of ${name}, which is ${fmtKm(d)} km away.`, caveats: [], distanceKm: d, review });
    }
  }
  // stated geography the registry does not encode, from the reviewable model: approximate only
  for (const part of model.noticeParts) {
    const r = activeRecords(ds).find((x) => x.id === part.noticeId);
    if (!r || out.some((x) => x.record.id === r.id && x.tier === "geographic")) continue;
    const outlines = outlinesFor(ds, part.outlineFrom);
    if (!outlines.length) continue;
    const d = Math.min(...outlines.map((f) => distanceToPolygonKm(lon, lat, f.geometry)));
    if (d > APPLICABILITY_KM.nearby) continue;
    const prev = out.findIndex((x) => x.record.id === r.id);
    // the part has no official boundary, so neither "applies" nor "outside" can be shown
    const assoc: Association = {
      record: r,
      tier: "unresolved",
      basis:
        d === 0
          ? `The notice also covers the ${part.part} ("${part.quote}"); these waters are inside Neritic's approximate outline of that area.`
          : `The notice also covers the ${part.part} ("${part.quote}"), about ${fmtKm(d)} km away. It gives no boundary for those waters, so whether they reach here is not stated.`,
      caveats: [`Neritic's outline for the ${part.part} is borrowed from another CDPH notice and is approximate for this one.`],
      distanceKm: d === 0 ? null : d,
      review: reviewOf(location, r.id, model.reviews),
    };
    if (prev >= 0) out.splice(prev, 1, assoc);
    else out.push(assoc);
  }
  for (const a of out) a.dispute = disputeOf(ds, a.record.id);
  const order = (x: Association) => ({ geographic: 0, unresolved: 1, outside: 2 })[x.tier] * 1000 + (x.distanceKm ?? 0);
  return out.sort((a, b) => order(a) - order(b));
}

/** CDFW recreational regulation region(s) at a latitude; two when within `edgeKm` of a boundary. */
export function cdfwRegionsAt(lat: number, edgeKm = 5) {
  const at = CDFW_REGIONS.filter((r) => lat <= r.north && lat > r.south);
  const near = CDFW_REGIONS.filter((r) => !at.includes(r) && (Math.abs(lat - r.north) * KM_PER_DEG <= edgeKm || Math.abs(lat - r.south) * KM_PER_DEG <= edgeKm));
  return { at, near };
}

/** Who a notice speaks to, in plain words, from its registry field. */
export const AUDIENCE: Record<OfficialRecord["fishery"], { label: string; groups: Array<"commercial" | "recreational" | "shellfish"> }> = {
  commercial: { label: "Commercial fishing", groups: ["commercial"] },
  recreational: { label: "Recreational fishing", groups: ["recreational", "shellfish"] },
  commercial_and_recreational: { label: "Commercial and recreational", groups: ["commercial", "recreational"] },
  sport_harvest: { label: "Recreational (sport) harvest", groups: ["recreational", "shellfish"] },
  consumption: { label: "Anyone eating it", groups: ["commercial", "recreational", "shellfish"] },
};

/**
 * The notices control's "local" view for any place on the map (a harbour, a chosen point or a model
 * cell): the notices whose official geography covers it and those whose reach there is not settled.
 * Pure; the same tiers as the briefing, so a point and a harbour are never counted differently.
 */
export function localNotices(ds: OfficialDataset | null, location: string, name: string, lat: number, lon: number, kind: LocationKind = "shore"): { name: string; covering: string[]; unsettled: string[] } | null {
  if (!ds) return null;
  const a = associationsAt(ds, location, lat, lon, APPLICABILITY, kind);
  return { name, covering: a.filter((x) => x.tier === "geographic").map((x) => x.record.id), unsettled: a.filter((x) => x.tier === "unresolved").map((x) => x.record.id) };
}
