import { OFFICIAL_LINKS } from "../content/links";
import { fmtKm } from "../lib/geo";
import { ageWords, fmtDay } from "../lib/time";
import { classRange, type DirState, fmtLat, shortAction } from "./shared";

/** Plain facts computed from the shared state; each direction phrases and arranges them its own way. */
export function facts(st: DirState) {
  const r = st.reading!;
  const covering = st.covering;
  const station = r.station!;
  const mr = r.model;
  const value = mr.status === "value" ? mr.value : null;
  const cls = value != null ? classRange(value) : null;
  const layer = st.layer!;
  const disputed = r.disputed[0] ?? null;
  const words = ["No", "One", "Two", "Three", "Four", "Five", "Six"];
  return {
    place: r.place.label,
    nCover: covering.length,
    nCoverWord: words[covering.length] ?? String(covering.length),
    covering,
    limit: st.limit,
    limitSpecies: st.limitNotice?.record.species[0] ?? null,
    limitSharedBy: st.limitNotice ? covering.filter((n) => n.nearestLimit && Math.abs(n.nearestLimit.lat - st.limitNotice!.nearestLimit!.lat) < 1e-4).length : 0,
    coverSummary: summarise(covering.map((n) => ({ species: (n.record.species[0] ?? "seafood").replace(/\s*\(.*?\)/g, ""), action: n.record.action }))),
    limitText: st.limit ? `${fmtLat(st.limit.lat)}, ${fmtKm(st.limit.km)} ${st.limit.which === "south" ? "south" : "north"} of this point` : null,
    station,
    stationText: `${station.station.name}, ${fmtKm(station.km)} ${station.bearing}`,
    sampled: station.lastDate ? `${fmtDay(station.lastDate, { weekday: true })} (${ageWords(station.lastDate)})` : "never",
    pDA: station.lastValue,
    pDADate: station.toxinDate,
    value,
    cls,
    validFor: layer.time.valid_date!,
    issued: layer.time.issued_date!,
    freshness: st.freshness,
    threshold: layer.threshold_text ?? "",
    disputed,
    gaps: [
      ...(mr.status === "gap" ? ["No forecast for this cell"] : []),
      ...(station.ageDays != null && station.ageDays > 7 ? [`Nothing measured at ${station.station.name} since ${fmtDay(station.lastDate!)}`] : []),
      "Toxin in seafood: only CDPH testing tells you that; CoastWatch does not have those results",
    ],
  };
}
export type Facts = ReturnType<typeof facts>;

export function noticeLabel(n: Facts["covering"][number]) {
  return `${n.record.agency} ${shortAction[n.record.action]?.toLowerCase() ?? n.record.action}`;
}

/** The official status line — present in every direction, never dismissible. */
export function StatusLine({ className }: { className?: string }) {
  return (
    <p className={`dir-status ${className ?? ""}`}>
      <span className="mark mark-rule" aria-hidden="true" />
      <span>Notices are copied from CDFW and CDPH and may be incomplete. Before harvesting or eating seafood, check{" "}
        <a href={OFFICIAL_LINKS.cdfw.url} target="_blank" rel="noopener" className="ext">CDFW</a> and{" "}
        <a href={OFFICIAL_LINKS.cdph.url} target="_blank" rel="noopener" className="ext">CDPH</a>.</span>
    </p>
  );
}

/** "mussels (quarantine) and northern anchovy (2 notices)" — grouped by the notices' own species. */
function summarise(items: { species: string; action: string }[]) {
  const g = new Map<string, string[]>();
  for (const it of items) g.set(it.species, [...(g.get(it.species) ?? []), it.action]);
  const parts = [...g].map(([sp, acts]) => (acts.length > 1 ? `${sp} (${acts.length} notices)` : `${sp} (${shortAction[acts[0]]?.toLowerCase() ?? acts[0]})`));
  return parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}` : parts[0] ?? "";
}
