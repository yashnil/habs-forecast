"use client";

import { motion, useReducedMotion } from "motion/react";
import type { Focus, OutlookDay, SatReading } from "@/components/coast/HabBriefing";
import { Mark, type MarkKind } from "@/components/ui/Mark";
import { ageWords, fmtMeasured, fmtPct, EVIDENCE_RADIUS_KM, NEIGHBOURHOOD_KM, type StationEvidence } from "@/lib/hab";
import { useWidth } from "@/lib/useWidth";

const DAY = 86400000;
const BACK = 90;

type Row = {
  id: "cells" | "toxin" | "satellite" | "model";
  kind: MarkKind;
  title: string;
  /** where it was measured or computed, with its distance */
  where: string;
  /** the newest value, in words */
  latest: string;
  /** no evidence of this kind describes these waters */
  far?: boolean;
  focus: Focus;
};

/**
 * HAB evidence near the waters on one time axis: the last 90 days of shore samples (Pseudo-nitzschia
 * cells, domoic acid in seawater), the latest clear satellite view, and NOAA's model days up to the
 * trip day. Each row is one source at one place, with its distance, so a sample 77 km away never reads
 * as a sample here. Rows open their source on the map. The limits are said once, under the chart.
 */
export function EvidenceTimeline({
  hab,
  sat,
  outlook,
  today,
  tripDate,
  focus,
  onFocus,
}: {
  hab: { cells: StationEvidence | null; toxin: StationEvidence | null } | null;
  sat: SatReading | "error";
  outlook: OutlookDay[] | null | "error";
  today: string | null;
  tripDate: string | null;
  focus: Focus;
  onFocus: (f: Focus) => void;
}) {
  // starts at a phone's width (never wider than the page before it is measured), then follows its box
  const [ref, w] = useWidth<HTMLDivElement>(300, 280);
  const reduce = useReducedMotion();
  if (!today || !hab) return <div className="h-[260px] animate-pulse rounded-2xl bg-surface-3/70" data-testid="evidence-loading" aria-label="Reading the evidence" role="status" />;

  const days = outlook && outlook !== "error" ? outlook : [];
  const tNow = Date.parse(`${today}T12:00:00Z`);
  const lastDay = [tripDate, ...days.map((d) => d.valid)].filter((d): d is string => !!d).reduce((a, b) => (b > a ? b : a), today);
  const t1 = Date.parse(`${lastDay}T12:00:00Z`) + DAY;
  const t0 = tNow - BACK * DAY;
  // two panels on one axis: the last three months, then (wider per day) the last week and the trip
  const tW = tNow - 7 * DAY;
  const LABEL = w < 460 ? 0 : 150; // phones: labels above each row
  const plotW = w - LABEL - 8;
  const GAP = 10;
  const pastW = plotW * 0.56;
  const xt = (t: number) => (t < tW ? LABEL + ((t - t0) / (tW - t0)) * pastW : LABEL + pastW + GAP + ((t - tW) / (t1 - tW)) * (plotW - pastW - GAP));
  const x = (iso: string) => xt(Date.parse(`${iso.slice(0, 10)}T12:00:00Z`));
  const dayW = (plotW - pastW - GAP) / ((t1 - tW) / DAY);
  const ROW = LABEL ? 52 : 74;
  const TOP = LABEL ? 0 : 22;

  const rows: (Row & { evidence?: StationEvidence | null })[] = [
    stationRow("cells", hab.cells),
    stationRow("toxin", hab.toxin),
    {
      id: "satellite",
      kind: "satellite",
      title: "Chlorophyll from space",
      where: sat && sat !== "error" ? sat.sensor : "NOAA CoastWatch satellites",
      latest:
        sat === "error"
          ? "Could not be read"
          : sat === null
            ? "Reading…"
            : sat.value == null
              ? sat.flagged
                ? `No open-water reading within ${sat.radiusKm} km`
                : `No clear view in ${sat.windowDays} days (cloud or fog)`
              : `${Number(sat.value.toPrecision(2))} mg/m³${sat.basis ? ` · median within ${sat.basis.radiusKm} km` : ""}${sat.date ? ` · ${shortDate(sat.date)}` : ""}`,
      focus: "satellite",
      far: !sat || sat === "error" || sat.value == null,
    },
    {
      id: "model",
      kind: "model",
      title: "NOAA model (toxin)",
      where: days.some((d) => d.pda?.here != null)
        ? "these waters' own 3 km cell"
        : days.some((d) => d.pda && d.pda.n > 0)
          ? `offshore cells within ${NEIGHBOURHOOD_KM} km, not these waters`
          : "C-HARM",
      latest: outlook === null ? "Reading…" : outlook === "error" || !days.length ? "No model run available" : modelLatest(days, tripDate),
      focus: "model",
      far: !days.some((d) => d.pda && (d.pda.here != null || d.pda.n > 0)),
    },
  ];
  const H = TOP + rows.length * ROW + 26;
  const tripX = tripDate ? x(tripDate) : null;
  const weekX0 = LABEL + pastW + GAP;
  const weekDays = Array.from({ length: Math.round((t1 - tW) / DAY) }, (_, i) => tW + (i + 0.5) * DAY).filter((t) => t < t1);

  return (
    <div ref={ref} className="relative" data-testid="evidence-timeline">
      <svg width={w} height={H} style={{ maxWidth: "100%" }} viewBox={`0 0 ${w} ${H}`} role="img" aria-labelledby="ev-desc" className="block overflow-visible">
        <desc id="ev-desc">{rows.map((r) => `${r.title}, ${r.where}: ${r.latest}.`).join(" ")}</desc>
        {/* the week panel: a finer scale, tinted so the change of scale is visible */}
        <rect x={weekX0 - 2} y={0} width={w - 8 - weekX0 + 2} height={H - 22} rx={6} fill="var(--cw-surface-3)" opacity={0.7} />
        {/* the trip day */}
        {tripX != null && (
          <g data-testid="ev-trip">
            <rect x={tripX - dayW / 2 + 1} y={0} width={Math.max(6, dayW - 2)} height={H - 22} fill="var(--cw-accent)" opacity={0.14} rx={4} />
            <text x={tripX} y={H - 6} textAnchor="middle" className="fill-accent text-[11px] font-semibold">
              trip
            </text>
          </g>
        )}
        {/* today */}
        <line x1={x(today) + dayW / 2} x2={x(today) + dayW / 2} y1={0} y2={H - 22} stroke="var(--cw-ink)" strokeOpacity={0.4} strokeDasharray="3 3" />
        {[90, 60, 30].map((d) => (
          <text key={d} x={xt(tNow - d * DAY)} y={H - 6} textAnchor={d === 90 ? "start" : "middle"} className="fill-ink-3 text-[11px]">
            {d / 30} mo{LABEL || d === 90 ? " ago" : ""}
          </text>
        ))}
        {weekDays.map((t) =>
          Math.abs(t - Date.parse(`${tripDate}T12:00:00Z`)) < DAY / 2 ? null : (
            <text key={t} x={xt(t)} y={H - 6} textAnchor="middle" className="fill-ink-3 text-[10.5px]">
              {Math.abs(t - tNow) < DAY / 2
                ? "today"
                : dayW >= 26
                  ? new Date(t).toLocaleDateString("en-US", {
                      day: "numeric",
                      timeZone: "UTC",
                    })
                  : ""}
            </text>
          ),
        )}
        {rows.map((r, i) => {
          const y0 = TOP + i * ROW;
          const mid = y0 + ROW / 2 + (LABEL ? 0 : 8);
          const on = focus === r.focus;
          return (
            <g key={r.id} data-testid={`ev-row-${r.id}`}>
              <line x1={LABEL} x2={w - 8} y1={mid} y2={mid} stroke={on ? "var(--cw-accent)" : "var(--cw-hairline-strong)"} strokeWidth={on ? 1.5 : 1} strokeDasharray={r.far ? "2 4" : undefined} />
              {r.far && <rect x={LABEL} y={mid - 9} width={plotW} height={18} fill="url(#ev-hatch)" opacity={0.55} rx={4} />}
              {(r.id === "cells" || r.id === "toxin") && r.evidence && !r.far && <Dots e={r.evidence} x={x} t0={t0} mid={mid} reduce={!!reduce} label={LABEL} />}
              {r.id === "satellite" && sat && sat !== "error" && sat.value != null && sat.date && Date.parse(sat.date) >= t0 && (
                <rect x={x(sat.date) - 5} y={mid - 5} width={10} height={10} fill="var(--cw-chl)" />
              )}
              {r.id === "model" &&
                days.map((d) => {
                  const v = d.pda?.here ?? d.pda?.median;
                  if (!d.valid || v == null) return null;
                  const bh = 4 + v * (ROW - 28);
                  const bw = Math.min(14, Math.max(4, dayW - 6));
                  return (
                    <motion.rect
                      key={`${d.lead}-${d.valid}`}
                      x={x(d.valid) - bw / 2}
                      y={mid + 9 - bh}
                      width={bw}
                      height={bh}
                      rx={1.5}
                      fill={d.pda?.here != null ? "var(--cw-forecast)" : "url(#ev-hatch-model)"}
                      stroke={d.pda?.here != null ? undefined : "var(--cw-forecast)"}
                      strokeWidth={d.pda?.here != null ? 0 : 1}
                      strokeDasharray={d.pda?.here != null ? undefined : "2 2"}
                      data-basis={d.pda?.here != null ? "own-cell" : "offshore-median"}
                      style={{
                        transformBox: "fill-box",
                        transformOrigin: "50% 100%",
                      }}
                      initial={reduce ? false : { scaleY: 0 }}
                      animate={{ scaleY: 1 }}
                      transition={{
                        type: "spring",
                        stiffness: 180,
                        damping: 24,
                        delay: d.lead * 0.05,
                      }}
                    />
                  );
                })}
            </g>
          );
        })}
        <defs>
          <pattern id="ev-hatch-model" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--cw-forecast)" strokeWidth="1.2" strokeOpacity="0.6" />
          </pattern>
          <pattern id="ev-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--cw-gap-hatch)" strokeWidth="2" />
          </pattern>
        </defs>
      </svg>
      {/* row labels and their buttons sit over the chart, as HTML (readable, focusable, large targets) */}
      {rows.map((r, i) => {
        const y0 = TOP + i * ROW;
        const on = focus === r.focus;
        return (
          <button
            key={r.id}
            type="button"
            onClick={() => onFocus(on ? null : r.focus)}
            aria-pressed={on}
            className={`cw-press absolute left-0 flex flex-col justify-center rounded-lg px-1 text-left transition-colors ${on ? "bg-accent/10" : "hover:bg-surface-3/60"}`}
            style={LABEL ? { top: y0, height: ROW, width: LABEL - 6 } : { top: y0 - 2 + (i === 0 ? 0 : 0), height: 24, width: "100%" }}
            data-testid={`ev-label-${r.id}`}
          >
            <span className="flex items-center gap-1.5 text-[13px] font-semibold leading-tight text-ink">
              <Mark kind={r.kind} aged={r.far} />
              {r.title}
              {!LABEL && <span className="truncate font-normal text-ink-3">· {r.latest}</span>}
            </span>
            {!!LABEL && <span className="truncate pl-[18px] text-[12px] leading-tight text-ink-3">{r.latest}</span>}
          </button>
        );
      })}
      <ul className="mt-1 space-y-0.5 text-[12.5px] leading-snug text-ink-3" data-testid="evidence-where">
        {rows.map((r) => (
          <li key={r.id}>
            <b className="font-medium text-ink-2">{r.title}:</b> {r.where}
          </li>
        ))}
      </ul>
    </div>
  );
}

function stationRow(kind: "cells" | "toxin", e: StationEvidence | null): Row & { evidence: StationEvidence | null } {
  const title = kind === "cells" ? "Pseudo-nitzschia (algae)" : "Domoic acid in seawater";
  if (!e)
    return {
      id: kind,
      kind: "gap",
      title,
      where: "No station measures it nearby",
      latest: "No samples",
      far: true,
      focus: null,
      evidence: null,
    };
  const km = e.km < 10 ? e.km.toFixed(1) : String(Math.round(e.km));
  const v =
    kind === "cells"
      ? fmtMeasured(
          "pn_seriata",
          e.values.reduce((a, x) => a + x.value, 0),
        )
      : fmtMeasured("pDA", e.values[0].value);
  if (!e.near)
    return {
      id: kind,
      kind: "gap",
      title,
      where: `nearest station ${e.station.name}, ${km} km away: too far to describe these waters`,
      latest: `None within ${EVIDENCE_RADIUS_KM} km`,
      far: true,
      focus: kind,
      evidence: e,
    };
  return {
    id: kind,
    kind: "measured",
    title,
    where: `${e.station.name}, ${km} km ${e.bearing}`,
    latest: `${v} · ${ageWords(e.ageDays)}`,
    far: e.ageDays > BACK,
    focus: kind,
    evidence: e,
  };
}

function modelLatest(days: OutlookDay[], tripDate: string | null): string {
  const d = days.find((x) => x.valid === tripDate);
  if (!d) return `Does not reach ${tripDate ? shortDate(tripDate) : "that day"} (last ${shortDate(days.at(-1)!.valid!)})`;
  if (d.pda?.here != null) return `${fmtPct(d.pda.here)} chance above threshold, ${shortDate(d.valid!)}`;
  if (d.pda && d.pda.n > 0) return `No value here · offshore cells ${fmtPct(d.pda.min)}–${fmtPct(d.pda.max)}, context only`;
  return `No cell within ${NEIGHBOURHOOD_KM} km`;
}

function shortDate(d: string) {
  return new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/** Samples in the window: a dot per sample on a log scale within the row, reported zeros as rings on the floor, the newest labelled. */
function Dots({ e, x, t0, mid, reduce, label }: { e: StationEvidence; x: (iso: string) => number; t0: number; mid: number; reduce: boolean; label: number }) {
  const pts = new Map<string, number>();
  for (const s of Object.values(e.series)) for (const [d, v] of s) if (v != null && Date.parse(`${d}T12:00:00Z`) >= t0) pts.set(d, (pts.get(d) ?? 0) + v);
  const list = [...pts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  const pos = list.map(([, v]) => v).filter((v) => v > 0);
  const lo = Math.log10(Math.min(...pos, 1));
  const hi = Math.log10(Math.max(...pos, 10));
  const y = (v: number) => (v <= 0 ? mid + 7 : mid + 7 - ((Math.log10(v) - lo) / Math.max(0.5, hi - lo)) * 18);
  if (!list.length) {
    const older = e.date;
    return (
      <text x={label + 4} y={mid - 6} className="fill-ink-3 text-[11px]">
        ← last sample {shortDate(older)}
      </text>
    );
  }
  return (
    <g>
      {list.map(([d, v], i) =>
        v > 0 ? (
          <motion.circle
            key={d}
            cx={x(d)}
            cy={y(v)}
            r={3.6}
            fill="var(--cw-chl)"
            initial={reduce ? false : { opacity: 0, scale: 0 }}
            animate={{ opacity: 0.9, scale: 1 }}
            transition={{ delay: 0.02 * i }}
          />
        ) : (
          <circle key={d} cx={x(d)} cy={y(0)} r={2.6} fill="none" stroke="var(--cw-chl)" strokeOpacity={0.6} />
        ),
      )}
      <circle cx={x(list.at(-1)![0])} cy={y(list.at(-1)![1])} r={6.5} fill="none" stroke="var(--cw-ink)" strokeWidth={1.3} />
    </g>
  );
}
