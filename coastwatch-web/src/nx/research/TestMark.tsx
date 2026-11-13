"use client";

import { ChartMarks, type ChartMark } from "@/nx/chart/Labels";
import type { Bounds } from "@/nx/chart/mercator";
import { Micro, usePhone } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import { useGates } from "@/nx/coast/useCoast";
import s from "./rx.module.css";

/**
 * The CDPH crab test on Research's chart (Toxin tests layer, the prototype's reading beside Monterey Bay): the latest
 * Dungeness test that informs the station's nearest port, read through My Coast's own gate (`/coast/data`, the same
 * record the trip draws), drawn where CDPH collected it when that is inside the framed area. Seafood tissue, ppm, against
 * CDPH's action level; never mixed with the shore station's water samples. Wide screens only, as in the prototype.
 */
export function TestMark({ place, within }: { place: string | null; within: Bounds }) {
  const phone = usePhone();
  const gates = useGates(phone ? null : place, "dungeness_commercial");
  const test = gates.state === "ok" ? gates.data.test : null;
  if (!test || test.lat == null || test.lon == null || phone) return null;
  const [[w, sth], [e, nth]] = within;
  if (test.lon < w || test.lon > e || test.lat < sth || test.lat > nth) return null;
  const old = test.stale || (test.ageDays ?? 0) > 30;
  const n = test.raw.length;
  const mark: ChartMark = {
    id: `research-test-${test.site}`,
    lon: test.lon,
    lat: test.lat,
    layer: "toxin",
    priority: 70,
    leader: true,
    offset: 22,
    // the station names stand to the west of their pins; the test reads on the land side
    anchors: ["right", "top-right", "bottom-right", "bottom"],
    point: <i className={s.testPoint} />,
    label: `CDPH test, ${test.site}, ${day(test.date)}: ${test.species}, highest ${test.max ?? "not detected"}${test.action != null ? ` of ${test.action} ppm` : " ppm"}, ${n} samples${old ? ", more than 30 days ago" : ""}`,
    content: (
      <div className={s.tag} data-testid="research-cdph-test">
        <Micro glyph={old ? "old" : "measured"}>
          CDPH test · {test.site} · {day(test.date)}
        </Micro>
        <p className={s.tagFig}>
          {test.species.replace(/ crab$/, "")}, highest {test.max == null ? "ND" : test.max} <small>{test.action != null ? `of ${test.action} ppm` : "ppm"}</small>
        </p>
        {test.max != null && test.action != null ? (
          <span className={s.tagGauge} aria-hidden="true">
            <i style={{ width: `${Math.min(100, (test.max / test.action) * 100)}%` }} />
          </span>
        ) : null}
        <p className={s.tagSub}>
          {n} samples{test.ageDays != null ? ` · ${test.ageDays} days ago${old ? " (old)" : ""}` : ""}
        </p>
      </div>
    ),
  };
  return <ChartMarks marks={[mark]} />;
}
