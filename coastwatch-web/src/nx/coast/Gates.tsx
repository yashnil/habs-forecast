"use client";

import { Loading, MarginRow, MarginRows, Stamp, Unavailable, type Provenance } from "@/nx/ds";
import { day, sig } from "@/nx/ds/format";
import { cx } from "@/nx/ds/util";
import type { CoastGates, TestGate } from "./brief";
import type { CoastFishery } from "./fisheries";
import type { Loadable } from "./useCoast";
import s from "./coast.module.css";

/**
 * The gates for the chosen fishery, in one ruled block: season status, whale measures, the latest test.
 * Each line carries its own stamp (kind, source, date and, for official records, "not verified"). The order
 * is the hierarchy (T44): the official status first; a test never softens it.
 */
export function Gates({ gates, fishery }: { gates: Loadable<CoastGates>; fishery: CoastFishery }) {
  if (gates.state === "idle") return null;
  // the loading line holds about the gates' own height, so the sea window and worksheet do not jump down when they land (CLS)
  if (gates.state === "loading")
    return (
      <div className={s.gatesLoading}>
        <Loading>Reading the season list, whale measures and CDPH tests…</Loading>
      </div>
    );
  if (gates.state === "error")
    return (
      <Unavailable
        what="The gates could not be read"
        why={`Neritic's dataset did not answer (${gates.error}). Unknown is not open.`}
        next={
          <a href="https://wildlife.ca.gov/Fishing/Commercial" className={s.link}>
            CDFW commercial fishing
          </a>
        }
      />
    );
  const g = gates.data;
  const now = g.now;
  return (
    <MarginRows label={`Gates for ${fishery.label}`} className={s.rows}>
      {g.season ? (
        <MarginRow
          label="Season"
          value={<span className={cx(s.statusText, g.season.tone === "regulation" && s.toneReg, g.season.tone === "closure" && s.toneClosure)}>{g.season.text}</span>}
          state={g.season.stale ? "stale" : "default"}
          stamp={<Stamp now={now} provenance={seasonProvenance(g, fishery)} />}
        />
      ) : (
        <MarginRow label="Season" state="unavailable" note="Neritic's season list did not load. Unknown is not open: check CDFW." />
      )}
      {g.whale ? (
        <MarginRow
          label="Whale measures"
          value={
            <>
              <span>
                Zone {g.whale.zone}, {g.whale.area}. {g.whale.measures}
              </span>
              {g.whale.next ? (
                <span className={s.next}>
                  <span className={s.nextKey}>Next assessment</span> {g.whale.next}
                </span>
              ) : null}
            </>
          }
          stamp={
            <Stamp
              now={now}
              provenance={{
                kind: "official",
                title: `Whale measures, RAMP Zone ${g.whale.zone}`,
                source: "CDFW RAMP",
                sourceLong: "CDFW Risk Assessment and Mitigation Program (whale entanglement), Whale-Safe Fisheries page",
                sourceUrl: g.whale.url ?? undefined,
                date: g.whale.assessed ?? now.slice(0, 10),
                dateRole: g.whale.assessed ? "assessed" : "checked",
                caveats: ["RAMP is about whale entanglement, not domoic acid.", "Assessment dates are CDFW's and subject to change."],
              }}
            />
          }
        />
      ) : null}
      {g.test ? (
        <MarginRow label="Latest test" value={<TestLine t={g.test} now={now} />} state={g.test.stale ? "stale" : "default"} stamp={<Stamp now={now} provenance={testProvenance(g.test)} />} />
      ) : (
        <MarginRow label="Latest test" value={<span className={s.quiet}>{g.testsError ? "CDPH's results did not load." : `No CDPH test for ${fishery.label.toLowerCase()} informs this port.`}</span>} />
      )}
    </MarginRows>
  );
}

function seasonProvenance(g: CoastGates, f: CoastFishery): Provenance {
  const se = g.season!;
  return {
    kind: "official",
    title: `Season status, ${f.label}`,
    value: se.text,
    source: se.source,
    sourceUrl: se.url ?? undefined,
    date: (se.checked ?? g.now).slice(0, 10),
    dateRole: "checked",
    stale: se.stale,
    caveats: [se.differ, "CDFW acts on fisheries; CDPH tests and advises.", "Transcribed from the agency's page; a later notice may be missing."].filter((x): x is string => !!x),
  };
}

function testProvenance(t: TestGate): Provenance {
  return {
    kind: "measured",
    title: `Domoic acid, ${t.site} ${t.species.toLowerCase()}`,
    value: t.max != null ? `${sig(t.max)} ppm highest` : undefined,
    source: t.source,
    sourceLong: "CDPH, Summary of Domoic Acid Levels in Crab",
    sourceUrl: t.url,
    date: t.date,
    dateRole: "sampled",
    stale: t.stale,
    resolution: `${t.raw.length} samples, ${t.tissue}`,
    caveats: ["A measured test of seafood tissue at one site, not a forecast.", "CDPH tests; CDFW decides the season. A test does not open or close one."],
  };
}

/**
 * "Monterey Bay, Sep 28: highest 9.8 ppm; none of 6 at or over the 30 ppm action level." with the samples drawn
 * against that level. No average (T8): CDPH's printed average counts results below detection as zero.
 */
function TestLine({ t, now }: { t: TestGate; now: string }) {
  const n = t.raw.length;
  const cens = t.raw.map((r) => r.trim()).filter((r) => /^(<|nd)/i.test(r));
  const censored = cens.length ? `${cens.length} reported as ${[...new Set(cens)].join(" or ")}` : null;
  const over = t.action == null ? null : t.over === 0 ? `none of ${n} at or over` : `${t.over} of ${n} at or over`;
  return (
    <>
      <span>
        {t.site} {/dungeness/i.test(t.species) ? t.species : t.species.toLowerCase()}, {day(t.date, { ref: now })}: {t.max != null ? `highest ${sig(t.max)} ppm` : "no value above detection"}
        {over ? `; ${over} the ${t.action} ppm action level` : ""}
        {censored ? ` (${censored})` : ""}.
        {t.ageDays != null && t.ageDays > 1 ? ` ${t.ageDays} days ago.` : ""}
      </span>
      {t.action != null ? <Gauge t={t} /> : null}
    </>
  );
}

/** Each sample a tick on a 0 to action-level rule; censored samples are not drawn as numbers (T7). */
function Gauge({ t }: { t: TestGate }) {
  const W = 240;
  const x = (v: number) => 1 + (Math.min(v, t.action! * 1.15) / t.action!) * (W - 30);
  const ax = x(t.action!);
  return (
    <svg className={s.gauge} viewBox={`0 0 ${W} 40`} role="img" aria-label={`Samples against the ${t.action} ppm action level`} focusable="false">
      <line x1="1" y1="14" x2={ax} y2="14" className={s.gaugeAxis} />
      {t.values.map((v, i) => (v == null ? null : <line key={i} x1={x(v)} x2={x(v)} y1="6" y2="22" className={s.gaugeTick} />))}
      <line x1={ax} x2={ax} y1="2" y2="26" className={s.gaugeLimit} />
      <text x="1" y="38" className={s.gaugeText}>
        0
      </text>
      <text x={ax} y="38" textAnchor="end" className={s.gaugeText}>
        {t.action} ppm
      </text>
    </svg>
  );
}
