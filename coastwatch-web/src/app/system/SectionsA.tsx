"use client";
import {
  Agenda,
  BigDate,
  Button,
  ChartNote,
  Legend,
  Margin,
  MarginRow,
  MarginRows,
  MarginSheet,
  Mark,
  ProvenanceSheet,
  RampLegend,
  Readout,
  SeriesChart,
  Sounding,
  Stamp,
  Statement,
  day,
  dayTime,
  fixed,
  hatchSegments,
  rampPosition,
  range,
  sig,
  svgId,
  type AgendaItem,
  type ChartMode,
} from "@/nx/ds";
import { useId, useState } from "react";
import { D, NOW, P, S, plates } from "./data";
import { Section, Specimen } from "./parts";
import { Plate } from "./Plate";
import s from "./catalog.module.css";

const recreational = S.fishery_status.records[1];
const anchovy = S.fishery_status.records[2];
const salmon = S.fishery_status.records.find((r) => r.id === "salmon-commercial-2026")!;
const ramp = S.fishery_status.ramp_zone_4;

const DEPTHS = ["--depth-0", "--depth-1", "--depth-2", "--depth-3", "--depth-4", "--depth-5", "--depth-6", "--depth-7", "--depth-8"];
const CHL = ["--chl-0", "--chl-1", "--chl-2", "--chl-3", "--chl-4", "--chl-5", "--chl-6", "--chl-7"];
const PROB = Array.from({ length: 10 }, (_, i) => `--prob-${i}`);
const SST = Array.from({ length: 8 }, (_, i) => `--sst-${i}`);
const INK = [
  ["--paper", "Paper", "Sheet and margin ground"],
  ["--ink", "Ink", "Text, interaction, rules"],
  ["--ink-2", "Ink 2", "Secondary text; there is no third level"],
  ["--water-name", "Water name", "Italic water names only"],
  ["--reg", "Regulation line", "Zone lines and rails; never text"],
  ["--reg-text", "Regulation text", "Delay and closure statements"],
  ["--closure", "Closure", "Official closures only"],
  ["--land", "Land", "With relief"],
];

export function SectionsA({ mode }: { mode: ChartMode }) {
  return (
    <>
      <Foundations mode={mode} />
      <Statements />
      <BigDates />
      <Agendas />
      <Marks />
      <Stamps />
      <Margins />
      <Sheet />
      <Soundings />
      <Ramp />
    </>
  );
}

function Foundations({ mode }: { mode: ChartMode }) {
  return (
    <Section id="foundations" name="Foundations" purpose={<>The tokens every component is built from, shown in the {mode} table. Colour is the last tool: letter style and shape carry class first.</>}>
      <Specimen state="Type" wide>
        <div className={s.typeSpec}>
          <p className={s.tsStatement}>Season delayed</p>
          <p className={s.tsLabel}>Statement · Literata 500, 64 / 48 / 36 (phone 40 / 32 / 28)</p>
          <p className={s.tsWater}>Monterey Canyon</p>
          <p className={s.tsLabel}>Water names · Literata italic 400, 15 to 22, tracked 0.04 em</p>
          <p className={s.tsSounding}>
            6.6<em>ft at 14 s</em>
          </p>
          <p className={s.tsLabel}>Sounding · Public Sans 600 tabular, 48 / 36, unit at 0.46 em</p>
          <p className={s.tsBody}>Interface and land are upright Public Sans: 16 body, 18 lead, 22 heads. Sentence case everywhere; no monospace.</p>
          <p className={s.tsLabel}>Small · 14, the floor for anything read: provenance, ticks</p>
        </div>
      </Specimen>
      <Specimen state="Ink and grounds" wide>
        <ul className={s.swatches}>
          {INK.map(([v, n, role]) => (
            <li key={v}>
              <span className={s.swatch} style={{ background: `var(${v})` }} />
              <span className={s.swName}>{n}</span>
              <span className={s.swRole}>{role}</span>
            </li>
          ))}
        </ul>
      </Specimen>
      <Specimen state="Depth tints" note="shoal to deep">
        <div className={s.strip}>
          {DEPTHS.map((v) => (
            <span key={v} style={{ background: `var(${v})` }} />
          ))}
        </div>
      </Specimen>
      <Specimen state="Chlorophyll ramp" note="log10 0.05 to 50 mg/m³, lightness monotone">
        <div className={s.strip}>
          {CHL.map((v) => (
            <span key={v} style={{ background: `var(${v})` }} />
          ))}
        </div>
      </Specimen>
      <Specimen state="C-HARM probability classes" note="cw-probability-classes-v1, ten 10-point display intervals, not risk tiers">
        <div className={s.strip}>
          {PROB.map((v) => (
            <span key={v} style={{ background: `var(${v})` }} />
          ))}
        </div>
      </Specimen>
      <Specimen state="Sea temperature ramp" note="sequential, its own hue family">
        <div className={s.strip}>
          {SST.map((v) => (
            <span key={v} style={{ background: `var(${v})` }} />
          ))}
        </div>
      </Specimen>
      <Specimen state="Rules and space" wide>
        <div className={s.rulesSpec}>
          <span className={s.ruleHead}>Head rule 1.5 px</span>
          <span className={s.ruleHair}>Hairline 1 px</span>
          <span className={s.spaceRow}>
            {[4, 8, 12, 16, 24, 32, 48, 64].map((n) => (
              <span key={n} className={s.spaceCell}>
                <span className={s.spaceBar} style={{ width: `var(--s-${n})` }} />
                <span>{n}</span>
              </span>
            ))}
          </span>
        </div>
      </Specimen>
    </Section>
  );
}

function Statements() {
  return (
    <Section id="statement" name="Statement" purpose="The page title plus its status sentence, in Literata. Regulation hue only for an official delay; rust only for an official closure.">
      <Specimen state="Regulation, with its line" wide>
        <Statement scope="Moss Landing · Commercial Dungeness" title="Season delayed to at least Jan 1, 2027" why="(humpback whale entanglements)" tone="regulation" rule as="h3">
          <p>
            Set {day(D.rampAssessed)} by CDFW. <Stamp provenance={P.season} now={NOW} />
          </p>
        </Statement>
      </Specimen>
      <Specimen state="Ink, large">
        <Statement size="lg" as="h3" title="Recreational crab: traps prohibited" scope="Moss Landing · Recreational Dungeness">
          <p>Open since {day(recreational.effective!)} with hoop nets and snares only, in CDFW&rsquo;s words.</p>
        </Statement>
      </Specimen>
      <Specimen state="Closure, medium">
        <Statement size="md" as="h3" tone="closure" title="Commercial ocean salmon season concluded">
          <p>CDFW, checked {day(salmon.checked_at)}. Not verified.</p>
        </Statement>
      </Specimen>
    </Section>
  );
}

function BigDates() {
  return (
    <Section id="bigdate" name="BigDate" purpose="The next date that matters, what it is, and whose date it is.">
      <Specimen state="Range with later dates">
        <BigDate label="Next whale assessment, subject to change" start={D.nextAssessment.start} end={D.nextAssessment.end} then={[D.thenAssessment]} now={NOW}>
          <Stamp provenance={{ ...P.season, title: "RAMP Zone 4 assessment schedule", value: range(D.nextAssessment.start, D.nextAssessment.end) }} now={NOW} />
        </BigDate>
      </Specimen>
      <Specimen state="Single date, next year, large">
        <BigDate size="lg" label="Commercial season delayed to at least" start={D.earliestCommercial} now={NOW} />
      </Specimen>
    </Section>
  );
}

function Agendas() {
  const items: AgendaItem[] = [
    { id: "anchovy", start: D.anchovyRestriction, title: "Anchovy: no take for human consumption", detail: anchovy.area_label, stamp: <Stamp provenance={{ ...P.season, title: anchovy.label, value: "No take for human consumption", sourceUrl: anchovy.source_url }} now={NOW} /> },
    { id: "crab", start: D.crabSampled, kind: "measured", title: "Crab tested, Monterey Bay", detail: "Dungeness highest 9.8 of 30 ppm; rock crab highest 20.", stamp: <Stamp provenance={P.crab} now={NOW} /> },
    { id: "ramp", start: D.rampAssessed, kind: "official", title: "Commercial season delayed", tone: "regulation", detail: "Humpback whale entanglements, Zones 1 to 5." },
    { id: "charm", start: D.charmIssued, kind: "model", title: "Newest C-HARM run issued", detail: "Its leads are all past; nothing newer since." },
    { id: "rec", start: D.recreationalOpen, title: "Recreational crab with hoop nets and snares", tone: "regulation" },
    { id: "next", start: D.nextAssessment.start, end: D.nextAssessment.end, kind: "official", title: "Whale assessment, Zone 4", tone: "regulation", tentative: true },
    { id: "jan", start: D.earliestCommercial, title: "Commercial delay runs to at least this date", tone: "regulation", detail: ramp.label },
    { id: "then", start: D.thenAssessment.start, end: D.thenAssessment.end, title: "Following assessment", tentative: true },
  ];
  return (
    <Section id="agenda" name="Agenda" purpose="Dated items on a rail, today ruled across: each mark is the item's kind as the legend draws it (measured solid, model hatched, official a ring), and time is its place against the now line. Dates are absolute and set in their own column.">
      <Specimen state="Past, today, coming" wide>
        <Agenda items={items} now={NOW} label="Dates for Moss Landing, commercial Dungeness" />
      </Specimen>
    </Section>
  );
}

function Marks() {
  return (
    <Section id="marks" name="Mark and Legend" purpose="The five semantic marks: shape first, colour second, so a greyscale print still separates measured from model. One legend per view replaces caveats beside each figure.">
      <Specimen state="The five marks">
        <ul className={s.markList}>
          <li>
            <Mark kind="measured" /> Measured
          </li>
          <li>
            <Mark kind="model" /> Model or prediction
          </li>
          <li>
            <Mark kind="official" withWords={false} /> Official, not verified
          </li>
          <li>
            <Mark kind="noview" /> No view
          </li>
          <li>
            9.8 ppm <Mark kind="stale" />
          </li>
        </ul>
      </Specimen>
      <Specimen state="Legend, column, with words for this view">
        <Legend layout="column" title="On this chart" kinds={["measured", "model", "official", "noview"]} words={{ model: "C-HARM model, past run (Nov 3)", noview: "No view: cloud or gap, not low" }} />
      </Specimen>
    </Section>
  );
}

function Stamps() {
  const [open, setOpen] = useState(false);
  return (
    <Section id="stamp" name="Stamp and ProvenanceSheet" purpose="Kind · source · date · verification: the provenance line every number carries. It opens the full record.">
      <Specimen state="Measured">
        <Stamp provenance={P.waves} now={NOW} />
      </Specimen>
      <Specimen state="Measured, old">
        <Stamp provenance={P.crab} now={NOW} />
      </Specimen>
      <Specimen state="Official, not verified">
        <Stamp provenance={P.season} now={NOW} />
      </Specimen>
      <Specimen state="Model, past run, old">
        <Stamp provenance={P.charm} now={NOW} />
      </Specimen>
      <Specimen state="Plain (inside a sheet)">
        <Stamp provenance={P.tide} now={NOW} interactive={false} />
      </Specimen>
      <Specimen state="ProvenanceSheet">
        <Button variant="secondary" onClick={() => setOpen(true)} aria-controls="prov-demo">
          Open the crab test&rsquo;s record
        </Button>
        <ProvenanceSheet provenance={P.crab} now={NOW} open={open} onOpenChange={setOpen} />
      </Specimen>
    </Section>
  );
}

function Margins() {
  const fc = S.marine_forecast.periods[0];
  const tide = S.tide.next_events_after_asOf;
  return (
    <Section id="margin" name="Margin and MarginRow" purpose="The ruled margin column beside the chart: the view's one sentence, its statement, then rows that expand in place.">
      <Specimen state="Rows in every state" wide>
        <div className={s.marginDemo}>
          <Margin label="Today at Moss Landing" imprint={<>Data as of {day(NOW)}. Relief NOAA ETOPO 2022. Not for navigation.</>}>
            <p>What changed at your port, and the next date that matters.</p>
            <MarginRows label="Conditions">
              <MarginRow
                label="Sea"
                value="Seas 4 to 6 ft"
                defaultOpen
                detail={<p>{fc.text}</p>}
                stamp={<Stamp provenance={P.forecast} now={NOW} />}
              />
              <MarginRow label="Tide" value={`High ${fixed(tide[0].ft)} ft, low ${fixed(tide[1].ft)} ft`} stamp={<Stamp provenance={P.tide} now={NOW} />} />
              <MarginRow label="Algae" value={<>{P.chl.value}, algae, not toxin</>} detail={<p>{S.chlorophyll.must_say}</p>} stamp={<Stamp provenance={P.chl} now={NOW} />} />
              <MarginRow label="Model" state="stale" value={<>C-HARM past run ({day(D.charmIssued)})</>} detail={<p>{S.charm.must_say}</p>} stamp={<Stamp provenance={P.charm} now={NOW} />} />
              <MarginRow label="Tests" state="unavailable" note="Crab tests aren't in this build" />
              <MarginRow label="Buoy" state="loading" note="Reading NDBC 46236…" />
              <MarginRow label="Wind" state="offline" value="9.7 kt from the ENE" note={`(saved ${day(D.savedAt)})`} />
            </MarginRows>
          </Margin>
        </div>
      </Specimen>
    </Section>
  );
}

function Sheet() {
  const [on, setOn] = useState(false);
  return (
    <Section id="sheet" name="MarginSheet" purpose="On a phone the margin becomes a bottom sheet. It rests where the status is visible and drags up with velocity; More and Less move between detents for keyboards.">
      <Specimen state="Resting, then dragged up">
        <Button variant="secondary" onClick={() => setOn((v) => !v)} aria-expanded={on}>
          {on ? "Remove the phone sheet" : "Show the phone sheet"}
        </Button>
        {on ? (
          <MarginSheet
            label="Today at Moss Landing"
            onClose={() => setOn(false)}
            peek={
              <div>
                <p className={s.sheetStatus}>Season delayed to at least Jan 1, 2027</p>
                <p className={s.sheetSub}>Next whale assessment {range(D.nextAssessment.start, D.nextAssessment.end)}, subject to change.</p>
              </div>
            }
          >
            <MarginRows label="Conditions">
              <MarginRow label="Sea" value="Seas 4 to 6 ft" stamp={<Stamp provenance={P.forecast} now={NOW} />} />
              <MarginRow label="Tide" value="High 5.7 ft" stamp={<Stamp provenance={P.tide} now={NOW} />} />
              <MarginRow label="Algae" value={<>{P.chl.value}, algae, not toxin</>} />
            </MarginRows>
          </MarginSheet>
        ) : null}
      </Specimen>
    </Section>
  );
}

function Soundings() {
  const pl = plates.bay.places;
  const [sel, setSel] = useState(false);
  return (
    <Section id="sounding" name="Sounding and ChartNote" purpose="Figures set where they were measured, notes on the water, each on a knockout halo. The chart host positions them and drops the lowest priority first when space is short.">
      <Specimen state="On the chart" wide note="host-placed, priorities 1 to 6">
        <Plate label="Monterey Bay with two soundings and its names" rail>
          {(k) => {
            // what the chart host does on a narrow sheet: drop priority 4 and below, set figures smaller
            const narrow = k < 0.6;
            const drop = (p: number) => narrow && p >= 3;
            return (
            <>
              <ChartNote x={pl.santaCruz[0] * k} y={pl.santaCruz[1] * k} variant="place" anchor="se" priority={4} dropped={drop(4)}>
                Santa Cruz
              </ChartNote>
              <ChartNote x={pl.monterey[0] * k} y={pl.monterey[1] * k} variant="place" anchor="e" priority={4} dropped={drop(4)}>
                Monterey
              </ChartNote>
              <ChartNote x={pl.mossLanding[0] * k} y={pl.mossLanding[1] * k} variant="port" anchor="w" priority={1}>
                Moss Landing
              </ChartNote>
              <ChartNote x={330 * k} y={470 * k} variant="water" size={22} priority={5} dropped={drop(5)}>
                Monterey Bay
              </ChartNote>
              <ChartNote x={420 * k} y={600 * k} variant="water" size={16} rotate={-38} priority={6} dropped={drop(6)}>
                Monterey Canyon
              </ChartNote>
              <ChartNote x={940 * k} y={820 * k} variant="regulation" rotate={-90} anchor="e" priority={3} dropped={drop(3)}>
                Whale zone 4, Pigeon Point to Lopez Point
              </ChartNote>
              <Sounding
                x={pl.cdphTestArea[0] * k}
                y={pl.cdphTestArea[1] * k}
                kind="measured"
                label="Dungeness viscera, highest of 6"
                value="9.8"
                unit="of 30 ppm"
                caption={`CDPH, Monterey Bay test area, ${day(D.crabSampled)}`}
                stale
                anchor="nw"
                leader={narrow ? { dx: -6, dy: -12 } : { dx: -24 * k - 8, dy: -90 * k }}
                size={narrow ? "md" : "lg"}
                width={narrow ? 250 : 260}
                priority={2}
                onActivate={() => setSel(true)}
                ariaLabel="Dungeness crab viscera, highest of 6 samples: 9.8 parts per million, action level 30. CDPH, Monterey Bay test area, Sep 28, old. Open its record."
              />
              <Sounding
                x={pl.buoy46236[0] * k}
                y={pl.buoy46236[1] * k}
                kind="measured"
                value={String(S.buoy.waves.wvht_ft)}
                unit={`ft at ${S.buoy.waves.period_dominant_s} s`}
                caption={`Waves, buoy 46236, ${dayTime(S.buoy.waves.asOf)}`}
                anchor={narrow ? "sw" : "se"}
                leader={narrow ? { dx: -12, dy: 44 } : { dx: 40 * k + 10, dy: 120 * k + 10 }}
                size={narrow ? "md" : "lg"}
                width={narrow ? 200 : 260}
                priority={1}
              />
            </>
            );
          }}
        </Plate>
        <ProvenanceSheet provenance={P.crab} now={NOW} open={sel} onOpenChange={setSel} />
      </Specimen>
      <Specimen state="Model figure, hollow mark" note="analysis, not an observation">
        <div className={s.miniWater}>
          <Sounding x={28} y={34} kind="analysis" size="md" value={sig(S.sst.gridded_analysis.median_within_15km, 3)} unit="°C" label="Sea surface, MUR analysis" caption={`Median within 15 km, ${day(S.sst.gridded_analysis.asOf)}`} anchor="se" leader={{ dx: 18, dy: 10 }} />
        </div>
      </Specimen>
      <Specimen state="Dropped by priority" note="the host sets dropped; the note leaves with register">
        <div className={s.miniWater}>
          <ChartNote x={24} y={40} variant="water" size={20} priority={5}>
            Monterey Bay
          </ChartNote>
          <ChartNote x={24} y={92} variant="note" priority={6} dropped>
            Shelf break
          </ChartNote>
          <p className={s.miniCap}>&ldquo;Shelf break&rdquo; (priority 6) is dropped at this width.</p>
        </div>
      </Specimen>
    </Section>
  );
}

function Ramp() {
  const g = plates.chl.grid;
  const id = svgId(useId());
  const { rowEdges: R, colEdges: C } = plates.chl;
  const cells: React.ReactNode[] = [];
  g.rows.forEach((row, i) =>
    row.forEach((v, j) => {
      const x = C[j], y = R[i], w = C[j + 1] - C[j] + 0.4, h = R[i + 1] - R[i] + 0.4;
      if (v === null) cells.push(<rect key={`${i}-${j}`} x={x} y={y} width={w} height={h} fill={`url(#${id})`} />);
      else {
        const step = Math.min(7, Math.floor(rampPosition(v, [0.05, 50]) * 8));
        cells.push(<rect key={`${i}-${j}`} x={x} y={y} width={w} height={h} style={{ fill: `var(--chl-${step})` }} />);
      }
    }),
  );
  const ri = 26, ci = 37;
  const v = g.rows[ri][ci]!;
  const age = g.ages[ri][ci]!;
  const lat = 37.05 - (ri + 0.5) * 0.01, lon = -122.2 + (ci + 0.5) * 0.01;
  return (
    <Section id="ramp" name="RampLegend and Readout" purpose="The legend is the active layer's title: stepped ramp, log ticks, and the no-view hatch. Tapping the water gives a readout with its kind, date and pixel age.">
      <Specimen state="Layer with its legend" wide>
        <div className={s.rampGrid}>
          <Plate
            which="chl"
            label="Chlorophyll, Sentinel-3 OLCI, latest clear view"
            under={
              <>
                <defs>
                  <pattern id={id} width="6" height="6" patternUnits="userSpaceOnUse">
                    <rect width="6" height="6" className={s.gapGround} />
                    <path d={hatchSegments(0, 0, 6, 6, 3)} className={s.gapHatch} />
                  </pattern>
                </defs>
                {cells}
              </>
            }
          >
            {(k) => (
              <ChartNote x={((plates.chl.colEdges[ci] + plates.chl.colEdges[ci + 1]) / 2) * k} y={((plates.chl.rowEdges[ri] + plates.chl.rowEdges[ri + 1]) / 2) * k} variant="place" anchor="w" priority={1}>
                Tapped point
              </ChartNote>
            )}
          </Plate>
          <div className={s.rampSide}>
            <RampLegend
              title="Chlorophyll"
              unit="mg/m³"
              steps={CHL}
              domain={[0.05, 50]}
              ticks={[0.1, 1, 10]}
              caption={<>Algae, not toxin. Sentinel-3 OLCI, latest clear view {range(D.olciFrom, D.olciTo)}; each pixel has its own date.</>}
            />
            <Readout
              title="Chlorophyll at this point"
              lat={lat}
              lon={lon}
              state="value"
              value={sig(v, 2)}
              unit="mg/m³"
              provenance={{ ...P.chl, title: "Chlorophyll at this point", value: `${sig(v, 2)} mg/m³`, place: "One 0.01° block, geometric mean of native 300 m pixels" }}
              now={NOW}
              note={<>Algae, not toxin. This pixel was seen {age} days before {day(NOW)}.</>}
              action={<Button variant="text" href="#series">See the record nearby</Button>}
            >
              <SeriesChart points={S.chlorophyll.port_intel_8day.history.map(([t, x]) => ({ t: t as string, v: x as number | null }))} label="VIIRS 8-day median near Moss Landing" unit="mg/m³" marks="both" height={84} showReadout={false} gapDays={6} />
            </Readout>
          </div>
        </div>
      </Specimen>
      <Specimen state="Readout, reading">
        <Readout title="Chlorophyll at this point" lat={36.712} lon={-121.98} state="loading" message={`Reading this point from the ${day(D.olciTo)} view…`} />
      </Specimen>
      <Specimen state="Readout, no view">
        <Readout title="Chlorophyll at this point" lat={36.93} lon={-121.91} state="noview" message={<>No view here: cloud on every day {range(D.olciFrom, D.olciTo)}. No value is not a low value.</>} />
      </Specimen>
      <Specimen state="Readout, error">
        <Readout title="Chlorophyll at this point" lat={36.66} lon={-122.05} state="error" />
      </Specimen>
    </Section>
  );
}
