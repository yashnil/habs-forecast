"use client";
import {
  Button,
  Dialog,
  Field,
  ICON_NAMES,
  Icon,
  Loading,
  Offline,
  Popover,
  PortFinder,
  Segmented,
  SensitivityTable,
  SeriesChart,
  SmallMultiples,
  SourcesDiffer,
  Stale,
  Stepper,
  Toggle,
  Tooltip,
  Unavailable,
  Waterfall,
  day,
  usd,
  type PortEntry,
  type WaterfallStep,
} from "@/nx/ds";
import { DIESEL_REF, PRICE_REFS } from "@/content/referenceValues";
import { useMemo, useState } from "react";
import { D, NOW, S, charmSeries, ports, record, regions, station } from "./data";
import { Section, Specimen } from "./parts";
import s from "./catalog.module.css";

const DAY = 86_400_000;
const lastSample = Date.parse(record.pDA.points[record.pDA.points.length - 1].t);

export function SectionsB() {
  return (
    <>
      <Series />
      <Multiples />
      <Worksheet />
      <Buttons />
      <Finder />
      <Overlays />
      <Icons />
      <States />
    </>
  );
}

function Series() {
  return (
    <Section id="series" name="SeriesChart" purpose="A measured record in SVG with a crosshair readout. Reported zeros and censored values sit in their own strip, never at 0 or at the limit; long gaps break the line.">
      <Specimen state="Log points with reported 0" wide note={`${station.name}, CalHABMAP`}>
        <SeriesChart points={record.pDA.points} label={`${record.pDA.label}, ${station.name}`} unit={record.pDA.units} xDomain={[lastSample - 3.75 * 365 * DAY, lastSample]} height={220} />
      </Specimen>
      <Specimen state="Linear line with gaps" note="samples more than 45 days apart break the line">
        <SeriesChart points={record.temp.points} label={`Water temperature, ${station.name}`} unit="°C" scale="linear" marks="line" height={140} />
      </Specimen>
      <Specimen state="Model, past run" note="hollow, dashed, a now divider">
        <SeriesChart points={charmSeries} label="C-HARM P(pDA > 500 ng/L), median within 15 km" unit="%" kind="model" scale="linear" yDomain={[0, 100]} marks="both" height={140} now={NOW} xDomain={[Date.parse(charmSeries[0].t), Date.parse(NOW)]} gapDays={4} />
      </Specimen>
      <Specimen state="Empty window">
        <SeriesChart points={[]} label="Dissolved domoic acid" unit="ng/mL" height={90} xDomain={[lastSample - 365 * DAY, lastSample]} />
      </Specimen>
    </Section>
  );
}

function Multiples() {
  return (
    <Section id="multiples" name="SmallMultiples and Brush" purpose="A station's record as strips on one time axis with one brush. The crosshair is shared; the brush filters every strip and is a slider for the keyboard.">
      <Specimen state="Four strips, one brush" wide note={`${station.name}, ${record.pDA.points.length} samples from ${day(record.pDA.points[0].t, { year: true })}`}>
        <SmallMultiples
          strips={[
            { id: "pda", label: record.pDA.label, unit: record.pDA.units, points: record.pDA.points },
            { id: "pn", label: "Pseudo-nitzschia, seriata group", unit: "cells/L", points: record.pn.points },
            { id: "chl", label: "Chlorophyll, extracted", unit: "mg/m³", points: record.chl.points },
            { id: "temp", label: "Water temperature", unit: "°C", points: record.temp.points, scale: "linear", marks: "line" },
          ]}
          initial={[lastSample - 2 * 365 * DAY, lastSample]}
        />
      </Specimen>
    </Section>
  );
}

/* ---------- worksheet ---------- */

type Who = "shared" | "boat";
interface Entry {
  lb: string;
  price: string;
  fees: string;
  gal: string;
  fuel: string;
  ice: string;
  crew: number;
  share: string;
}
const num = (v: string) => (v.trim() === "" || !Number.isFinite(Number(v)) ? null : Number(v));

function trip(e: Entry, who: Who | null, rf = 1, cf = 1) {
  const lb = num(e.lb), price = num(e.price), fees = num(e.fees), gal = num(e.gal), fuel = num(e.fuel), ice = num(e.ice), share = num(e.share);
  const gross = lb !== null && price !== null ? lb * rf * price * cf : null;
  const fee = gross !== null && fees !== null ? (gross * fees) / 100 : null;
  const fuelUsd = gal !== null && fuel !== null ? gal * fuel : null;
  const n = gross !== null && fee !== null && fuelUsd !== null && ice !== null && who ? gross - fee - ice - (who === "shared" ? fuelUsd : 0) : null;
  const crew = n !== null && share !== null ? Math.max(0, n) * ((e.crew * share) / 100) : null;
  const net = n !== null && crew !== null && fuelUsd !== null ? n - crew - (who === "boat" ? fuelUsd : 0) : null;
  return { gross, fee, fuelUsd, ice, crew, net };
}

const dungeness = PRICE_REFS.dungeness;

function Worksheet() {
  const [e, setE] = useState<Entry>({ lb: "800", price: "4.75", fees: "5", gal: "120", fuel: "4.60", ice: "180", crew: 2, share: "15" });
  const [who, setWho] = useState<Who | null>("shared");
  const [cell, setCell] = useState<[number, number] | null>(null);
  const rows = [
    { label: "+30 %", factor: 1.3 },
    { label: "As entered", factor: 1 },
    { label: "−30 %", factor: 0.7 },
  ];
  const cols = [
    { label: "−20 %", factor: 0.8 },
    { label: "As entered", factor: 1 },
    { label: "+20 %", factor: 1.2 },
  ];
  const t = cell ? trip(e, who, rows[cell[0]].factor, cols[cell[1]].factor) : trip(e, who);
  const set = (k: keyof Entry) => (v: string) => setE((x) => ({ ...x, [k]: v }));
  const steps: WaterfallStep[] = [
    { id: "gross", label: "Landed", kind: "gross", amount: t.gross, waiting: "waiting for catch and price" },
    { id: "fees", label: "Fees", kind: "cost", amount: t.fee, waiting: "waiting for fees" },
    { id: "fuel", label: "Fuel", kind: "cost", amount: t.fuelUsd, waiting: "waiting for fuel" },
    { id: "ice", label: "Ice and bait", kind: "cost", amount: t.ice, waiting: "waiting for ice and bait" },
    { id: "crew", label: "Crew", kind: "cost", amount: t.crew, waiting: who ? "waiting for crew share" : "waiting for who pays fuel" },
    { id: "net", label: "Net to boat", kind: "net", amount: t.net, waiting: "waiting for your numbers" },
  ];
  const cells = {
    gross: (
      <span className={s.ledgerIn}>
        Landed <Field layout="inline" hideLabel label="Catch, pounds" value={e.lb} onChange={set("lb")} suffix="lb ×" size={4} />
        <Field layout="inline" hideLabel label="Price per pound" value={e.price} onChange={set("price")} prefix="$" size={4} />
      </span>
    ),
    fees: (
      <span className={s.ledgerIn}>
        Fees <Field layout="inline" hideLabel label="Fees, percent of gross" value={e.fees} onChange={set("fees")} suffix="% of gross" size={2} />
      </span>
    ),
    fuel: (
      <span className={s.ledgerIn}>
        Fuel <Field layout="inline" hideLabel label="Fuel, gallons" value={e.gal} onChange={set("gal")} suffix="gal × $" size={3} />
        <Field layout="inline" hideLabel label="Fuel price per gallon" value={e.fuel} onChange={set("fuel")} size={4} />
      </span>
    ),
    ice: (
      <span className={s.ledgerIn}>
        Ice and bait <Field layout="inline" hideLabel label="Ice and bait, dollars" value={e.ice} onChange={set("ice")} prefix="$" size={4} />
      </span>
    ),
  };
  return (
    <Section id="worksheet" name="Worksheet" purpose="A ruled ledger the reader fills in: fields with units, a live waterfall from gross to net, and a fixed sensitivity grid that moves the waterfall on hover. Every output is hypothetical, from the reader's numbers.">
      <Specimen state="Example entries, live" wide note="Example entries. Hypothetical.">
        <div className={s.ws}>
          <div>
            <Waterfall steps={steps} cells={cells} label="Trip, landed to net to boat, hypothetical" />
            <p className={s.wsNote}>
              {t.net !== null && t.gross !== null ? (
                <>
                  Net to boat, hypothetical, from your numbers: <strong>{usd(t.net)}</strong>
                  {cell ? `, at ${rows[cell[0]].label.toLowerCase()} catch and ${cols[cell[1]].label.toLowerCase()} price` : ""}. Before insurance, permits, upkeep and loan payments; not profit.
                </>
              ) : (
                <>Net to boat is waiting for your numbers.</>
              )}
            </p>
          </div>
          <SensitivityTable
            rows={rows}
            cols={cols}
            corner="Catch \ price"
            value={(rf, cf) => trip(e, who, rf, cf).net}
            selected={cell}
            onSelect={setCell}
            caption="Net to boat, hypothetical. Fuel, ice and bait are held fixed as catch moves."
          />
        </div>
        <div className={s.wsControls}>
          <Stepper label="Crew, besides the skipper" value={e.crew} onChange={(n) => setE((x) => ({ ...x, crew: n }))} min={0} max={8} unit="people" />
          <Field label="Crew share, each" value={e.share} onChange={set("share")} suffix="% of net after expenses" size={3} />
          <Segmented label="Who pays fuel" options={[{ value: "shared", label: "Comes off the top" }, { value: "boat", label: "The boat pays" }]} value={who} onChange={setWho} />
        </div>
      </Specimen>
      <Specimen state="Field states" wide>
        <div className={s.fieldGrid}>
          <Field label="Price, Dungeness" value="" onChange={() => undefined} prefix="$" suffix="/lb" reference={{ value: dungeness.usdPerLb.toFixed(2), words: `CDFW's 2024 statewide average, $${dungeness.usdPerLb.toFixed(2)}/lb`, source: "mixes two seasons" }} />
          <Field label="Fuel price" value={DIESEL_REF.usdPerGal.toFixed(2)} onChange={() => undefined} prefix="$" suffix="/gal" fromReference={`EIA, week of ${day(DIESEL_REF.week)}`} />
          <Field label="Catch" value="eight hundred" onChange={() => undefined} suffix="lb" size={12} error="Use numbers only, like 800." />
          <Field label="Pots" value="" onChange={() => undefined} suffix="pots" disabled hint="Pots apply to crab gear only." />
        </div>
      </Specimen>
      <Specimen state="Waiting" note="a blank reads as waiting, never 0">
        <Waterfall
          label="Trip waiting for a price, hypothetical"
          steps={[
            { id: "g", label: "Landed", kind: "gross", amount: null, waiting: "waiting for price" },
            { id: "f", label: "Fuel", kind: "cost", amount: 552 },
            { id: "n", label: "Net to boat", kind: "net", amount: null, waiting: "waiting for price" },
          ]}
        />
      </Specimen>
      <Specimen state="Costs over gross" note="the domain follows the running low (E3)">
        <Waterfall
          label="Trip at a loss, hypothetical"
          steps={[
            { id: "g", label: "Landed", kind: "gross", amount: 420 },
            { id: "f", label: "Fuel", kind: "cost", amount: 552 },
            { id: "i", label: "Ice and bait", kind: "cost", amount: 180 },
            { id: "n", label: "Net to boat", kind: "net", amount: -312 },
          ]}
        />
      </Specimen>
      <Specimen state="Toggle">
        <ToggleDemo />
      </Specimen>
    </Section>
  );
}

function ToggleDemo() {
  const [a, setA] = useState(true);
  const [b, setB] = useState(false);
  return (
    <div className={s.stack}>
      <Toggle label="Show temperatures in °C" checked={b} onChange={setB} />
      <Toggle label="Save this port's page for offline use" checked={a} onChange={setA} onWord="Saved" offWord="Not saved" />
      <Toggle label="Alerts by text message" checked={false} onChange={() => undefined} disabled offWord="Off; needs an account" />
    </div>
  );
}

function Buttons() {
  const [busy, setBusy] = useState(false);
  return (
    <Section id="buttons" name="Button" purpose="Three kinds and no more: primary ink (one per view), secondary ruled, and a text link. Busy says what it is doing; disabled stays focusable so its reason can be read.">
      <Specimen state="Primary">
        <Button variant="primary">Export these entries as CSV</Button>
      </Specimen>
      <Specimen state="Primary, busy">
        <Button variant="primary" busy busyLabel="Exporting 194 entries…">
          Export these entries as CSV
        </Button>
      </Specimen>
      <Specimen state="Secondary, with icon">
        <Button variant="secondary" icon="share" href="https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/pzz535.txt" external>
          Open the source
        </Button>
      </Specimen>
      <Specimen state="Secondary, pressed to busy">
        <Button variant="secondary" busy={busy} busyLabel="Saving for offline…" onClick={() => setBusy((b) => !b)}>
          Save for offline
        </Button>
      </Specimen>
      <Specimen state="Disabled">
        <Button variant="secondary" disabled aria-describedby="why-disabled">
          Subscribe to alerts
        </Button>
        <p id="why-disabled" className={s.miniCap}>
          Alerts need an account.
        </p>
      </Specimen>
      <Specimen state="Text link">
        <Button variant="text" href="#series">
          See the record nearby
        </Button>
      </Specimen>
    </Section>
  );
}

function Finder() {
  const [picked, setPicked] = useState<PortEntry | null>(null);
  return (
    <Section id="portfinder" name="PortFinder" purpose="Find a port by typing, or read down the coast: CDFW landing ports grouped by region, north to south, with recent ports first.">
      <Specimen state="Open, with a recent port" note={picked ? `chose ${picked.name}` : "type “moss” or use the arrows"}>
        <PortFinder ports={ports} regions={regions} recent={[592]} current={picked?.code ?? 592} onSelect={setPicked} />
      </Specimen>
    </Section>
  );
}

function Overlays() {
  const [port, setPort] = useState("Moss Landing");
  const near = useMemo(() => ports.filter((p) => p.region === "monterey_bay"), []);
  return (
    <Section id="overlays" name="Popover, Dialog, Tooltip" purpose="Radix primitives dressed as paper: a 1.5 px rule, no shadow, no radius. The page fades under a paper halo, not a dark scrim.">
      <Specimen state="Popover">
        <Popover
          label="Ports nearby"
          trigger={
            <button type="button" className={s.portButton}>
              {port}
              <Icon name="chevron" size={20} />
            </button>
          }
        >
          <p className={s.popHead}>Ports in Monterey Bay</p>
          <ul className={s.popList}>
            {near.map((p) => (
              <li key={p.code}>
                <button type="button" onClick={() => setPort(p.name)} aria-current={p.name === port || undefined}>
                  {p.name}
                </button>
              </li>
            ))}
          </ul>
        </Popover>
      </Specimen>
      <Specimen state="Dialog">
        <Dialog
          title="Save this trip on this device"
          description="The worksheet stays in this browser only. Nothing is sent anywhere."
          trigger={<Button variant="secondary">Save this trip</Button>}
          footer={
            <>
              <Button variant="primary">Save on this device</Button>
              <Button variant="text" href="#worksheet">
                Keep editing
              </Button>
            </>
          }
        >
          <p>Saved trips are listed under My Coast. Clearing this browser&rsquo;s data removes them.</p>
        </Dialog>
      </Specimen>
      <Specimen state="Tooltip" note="supplementary words only">
        <Tooltip content="Zoom in on the chart">
          <button type="button" className={s.iconButton} aria-label="Zoom in">
            <Icon name="plus" />
          </button>
        </Tooltip>
      </Specimen>
    </Section>
  );
}

function Icons() {
  return (
    <Section id="icons" name="Icons" purpose="Drawn for Neritic on a 24 grid at 1.5 px, from chart symbology. Ink only, and never without a word unless the control is universal.">
      <Specimen state="The set" wide>
        <ul className={s.icons}>
          {ICON_NAMES.map((n) => (
            <li key={n}>
              <Icon name={n} size={32} />
              <span>{n}</span>
            </li>
          ))}
        </ul>
      </Specimen>
    </Section>
  );
}

function States() {
  const conflict = S.official_notices.records_for_port_592.find((r) => r.disputed)!;
  return (
    <Section id="states" name="Designed states" purpose="Written for the reader: what is missing, why, and what to do. Never a lone spinner, never a blank.">
      <Specimen state="Unavailable">
        <Unavailable what="Crab tests aren't available in this build." why="CDPH test results are read on the staging channel only; production has no test file yet." next={<Button variant="text" href="https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx" external>Read CDPH&rsquo;s results</Button>} />
      </Specimen>
      <Specimen state="Stale">
        <Stale date={D.crabSampled} now={NOW} why="CDPH tests on no fixed schedule; the next result has no date.">
          Dungeness viscera, highest 9.8 of 30 ppm
        </Stale>
      </Specimen>
      <Specimen state="Offline">
        <Offline savedAt={D.savedAt} now={NOW} />
      </Specimen>
      <Specimen state="Loading">
        <Loading>Reading buoy 46236…</Loading>
      </Specimen>
      <Specimen state="Sources differ" wide>
        <SourcesDiffer
          subject={conflict.title}
          checked={D.checked}
          now={NOW}
          claims={[
            { agency: "CDPH", quote: conflict.summary, reference: `SN26-018, ${day(conflict.effective_date)}`, url: conflict.source_url },
            { agency: "CDPH", quote: "CDPH Lifts Warnings about Sport-Harvested Bivalve Shellfish from Monterey County", reference: "SN26-021", url: "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx" },
          ]}
          next={<>Check CDPH before harvesting.</>}
        />
      </Specimen>
    </Section>
  );
}
