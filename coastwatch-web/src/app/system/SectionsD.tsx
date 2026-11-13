"use client";
import { useState } from "react";
import {
  Button,
  Chip,
  DatesLine,
  Dock,
  Dumbbell,
  FollowButton,
  GLYPH_WORDS,
  Glyph,
  LimitGauge,
  MAP_LAYERS,
  Micro,
  Pass,
  Pennant,
  RollingNumber,
  SeaBars,
  Slot,
  StampLine,
  Statement,
  TabPanel,
  Tabs,
  Ticket,
  TicketMain,
  TicketStub,
  Waterfall,
  WindArrow,
  type GlyphKind,
} from "@/nx/ds";
import { Section, Specimen } from "./parts";

/*
 * v14 synthesis parts (docs/coastwatch/v14/design-system/README.md). The trip numbers are the approved brief's own
 * worked example (400 lb × $9.00, 30 gal × $8.24, $100 bait, 1 crew at 20 % after costs) and are hypothetical; the Today
 * and sea values are the synthesis audit's Moss Landing and Fort Bragg readings, set here as specimens, never as live data.
 */
const usd = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const D = (y: number, m: number, d: number, h = 0) => Date.UTC(y, m - 1, d, h);
const ML = { lon: -121.787, lat: 36.806 };

function TicketSpecimen({ followed }: { followed: boolean }) {
  const [tab, setTab] = useState("status");
  const [f, setF] = useState(followed);
  return (
    <Ticket inline at={ML} labelledBy={followed ? "sys-t2" : "sys-t1"}>
      <TicketMain>
        <Statement as="h3" id={followed ? "sys-t2" : "sys-t1"} tone="regulation" scope="Moss Landing · Dungeness crab, commercial" lede="Season delayed to at least" hero="Jan 1, 2027" />
        <StampLine>CDFW · checked Nov 8 · not verified</StampLine>
      </TicketMain>
      <TicketStub>
        <Tabs label="Detail" idBase={followed ? "sys-tb2" : "sys-tb1"} value={tab} onChange={setTab} items={[{ id: "status", label: "Status" }, { id: "record", label: "Record" }, { id: "fisheries", label: "Fisheries", count: 8 }, { id: "notices", label: "Notices", count: 3 }]} />
        <TabPanel idBase={followed ? "sys-tb2" : "sys-tb1"} id={tab} active>
          <p>
            <Micro>Why</Micro> <b>Humpback whale entanglements</b>
          </p>
          <DatesLine
            label="Dates on record: the delay took effect Oct 30; today Nov 12; next whale assessment Dec 15 to 18, subject to change; earliest start Jan 1, 2027."
            domain={[D(2026, 10, 27), D(2027, 1, 21)]}
            span={{ from: D(2026, 10, 30), to: D(2027, 1, 1), fromLabel: "Oct 30", toLabel: "earliest start" }}
            window={{ from: D(2026, 12, 15), to: D(2026, 12, 19) }}
            today={{ t: D(2026, 11, 12) }}
            ticks={[D(2026, 11, 1), D(2026, 12, 1), D(2027, 1, 1)]}
          />
          <p style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <Micro>Domoic acid test</Micro>
            <Slot>No test site for this port</Slot>
          </p>
        </TabPanel>
        <div style={{ display: "flex", gap: 14, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
          {f ? (
            <>
              <Button variant="primary" href="/coast?place=port-592&view=trip">
                Plan a trip
              </Button>
              <FollowButton name="Moss Landing" following={f} onChange={setF} variant="toggle" />
            </>
          ) : (
            <FollowButton name="Moss Landing" following={f} onChange={setF} />
          )}
        </div>
      </TicketStub>
    </Ticket>
  );
}

export function SectionsD() {
  const [layers, setLayers] = useState<Record<string, boolean>>({ regulation: true, depth: true, sea: true, toxin: false });
  const [catchLb, setCatch] = useState(400);
  const [price, setPrice] = useState(9);
  const fixed = 247.2 + 100;
  const gross = catchLb * price;
  const crew = (gross - fixed) * 0.2;
  const net = gross - fixed - crew;
  const netAt = (c: number, p: number) => (c * p - fixed) * 0.8;
  const glyphs = Object.keys(GLYPH_WORDS) as GlyphKind[];
  return (
    <>
      <Section id="v14-grammar" name="The data-state grammar (v14)" purpose="Marks, not sentences: one glyph per state, on the chart and off it.">
        <Specimen state="every kind" wide>
          <ul style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 10, listStyle: "none", padding: 0, margin: 0 }}>
            {glyphs.map((k) => (
              <li key={k} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Glyph kind={k} />
                <span>{GLYPH_WORDS[k]}</span>
              </li>
            ))}
          </ul>
        </Specimen>
        <Specimen state="micro, stamp, slot, pennants, wind, gauge" wide>
          <div style={{ display: "grid", gap: 12 }}>
            <Micro glyph="old">CDPH test · Monterey Bay · Sep 28</Micro>
            <StampLine>CDFW · checked Nov 8 · not verified</StampLine>
            <Slot>No test site for this port</Slot>
            <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <Pennant n={1} /> Small craft advisory <Pennant n={2} /> Gale warning <WindArrow fromDeg={160} /> SSE 25 kt
            </span>
            <LimitGauge value={9.8} limit={30} label="9.8 against an action level of 30 ppm" />
          </div>
        </Specimen>
      </Section>

      <Section id="v14-buttons" name="Buttons, Follow, chips, tabs, rolling figure (v14)" purpose="A primary is a small ticket whose stub's glyph says what happens. One filled button per screen.">
        <Specimen state="primary, secondary, quiet">
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
            <Button variant="primary">Sample record</Button>
            <Button variant="primary" glyph="↓">
              Save trip
            </Button>
            <Button variant="secondary" glyph="✓">
              Saved
            </Button>
            <Button variant="quiet" href="https://wildlife.ca.gov" external>
              CDFW
            </Button>
            <Button variant="primary" disabled>
              Save trip
            </Button>
          </div>
        </Specimen>
        <Specimen state="follow: not followed, followed (quiet toggle beside the next action)">
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            <FollowButton name="Moss Landing" following={false} onChange={() => {}} />
            <FollowButton name="Fort Bragg" following onChange={() => {}} variant="toggle" />
          </div>
        </Specimen>
        <Specimen state="layer chips (the legend is the control)" wide>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 2, padding: 4, background: "var(--paper)", borderRadius: 13, boxShadow: "var(--shadow-sm)", width: "fit-content", maxWidth: "100%" }}>
            {MAP_LAYERS.map((l) => (
              <Chip key={l.id} pressed={!!layers[l.id]} onPress={() => setLayers((o) => ({ ...o, [l.id]: !o[l.id] }))} swatch={l.swatch} keyText={{ question: l.question, key: l.key }} data-layer={l.id}>
                {l.name}
              </Chip>
            ))}
          </div>
        </Specimen>
        <Specimen state="rolling figure (press to change the catch)">
          <div style={{ display: "flex", alignItems: "baseline", gap: 18 }}>
            <span style={{ fontSize: 54, fontWeight: 750, letterSpacing: "-0.05em" }}>
              <RollingNumber value={net} cents format={usd} />
            </span>
            <Button variant="secondary" onClick={() => setCatch((c) => (c === 400 ? 450 : 400))}>
              Catch {catchLb === 400 ? "450" : "400"} lb
            </Button>
          </div>
        </Specimen>
      </Section>

      <Section id="v14-ticket" name="Ticket (v14)" purpose="The card beside its place: main part, perforated stub. On the chart it floats a stub's length from its port; here it stands inline.">
        <Specimen state="Today, not followed">
          <TicketSpecimen followed={false} />
        </Specimen>
        <Specimen state="Today, followed: the next action leads">
          <TicketSpecimen followed />
        </Specimen>
      </Section>

      <Section id="v14-dock" name="Dock and SeaBars (v14)" purpose="Time lives in the dock: NWS seas per period (dashed: predicted), the buoy's measured point, advisories in NWS's words.">
        <Specimen state="sea dock" wide>
          <Dock inline label="Sea, next four days" cover={false}>
            <Micro>Sea · NWS PZZ455</Micro>
            <SeaBars
              label="Seas forecast for four days, with the buoy reading and the advisories"
              domain={[D(2026, 11, 12, 14), D(2026, 11, 16, 14)]}
              measured={{ t: D(2026, 11, 12, 18), v: 13.1, label: "Buoy 46014 · now", aria: "Buoy 46014, measured 10:00 AM" }}
              advisories={[
                { id: "sca", from: D(2026, 11, 12, 14), to: D(2026, 11, 13, 8), label: "Small craft", level: 1 },
                { id: "gale", from: D(2026, 11, 13, 8), to: D(2026, 11, 14, 2), label: "Gale", level: 2 },
              ]}
              days={[0, 1, 2, 3].map((d) => ({ t: D(2026, 11, 12 + d, 20), label: ["Thu 12", "Fri 13", "Sat 14", "Sun 15"][d] }))}
              periods={[11, 11, 11, 18, 21, 15, 11, 16].map((v, i) => ({ id: String(i), from: D(2026, 11, 12, 14 + 12 * i), to: D(2026, 11, 12, 26 + 12 * i), lo: v, hi: v, name: `Period ${i + 1}`, text: `Seas ${v} ft.`, night: i % 2 === 1 }))}
            />
          </Dock>
        </Specimen>
      </Section>

      <Section id="v14-pass" name="Pass, Waterfall and Dumbbell (v14)" purpose="The trip as one bar carved from gross to net, the break-even caliper on the gross column, what-if ends that set the input.">
        <Specimen state="the brief's worked trip, hypothetical" wide>
          <Pass
            inline
            label="A hypothetical trip"
            stub={
              <>
                <Micro glyph="hypothetical">Hypothetical · net to boat</Micro>
                <span style={{ fontSize: 54, fontWeight: 750, letterSpacing: "-0.05em", margin: "8px 0" }}>
                  <RollingNumber value={net} cents format={usd} />
                </span>
                <Button variant="primary" glyph="↓" full>
                  Save trip
                </Button>
              </>
            }
          >
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))", gap: 28, padding: "20px 26px 14px 28px" }}>
              <Waterfall
                label="Hypothetical trip, gross to net to boat"
                format={usd}
                breakEven={{ at: fixed, label: `Break-even ${Math.ceil(fixed / price)} lb` }}
                steps={[
                  { id: "gross", label: "Gross", basis: `${catchLb} lb × ${usd(price)}`, amount: gross, kind: "start" },
                  { id: "fuel", label: "Fuel", basis: "30 gal × $8.24", amount: 247.2, kind: "cost" },
                  { id: "bait", label: "Bait, ice", amount: 100, kind: "cost" },
                  { id: "fees", label: "Fees", basis: "0% of gross", amount: 0, kind: "cost" },
                  { id: "crew", label: "Crew", basis: "1 × 20%", amount: crew, kind: "share" },
                  { id: "net", label: "Net to boat", amount: net, kind: "end", valueText: `${Math.round((net / gross) * 100)}% of gross` },
                ]}
              />
              <Dumbbell
                label="What if: catch and price"
                heading="What if"
                caliper={{ value: net, label: "as entered" }}
                rows={[
                  { id: "catch", label: "Catch", lo: { value: netAt(catchLb * 0.75, price), label: `${catchLb * 0.75} lb`, text: usd(netAt(catchLb * 0.75, price)), onPress: () => setCatch(catchLb * 0.75), pressLabel: "Apply" }, hi: { value: netAt(catchLb * 1.25, price), label: `${catchLb * 1.25} lb`, text: usd(netAt(catchLb * 1.25, price)), onPress: () => setCatch(catchLb * 1.25), pressLabel: "Apply" } },
                  { id: "price", label: "Price", lo: { value: netAt(catchLb, price - 1), label: `${usd(price - 1)}/lb`, text: usd(netAt(catchLb, price - 1)), onPress: () => setPrice(price - 1), pressLabel: "Apply" }, hi: { value: netAt(catchLb, price + 1), label: `${usd(price + 1)}/lb`, text: usd(netAt(catchLb, price + 1)), onPress: () => setPrice(price + 1), pressLabel: "Apply" } },
                ]}
              />
            </div>
          </Pass>
        </Specimen>
        <Specimen state="an alert's change: was (unverified) to now (illustrative, hatched)" wide>
          <Dumbbell
            label="Earliest start moved from January 1 to January 15, 2027 (illustrative)"
            slide
            domain={[D(2026, 12, 9), D(2027, 1, 21)]}
            ticks={[
              { value: D(2026, 12, 15), label: "Dec 15" },
              { value: D(2027, 1, 1), label: "Jan 1" },
              { value: D(2027, 1, 15), label: "Jan 15" },
            ]}
            delta={{ big: "+14 days", small: "later start" }}
            rows={[{ id: "start", lo: { value: D(2027, 1, 1), label: "was", mark: "hollow", tone: "ink" }, hi: { value: D(2027, 1, 15), label: "now", text: "sample", mark: "hatched", tone: "coral" } }]}
          />
        </Specimen>
      </Section>
    </>
  );
}
