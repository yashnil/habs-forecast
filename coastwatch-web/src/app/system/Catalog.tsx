"use client";
import { CHART_MODE_LABEL, CHART_MODES, ChartModeRoot, Legend, Segmented, TooltipProvider, day, type ChartMode } from "@/nx/ds";
import { useState } from "react";
import { NOW } from "./data";
import { SectionsA } from "./SectionsA";
import { SectionsB } from "./SectionsB";
import { SectionsC } from "./SectionsC";
import { SectionsD } from "./SectionsD";
import s from "./catalog.module.css";

export const INDEX: Array<[string, string]> = [
  ["v14-grammar", "Data-state grammar (v14)"],
  ["v14-buttons", "Buttons, chips, tabs (v14)"],
  ["v14-ticket", "Ticket (v14)"],
  ["v14-dock", "Dock, SeaBars (v14)"],
  ["v14-pass", "Pass, Waterfall, Dumbbell (v14)"],
  ["foundations", "Foundations"],
  ["statement", "Statement"],
  ["bigdate", "BigDate"],
  ["agenda", "Agenda"],
  ["marks", "Mark and Legend"],
  ["stamp", "Stamp and ProvenanceSheet"],
  ["margin", "Margin and MarginRow"],
  ["sheet", "MarginSheet"],
  ["sounding", "Sounding and ChartNote"],
  ["ramp", "RampLegend and Readout"],
  ["series", "SeriesChart"],
  ["multiples", "SmallMultiples and Brush"],
  ["worksheet", "Worksheet"],
  ["buttons", "Button"],
  ["portfinder", "PortFinder"],
  ["overlays", "Popover, Dialog, Tooltip"],
  ["icons", "Icons"],
  ["states", "Designed states"],
  ["desk", "Desk (v13)"],
  ["controls", "Controls (v13)"],
  ["figure", "Figure"],
  ["datastate", "DataState, Availability"],
  ["plot", "Plot kit"],
  ["money", "Waterfall, grid, sweep"],
  ["gates", "GateList, Advisories"],
];

export function Catalog({ initialMode }: { initialMode: ChartMode }) {
  const [mode, setMode] = useState<ChartMode>(initialMode);
  const choose = (m: ChartMode) => {
    setMode(m);
    const u = new URL(window.location.href);
    u.searchParams.set("mode", m);
    window.history.replaceState(null, "", u);
  };
  return (
    <ChartModeRoot mode={mode} className={s.page}>
      <TooltipProvider>
        <header className={s.masthead}>
          <div className={s.mastLeft}>
            <p className={s.wordmark}>Neritic</p>
            <p className={s.mastNote}>Design system · v14 synthesis, with the v12 and v13 parts</p>
          </div>
          <Segmented label="Chart mode" hideLabel options={CHART_MODES.map((m) => ({ value: m, label: CHART_MODE_LABEL[m] }))} value={mode} onChange={choose} />
        </header>
        <div className={s.layout}>
          <nav className={s.index} aria-label="Components">
            <ol>
              {INDEX.map(([id, name]) => (
                <li key={id}>
                  <a href={`#${id}`}>{name}</a>
                </li>
              ))}
            </ol>
          </nav>
          <main className={s.main}>
            <div className={s.intro}>
              <h1 className={s.title}>The component catalog</h1>
              <p className={s.lede}>
                Every component of the chart system, in each of its states, set with Moss Landing data as of {day(NOW, { year: true })}. Switch the chart mode above; the page is the review and screenshot surface, and it is not indexed.
              </p>
              <Legend kinds={["measured", "model", "official", "noview", "stale"]} title="The five marks, stated once for this page" />
            </div>
            <SectionsD />
            <SectionsA mode={mode} />
            <SectionsB />
            <SectionsC />
          </main>
        </div>
      </TooltipProvider>
    </ChartModeRoot>
  );
}
