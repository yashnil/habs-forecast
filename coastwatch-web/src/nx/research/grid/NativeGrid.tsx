"use client";

/**
 * v11's native-grid instrument, kept whole for v12 Research (capability inventory: "Native grid: cells,
 * transects, area stats"; "Exports CSV with provenance (cell, transect, area)"; "Lineage"). It reads the
 * published C-HARM, satellite-chlorophyll and HF-radar grids near a station, cell by cell, with the same
 * reads, exports and lineage as v11 (src/v4/research). It opens on demand in a dialog with its own small map,
 * loaded only then, so the record's first load carries none of it. Its v4 parts are dressed in the chart
 * mode's tokens (grid.module.css); restyling them natively is a follow-up.
 */
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, satelliteLatest, tileIndexUrl, type CharmVariable } from "@/lib/layers";
import { currentsHourly, fieldFeatures } from "@/lib/currents";
import { cellEdges, loadGrid } from "@/lib/grid";
import { labelledGeometry } from "@/lib/officialGeometry";
import { Dialog } from "@/nx/ds/Overlay";
import { literata, publicSans } from "@/nx/ds/fonts";
import { day } from "@/nx/ds/format";
import { ReadinessProvider } from "@/v4/readiness/store";
import { AreaModel, CellBody } from "@/v4/explore/Inspect";
import { useCellRead } from "@/v4/explore/reads";
import { VARIABLE_WORDS } from "@/v4/explore/layers";
import { boxFeature, useAreaDrag, type BBox } from "@/v4/explore/AreaDrag";
import { transect, type TransectCell } from "@/v4/research/transect";
import { ExportButton, QUIET, Seg, TransectChart } from "@/v4/research/parts";
import { areaCsv, cellLeadsCsv, download, transectCsv } from "@/v4/research/exports";
import { Lineage } from "@/v4/research/Lineage";
import { footprintOf, footprintOutline, obsArea, obsPoint, obsTransect, pixelPolygon, radarLayerFor, satLayerFor, type BBox as ObsBox, type RadarChoice, type SatChoice } from "@/v4/research/obsGrid";
import { ObsArea, ObsControls, ObsNote, ObsPointPanel, ObsTransect, obsSensor, obsWhen, useObsSampler, type ObsSource } from "@/v4/research/ObsTools";
import g from "./grid.module.css";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

export type NativeGridProps = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  baseUrl: string;
  station: { station_id: string; name: string; lat: number; lon: number };
  now: string;
  open: boolean;
  onOpenChange: (o: boolean) => void;
};

export function NativeGrid(props: NativeGridProps) {
  return (
    <Dialog
      title={`The native grid near ${props.station.name}`}
      description="Published model and observation grids, cell by cell. A cell without a value is a gap, never a low value."
      open={props.open}
      onOpenChange={props.onOpenChange}
      // the dialog portals outside the route group's font scope: carry the two faces' variables with it
      className={`${g.dialog} ${literata.variable} ${publicSans.variable}`}
    >
      <ReadinessProvider>
        <GridInstrument {...props} />
      </ReadinessProvider>
    </Dialog>
  );
}

function GridInstrument({ manifest, ports, official, baseUrl, station: st, now: nowIso }: NativeGridProps) {
  const now = useMemo(() => new Date(nowIso), [nowIso]);
  const run = charmRun(manifest);
  const leads = useMemo(() => (run?.leads_available ?? []).filter((l) => (CHARM_LEADS as readonly number[]).includes(l)).sort((a, b) => a - b), [run]);
  const [gridVar, setGridVar] = useState<CharmVariable>("particulate_domoic");
  const [leadSel, setLead] = useState<number | null>(null);
  const lead = leadSel != null && leads.includes(leadSel) ? leadSel : leads.includes(0) ? 0 : (leads[0] ?? 0);
  const [showEdges, setShowEdges] = useState(true);
  const [tool, setTool] = useState<"cell" | "transect" | "area">("cell");
  const [cellAtPt, setCellAtPt] = useState<{ lat: number; lon: number } | null>(null);
  const [line, setLine] = useState<{ lat: number; lon: number }[]>([]);
  const [area, setArea] = useState<BBox | null>(null);
  const [areaArmed, setAreaArmed] = useState(true);
  const mapRef = useRef<MlMap | null>(null);
  const { dragBox, handlers: areaHandlers } = useAreaDrag(() => mapRef.current, setArea, () => setAreaArmed(false));
  const cellRead = useCellRead(manifest, baseUrl, ports, cellAtPt, gridVar, lead);
  const forecastLayer = charmLayer(manifest, gridVar, lead);
  const [codes, setCodes] = useState<{ id: string; codes: Uint16Array } | null>(null);
  useEffect(() => {
    const l = forecastLayer;
    if (!l?.grid || codes?.id === l.layer_id) return;
    let live = true;
    loadGrid(artifactUrl(baseUrl, l.grid.url))
      .then((c) => live && setCodes({ id: l.layer_id, codes: c }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [forecastLayer, baseUrl, codes?.id]);
  const active = forecastLayer?.grid && codes?.id === forecastLayer.layer_id ? codes.codes : null;
  const edges = useMemo(() => (forecastLayer?.grid && active ? cellEdges(forecastLayer.grid, active) : null), [forecastLayer, active]);
  const profile = useMemo<TransectCell[] | null>(() => (line.length === 2 && forecastLayer?.grid && active ? transect(forecastLayer.grid, active, line[0], line[1]) : null), [line, forecastLayer, active]);

  const [source, setSource] = useState<ObsSource>("charm");
  const [sat, setSat] = useState<SatChoice>(() => ({ product: satelliteLatest(manifest, "multi") ? "multi" : "olci300", day: null }));
  const [radar, setRadar] = useState<RadarChoice>({ hour: null, mean: false });
  const hasSat = !!(satelliteLatest(manifest, "multi") ?? satelliteLatest(manifest, "olci300") ?? satelliteLatest(manifest, "viirs750"));
  const hasRadar = currentsHourly(manifest).length > 0;
  const obsLayer = source === "satellite" ? satLayerFor(manifest, sat) : source === "radar" ? radarLayerFor(manifest, radar) : null;
  const obsBox = useMemo<ObsBox | null>(() => {
    if (tool === "cell" && cellAtPt) return [cellAtPt.lon - 0.01, cellAtPt.lat - 0.01, cellAtPt.lon + 0.01, cellAtPt.lat + 0.01];
    if (tool === "transect" && line.length === 2) return [Math.min(line[0].lon, line[1].lon), Math.min(line[0].lat, line[1].lat), Math.max(line[0].lon, line[1].lon), Math.max(line[0].lat, line[1].lat)];
    if (tool === "area" && area) return [area[0] - 0.02, area[1] - 0.02, area[2] + 0.02, area[3] + 0.02];
    return null;
  }, [tool, cellAtPt, line, area]);
  const obsRead = useObsSampler(manifest, baseUrl, obsLayer, obsBox);
  const obsS = obsRead.sampler;
  const obsPx = useMemo(() => (obsS && tool === "cell" && cellAtPt ? obsPoint(obsS, cellAtPt.lat, cellAtPt.lon) : null), [obsS, tool, cellAtPt]);
  const obsProfile = useMemo(() => (obsS && tool === "transect" && line.length === 2 ? obsTransect(obsS, line[0], line[1]) : null), [obsS, tool, line]);
  const obsCells = useMemo(() => (obsS && tool === "area" && area ? obsArea(obsS, area) : null), [obsS, tool, area]);
  const obsFootprint = useMemo<GeoJSON.Feature[]>(() => {
    if (!obsS) return [];
    if (obsPx?.value != null) {
      const fp = footprintOf(obsS, obsPx);
      return [{ type: "Feature", geometry: pixelPolygon(fp.g, fp.row, fp.col), properties: { kind: "pixel" } }];
    }
    const on = (obsProfile ?? obsCells ?? []).filter((c) => c.value != null && ("inDomain" in c ? c.inDomain : true));
    return on.length ? [{ type: "Feature", geometry: footprintOutline(obsS.g, on), properties: { kind: "footprint" } }] : [];
  }, [obsS, obsPx, obsProfile, obsCells]);
  const satRaster = useMemo(() => {
    const t = source === "satellite" ? obsLayer?.tiles : null;
    return obsLayer && t ? { id: `r-${obsLayer.layer_id}`, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) } : null;
  }, [source, obsLayer, baseUrl]);
  const radarField = source === "radar" && obsS?.kind === "radar" ? obsS.field : null;
  const currentsOverlay = useMemo(() => (radarField && obsS ? { id: `r-${obsS.layer.layer_id}`, field: radarField, features: fieldFeatures(radarField), mode: "arrows" as const } : null), [radarField, obsS]);
  const highlight = useMemo<GeoJSON.FeatureCollection>(() => {
    const f: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [st.lon, st.lat] }, properties: { kind: "station", label: st.name } }];
    if (source === "charm" && tool === "cell" && cellRead?.status === "ready" && cellRead.polygon) f.push({ type: "Feature", geometry: cellRead.polygon, properties: { kind: "cell" } });
    if (source !== "charm" && tool === "cell" && cellAtPt) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [cellAtPt.lon, cellAtPt.lat] }, properties: { kind: "point" } });
    if (source !== "charm") f.push(...obsFootprint);
    if (tool === "transect" && line.length === 2) f.push({ type: "Feature", geometry: { type: "LineString", coordinates: line.map((p) => [p.lon, p.lat]) }, properties: { kind: "transect" } });
    else if (tool === "transect" && line.length === 1) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [line[0].lon, line[0].lat] }, properties: { kind: "point" } });
    const box = tool === "area" ? (dragBox ?? area) : null;
    if (box) f.push(boxFeature(box));
    return { type: "FeatureCollection", features: f };
  }, [st, cellRead, line, tool, dragBox, area, source, cellAtPt, obsFootprint]);
  const runLayers = CHARM_LEADS.map((l) => charmLayer(manifest, gridVar, l)).filter((l): l is NonNullable<typeof l> => !!l);
  const fileTag = `${gridVar}-${forecastLayer?.time.valid_date ?? "na"}`;
  const pickSource = (x: ObsSource) => {
    setSource(x);
    setLine([]);
  };

  return (
    <div className={g.scope} data-testid="research-grid">
      <h3 className={g.heading} data-testid="grid-heading">
        {source === "charm" ? `C-HARM · ${VARIABLE_WORDS[gridVar].short} · ${forecastLayer?.time.valid_date ? `${lead === 0 ? "nowcast" : "forecast"} for ${day(forecastLayer.time.valid_date, { year: true })}` : "no run"}` : `${source === "satellite" ? "Satellite chlorophyll" : "Surface currents"} · ${obsSensor(obsLayer)} · ${obsWhen(obsLayer)}`}
      </h3>
      <div className={g.controls}>
        <Field label="Data source">
          <Seg
            label="Data source"
            value={source}
            onChange={pickSource}
            options={(
              [
                ["charm", "C-HARM model", true],
                ["satellite", "Satellite chlorophyll", hasSat],
                ["radar", "Surface currents", hasRadar],
              ] as const
            )
              .filter(([, , ok]) => ok)
              .map(([id, label]) => ({ id, label, testId: `gsrc-${id}` }))}
          />
        </Field>
        <Field label="Tool">
          <Seg
            label="Map tool"
            value={tool}
            onChange={(t) => {
              setTool(t);
              setLine([]);
              if (t === "area") setAreaArmed(true);
            }}
            options={(["cell", "transect", "area"] as const).map((t) => ({ id: t, testId: `tool-${t}`, label: t === "cell" ? (source === "satellite" ? "Inspect a pixel" : "Inspect a cell") : t === "transect" ? "Draw a transect" : "Select an area" }))}
          />
        </Field>
      </div>
      {source !== "charm" ? (
        <>
          <div className={g.controls} data-testid="obs-controls">
            <ObsControls manifest={manifest} source={source} sat={sat} onSat={setSat} radar={radar} onRadar={setRadar} now={now} />
          </div>
          <ObsNote source={source} />
        </>
      ) : (
        <div className={g.controls} data-testid="grid-controls">
          <Field label="C-HARM variable">
            <Seg label="C-HARM variable" value={gridVar} onChange={setGridVar} options={CHARM_VARIABLES.map((v) => ({ id: v, label: VARIABLE_WORDS[v].short, testId: `gvar-${v}` }))} />
          </Field>
          <Field label={`Day of this run (issued ${run ? `${day(run.issued_date, { year: true })}${run.issued_date_derived ? ", inferred" : ""}` : "—"})`}>
            <Seg label="Day of the run" value={lead} onChange={setLead} options={leads.map((l) => ({ id: l, testId: `lead-${l}`, label: `${charmLayer(manifest, gridVar, l)?.time.valid_date ? day(charmLayer(manifest, gridVar, l)!.time.valid_date!) : "—"} · ${l === 0 ? "nowcast" : `+${l}`}` }))} />
          </Field>
          <label className={g.check}>
            <input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} data-testid="grid-edges" />
            Cell edges
          </label>
        </div>
      )}
      <div className={g.body}>
        <div className={g.mapCol}>
          <div className={g.map} data-testid="research-map">
            <MapCanvas
              forecast={source === "charm" && forecastLayer?.image ? { id: `r-${forecastLayer.layer_id}`, url: artifactUrl(baseUrl, forecastLayer.image.url), corners: forecastLayer.image.corners_lnglat, grid: forecastLayer.grid ? { layer: forecastLayer, url: artifactUrl(baseUrl, forecastLayer.grid.url) } : null } : null}
              satellite={satRaster}
              satelliteAge={null}
              currents={currentsOverlay}
              imagery={null}
              ports={ports}
              showPorts
              officialGeometry={labelledGeometry(official)}
              showOfficial
              selectedPort={null}
              inspect={null}
              chrome="none"
              highlight={highlight}
              cellEdges={source === "charm" && showEdges ? edges : null}
              initialBounds={[[st.lon - 0.35, st.lat - 0.25], [st.lon + 0.35, st.lat + 0.25]]}
              initialPadding={16}
              onPort={() => undefined}
              onReady={(m) => {
                mapRef.current = m;
              }}
              onPoint={(p) => {
                if (tool === "cell") setCellAtPt(p);
                else if (tool === "transect") setLine((l) => (l.length >= 2 ? [p] : [...l, p]));
              }}
            />
            {tool === "area" && areaArmed ? (
              <div {...areaHandlers} className={g.areaOverlay} data-testid="area-overlay">
                <p>Drag across the water to select an area</p>
              </div>
            ) : null}
          </div>
          <p className={g.small}>© OpenStreetMap · OpenMapTiles · NOAA relief · not for navigation</p>
          {tool === "transect" ? (
            <div data-testid="transect">
              {line.length < 2 ? (
                <p className={g.small}>Click two points on the map. The profile lists every published {source === "satellite" ? "pixel" : "cell"} the line crosses; gaps stay gaps.</p>
              ) : source !== "charm" ? (
                <ObsTransect manifest={manifest} read={obsRead} cells={obsProfile} a={line[0]} b={line[1]} />
              ) : profile && forecastLayer ? (
                <>
                  <TransectChart cells={profile} a={line[0]} b={line[1]} />
                  <ExportButton testid="export-transect" onClick={() => download(`neritic-transect-${fileTag}.csv`, transectCsv(line[0], line[1], profile, forecastLayer, manifest))}>
                    Transect cells (CSV)
                  </ExportButton>
                </>
              ) : (
                <p className={g.small}>Reading the grid…</p>
              )}
            </div>
          ) : null}
        </div>
        <div className={g.readCol}>
          {tool === "cell" && source !== "charm" ? (
            cellAtPt ? (
              <ObsPointPanel manifest={manifest} baseUrl={baseUrl} read={obsRead} px={obsPx} point={cellAtPt} sat={sat} now={now} />
            ) : (
              <p>{source === "satellite" ? "Click the water to read one published pixel: its own observation date, sensor, native size and footprint. An empty pixel is no observation, never low chlorophyll." : "Click the water to read one published 2 km radar cell. An empty cell is no observation, never calm water."}</p>
            )
          ) : null}
          {tool === "cell" && source === "charm" ? (
            cellAtPt ? (
              <div className={g.stack}>
                <CellBody read={cellRead} ctx={{ manifest, baseUrl, ports, portIntel: null, stations: null, now, variable: gridVar, lead, onSelect: (x) => x.kind === "cell" && setCellAtPt({ lat: x.lat, lon: x.lon }) }} variable={gridVar} precise={{ quantErr: forecastLayer?.grid?.max_quantization_error ?? null }} />
                {cellRead?.status === "ready" && cellRead.cell ? (
                  <ExportButton testid="export-cell" onClick={() => download(`neritic-cell-${cellRead.cell!.row}-${cellRead.cell!.col}-${gridVar}.csv`, cellLeadsCsv(cellRead.cell!, cellRead.leads, runLayers, manifest, gridVar))}>
                    This cell, every day of the run (CSV)
                  </ExportButton>
                ) : null}
              </div>
            ) : (
              <p>Click the water to read a native 3 km cell: its published value, footprint, the run&rsquo;s days and whether it represents a harbour. Each value is model output for one cell, not a toxin measurement.</p>
            )
          ) : null}
          {tool === "area" ? (
            area ? (
              <div className={g.stack} data-testid="inspect-area">
                {source !== "charm" ? (
                  <ObsArea manifest={manifest} baseUrl={baseUrl} read={obsRead} cells={obsCells} bbox={area} now={now} />
                ) : (
                  <AreaModel
                    layer={forecastLayer}
                    baseUrl={baseUrl}
                    bbox={area}
                    variable={gridVar}
                    footer={(cells) =>
                      forecastLayer && (
                        <ExportButton testid="export-area" onClick={() => download(`neritic-area-${fileTag}.csv`, areaCsv(area, cells, forecastLayer, manifest))}>
                          Cells in the box (CSV)
                        </ExportButton>
                      )
                    }
                  />
                )}
                <button type="button" onClick={() => setAreaArmed(true)} className={`${QUIET} self-start`} data-testid="area-redraw">
                  Draw another box
                </button>
              </div>
            ) : (
              <p>Drag a box across the water. It counts the cells with and without a published value and gives the spread of the values; gaps are counted, never filled.</p>
            )
          ) : null}
        </div>
      </div>
      <details className={g.lineage}>
        <summary>Lineage and citation of this grid</summary>
        {source === "charm" ? forecastLayer && <Lineage layer={forecastLayer} manifest={manifest} /> : obsLayer && <Lineage layer={obsLayer} manifest={manifest} />}
      </details>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={g.field}>
      <span className={g.fieldLabel}>{label}</span>
      {children}
    </div>
  );
}
