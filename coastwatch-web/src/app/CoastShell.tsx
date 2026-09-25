"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import CoastMap from "@/components/CoastMap";
import InsightPanel from "@/components/InsightPanel";
import ComplianceBanner from "@/components/ComplianceBanner";
import CoastalBrief from "@/components/CoastalBrief";
import ZoneDrawer from "@/components/ZoneDrawer";
import ProvenanceStrip from "@/components/ProvenanceStrip";
import { nearestHarbor } from "@/lib/geo";
import { fallbackSatelliteDate } from "@/lib/gibs";
import type {
  FisheriesContext,
  HarborFeature,
  HarborsGeoJSON,
  Snapshot,
} from "@/lib/types";

function formatGenerated(iso?: string) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "UTC",
    }) + " UTC";
  } catch {
    return iso;
  }
}

export default function CoastShell() {
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [fisheries, setFisheries] = useState<FisheriesContext | null>(null);
  const [harbors, setHarbors] = useState<HarborsGeoJSON | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [userPos, setUserPos] = useState<[number, number] | null>(null);

  const [showGibs, setShowGibs] = useState(true);
  const [showHarbors, setShowHarbors] = useState(true);
  const [gibsOp, setGibsOp] = useState(0.68);

  const [viirsDate, setViirsDate] = useState<string>(() => fallbackSatelliteDate(2));
  const [paceDate, setPaceDate] = useState<string>(() => fallbackSatelliteDate(2));

  const regionKeys = useMemo(
    () => Object.keys(fisheries?.regions ?? {}),
    [fisheries],
  );
  const [region, setRegion] = useState("");
  const [didLocateRegion, setDidLocateRegion] = useState(false);
  const [drawerHarbor, setDrawerHarbor] = useState<HarborFeature | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [s, f, h] = await Promise.all([
          fetch("/data/snapshot.json").then((r) => r.json()),
          fetch("/data/fisheries_context.json").then((r) => r.json()),
          fetch("/data/ca_harbors.geojson").then((r) => r.json()),
        ]);
        if (!cancelled) {
          setSnapshot(s as Snapshot);
          setFisheries(f as FisheriesContext);
          setHarbors(h as HarborsGeoJSON);
        }
      } catch (e) {
        if (!cancelled) setErr(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/gibs-chl-meta")
      .then((r) => r.json())
      .then(
        (j: {
          date?: string;
          viirsDate?: string;
          paceDate?: string;
        }) => {
          if (cancelled) return;
          const v = j.viirsDate ?? j.date;
          const p = j.paceDate ?? v;
          if (v && /^\d{4}-\d{2}-\d{2}$/.test(v)) setViirsDate(v);
          if (p && /^\d{4}-\d{2}-\d{2}$/.test(p)) setPaceDate(p);
        },
      )
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const onUser = useCallback((p: [number, number] | null) => {
    setUserPos(p);
  }, []);

  useEffect(() => {
    if (!fisheries) return;
    if (regionKeys.length && !regionKeys.includes(region)) {
      setRegion(regionKeys[0]);
    }
  }, [fisheries, regionKeys, region]);

  useEffect(() => {
    if (!userPos || !harbors?.features.length || didLocateRegion) return;
    const n = nearestHarbor(userPos, harbors.features);
    if (n && regionKeys.includes(n.feature.properties.region_key)) {
      setRegion(n.feature.properties.region_key);
      setDidLocateRegion(true);
    }
  }, [userPos, harbors, regionKeys, didLocateRegion]);

  const onHarborSelect = useCallback((h: HarborFeature) => {
    setRegion(h.properties.region_key);
    setDrawerHarbor(h);
    setDrawerOpen(true);
  }, []);

  const closeDrawer = useCallback(() => {
    setDrawerOpen(false);
  }, []);

  if (err) {
    return (
      <div className="p-8 text-center text-red-300">
        Failed to load data: {err}. Run{" "}
        <code className="text-cyan-400">npm run sync-data</code> from{" "}
        <code className="text-cyan-400">coastwatch-web</code>.
      </div>
    );
  }

  if (!snapshot || !fisheries || !harbors) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center text-slate-400">
        Loading coast data…
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950">
        <div className="mx-auto max-w-6xl px-4 py-4">
          <p className="text-[10px] font-medium uppercase tracking-widest text-cyan-500/90">
            California Coastwatch
          </p>
          <h1 className="mt-0.5 text-xl font-bold tracking-tight sm:text-2xl">
            Coastal chlorophyll &amp; port context
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            <strong className="text-slate-300">Map:</strong> NASA GIBS VIIRS chlorophyll near the
            California coast (VIIRS {viirsDate}, PACE underlay {paceDate}). The view stays over
            shore and nearby ocean so it
            loads faster and matches how people fish. Not a toxin, closure, or legal map.
          </p>
          <p className="mt-2 text-[11px] text-slate-500">
            App bundle: {formatGenerated(snapshot.generated_at)} (metadata for this build only)
          </p>
        </div>
      </header>

      <ComplianceBanner />

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-6">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/30 p-4 shadow-lg">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
            <h2 className="text-sm font-semibold text-slate-200">Map</h2>
            <p className="text-[11px] text-slate-500">
              Open NASA browse tiles. Zoom to your grounds; use the official color scale (sidebar)
              to read chlorophyll.
            </p>
          </div>

          <div className="mt-3 space-y-3 rounded-xl border border-slate-800/80 bg-slate-950/50 p-3">
            <label className="flex cursor-pointer items-start gap-3 text-sm text-slate-200">
              <input
                type="checkbox"
                className="mt-1"
                checked={showGibs}
                onChange={(e) => setShowGibs(e.target.checked)}
              />
              <span>
                <span className="font-medium text-cyan-300">NASA GIBS chlorophyll</span>
                <span className="block text-[11px] font-normal text-slate-500">
                  VIIRS + PACE chlorophyll (see map footnote for dates)
                </span>
              </span>
            </label>
            <label className="flex cursor-pointer items-start gap-3 text-sm text-slate-200">
              <input
                type="checkbox"
                className="mt-1"
                checked={showHarbors}
                onChange={(e) => setShowHarbors(e.target.checked)}
              />
              <span>
                <span className="font-medium text-slate-200">Ports</span>
                <span className="block text-[11px] font-normal text-slate-500">
                  Tap a port; zoom in for names. Drawer is context only — not legal go/no-go
                </span>
              </span>
            </label>
            {showGibs && (
              <label className="flex items-center gap-2 pl-7 text-xs text-slate-500">
                Chlorophyll opacity
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.02}
                  value={gibsOp}
                  onChange={(e) => setGibsOp(Number(e.target.value))}
                />
              </label>
            )}
          </div>

          <div className="mt-4">
            <CoastMap
              token={token}
              harbors={harbors}
              viirsDate={viirsDate}
              paceDate={paceDate}
              showGibs={showGibs}
              gibsOpacity={gibsOp}
              showHarbors={showHarbors}
              onUserLocation={onUser}
              onHarborSelect={onHarborSelect}
            />
          </div>

          <p className="mt-3 text-center text-[11px] text-slate-500">
            <a
              className="text-cyan-500 hover:underline"
              href="https://wiki.earthdata.nasa.gov/display/GIBS/"
              target="_blank"
              rel="noreferrer"
            >
              NASA GIBS
            </a>{" "}
            (open) ·{" "}
            <a
              className="text-cyan-500 hover:underline"
              href="https://account.mapbox.com/access-tokens/"
              target="_blank"
              rel="noreferrer"
            >
              Mapbox token
            </a>{" "}
            for basemap
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
          <InsightPanel
            snapshot={snapshot}
            fisheries={fisheries}
            userPos={userPos}
            region={regionKeys.includes(region) ? region : regionKeys[0] ?? ""}
            onRegionChange={setRegion}
          />
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
              <h2 className="text-sm font-semibold text-slate-200">Bundled notes</h2>
              <p className="mt-2 text-sm text-slate-400">{snapshot.viewer?.time_headline}</p>
              <p className="mt-1 text-xs text-slate-500">{snapshot.viewer?.time_detail}</p>
              {snapshot.viewer?.composite_note && (
                <p className="mt-2 text-xs text-slate-500">{snapshot.viewer.composite_note}</p>
              )}
            </div>
            <CoastalBrief
              fisheries={fisheries}
              region={regionKeys.includes(region) ? region : regionKeys[0] ?? region}
              viirsDate={viirsDate}
              paceDate={paceDate}
            />
          </div>
        </div>
      </main>

      <ZoneDrawer
        open={drawerOpen}
        harbor={drawerHarbor}
        fisheries={fisheries}
        onClose={closeDrawer}
      />

      <ProvenanceStrip
        snapshot={snapshot}
        viirsDate={viirsDate}
        paceDate={paceDate}
      />
    </div>
  );
}
