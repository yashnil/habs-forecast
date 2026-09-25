"use client";

import { useMemo } from "react";
import type { FisheriesContext, Snapshot } from "@/lib/types";
import { officialChecklist } from "@/lib/recommendations";
import { marineForecastUrlForRegion, regionForecastLabel } from "@/lib/nws";
import { GIBS_PACE_CHL, GIBS_SATELLITE } from "@/lib/gibs";

type Props = {
  snapshot: Snapshot;
  fisheries: FisheriesContext;
  userPos: [number, number] | null;
  region: string;
  onRegionChange: (regionKey: string) => void;
};

function SatelliteGuide() {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/90 p-3">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-cyan-400/90">
        Reading VIIRS chlorophyll (map)
      </p>
      <ul className="list-disc space-y-1.5 pl-4 text-[11px] leading-relaxed text-slate-400">
        <li>
          The map stacks{" "}
          <a
            href={GIBS_PACE_CHL.legendHorizontalSvg}
            target="_blank"
            rel="noreferrer"
            className="text-cyan-400 hover:underline"
          >
            PACE Chl-a
          </a>{" "}
          (under) and{" "}
          <a
            href={GIBS_SATELLITE.legendHorizontalSvg}
            target="_blank"
            rel="noreferrer"
            className="text-cyan-400 hover:underline"
          >
            VIIRS Chl-a
          </a>{" "}
          (on top). Scales differ slightly — use each legend for quantitative read.
        </li>
        <li>
          <strong className="text-slate-300">Bays &amp; plumes:</strong> PACE often fills
          nearshore / turbid areas (e.g. SF Bay) where VIIRS L3 is masked; still compare to
          neighbors and official HAB notices.
        </li>
        <li>
          Zoom until your grounds fill the frame. If the basemap goes blank after heavy zooming,
          refresh the page — the app caps zoom to keep GIBS tiles stable.
        </li>
      </ul>
    </div>
  );
}

export default function InsightPanel({
  snapshot,
  fisheries,
  userPos,
  region,
  onRegionChange,
}: Props) {
  const regionKeys = useMemo(
    () => Object.keys(fisheries.regions ?? {}),
    [fisheries],
  );

  const fish = fisheries.regions?.[region];

  return (
    <div className="flex flex-col gap-4">
      <SatelliteGuide />

      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <h3 className="text-sm font-semibold text-slate-100">Your area</h3>
        <p className="mt-1 text-xs text-slate-400">
          Pick a coast band or use <strong>locate</strong> to snap to the nearest port region.
        </p>
        <select
          className="mt-3 w-full rounded-lg border border-slate-600 bg-slate-950 px-3 py-2 text-sm text-slate-100"
          value={regionKeys.includes(region) ? region : regionKeys[0] ?? ""}
          onChange={(e) => onRegionChange(e.target.value)}
        >
          {regionKeys.map((k) => (
            <option key={k} value={k}>
              {fisheries.regions?.[k]?.headline ?? k}
            </option>
          ))}
        </select>
        {userPos && (
          <p className="mt-2 text-[11px] text-slate-500">
            Your location: {userPos[1].toFixed(3)}°N, {Math.abs(userPos[0]).toFixed(3)}°W
          </p>
        )}
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <h3 className="text-sm font-semibold text-slate-100">Target species (general)</h3>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">
          {fish?.typical_targets ||
            "Add regional text in fisheries_context.json for this app."}
        </p>
        {fish?.operational_note && (
          <p className="mt-2 text-xs leading-relaxed text-slate-500">{fish.operational_note}</p>
        )}
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <h3 className="text-sm font-semibold text-slate-100">Operational tips</h3>
        <ul className="mt-2 list-disc space-y-2 pl-4 text-sm text-slate-300">
          <li>
            Pair chlorophyll with <strong>wind, swell, and biotoxin bulletins</strong> — blooms in
            the news do not always match a single satellite overpass.
          </li>
          <li>
            If buyers are sensitive to “red tide” headlines, keep links to{" "}
            <strong>official</strong> state bulletins handy when discussing landings.
          </li>
          <li>
            Higher chlorophyll near the coast often reflects upwelling or river plumes — use local
            knowledge and sampling programs, not color alone.
          </li>
        </ul>
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <h3 className="text-sm font-semibold text-slate-100">Official checks</h3>
        <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-slate-400">
          {officialChecklist().map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
        <div className="mt-3 space-y-1">
          <a
            href={marineForecastUrlForRegion(region)}
            target="_blank"
            rel="noreferrer"
            className="block text-xs text-cyan-400 hover:underline"
          >
            NWS marine forecast ({regionForecastLabel(region)}) →
          </a>
          {fish?.links?.map((l) => (
            <a
              key={l.url}
              href={l.url}
              target="_blank"
              rel="noreferrer"
              className="block text-xs text-cyan-400 hover:underline"
            >
              {l.label}
            </a>
          ))}
        </div>
      </div>

      {snapshot.viewer?.what_map_shows && (
        <p className="text-[11px] leading-relaxed text-slate-600">{snapshot.viewer.what_map_shows}</p>
      )}

      {fisheries.disclaimer && (
        <p className="text-[11px] leading-relaxed text-slate-600">{fisheries.disclaimer}</p>
      )}
    </div>
  );
}
