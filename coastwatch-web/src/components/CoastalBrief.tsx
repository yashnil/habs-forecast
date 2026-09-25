"use client";

import type { FisheriesContext } from "@/lib/types";
import { GIBS_SATELLITE, GIBS_PACE_CHL } from "@/lib/gibs";
import { marineForecastUrlForRegion, regionForecastLabel } from "@/lib/nws";

type Props = {
  fisheries: FisheriesContext;
  region: string;
  viirsDate: string;
  paceDate: string;
};

export default function CoastalBrief({
  fisheries,
  region,
  viirsDate,
  paceDate,
}: Props) {
  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 shadow-lg">
      <h2 className="text-sm font-semibold text-slate-100">Coastal brief</h2>
      <p className="mt-1 text-[11px] text-slate-500">
        Map stacks <strong>PACE</strong> ({paceDate}) under <strong>VIIRS</strong> ({viirsDate}).
        PACE often retains signal in <strong>turbid bays</strong> (e.g. SF Bay margins) where VIIRS
        L3 masks water; still not toxin or legal advice.
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-300">
          VIIRS · {viirsDate}
        </span>
        <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-300">
          PACE · {paceDate}
        </span>
        <a
          href={GIBS_SATELLITE.legendHorizontalSvg}
          target="_blank"
          rel="noreferrer"
          className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-cyan-400 hover:underline"
        >
          VIIRS legend →
        </a>
        <a
          href={GIBS_PACE_CHL.legendHorizontalSvg}
          target="_blank"
          rel="noreferrer"
          className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-cyan-400 hover:underline"
        >
          PACE legend →
        </a>
      </div>

      <p className="mt-3 text-sm text-slate-300">
        <span className="text-slate-400">Region:</span>{" "}
        <strong>{fisheries.regions?.[region]?.headline ?? region}</strong>
      </p>

      <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-slate-400">
        <li>CDPH / OEHHA for shellfish safety; CDFW for seasons, closures, MPAs.</li>
        <li>Chlorophyll is biomass proxy — not domoic acid or saxitoxin.</li>
        <li>Zoom until your grounds fill the frame; compare to water next door on the same day.</li>
      </ul>

      <div className="mt-3 border-t border-slate-700 pt-3">
        <a
          href={marineForecastUrlForRegion(region)}
          target="_blank"
          rel="noreferrer"
          className="text-sm text-cyan-400 hover:underline"
        >
          NWS marine forecast — {regionForecastLabel(region)} →
        </a>
      </div>
    </section>
  );
}
