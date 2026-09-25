"use client";

import type { FisheriesContext, HarborFeature } from "@/lib/types";
import { marineForecastUrlForRegion, pointForecastUrl } from "@/lib/nws";

type Props = {
  open: boolean;
  harbor: HarborFeature | null;
  fisheries: FisheriesContext;
  onClose: () => void;
};

export default function ZoneDrawer({
  open,
  harbor,
  fisheries,
  onClose,
}: Props) {
  if (!open || !harbor) return null;

  const rk = harbor.properties.region_key;
  const fish = fisheries.regions?.[rk];
  const [lon, lat] = harbor.geometry.coordinates;

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-40 bg-black/50 md:bg-black/40"
        aria-label="Close zone details"
        onClick={onClose}
      />
      <aside
        className="fixed bottom-0 right-0 z-50 max-h-[85vh] w-full max-w-md overflow-y-auto border-l border-t border-slate-700 bg-slate-950 p-4 shadow-2xl md:top-0 md:h-full md:max-h-none md:border-t-0"
        role="dialog"
        aria-labelledby="zone-drawer-title"
      >
        <div className="flex items-start justify-between gap-2">
          <h2 id="zone-drawer-title" className="text-lg font-bold text-white">
            {harbor.properties.name}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            Close
          </button>
        </div>
        <p className="mt-1 font-mono text-[11px] text-slate-500">
          {lat.toFixed(4)}°N, {Math.abs(lon).toFixed(4)}°W
        </p>

        <div className="mt-4 space-y-4 text-sm">
          <div className="rounded-lg border border-amber-900/50 bg-amber-950/30 p-3 text-xs text-amber-100/90">
            <strong>Legal / compliance:</strong> This panel does <strong>not</strong> tell you if
            fishing or harvest is allowed. Verify <strong>CDFW, MPAs, and CDPH/OEHHA</strong> for
            your species and area. Satellite color is <strong>not</strong> a go/no-go for legality
            or toxins.
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase text-cyan-400">Port &amp; band</h3>
            <p className="mt-1 text-slate-300">
              Marker from the California harbor list. Coast band:{" "}
              <strong>{fish?.headline ?? rk}</strong>. On the map, read{" "}
              <strong>VIIRS chlorophyll</strong> in the water near this port — zoom in until your
              fishing grounds are in frame.
            </p>
          </div>

          {fish?.typical_targets && (
            <div>
              <h3 className="text-xs font-semibold text-slate-300">Typical targets (general)</h3>
              <p className="mt-1 text-slate-300">{fish.typical_targets}</p>
            </div>
          )}

          {fish?.operational_note && (
            <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-3 text-xs text-slate-400">
              {fish.operational_note}
            </div>
          )}

          <div className="border-t border-slate-800 pt-3">
            <h3 className="text-xs font-semibold text-slate-300">Weather</h3>
            <a
              href={marineForecastUrlForRegion(rk)}
              target="_blank"
              rel="noreferrer"
              className="mt-1 block text-cyan-400 hover:underline"
            >
              NWS marine forecast (regional point) →
            </a>
            <a
              href={pointForecastUrl(lat, lon)}
              target="_blank"
              rel="noreferrer"
              className="mt-1 block text-cyan-400 hover:underline"
            >
              NWS grid forecast at harbor →
            </a>
          </div>

          {fish?.links && fish.links.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-slate-300">Official links</h3>
              <div className="mt-1 flex flex-col gap-1">
                {fish.links.map((l) => (
                  <a
                    key={l.url}
                    href={l.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-cyan-400 hover:underline"
                  >
                    {l.label}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
