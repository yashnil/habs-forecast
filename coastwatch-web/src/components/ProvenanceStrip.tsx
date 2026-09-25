"use client";

import type { Snapshot } from "@/lib/types";
import { GIBS_SATELLITE, GIBS_PACE_CHL } from "@/lib/gibs";

function fmt(iso?: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toISOString().replace("T", " ").slice(0, 19) + " UTC";
  } catch {
    return iso;
  }
}

type Props = {
  snapshot: Snapshot;
  viirsDate: string;
  paceDate: string;
};

export default function ProvenanceStrip({ snapshot, viirsDate, paceDate }: Props) {
  const prov = snapshot.provenance;

  return (
    <footer className="border-t border-slate-800 bg-slate-950/95">
      <div className="mx-auto max-w-6xl px-4 py-3">
        <details className="group">
          <summary className="cursor-pointer list-none text-[10px] font-semibold uppercase tracking-wide text-slate-500 marker:hidden [&::-webkit-details-marker]:hidden">
            Technical provenance (expand)
          </summary>
          <dl className="mt-2 grid gap-x-6 gap-y-1 text-[11px] text-slate-400 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <dt className="text-slate-600">NASA GIBS — VIIRS (top)</dt>
              <dd className="break-all font-mono text-slate-300">
                {GIBS_SATELLITE.layerId} · {viirsDate}
              </dd>
            </div>
            <div>
              <dt className="text-slate-600">NASA GIBS — PACE (underlay)</dt>
              <dd className="break-all font-mono text-slate-300">
                {GIBS_PACE_CHL.layerId} · {paceDate}
              </dd>
            </div>
            <div>
              <dt className="text-slate-600">Tile matrix</dt>
              <dd className="font-mono text-slate-300">{GIBS_SATELLITE.tileMatrixSet}</dd>
            </div>
            <div>
              <dt className="text-slate-600">Map view</dt>
              <dd className="text-slate-300">
                California coast + nearshore Pacific (pan limited)
              </dd>
            </div>
            <div>
              <dt className="text-slate-600">Legends</dt>
              <dd className="break-all font-mono text-slate-300">
                <a
                  href={GIBS_SATELLITE.legendHorizontalSvg}
                  target="_blank"
                  rel="noreferrer"
                  className="text-cyan-400 hover:underline"
                >
                  VIIRS
                </a>
                {" · "}
                <a
                  href={GIBS_PACE_CHL.legendHorizontalSvg}
                  target="_blank"
                  rel="noreferrer"
                  className="text-cyan-400 hover:underline"
                >
                  PACE
                </a>
              </dd>
            </div>
            <div>
              <dt className="text-slate-600">App bundle generated</dt>
              <dd className="font-mono text-slate-300">{fmt(snapshot.generated_at)}</dd>
            </div>
            <div>
              <dt className="text-slate-600">Bundle data_source</dt>
              <dd className="text-slate-300">{snapshot.data_source ?? "—"}</dd>
            </div>
            {prov?.checkpoint && (
              <div>
                <dt className="text-slate-600">Checkpoint</dt>
                <dd className="break-all font-mono text-slate-300">{prov.checkpoint}</dd>
              </div>
            )}
            {prov?.export_command && (
              <div className="sm:col-span-2">
                <dt className="text-slate-600">Export command</dt>
                <dd className="break-all font-mono text-slate-300">{prov.export_command}</dd>
              </div>
            )}
            {prov?.notes && (
              <div className="sm:col-span-2 lg:col-span-3">
                <dt className="text-slate-600">Notes</dt>
                <dd className="text-slate-400">{prov.notes}</dd>
              </div>
            )}
          </dl>
        </details>
      </div>
    </footer>
  );
}
