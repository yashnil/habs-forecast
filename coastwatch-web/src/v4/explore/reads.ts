"use client";

import { useEffect, useState } from "react";
import type { Manifest, PortsCollection, ValueGrid } from "@/generated/schema";
import { CHARM_LEADS, artifactUrl, charmLayer, type CharmVariable } from "@/lib/layers";
import { cellAt, distanceKm, gapAt, loadGrid, valueOf, type Cell, type Gap } from "@/lib/grid";
import { cellPolygon } from "@/lib/coast";
import type { EvidenceGraph } from "@/lib/harborEvidence";
import type { Conditions } from "@/lib/conditions";

export type LeadValue = { lead: number; date: string | null; value: number | null; issued: boolean };
export type HarborRelation = { name: string; code: number; km: number; ownCell: "same" | "valued-other" | "no-value" | "outside" };

export type CellRead =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      variable: CharmVariable;
      cell: Cell | null;
      polygon: GeoJSON.Polygon | null;
      /** km per cell side (the native resolution) */
      sizeKm: number;
      leads: LeadValue[];
      /** for the chosen lead: the gap at the point when its own cell has no value */
      gap: Gap;
      /** the closest harbour, and whether this cell represents its waters */
      harbor: HarborRelation | null;
    };

const kmOf = (g: ValueGrid, lat: number) => Math.abs(g.lat_step) * 111.32 * 0.5 + Math.abs(g.lon_step) * 111.32 * Math.cos((lat * Math.PI) / 180) * 0.5;

/**
 * Everything about one C-HARM cell for one variable: its footprint, the published value on each
 * day of the run (the run's own forecast horizon, nothing beyond it), the gap when it has none, and
 * whether it is a harbour's own cell. Grids are fetched once and cached (lib/grid); a newer selection
 * cancels the older read.
 */
export function useCellRead(manifest: Manifest, baseUrl: string, ports: PortsCollection | null, at: { lat: number; lon: number } | null, variable: CharmVariable, lead: number): CellRead | null {
  const [r, setR] = useState<CellRead | null>(null);
  const lat = at?.lat ?? null;
  const lon = at?.lon ?? null;
  useEffect(() => {
    if (lat == null || lon == null) return setR(null);
    let live = true;
    setR({ status: "loading" });
    const layers = CHARM_LEADS.map((l) => charmLayer(manifest, variable, l));
    const g = layers.find((l) => l?.grid)?.grid;
    if (!g) {
      setR({ status: "error", message: "No C-HARM grid is published for this variable." });
      return;
    }
    Promise.all(layers.map((l) => (l?.grid ? loadGrid(artifactUrl(baseUrl, l.grid.url)) : Promise.resolve(null))))
      .then((codes) => {
        if (!live) return;
        const cell = cellAt(g, lat, lon);
        const leads = layers.map((l, i): LeadValue => ({ lead: i, date: l?.time.valid_date ?? null, issued: !!l, value: cell && codes[i] ? valueOf(l!.grid!, codes[i]!, cell) : null }));
        const activeCodes = codes[lead] ?? codes.find(Boolean)!;
        const gap = gapAt(g, activeCodes, lat, lon);
        // the nearest harbour (within 25 km) and what its own cell holds
        let harbor: HarborRelation | null = null;
        for (const f of ports?.features ?? []) {
          const [plon, plat] = f.geometry.coordinates as number[];
          const km = distanceKm(lat, lon, plat, plon);
          if (km > 25 || (harbor && km >= harbor.km)) continue;
          const pc = cellAt(g, plat, plon);
          const own = pc ? valueOf(g, activeCodes, pc) : null;
          harbor = {
            name: f.properties.display_name ?? f.properties.name,
            code: f.properties.port_code,
            km,
            ownCell: !pc ? "outside" : cell && pc.row === cell.row && pc.col === cell.col ? "same" : own != null ? "valued-other" : "no-value",
          };
        }
        setR({ status: "ready", variable, cell, polygon: cell ? cellPolygon(g, cell) : null, sizeKm: kmOf(g, lat), leads, gap, harbor });
      })
      .catch((e) => live && setR({ status: "error", message: (e as Error).message }));
    return () => {
      live = false;
    };
  }, [manifest, baseUrl, ports, lat, lon, variable, lead]);
  return r;
}

export type Fetched<T> = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: T; at: string };

/** A JSON route, cancelled when the key changes; `retry` re-requests it. */
export function useJson<T>(url: string | null): [Fetched<T> | null, () => void] {
  const [r, setR] = useState<Fetched<T> | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!url) return setR(null);
    const ac = new AbortController();
    setR({ status: "loading" });
    fetch(url, { signal: ac.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(((await res.json().catch(() => null)) as { error?: string } | null)?.error ?? `HTTP ${res.status}`);
        return res.json() as Promise<T>;
      })
      .then((data) => setR({ status: "ready", data, at: new Date().toISOString() }))
      .catch((e) => {
        if ((e as Error).name !== "AbortError") setR({ status: "error", message: (e as Error).message === "Failed to fetch" ? "No connection" : (e as Error).message });
      });
    return () => ac.abort();
  }, [url, n]);
  return [r, () => setN((x) => x + 1)];
}

export const useEvidence = (place: string | null) => useJson<EvidenceGraph>(place ? `/api/evidence?place=${encodeURIComponent(place)}` : null);
export const useConditions = (at: { lat: number; lon: number } | null) => useJson<Conditions>(at ? `/api/conditions?lat=${at.lat.toFixed(4)}&lon=${at.lon.toFixed(4)}` : null);
