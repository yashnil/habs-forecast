"use client";

import { useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";

export type BBox = [number, number, number, number];

/** The box as a highlight feature (MapCanvas draws kind "area" dashed). */
export function boxFeature(b: BBox): GeoJSON.Feature {
  const [w, s, e, n] = b;
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }, properties: { kind: "area" } };
}

/**
 * The area tool shared by Explore and Research: a transparent overlay over the map that turns a
 * drag into a box in longitude/latitude (west, south, east, north). A box smaller than 0.01° on a
 * side is treated as a stray click and ignored. `dragBox` is the box while dragging, for drawing.
 */
export function useAreaDrag(getMap: () => MlMap | null, onBox: (b: BBox) => void, onEnd: () => void) {
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);
  const toLngLat = (e: React.PointerEvent): [number, number] | null => {
    const m = getMap();
    const el = ref.current;
    if (!m || !el) return null;
    const r = el.getBoundingClientRect();
    const p = m.unproject([e.clientX - r.left, e.clientY - r.top]);
    return [p.lng, p.lat];
  };
  const dragBox = useMemo(() => (drag ? ([Math.min(drag.a[0], drag.b[0]), Math.min(drag.a[1], drag.b[1]), Math.max(drag.a[0], drag.b[0]), Math.max(drag.a[1], drag.b[1])] as BBox) : null), [drag]);
  const handlers = {
    ref,
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => {
      const p = toLngLat(e);
      if (p) setDrag({ a: p, b: p });
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => {
      const p = toLngLat(e);
      if (p && drag) setDrag({ ...drag, b: p });
    },
    onPointerUp: () => {
      if (dragBox && dragBox[2] - dragBox[0] > 0.01 && dragBox[3] - dragBox[1] > 0.01) onBox(dragBox);
      setDrag(null);
      onEnd();
    },
  };
  return { dragBox, handlers };
}
