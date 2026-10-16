/**
 * The one thing selected in a workspace, shared by the map, the inspection surface, the charts
 * and the briefing. Encoded in the URL (`sel=`) so a view can be saved and shared; v2 links
 * (`port=`, `inspect=`) still resolve to the same selections.
 */
export type Selection =
  | { kind: "harbor"; port: number }
  | { kind: "station"; id: string }
  /** a C-HARM model cell, addressed by any point inside it (the cell is resolved from the grid) */
  | { kind: "cell"; lat: number; lon: number }
  /** a point in the water that is not a station or a model cell (no forecast layer on) */
  | { kind: "point"; lat: number; lon: number }
  /** west, south, east, north */
  | { kind: "area"; bbox: [number, number, number, number] };

const r4 = (n: number) => Number(n.toFixed(4));
const okLat = (n: number) => Number.isFinite(n) && n > 30 && n < 44;
const okLon = (n: number) => Number.isFinite(n) && n > -128 && n < -114;

export function encodeSelection(s: Selection | null): string | null {
  if (!s) return null;
  switch (s.kind) {
    case "harbor":
      return `harbor:${s.port}`;
    case "station":
      return `station:${s.id}`;
    case "cell":
      return `cell:${r4(s.lat)},${r4(s.lon)}`;
    case "point":
      return `point:${r4(s.lat)},${r4(s.lon)}`;
    case "area":
      return `area:${s.bbox.map(r4).join(",")}`;
  }
}

export function decodeSelection(q: URLSearchParams): Selection | null {
  const v = q.get("sel");
  if (v) {
    const i = v.indexOf(":");
    const kind = v.slice(0, i);
    const rest = v.slice(i + 1);
    if (kind === "harbor" && /^\d+$/.test(rest)) return { kind, port: Number(rest) };
    if (kind === "station" && /^[A-Za-z0-9_.-]{1,80}$/.test(rest)) return { kind, id: rest };
    if (kind === "cell" || kind === "point") {
      const [lat, lon] = rest.split(",").map(Number);
      if (okLat(lat) && okLon(lon)) return { kind, lat, lon };
    }
    if (kind === "area") {
      const b = rest.split(",").map(Number);
      if (b.length === 4 && okLon(b[0]) && okLat(b[1]) && okLon(b[2]) && okLat(b[3]) && b[0] < b[2] && b[1] < b[3]) return { kind, bbox: b as [number, number, number, number] };
    }
  }
  // v2 links
  const port = q.get("port");
  if (port && /^\d+$/.test(port)) return { kind: "harbor", port: Number(port) };
  const ins = q.get("inspect")?.split(",").map(Number);
  if (ins && ins.length === 2 && okLat(ins[0]) && okLon(ins[1])) return { kind: "point", lat: ins[0], lon: ins[1] };
  return null;
}

export function sameSelection(a: Selection | null, b: Selection | null): boolean {
  return encodeSelection(a) === encodeSelection(b);
}

/** The camera the selection was made from, so "Back" returns to the previous map state. */
export type Camera = { center: [number, number]; zoom: number };
export type HistoryEntry = { selection: Selection | null; camera: Camera | null };
