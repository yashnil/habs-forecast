import { AJAXError, addProtocol } from "maplibre-gl";
import { EMPTY_TILE_PNG_BASE64 } from "@/lib/emptyTile";
import { clipToWater, waterMask } from "@/lib/waterMask";

/**
 * Skip tiles that do not exist (M5). CoastWatch tile sets publish tiles/index.json listing
 * every tile written; tiles with no valid pixel are never written. Without the index the map
 * requests them anyway and the server answers 404, which fills the console. With it, a
 * missing tile is answered locally with a transparent image and never requested.
 *
 * Tile sets without an index (published before M5, or third-party imagery) are fetched
 * directly, as before. Tiles that go through here are clipped to the basemap's water
 * (lib/waterMask), so ocean data stop at the drawn shoreline.
 */
const PROTOCOL = "cwtile";
const SEP = "#cw-index=";
const indexes = new Map<string, Promise<Set<string> | null>>();
const EMPTY = Uint8Array.from(atob(EMPTY_TILE_PNG_BASE64), (c) => c.charCodeAt(0)).buffer;

function loadIndex(url: string): Promise<Set<string> | null> {
  let p = indexes.get(url);
  if (!p) {
    // an unreadable index never hides tiles: fall back to requesting every tile
    p = fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((list: unknown) => (Array.isArray(list) ? new Set(list.map(String)) : null))
      .catch(() => null);
    indexes.set(url, p);
  }
  return p;
}

let registered = false;
function register() {
  if (registered) return;
  registered = true;
  addProtocol(PROTOCOL, async (params, abort) => {
    const raw = params.url.slice(PROTOCOL.length + 3);
    const [url, indexUrl] = raw.split(SEP);
    const zxy = /\/(\d+)\/(\d+)\/(\d+)\.png$/.exec(url);
    if (indexUrl && zxy) {
      const set = await loadIndex(decodeURIComponent(indexUrl));
      if (set && !set.has(`${zxy[1]}/${zxy[2]}/${zxy[3]}`)) return { data: EMPTY.slice(0) }; // a copy: MapLibre may transfer (detach) the buffer
    }
    const r = await fetch(url, { signal: abort.signal });
    // A tile that is not there (no index yet, or the manifest ahead of its tiles while a
    // publication propagates) is drawn as no data, like a tile the index leaves out. Thrown
    // instead, the failure is the last thing to settle and MapLibre never repaints, so its
    // load event (and everything waiting on it) can hang until someone moves the map.
    if (r.status === 404) return { data: EMPTY.slice(0) };
    if (!r.ok) throw new AJAXError(r.status, r.statusText, url, await r.blob());
    const buf = await r.arrayBuffer();
    // satellite data are ocean products: clip them to the basemap's water, like the forecast
    if (zxy && typeof OffscreenCanvas !== "undefined") {
      try {
        const bmp = await createImageBitmap(new Blob([buf]));
        const water = await waterMask(Number(zxy[1]), Number(zxy[2]), Number(zxy[3]), bmp.width);
        if (water && bmp.width === bmp.height) {
          const c = new OffscreenCanvas(bmp.width, bmp.height);
          const g = c.getContext("2d")!;
          g.drawImage(bmp, 0, 0);
          const img = g.getImageData(0, 0, bmp.width, bmp.height);
          clipToWater(img.data, water);
          g.putImageData(img, 0, 0);
          return { data: c.transferToImageBitmap() };
        }
      } catch {
        // drawn unclipped: data is never hidden because the basemap could not be read
      }
    }
    return { data: buf };
  });
}

/** The tile URL template to give MapLibre: through the index when the tile set has one. */
export function indexedTiles(template: string, indexUrl: string | null | undefined): string {
  if (!indexUrl || typeof window === "undefined") return template;
  register();
  return `${PROTOCOL}://${template}${SEP}${encodeURIComponent(indexUrl)}`;
}
