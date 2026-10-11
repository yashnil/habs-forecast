import { addProtocol } from "maplibre-gl";
import { EMPTY_TILE_PNG_BASE64 } from "@/lib/emptyTile";

/**
 * Skip tiles that do not exist (M5). CoastWatch tile sets publish tiles/index.json listing
 * every tile written; tiles with no valid pixel are never written. Without the index the map
 * requests them anyway and the server answers 404, which fills the console. With it, a
 * missing tile is answered locally with a transparent image and never requested.
 *
 * Tile sets without an index (published before M5, or third-party imagery) are fetched
 * directly, as before.
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
    if (!r.ok) throw new Error(`tile ${r.status}: ${url}`);
    return { data: await r.arrayBuffer() };
  });
}

/** The tile URL template to give MapLibre: through the index when the tile set has one. */
export function indexedTiles(template: string, indexUrl: string | null | undefined): string {
  if (!indexUrl || typeof window === "undefined") return template;
  register();
  return `${PROTOCOL}://${template}${SEP}${encodeURIComponent(indexUrl)}`;
}
