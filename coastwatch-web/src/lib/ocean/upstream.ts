import { readFile } from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";

/**
 * Upstream reads for the ocean model routes: a timeout on every request, Next's fetch cache as a second tier, and a
 * recorded-fixture mode so tests and the e2e server never touch the network.
 *
 * Fixture mode: CW_OCEAN_FIXTURE names a directory holding `fixture.json`
 * ({ recorded_at, files: { "<url>": "<file>" } }; a file ending in .gz is gunzipped). It is also on, with
 * tests/ocean-fixtures, whenever CW_CONDITIONS_FIXTURE is set (the e2e server), unless CW_OCEAN_FIXTURE=off.
 * A URL that was not recorded fails like an outage. In fixture mode the clock is CW_OCEAN_FIXED_NOW, else the
 * recording time, so the recorded cycle reads as current.
 */
const UA = "Neritic (formerly CoastWatch; California coastal information research project; https://github.com/yashnil/habs-forecast)";

type Fixture = { recorded_at: string; files: Record<string, string> };
let fixture: { dir: string; p: Promise<Fixture> } | null = null;

export function fixtureDir(): string | null {
  const v = process.env.CW_OCEAN_FIXTURE;
  if (v === "off") return null;
  if (v) return v;
  return process.env.CW_CONDITIONS_FIXTURE ? "tests/ocean-fixtures" : null;
}

function loadFixture(dir: string): Promise<Fixture> {
  if (!fixture || fixture.dir !== dir) {
    fixture = { dir, p: readFile(path.join(dir, "fixture.json"), "utf8").then((t) => JSON.parse(t) as Fixture) };
  }
  return fixture.p;
}

/** Now, for the ocean routes (see fixture mode above). */
export async function oceanNow(): Promise<Date> {
  if (process.env.CW_OCEAN_FIXED_NOW) return new Date(process.env.CW_OCEAN_FIXED_NOW);
  const dir = fixtureDir();
  if (dir) return new Date((await loadFixture(dir)).recorded_at);
  return new Date();
}

export type GetOpts = { timeoutMs: number; revalidate: number };

/** GET a URL's body as bytes. Throws on timeout, HTTP error or an unrecorded fixture URL. */
export async function getBytes(url: string, o: GetOpts): Promise<Uint8Array> {
  const dir = fixtureDir();
  if (dir) {
    const f = (await loadFixture(dir)).files[url];
    if (!f) throw new Error("HTTP 503 (not recorded)");
    const b = await readFile(path.join(dir, f));
    return new Uint8Array(f.endsWith(".gz") ? gunzipSync(b) : b);
  }
  const r = await fetch(url, {
    headers: { "User-Agent": UA },
    signal: AbortSignal.timeout(o.timeoutMs),
    next: { revalidate: o.revalidate },
  } as RequestInit);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export async function getText(url: string, o: GetOpts): Promise<string> {
  return new TextDecoder().decode(await getBytes(url, o));
}

export const errMsg = (e: unknown) => (e instanceof Error ? (e.name === "TimeoutError" ? "timed out" : e.message) : String(e)).slice(0, 160);
