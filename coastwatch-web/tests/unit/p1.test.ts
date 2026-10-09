import { readFileSync } from "node:fs";
import path from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LayerArtifact, Palette, ValueGrid } from "@/generated/schema";
import { decodeCode, sampleChunked } from "@/lib/grid";
import { nativeLabel } from "@/lib/layers";
import { AGE_COLOURS, colorAt } from "@/lib/palette";

const repo = path.resolve(__dirname, "../../..");
const manifest = JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-data/v1/manifest.json"), "utf8"));

function u16gz(codes: number[]): ArrayBuffer {
  const b = Buffer.alloc(codes.length * 2);
  codes.forEach((c, i) => b.writeUInt16LE(c, i * 2));
  const gz = gzipSync(b);
  return gz.buffer.slice(gz.byteOffset, gz.byteOffset + gz.byteLength) as ArrayBuffer;
}

afterEach(() => vi.unstubAllGlobals());

describe("chunked satellite grids", () => {
  // 4 x 6 grid, chunks of 2 x 4: keys 0_0, 0_1, 1_0 present; 1_1 absent (all no value)
  const S = 6 / 65534;
  const code = (v: number) => Math.round((Math.log10(v) + 3) / S);
  const g: ValueGrid = {
    url: "g/{row}_{col}.u16.gz",
    encoding: "uint16le+gzip",
    width: 6,
    height: 4,
    lat_first: 37.0,
    lat_step: -0.0025,
    lon_first: -122.0,
    lon_step: 0.0025,
    scale_factor: S,
    add_offset: -3,
    nodata: 65535,
    max_quantization_error: S / 2,
    transform: "log10",
    chunks: { rows: 2, cols: 4, url_template: "g/{row}_{col}.u16.gz", present: ["0_0", "0_1", "1_0"] },
  };
  const age: ValueGrid = { ...g, url: "a/{row}_{col}.u16.gz", scale_factor: 1, add_offset: 0, transform: "none", max_quantization_error: 0.5, chunks: { ...g.chunks!, url_template: "a/{row}_{col}.u16.gz" } };
  const N = 65535;
  const files: Record<string, number[]> = {
    // chunk 0_0: rows 0-1, cols 0-3
    "g/0_0.u16.gz": [code(1.5), N, N, N, N, N, N, code(0.2)],
    // chunk 0_1: rows 0-1, cols 4-5 (edge chunk, 2 columns)
    "g/0_1.u16.gz": [N, code(12), N, N],
    "g/1_0.u16.gz": [N, N, N, N, N, N, N, N].map((x, i) => (i === 0 ? code(3) : x)),
    "a/0_0.u16.gz": [2, N, N, N, N, N, N, 0],
    "a/0_1.u16.gz": [N, 5, N, N],
    "a/1_0.u16.gz": [1, N, N, N, N, N, N, N],
  };
  const stubFetch = () =>
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const key = url.replace(/^.*?\/(g|a)\//, "$1/");
        const body = files[key];
        return body ? new Response(u16gz(body)) : new Response("", { status: 404 });
      }),
    );

  it("decodes log10-quantized codes back to mg m-3 within quantization error", () => {
    for (const v of [0.05, 1, 37.5]) expect(Math.abs(Math.log10(decodeCode(g, code(v))!) - Math.log10(v))).toBeLessThanOrEqual(g.max_quantization_error + 1e-12);
    expect(decodeCode(g, 65535)).toBeNull();
  });

  it("reads the value and its age at a cell, from the right chunk only", async () => {
    stubFetch();
    const r = await sampleChunked("https://x/v1", g, 37.0, -122.0, age, 0);
    expect(r.kind).toBe("value");
    if (r.kind === "value") {
      expect(r.value).toBeCloseTo(1.5, 3);
      expect(r.ageDays).toBe(2);
      expect(r.nearest).toBe(false);
    }
    const e = await sampleChunked("https://x/v1", g, 37.0, -122.0 + 5 * 0.0025, age, 0); // edge chunk 0_1, col 5
    expect(e.kind === "value" && Math.round(e.value) === 12 && e.ageDays === 5).toBe(true);
  });

  it("with no value at the point, reports the nearest valid cell and its distance", async () => {
    stubFetch();
    const r = await sampleChunked("https://x/v1", g, 37.0, -122.0 + 2 * 0.0025, age, 2);
    expect(r.kind).toBe("value");
    if (r.kind === "value") {
      expect(r.nearest).toBe(true);
      expect(r.distanceKm).toBeGreaterThan(0.2);
      expect(r.distanceKm).toBeLessThan(0.6);
    }
  });

  it("cells in an unwritten chunk have no value, and nothing is fetched for them", async () => {
    stubFetch();
    const r = await sampleChunked("https://x/v1", g, 37.0 - 3 * 0.0025, -122.0 + 5 * 0.0025, null, 0);
    expect(r.kind).toBe("none");
  });
});

describe("palettes match the pipeline", () => {
  const charm: Palette = manifest.layers.find((l: LayerArtifact) => l.layer_id === "charm_particulate_domoic_lead1").palette;
  const chl: Palette = manifest.layers.find((l: LayerArtifact) => l.layer_id === "olci300_chl_latest").palette;

  it("C-HARM classes are stepped 10-point bins", () => {
    expect(charm.interpolation).toBe("step");
    expect(colorAt(charm, 0.0)).toBe(charm.stops[0].color);
    expect(colorAt(charm, 0.0999)).toBe(charm.stops[0].color);
    expect(colorAt(charm, 0.1)).toBe(charm.stops[1].color);
    expect(colorAt(charm, 1.0)).toBe(charm.stops[9].color);
  });

  it("chlorophyll is on a fixed log scale, clipped at the domain", () => {
    expect(chl.scale).toBe("log10");
    expect(colorAt(chl, 0.001)).toBe(colorAt(chl, 0.05));
    expect(colorAt(chl, 1000)).toBe(colorAt(chl, 50));
    expect(colorAt(chl, 0.05)).not.toBe(colorAt(chl, 5));
  });

  it("observation-age colours equal the pipeline's", () => {
    const py = readFileSync(path.join(repo, "pipeline/coastwatch_pipeline/process/palette.py"), "utf8");
    for (const c of AGE_COLOURS) expect(py).toContain(`"${c}"`);
  });
});

describe("native resolution labels", () => {
  it("say what one source cell is", () => {
    const byId = (id: string) => manifest.layers.find((l: LayerArtifact) => l.layer_id === id);
    expect(nativeLabel(byId("charm_particulate_domoic_lead1"))).toBe("3 km");
    expect(nativeLabel(byId("olci300_chl_latest"))).toBe("300 m");
    expect(nativeLabel(byId("viirs750_chl_latest"))).toBe("750 m");
  });
});
