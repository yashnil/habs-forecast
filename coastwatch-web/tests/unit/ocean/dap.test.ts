import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { decodeDods, parseDds } from "@/lib/ocean/dap";

const fx = (f: string) => new Uint8Array(readFileSync(path.join(__dirname, "fixtures", f)));
const big = (f: string) => new Uint8Array(gunzipSync(readFileSync(path.join(__dirname, "../../ocean-fixtures", f))));

describe("DAP2 binary decoding", () => {
  it("reads a recorded THREDDS subset (WCOFS f024, 7×15 cells off Monterey) in DDS order", () => {
    const v = decodeDods(fx("wcofs-monterey-f024.dods"));
    expect([...v.keys()]).toEqual(["time", "u_eastward", "v_northward", "temp"]);
    const t = v.get("temp")!;
    expect(t.type).toBe("Float32");
    expect(t.dims.map((d) => d.size)).toEqual([1, 1, 7, 15]);
    expect(t.data.length).toBe(105);
    // the file's own time: seconds since 2016-01-01 → 2026-11-15T03:00Z (cycle 11-14 03z + 24 h)
    expect(new Date(Date.UTC(2016, 0, 1) + Number(v.get("time")!.data[0]) * 1000).toISOString()).toBe("2026-11-15T03:00:00.000Z");
    // row 3 (ny 457 = 36.70°N), column 6 (nx 582 = 122.10°W): the audit's sample point
    const k = 3 * 15 + 6;
    expect(v.get("u_eastward")!.data[k]).toBeCloseTo(0.049, 3);
    expect(v.get("v_northward")!.data[k]).toBeCloseTo(0.094, 3);
    expect(t.data[k]).toBeCloseTo(19.71, 2);
    // the subset reaches the Monterey shore: land is the fill value −99999
    expect([...t.data].some((x) => x === -99999)).toBe(true);
  });

  it("matches NOAA's own point service (NCSS CSV, recorded) at 36.70°N 122.10°W", () => {
    const row = readFileSync(path.join(__dirname, "fixtures", "wcofs-ncss-36.70N-122.10W-f024.csv"), "utf8").trim().split("\n")[1].split(",");
    const v = decodeDods(fx("wcofs-monterey-f024.dods"));
    const k = 3 * 15 + 6;
    expect(v.get("temp")!.data[k]).toBeCloseTo(Number(row[4]), 5);
    expect(v.get("u_eastward")!.data[k]).toBeCloseTo(Number(row[5]), 6);
    expect(v.get("v_northward")!.data[k]).toBeCloseTo(Number(row[6]), 6);
  });

  it("reads an ERDDAP GRID with its map vectors (DINEOF, latitude descending)", () => {
    const v = decodeDods(big("dineof-20261102.dods.gz"));
    const chl = v.get("chlor_a")!;
    expect(chl.dims.map((d) => d.size)).toEqual([1, 1, 466, 409]);
    const lat = v.get("latitude")!.data, lon = v.get("longitude")!.data;
    expect(v.get("chlor_a.latitude")).toBe(v.get("latitude"));
    expect(lat.length).toBe(466);
    expect(lat[0]).toBeGreaterThan(lat[465]);
    expect(lon[0]).toBeCloseTo(-125.4896, 3);
    expect(new Date(Number(v.get("time")!.data[0]) * 1000).toISOString()).toBe("2026-11-02T12:00:00.000Z");
  });

  it("decodes Int16 as 32-bit words and pads Byte arrays", () => {
    const dds = "Dataset {\n    Int16 a[x = 2];\n    Byte b[y = 3];\n    Float64 s;\n} t;\n";
    const head = new TextEncoder().encode(dds + "\nData:\n");
    const body = new DataView(new ArrayBuffer(8 + 8 + 8 + 4 + 8));
    let o = 0;
    body.setUint32(o, 2); body.setUint32(o + 4, 2); o += 8;
    body.setInt32(o, -7); body.setInt32(o + 4, 300); o += 8;
    body.setUint32(o, 3); body.setUint32(o + 4, 3); o += 8;
    body.setUint8(o, 1); body.setUint8(o + 1, 2); body.setUint8(o + 2, 255); o += 4;
    body.setFloat64(o, 1.5);
    const all = new Uint8Array(head.length + body.byteLength);
    all.set(head);
    all.set(new Uint8Array(body.buffer), head.length);
    const v = decodeDods(all);
    expect([...v.get("a")!.data]).toEqual([-7, 300]);
    expect([...v.get("b")!.data]).toEqual([1, 2, 255]);
    expect(v.get("s")!.data[0]).toBe(1.5);
  });

  it("throws on an error body and on a truncated body", () => {
    const err = new TextEncoder().encode('Error {\n    code = 400;\n    message = "Constraint expression is bad";\n};\n');
    expect(() => decodeDods(err)).toThrow(/Constraint expression is bad/);
    const whole = fx("wcofs-monterey-f024.dods");
    expect(() => decodeDods(whole.subarray(0, whole.length - 10))).toThrow(/ends inside temp/);
  });

  it("refuses structures rather than guessing", () => {
    expect(() => parseDds("Dataset {\n Structure {\n Int32 a;\n } s;\n} t;")).toThrow(/not supported/);
  });
});
