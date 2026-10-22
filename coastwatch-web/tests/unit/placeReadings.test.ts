import { describe, expect, it } from "vitest";
import type { Conditions } from "@/lib/conditions";
import { placeReadings } from "@/lib/placeReadings";

/**
 * v8 fix pass: the front door's cards credited one buoy for all three readings and dated them with the
 * newest one's age. Monterey's water came from the harbour's tide gauge while the card named the Monterey
 * Canyon Outer buoy, and a two-hour-old wave height read "11 min ago". Each reading now has its own.
 */
const NONE = { status: "none", reason: "", source: "" } as const;
const buoy = (data: Record<string, { value: number; at: string }>) =>
  ({ status: "ok", data, source: "NDBC", url: "", fetchedAt: "", station: { id: "46236", name: "MONTEREY CANYON OUTER, CA (236)", km: 17.6 } }) as const;
const gauge = (at: string, stale = false) =>
  ({ status: "ok", data: { waterC: { value: 18.9, at }, flags: "0,0,0", stale }, source: "CO-OPS", url: "", fetchedAt: "", station: { id: "9413450", name: "Monterey", km: 1.0 }, basis: "in-harbour" }) as const;
const cond = (p: Partial<Conditions>): Conditions => ({ lat: 36.6, lon: -121.9, marine: NONE, alerts: NONE, tides: NONE, buoy: NONE, ...p }) as Conditions;

describe("placeReadings", () => {
  it("credits the harbour gauge for water and the buoy for waves, each with its own time", () => {
    const r = placeReadings(
      cond({
        harbor: gauge("2026-10-22T13:48:00Z"),
        buoy: buoy({ waterC: { value: 17, at: "2026-10-22T13:56:00Z" }, waveFt: { value: 4.3, at: "2026-10-22T13:00:00Z" }, periodS: { value: 10, at: "2026-10-22T13:00:00Z" } }),
      }),
    );
    expect(r.water).toMatchObject({ ok: true, c: 18.9, at: "2026-10-22T13:48:00Z", source: { kind: "gauge", short: "Tide gauge" } });
    expect(r.waves).toMatchObject({ ok: true, ft: 4.3, s: 10, at: "2026-10-22T13:00:00Z", source: { kind: "buoy", short: "Buoy" } });
    // the buoy reports no wind: said so, and by which instrument, not a bare dash
    expect(r.wind).toMatchObject({ ok: false, why: "Not reported", source: { kind: "buoy" } });
    expect(r.sources.map((s) => s.full)).toEqual(["Monterey tide gauge (NOAA CO-OPS), 1.0\u00a0km", "Monterey Canyon Outer buoy (NDBC 46236), 18\u00a0km"]);
  });

  it("each buoy field keeps its own observation time", () => {
    const r = placeReadings(cond({ buoy: buoy({ windKt: { value: 6, at: "2026-10-22T14:10:00Z" }, windDirDeg: { value: 135, at: "2026-10-22T14:10:00Z" }, waveFt: { value: 2, at: "2026-10-22T12:40:00Z" } }) }));
    expect(r.wind).toMatchObject({ ok: true, kt: 6, dir: 135, at: "2026-10-22T14:10:00Z" });
    expect(r.waves).toMatchObject({ ok: true, at: "2026-10-22T12:40:00Z" });
    expect(r.water).toMatchObject({ ok: false, why: "Not reported" });
    expect(r.sources).toHaveLength(1);
  });

  it("a stale gauge gives way to the buoy, and is used with its true age only when the buoy has no water", () => {
    const stale = gauge("2026-10-22T10:00:00Z", true);
    expect(placeReadings(cond({ harbor: stale, buoy: buoy({ waterC: { value: 17, at: "2026-10-22T13:56:00Z" } }) })).water).toMatchObject({ source: { kind: "buoy" }, at: "2026-10-22T13:56:00Z" });
    expect(placeReadings(cond({ harbor: stale, buoy: buoy({}) })).water).toMatchObject({ ok: true, source: { kind: "gauge" }, at: "2026-10-22T10:00:00Z" });
  });

  it("says why when there is no buoy, or its feed did not answer", () => {
    const none = placeReadings(cond({}));
    expect([none.water, none.waves, none.wind].map((x) => (x.ok ? null : x.why))).toEqual(["Not measured nearby", "No buoy nearby", "No buoy nearby"]);
    expect(none.sources).toEqual([]);
    const down = placeReadings(cond({ buoy: { status: "unavailable", reason: "timeout", source: "NDBC" } }));
    expect([down.water, down.waves, down.wind].map((x) => (x.ok ? null : x.why))).toEqual(["Buoy not reached", "Buoy not reached", "Buoy not reached"]);
  });
});
