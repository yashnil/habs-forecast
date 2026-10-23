import { describe, expect, it } from "vitest";
import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { chooseHour, defaultHour, usableHoursIn, type CurrentField } from "@/lib/currents";
import { decodeFootprint, encodeFootprint, footprintCell, footprintKm, footprintPolygon, productOf } from "@/lib/footprint";
import { FlowSim, SPEED_RAMP, speedBucket } from "@/lib/flowSim";
import { SPEED_CLASSES } from "@/lib/basemap";
import { resolve } from "@/v4/explore/layers";
import { timeSteps } from "@/v4/explore/Timebar";

// ---------------------------------------------------------------- the currents hour, by the region in view
const region = (id: string, f: number) => ({ region_id: id, label: id === "monterey_bay" ? "Monterey Bay" : "SF Bay", observed_fraction: f, observed_km2: 0, reference_cells: 575 });
const hour = (h: string, domain: number, regions: ReturnType<typeof region>[]): LayerArtifact =>
  ({ layer_id: `hfr2km_currents_20261021T${h}Z`, group_id: "currents", vectors: { u_grid: { lat_first: 30, lat_step: 0.018, lon_first: -130, lon_step: 0.0208, width: 540, height: 408 } }, time: { valid_time: `2026-10-21T${h}:00:00Z` }, coverage: { domain_observed_fraction: domain, domain_reference_cells: 100000, reference: "", regions } }) as unknown as LayerArtifact;
// the staging data of Oct 21: the newest hour (02Z) saw the domain (0.0926) but none of Monterey Bay
const m = {
  layers: [
    hour("00", 0.1, [region("monterey_bay", 0.73), region("sf_bay_farallones", 0.5)]),
    hour("01", 0.098, [region("monterey_bay", 0.71), region("sf_bay_farallones", 0.52)]),
    hour("02", 0.0926, [region("monterey_bay", 0), region("sf_bay_farallones", 0.55)]),
  ],
} as unknown as Manifest;

describe("v10: the currents open on the newest hour that observed the region in view", () => {
  it("Monterey Bay never opens on an hour with no cells there; the dock is told why", () => {
    const c = chooseHour(m, "monterey_bay");
    expect(c.layer?.layer_id).toBe("hfr2km_currents_20261021T01Z");
    expect(c.newest?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    expect(c.newestMissed).toBe(true);
    expect(c.newestFraction).toBe(0);
    expect(c.regionLabel).toBe("Monterey Bay");
    expect(usableHoursIn(m, "monterey_bay")?.get("hfr2km_currents_20261021T02Z")).toBe(false);
  });
  it("another region that the newest hour did see opens on it", () => {
    const c = chooseHour(m, "sf_bay_farallones");
    expect(c.layer?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    expect(c.newestMissed).toBe(false);
  });
  it("statewide (no region) and a region no hour publishes fall back to the domain rule", () => {
    expect(defaultHour(m)?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    const c = chooseHour(m, "north_coast");
    expect(c.layer?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    expect(c.regionId).toBeNull();
    expect(c.newestMissed).toBe(false);
  });
  it("an hour that saw a sliver of the region (under half its median) is passed over too", () => {
    const m2 = { layers: [hour("00", 0.1, [region("monterey_bay", 0.7)]), hour("01", 0.1, [region("monterey_bay", 0.72)]), hour("02", 0.1, [region("monterey_bay", 0.05)])] } as unknown as Manifest;
    const c = chooseHour(m2, "monterey_bay");
    expect(c.layer?.layer_id).toBe("hfr2km_currents_20261021T01Z");
    expect(c.newestFraction).toBe(0.05);
  });
  it("resolve and the time bar follow the region; an hour the reader chose stays chosen", () => {
    expect(resolve(m, { kind: "currents", hour: null, mean: false }, "", null, { region: "monterey_bay" }).curLayer?.layer_id).toBe("hfr2km_currents_20261021T01Z");
    expect(resolve(m, { kind: "currents", hour: null, mean: false }, "").curLayer?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    expect(resolve(m, { kind: "currents", hour: "20261021T02Z", mean: false }, "", null, { region: "monterey_bay" }).curLayer?.layer_id).toBe("hfr2km_currents_20261021T02Z");
    expect(timeSteps(m, { kind: "currents", hour: null, mean: false }, null, null, "monterey_bay")!.current).toBe("20261021T01Z");
  });
});

// ---------------------------------------------------------------- footprint geometry
const radar = { lat_first: 30.25, lat_step: 0.01798, lon_first: -130.36, lon_step: 0.02082, width: 540, height: 408 };
const charm = { lat_first: 31.3, lat_step: 0.03, lon_first: -127.5, lon_step: 0.03, width: 351, height: 391 };
const olci = { lat_first: 32.0, lat_step: -0.0027, lon_first: -125.0, lon_step: 0.0027, width: 2000, height: 2000 };

describe("v10: the selected footprint is the published cell or pixel, at its native size", () => {
  it("a radar cell is centre ± half a step: 2.0 km north-south, about 1.86 km east-west at 36.8° N", () => {
    const c = footprintCell(radar, 36.8, -122.0)!;
    const ring = footprintPolygon(radar, c.row, c.col).coordinates[0];
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]);
    const [w, s] = ring[0];
    const [e, n] = ring[2];
    expect(e - w).toBeCloseTo(radar.lon_step, 5);
    expect(n - s).toBeCloseTo(radar.lat_step, 5);
    // the point is inside its own cell, and the cell's centre is on the lattice
    expect(-122.0).toBeGreaterThanOrEqual(w);
    expect(-122.0).toBeLessThanOrEqual(e);
    expect(36.8).toBeGreaterThanOrEqual(s);
    expect(36.8).toBeLessThanOrEqual(n);
    expect((w + e) / 2).toBeCloseTo(radar.lon_first + c.col * radar.lon_step, 6);
    const km = footprintKm(radar, c.row);
    expect(km.ns).toBeCloseTo(2.0, 1);
    expect(km.ew).toBeCloseTo(1.86, 1);
  });
  it("a C-HARM cell is 0.03°; a satellite pixel keeps its own (descending) lattice and native size", () => {
    const c = footprintCell(charm, 36.6045, -121.8908)!;
    const [[w, s], , [e, n]] = footprintPolygon(charm, c.row, c.col).coordinates[0];
    expect(e - w).toBeCloseTo(0.03, 6);
    expect(n - s).toBeCloseTo(0.03, 6);
    const p = footprintCell(olci, 31.5, -124.5)!;
    const [[pw, ps], , [pe, pn]] = footprintPolygon(olci, p.row, p.col).coordinates[0];
    expect(pe - pw).toBeCloseTo(0.0027, 6);
    expect(pn - ps).toBeCloseTo(0.0027, 6);
    expect(pn).toBeGreaterThan(ps);
    expect(31.5).toBeGreaterThanOrEqual(ps);
    expect(31.5).toBeLessThanOrEqual(pn);
  });
  it("cell edges belong to one cell only; a point outside the grid has no footprint", () => {
    const a = footprintCell(charm, 31.3 + 0.015 - 1e-9, -127.5)!;
    const b = footprintCell(charm, 31.3 + 0.015 + 1e-9, -127.5)!;
    expect([a.row, b.row]).toEqual([0, 1]);
    expect(footprintCell(charm, 20, -127.5)).toBeNull();
    expect(footprintCell(radar, 36.8, -100)).toBeNull();
  });
  it("travels in the URL as product:row:col, and nothing else is accepted", () => {
    const f = { product: "hfr2km" as const, row: 362, col: 402 };
    expect(encodeFootprint(f)).toBe("hfr2km:362:402");
    expect(decodeFootprint("hfr2km:362:402")).toEqual(f);
    for (const bad of ["", "foo:1:2", "hfr2km:1", "hfr2km:-1:2", "charm:1:2:3", "olci300:a:b"]) expect(decodeFootprint(bad)).toBeNull();
    expect(productOf("olci300_latest")).toBe("olci300");
    expect(productOf("viirs750_20261019")).toBe("viirs750");
    expect(productOf("hfr2km_currents_20261021T01Z")).toBe("hfr2km");
    expect(productOf("multisensor_chl_latest")).toBeNull();
  });
});

// ---------------------------------------------------------------- the particles never leave observed cells
const g = { url: "", width: 30, height: 30, lat_first: 36.5, lat_step: 0.018, lon_first: -122.4, lon_step: 0.0208, scale_factor: 1, add_offset: 0 } as unknown as ValueGrid;
function field(): CurrentField {
  const u = new Float32Array(900).fill(NaN);
  const v = new Float32Array(900).fill(NaN);
  for (let r = 0; r < 30; r++)
    for (let c = 0; c < 30; c++) {
      // a ring of coverage around a hole, a gap column, and a fast eastward flow straight at the gap
      const hole = r >= 12 && r <= 17 && c >= 12 && c <= 17;
      if (hole || c === 22 || r < 3) continue;
      u[r * 30 + c] = 0.8;
      v[r * 30 + c] = 0.3 * Math.sin(r);
    }
  return { g, u, v };
}
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

describe("v10: particles move only inside observed cells", () => {
  it("over many steps, at harbour zoom, no particle head or trail point is ever in a cell without an observation", () => {
    const f = field();
    const sim = new FlowSim(600, rand);
    sim.setField(f);
    sim.setView({ west: -123, south: 36, east: -121, north: 37.5 });
    sim.setCount(600);
    sim.respawnAll();
    const worldPx = 512 * 2 ** 11;
    const observed = (lon: number, lat: number) => {
      const r = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
      const c = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
      return r >= 0 && c >= 0 && r < 30 && c < 30 && Number.isFinite(f.u[r * 30 + c]);
    };
    let checked = 0;
    for (let i = 0; i < 600; i++) {
      // 1/60 s per step (exaggerated 40x so the particles reach the gaps quickly)
      sim.step(40 / 60, worldPx, i % 3 === 0);
      if (i % 20 === 0)
        for (const [lon, lat] of sim.points()) {
          expect(observed(lon, lat)).toBe(true);
          checked++;
        }
    }
    expect(checked).toBeGreaterThan(5000);
  });
  it("a particle never hops over a one-cell gap, however far it moves in a step (zoomed out)", () => {
    const u = new Float32Array(900).fill(NaN);
    const v = new Float32Array(900).fill(NaN);
    for (let r = 0; r < 30; r++) for (const c of [21, 23]) [u[r * 30 + c], v[r * 30 + c]] = [1.5, 0];
    const sim = new FlowSim(50, rand);
    sim.setField({ g, u, v });
    const lon21 = g.lon_first + 21 * g.lon_step;
    sim.setView({ west: lon21 - 0.001, south: 36, east: lon21 + 0.001, north: 37.5 });
    sim.setCount(50);
    sim.respawnAll();
    const colOf = (lon: number) => Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
    for (let i = 0; i < 200; i++) {
      // zoom 5: a step moves far more than a cell
      sim.step(0.05, 512 * 2 ** 5, true);
      for (const [lon] of sim.points()) expect(colOf(lon)).toBe(21);
    }
  });
  it("a new hour is a cut: every particle is reborn in the new field; with nothing in view, nothing is born", () => {
    const sim = new FlowSim(100, rand);
    sim.setField(field());
    sim.setCount(100);
    sim.respawnAll();
    expect(sim.heads()).toHaveLength(100);
    sim.setField(field());
    expect(sim.heads()).toHaveLength(0);
    sim.setView({ west: 0, south: 0, east: 1, north: 1 });
    sim.respawnAll();
    expect(sim.visible.length).toBe(0);
    expect(sim.heads()).toHaveLength(0);
  });
  it("the still frame (reduced motion, paused) has trails without animating", () => {
    const sim = new FlowSim(200, rand);
    sim.setField(field());
    sim.setCount(200);
    sim.settle(512 * 2 ** 10);
    const out = new Float32Array(200 * 10 * 8);
    expect(sim.build(out, { colour: true, alpha: 1 })).toBeGreaterThan(200);
  });
  it("speed colours: one per speed class, sequential and lightness-monotone (never a rainbow)", () => {
    expect(SPEED_RAMP).toHaveLength(SPEED_CLASSES.length);
    expect([0, 0.09, 0.1, 0.3, 0.6, 1.4].map(speedBucket)).toEqual([0, 0, 1, 2, 3, 4]);
    const lum = SPEED_RAMP.map((h) => {
      const [r, gg, b] = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * gg + 0.0722 * b;
    });
    for (let i = 1; i < lum.length; i++) expect(lum[i]).toBeGreaterThan(lum[i - 1]);
  });
});

// ---------------------------------------------------------------- v10 rework: speed is encoded, honestly
import { FLOW_SKY, skyColour, speedShare, SPEED_FULL } from "@/lib/flowSim";
describe("v10 rework: streak length, brightness and colour follow the cell's observed speed", () => {
  it("a fast cell's streak is longer than a slow cell's at the same zoom (screen speed proportional to speed)", () => {
    const g2 = { url: "", lat_first: 36, lat_step: 0.018, lon_first: -123, lon_step: 0.0208, width: 40, height: 40, scale_factor: 1, add_offset: 0 } as unknown as ValueGrid;
    const u = new Float32Array(1600), v = new Float32Array(1600);
    // west half slow (0.05 m/s), east half fast (0.5 m/s), all northward, so no particle leaves its half
    for (let r = 0; r < 40; r++) for (let c = 0; c < 40; c++) [u[r * 40 + c], v[r * 40 + c]] = [0, c < 20 ? 0.05 : 0.5];
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const sim = new FlowSim(300, rnd);
    sim.setField({ g: g2, u, v });
    sim.setCount(300);
    sim.settle(512 * 2 ** 10);
    const s = sim.streaks(512 * 2 ** 10);
    const slow = s.filter((x) => x.speed < 0.1), fast = s.filter((x) => x.speed > 0.4);
    expect(slow.length).toBeGreaterThan(20);
    expect(fast.length).toBeGreaterThan(20);
    const mean = (xs: { px: number }[]) => xs.reduce((a, x) => a + x.px, 0) / xs.length;
    expect(mean(fast) / mean(slow)).toBeGreaterThan(5);
  });
  it("the default colour is the conditions (sky) family, getting lighter with speed, and saturates at SPEED_FULL", () => {
    const lum = ([r, g, b]: [number, number, number]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const ls = [0, 0.1, 0.2, 0.3, 0.45, SPEED_FULL].map((x) => lum(skyColour(x)));
    for (let i = 1; i < ls.length; i++) expect(ls[i]).toBeGreaterThan(ls[i - 1]);
    expect(skyColour(2)).toEqual(skyColour(SPEED_FULL));
    expect(speedShare(-1)).toBe(0);
    expect(FLOW_SKY.some(([, c]) => c === "#74c7ff")).toBe(true); // --nx-conditions
    for (const [, c] of FLOW_SKY) {
      const [r, , b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
      expect(b).toBeGreaterThan(r); // blue family, never a rainbow
    }
  });
});
