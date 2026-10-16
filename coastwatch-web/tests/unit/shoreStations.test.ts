import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { Conditions } from "@/lib/conditions";
import { checkGraph } from "@/lib/evidenceCheck";
import { harborEvidence } from "@/lib/harborEvidence";
import { parseShoreCsv, shoreCandidate, shoreUrl, SHORE_STATIONS } from "@/lib/shoreStations";

/**
 * CeNCOOS shore stations, from answers recorded at the end-to-end clock (2026-10-13T06:00Z) with
 * the exact URLs the route builds: only QARTOD-passed values become readings; any other flag is
 * withheld and said; a silent station is said with the date of its last report.
 */
const NOW = new Date("2026-10-13T06:00:00Z");
const csv = (id: string) => readFileSync(path.join(__dirname, "fixtures/shore", `${id}-2026-10-13T0600Z.csv`), "utf8");
const fx = JSON.parse(readFileSync(path.join(__dirname, "../conditions-fixtures.json"), "utf8")) as Record<string, string>;

const conditions = (shore: Conditions["shore"]): Conditions => ({
  lat: 35.366,
  lon: -120.851,
  marine: { status: "none", reason: "", source: "NWS" },
  alerts: { status: "none", reason: "", source: "NWS" },
  buoy: { status: "none", reason: "", source: "NDBC" },
  tides: { status: "none", reason: "", source: "CO-OPS" },
  harbor: { status: "none", reason: "", source: "CO-OPS" },
  shore,
});
const graph = (shore: Conditions["shore"]) =>
  harborEvidence({ place: { name: "Morro Bay", lat: 35.366, lon: -120.851 }, now: NOW, tripDate: "2026-10-13", measured: null, model: null, sat: null, conditions: conditions(shore), official: null });

describe("CeNCOOS shore stations", () => {
  it("every recorded answer is in the end-to-end replay, under the route's own URL", async () => {
    const { createHash } = await import("node:crypto");
    for (const s of SHORE_STATIONS) expect(fx[createHash("sha1").update(shoreUrl(s, NOW)).digest("hex") + ".txt"]).toBe(csv(s.id));
  });

  it("reads the newest row, keeping only values that passed QARTOD", () => {
    const morro = parseShoreCsv(csv("edu_calpoly_marine_morro"))!;
    expect(morro.at).toBe("2026-10-13T06:00:00Z");
    expect(morro.readings).toEqual({ water_temperature: { value: 18.97, at: morro.at }, salinity: { value: 33.367, at: morro.at }, chlorophyll_fluorescence: { value: 1.38, at: morro.at } });
    // Bodega's chlorophyll is NaN and "not evaluated": neither a reading nor a withheld number
    const bodega = parseShoreCsv(csv("bodega-bay-bml_wts"))!;
    expect(bodega.readings.chlorophyll_fluorescence).toBeUndefined();
    // Santa Cruz Wharf's last report (August 19): salinity suspect, chlorophyll failed
    const sc = parseShoreCsv(csv("edu_ucsc_scwharf1"))!;
    expect(sc.at).toBe("2026-08-19T15:55:00Z");
    expect(sc.withheld).toEqual({ salinity: "suspect", chlorophyll_fluorescence: "failed" });
    expect(Object.keys(sc.readings)).toEqual(["water_temperature"]);
  });

  it("serves only harbours within 6 km", () => {
    expect(shoreCandidate(35.366, -120.851)?.id).toBe("edu_calpoly_marine_morro"); // Morro Bay port, 1.0 km
    expect(shoreCandidate(40.8, -124.165)?.id).toBe("edu_humboldt_humboldt"); // Eureka, 3.6 km
    expect(shoreCandidate(36.60024, -121.89469)).toBeNull(); // Monterey: none
  });

  it("puts each reading in the graph as its own node and the temperature on the conditions line", () => {
    const data = parseShoreCsv(csv("edu_calpoly_marine_morro"))!;
    const g = graph({ status: "ok", data, source: "CeNCOOS", url: "https://erddap.cencoos.org/erddap/tabledap/edu_calpoly_marine_morro.html", fetchedAt: NOW.toISOString(), station: { id: "edu_calpoly_marine_morro", name: "Morro Bay shore station", km: 1.0, lat: 35.37085, lon: -120.8589 } });
    expect(g.nodes.filter((n) => n.id.startsWith("cencoos:")).map((n) => n.text)).toEqual(["19.0 °C", "33.4 PSU", "1.4 µg/L"]);
    const chl = g.nodes.find((n) => n.id.endsWith(":chlorophyll_fluorescence"))!;
    expect(chl.caveats[0]).toMatch(/not toxin/);
    expect(chl.where.setting).toBe("sheltered-harbor");
    const line = g.claims.find((c) => c.id === "conditions")!;
    expect(line.headline).toBe("Water 19.0 °C at Morro Bay shore station");
    // chlorophyll is never on the conditions line
    expect(`${line.headline} ${line.detail} ${line.text}`).not.toMatch(/chlorophyll/i);
    expect(checkGraph(g)).toEqual([]);
  });

  it("says withheld values and a silent station as gaps, never as numbers", () => {
    const data = parseShoreCsv(csv("edu_ucsc_scwharf1"))!;
    const g = graph({ status: "ok", data: { ...data, at: NOW.toISOString() }, source: "CeNCOOS", url: "u", fetchedAt: NOW.toISOString(), station: { id: "edu_ucsc_scwharf1", name: "Santa Cruz Wharf shore station", km: 1.8 } });
    expect(g.gaps.find((x) => x.id === "shore-chlorophyll_fluorescence")?.reason).toBe("Santa Cruz Wharf shore station: its latest chlorophyll fluorescence reading was flagged “failed” by its quality tests, so it is not shown.");
    expect(g.nodes.some((n) => n.id.endsWith(":chlorophyll_fluorescence"))).toBe(false);
    const silent = graph({ status: "unavailable", reason: "Santa Cruz Wharf shore station (1.8 km) has not reported since 2026-08-19.", source: "CeNCOOS" });
    expect(silent.gaps.find((x) => x.id === "shore")?.reason).toMatch(/has not reported since 2026-08-19/);
    expect(silent.nodes.some((n) => n.id.startsWith("cencoos:"))).toBe(false);
  });
});
