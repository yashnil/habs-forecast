import { describe, expect, it } from "vitest";
import type { LayerArtifact } from "@/generated/schema";
import { stampLines } from "@/lib/stamp";

const cur = (fraction: number | null) =>
  ({
    layer_id: "hfr2km_currents_20261009T20Z",
    time: { valid_time: "2026-10-09T20:00:00Z", observed_times: ["2026-10-09T20:00:00Z"] },
    coverage: { regions: fraction == null ? [] : [{ region_id: "north_coast", label: "North Coast", observed_fraction: fraction }] },
  }) as unknown as LayerArtifact;

describe("map timestamp lines", () => {
  it("currents: observation time and age, never forecast wording", () => {
    const l = stampLines({ group: "currents", currents: cur(0.4), now: new Date("2026-10-10T03:00:00Z"), region: { id: "north_coast", label: "North Coast" } });
    expect(l.map((x) => x.kind)).toEqual(["observation"]);
    expect(l[0].text).toMatch(/^Currents, HF radar · observed Oct 9, 1:00 PM PDT \(7 h ago\)$/);
    expect(l[0].text).not.toMatch(/forecast/i);
  });
  it("missing coverage in the selected region is stated, also when the region is outside the domain", () => {
    for (const f of [0, null]) {
      const l = stampLines({ group: "currents", currents: cur(f), region: { id: "north_coast", label: "North Coast" } });
      expect(l.find((x) => x.kind === "gap")?.text).toBe("No radar observations in North Coast this hour: no data, not calm water");
    }
    expect(stampLines({ group: "currents", currents: cur(0.4), region: null }).some((x) => x.kind === "gap")).toBe(false);
  });
  it("forecast lines are marked as model", () => {
    const f = { layer_id: "charm_x", time: { valid_date: "2026-10-08", lead_days: 1 } } as unknown as LayerArtifact;
    const l = stampLines({ group: "forecast", forecast: f, run: { issued_date: "2026-10-08" } as never });
    expect(l).toEqual([{ kind: "model", testid: "stamp-forecast", text: "C-HARM forecast for Thu, Oct 8 · issued Oct 8" }]);
  });
  it("satellite lines name the sensor and resolution of the layer drawn", () => {
    const sat = (id: string, platforms: string[], res: number) =>
      ({
        layer_id: id,
        short_title: "Chlorophyll",
        platforms,
        native_resolution_m: res,
        time: { observed_date: "2026-10-04" },
        composite: { oldest_observed_date: "2026-10-03", newest_observed_date: "2026-10-04" },
      }) as unknown as LayerArtifact;
    expect(stampLines({ group: "satellite", satellite: sat("viirs750_chl_latest", ["VIIRS"], 750) })[0].text).toBe(
      "Chlorophyll, VIIRS 750 m · pixels observed Oct 3–Oct 4",
    );
    expect(stampLines({ group: "satellite", satellite: sat("olci300_chl_latest", ["Sentinel-3A", "Sentinel-3B"], 300) })[0].text).toBe(
      "Chlorophyll, Sentinel-3 300 m · pixels observed Oct 3–Oct 4",
    );
    const day = { ...sat("olci300_chl_2026-10-04", ["Sentinel-3A"], 300), composite: null } as unknown as LayerArtifact;
    expect(stampLines({ group: "satellite", satellite: day })[0].text).toBe("Chlorophyll, Sentinel-3 300 m · overpass Sun, Oct 4");
    const ms = { ...sat("multisensor_chl_latest", ["Sentinel-3A", "Sentinel-3B", "VIIRS"], 0), native_resolution_m: null, composite: null, multisensor: {}, time: { observed_date: "2026-10-08" } } as unknown as LayerArtifact;
    expect(stampLines({ group: "satellite", satellite: ms })[0].text).toBe("Chlorophyll, Sentinel-3 300 m + VIIRS 750 m · newest pixel Oct 8; each pixel has its own date");
  });
});
