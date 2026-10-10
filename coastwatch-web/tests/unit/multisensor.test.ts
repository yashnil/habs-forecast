import { describe, expect, it } from "vitest";
import { multiSensorPick, observedDate, ratioPhrase } from "@/lib/multisensor";
import { SENSOR_COLOURS } from "@/lib/palette";
import { readFileSync } from "node:fs";
import path from "node:path";

const D = "2026-10-06";
const back = (n: number) => observedDate(D, n);

describe("multi-sensor rule (same table as pipeline tests/test_multisensor.py::test_rule_table)", () => {
  it.each([
    [{ value: 1, date: D }, null, 1],
    [null, { value: 2, date: D }, 2],
    [{ value: 1, date: D }, { value: 2, date: D }, 1],
    [{ value: 1, date: back(2) }, { value: 2, date: D }, 1],
    [{ value: 1, date: back(3) }, { value: 2, date: D }, 2],
    [{ value: 1, date: D }, { value: 2, date: back(5) }, 1],
    [null, null, 0],
  ] as const)("case %#", (p, s, want) => {
    expect(multiSensorPick(p, s, 2)).toBe(want);
  });

  it("dates from ages", () => {
    expect(observedDate("2026-10-09", 0)).toBe("2026-10-09");
    expect(observedDate("2026-10-09", 7)).toBe("2026-10-02");
  });

  it("describes the agreement ratio in words", () => {
    expect(ratioPhrase(-0.187)).toBe("Sentinel-3 read about 35% lower than VIIRS");
    expect(ratioPhrase(0.1)).toBe("Sentinel-3 read about 26% higher than VIIRS");
    expect(ratioPhrase(0.01)).toBe("Sentinel-3 and VIIRS read about the same");
  });

  it("sensor colours mirror the pipeline", () => {
    const py = readFileSync(path.resolve(__dirname, "../../../pipeline/coastwatch_pipeline/process/palette.py"), "utf8");
    const m = py.match(/SENSOR_COLOURS = \[([^\]]+)\]/);
    expect(m && JSON.parse(`[${m[1]}]`)).toEqual(SENSOR_COLOURS);
  });
});
