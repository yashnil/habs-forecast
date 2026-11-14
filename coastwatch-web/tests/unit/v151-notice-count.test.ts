import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import { currentRecords, disputedRecords } from "@/lib/official";
import { buildNotices, noticesLine } from "@/nx/notices/model";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const ds = JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith("official-"))!), "utf8")) as OfficialDataset;
const NOW = new Date("2026-10-13T06:00:00Z");

/**
 * v15.1 review: the informational pages' registry line (src/nx/pages/registry.server.ts, currentRecords) said 10 and
 * the Notices page said 11 on data with one disputed record. One rule now: a current notice is lib/official
 * currentRecords (active, and not one an agency release appears to lift); a disputed record is shown apart
 * ("where sources differ") and never counted as in effect.
 */
describe("the coast's notice count", () => {
  it("on fixture data, Notices' headline count is the registry line's count", () => {
    const v = buildNotices(ds, null, NOW)!;
    expect(v.total).toBe(currentRecords(ds).length);
    expect(noticesLine(v)).toMatch(new RegExp(`^${currentRecords(ds).length} on the coast`));
  });

  it("a record an agency release appears to lift is listed apart and counted by neither", () => {
    const first = currentRecords(ds)[0];
    const withDispute = {
      ...ds,
      conflicts: [...ds.conflicts, `Agency source discrepancy: CDFW release 2026-10-12 ("Fishery reopened", https://wildlife.ca.gov/x) appears to lift active record ${first.id}.`],
    } as OfficialDataset;
    expect(disputedRecords(withDispute)).toHaveLength(1);
    const v = buildNotices(withDispute, null, NOW)!;
    expect(v.differ).toHaveLength(1);
    expect(v.total).toBe(currentRecords(withDispute).length);
    expect(v.total).toBe(currentRecords(ds).length - 1);
  });
});
