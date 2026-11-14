import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { loadBiotoxin, loadFisheriesStatus, overAction, parseBiotoxin, parseFisheriesStatus, rampZoneForPlace, recordsForPlace, reviewWords, sitesForPlace } from "@/pilot/data";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const read = (rel: string): unknown => JSON.parse(readFileSync(path.join(FIX, rel), "utf8"));
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));

describe("pilot biotoxin loader", () => {
  const raw = read("biotoxin/index.json") as Record<string, unknown>;
  const idx = parseBiotoxin(raw)!;

  it("parses the fixture (CDPH crab PDF as of 2026-11-02)", () => {
    expect(idx).not.toBeNull();
    expect(idx.source.as_of).toBe("2026-11-02");
    expect(idx.parse_warnings).toEqual([]);
    const mb = idx.sites.find((s) => s.id === "bma-g-monterey-bay-dungeness-crab")!;
    expect(mb.sets[0]).toMatchObject({ date: "2026-09-28", avg_ppm: 5.5, pct_over_action: 0, action_level_ppm: 30 });
    const usal = idx.sites.find((s) => s.id === "bma-c-usal-dungeness-crab")!;
    expect(usal.sets[0].max_ppm).toBe(99);
    expect(overAction(usal.sets[0])).toBe(3);
  });

  it("maps sites to pilot places conservatively", () => {
    expect(sitesForPlace(idx, "port-592").map((s) => s.id).sort()).toEqual(["bma-g-monterey-bay-dungeness-crab", "bma-g-monterey-bay-rock-crab"]);
    expect(sitesForPlace(idx, "port-452").map((s) => s.site_label).sort()).toEqual(["Pigeon Point", "Pillar Point"]);
  });

  it("keeps ND and below-limit values null, never zero", () => {
    const rock = idx.sites.find((s) => s.id === "bma-g-monterey-bay-rock-crab")!.sets[0];
    expect(rock.values_ppm.filter((v) => v === null)).toHaveLength(4);
    expect(rock.values_ppm).not.toContain(0);
    expect(rock.nd_count).toBe(4);
  });

  it("drops malformed sites and rejects a wrong version or shape", () => {
    const bad = clone(raw) as { sites: Record<string, unknown>[] };
    bad.sites[0].sets = [{ date: "yesterday" }];
    expect(parseBiotoxin(bad)!.sites).toHaveLength(idx.sites.length - 1);
    expect(parseBiotoxin({ ...raw, version: 2 })).toBeNull();
    expect(parseBiotoxin({ version: 1 })).toBeNull();
    expect(parseBiotoxin(null)).toBeNull();
    expect(parseBiotoxin("<html>")).toBeNull();
  });

  it("a missing or failing file is null, never a throw", async () => {
    const notFound = (async () => new Response("nope", { status: 404 })) as typeof fetch;
    const boom = (async () => { throw new Error("offline"); }) as typeof fetch;
    const html = (async () => new Response("<html>", { status: 200 })) as typeof fetch;
    expect(await loadBiotoxin("https://a.example/v1", notFound)).toBeNull();
    expect(await loadBiotoxin("https://b.example/v1", boom)).toBeNull();
    expect(await loadBiotoxin("https://c.example/v1", html)).toBeNull();
    const ok = (async () => new Response(JSON.stringify(raw), { status: 200 })) as typeof fetch;
    expect((await loadBiotoxin("https://d.example/v1/", ok))?.sites.length).toBe(idx.sites.length);
  });
});

describe("pilot fisheries-status loader", () => {
  const raw = read("fisheries-status/index.json") as Record<string, unknown>;
  const idx = parseFisheriesStatus(raw)!;

  it("parses the fixture; every record is pending human review and says Not verified", () => {
    expect(idx.records.length).toBeGreaterThanOrEqual(6);
    for (const r of idx.records) {
      expect(r.review_state).toBe("pending_human_review");
      expect(reviewWords(r)).toBe("Not verified");
      expect(r.source_url).toMatch(/^https:\/\//);
    }
    const dc = idx.records.find((r) => r.id === "dungeness-commercial-2026-27")!;
    expect(dc).toMatchObject({ level: "delayed", cause: "whale_entanglement", next_review: "2026-12-15" });
    expect(idx.ramp?.review_state).toBe("pending_human_review");
  });

  it("selects the records and RAMP zone for a place", () => {
    const ids = recordsForPlace(idx, "port-452").map((r) => r.id);
    expect(ids).toContain("dungeness-commercial-2026-27"); // statewide
    expect(ids).not.toContain("anchovy-monterey-bay-2026-09"); // Princeton is north of Pigeon Point
    expect(recordsForPlace(idx, "port-592").map((r) => r.id)).toContain("anchovy-monterey-bay-2026-09");
    expect(rampZoneForPlace(idx, "port-592")?.zone).toBe("4");
    expect(rampZoneForPlace(idx, "port-452")?.zone).toBe("3");
  });

  it("never treats a record as verified without a named reviewer", () => {
    const forged = clone(raw) as { records: Record<string, unknown>[]; ramp: Record<string, unknown> };
    forged.records[0].review_state = "human_verified";
    forged.ramp.review_state = "human_verified";
    const p = parseFisheriesStatus(forged)!;
    expect(p.records[0].review_state).toBe("pending_human_review");
    expect(p.ramp?.review_state).toBe("pending_human_review");
    forged.records[0].reviewed_by = "Jane Doe";
    expect(parseFisheriesStatus(forged)!.records[0].review_state).toBe("human_verified");
  });

  it("drops malformed records and rejects other versions", () => {
    const bad = clone(raw) as { records: Record<string, unknown>[] };
    bad.records[0].level = "safe";
    expect(parseFisheriesStatus(bad)!.records).toHaveLength(idx.records.length - 1);
    expect(parseFisheriesStatus({ ...raw, version: 2 })).toBeNull();
    expect(parseFisheriesStatus({})).toBeNull();
  });

  it("a missing file is null", async () => {
    const notFound = (async () => new Response("", { status: 404 })) as typeof fetch;
    expect(await loadFisheriesStatus("https://e.example/v1", notFound)).toBeNull();
  });
});
