import { describe, expect, it } from "vitest";
import { SEASON_REGIONS, SPECIES_SEASON_IDS } from "@/content/seasons";
import { LANDINGS } from "@/content/landings";
import { SPECIES } from "@/content/species";
import { groundfishOn, portAreaValues, regionFor, regionSeasons, seasonsAgeDays, speciesValues } from "@/lib/speciesSeasons";
import { datePeriods, marineMaxima, overLimits, parseLimits, periodFor } from "@/lib/trip";

describe("CDFW regions", () => {
  it("covers the coast without gaps, by CDFW's latitude lines", () => {
    const sorted = [...SEASON_REGIONS].sort((a, b) => a.south - b.south);
    for (let i = 1; i < sorted.length; i++) expect(sorted[i].south).toBeCloseTo(sorted[i - 1].north, 4);
    expect(regionFor(41.749)?.id).toBe("northern"); // Crescent City
    expect(regionFor(40.0255)?.id).toBe("mendocino"); // Shelter Cove, south of 40°10′
    expect(regionFor(37.5037)?.id).toBe("san-francisco"); // Half Moon Bay
    expect(regionFor(37.15)?.id).toBe("central"); // just south of Pigeon Point
    expect(regionFor(36.6)?.id).toBe("central"); // Monterey
    expect(regionFor(34.407)?.id).toBe("southern"); // Santa Barbara
    expect(regionFor(30)).toBeNull();
  });
  it("keeps every entry quoted, marked and dated", () => {
    for (const r of SEASON_REGIONS) {
      expect(r.updated).toMatch(/^[A-Z][a-z]+ \d{1,2}, 20\d\d$/);
      expect(r.url).toMatch(/^https:\/\/wildlife\.ca\.gov\//);
      expect(r.items.length).toBeGreaterThan(15);
      for (const i of r.items) {
        expect(["open", "alert", "closed"]).toContain(i.mark);
        expect(i.mark === "closed").toBe(i.list === "closed");
        expect(i.quote.length).toBeGreaterThan(20);
      }
    }
  });
  it("maps species to CDFW's own entries", () => {
    const sf = regionSeasons(SEASON_REGIONS.find((r) => r.id === "san-francisco")!, "dungeness");
    expect(sf.mine.map((i) => i.list)).toEqual(["closed"]);
    expect(sf.mine[0].quote).toMatch(/scheduled to reopen/);
    const so = regionSeasons(SEASON_REGIONS.find((r) => r.id === "southern")!, "lobster");
    expect(so.mine[0].id).toBe("lobster");
    // the headline never repeats the reader's own species
    expect(sf.headline.some((h) => sf.mine.includes(h))).toBe(false);
    // every mapped id exists somewhere on CDFW's pages
    const all = new Set(SEASON_REGIONS.flatMap((r) => r.items.map((i) => i.id)));
    for (const ids of Object.values(SPECIES_SEASON_IDS)) expect(ids!.some((x) => all.has(x))).toBe(true);
    for (const s of SPECIES) expect(typeof s.label).toBe("string");
  });
  it("reads CDFW's groundfish table for the trip day, and only for the year it covers", () => {
    expect(groundfishOn("central", "2026-10-20")?.nearshore).toBe("Open all depths");
    expect(groundfishOn("central", "2026-02-01")?.nearshore).toMatch(/^Closed/);
    expect(groundfishOn("southern", "2026-10-20")).toMatchObject({ nearshore: expect.stringMatching(/^Closed/), shelf: "50 Fathom - Offshore Only" });
    expect(groundfishOn("central", "2027-04-01")).toBeNull();
    expect(seasonsAgeDays("2026-11-05")).toBe(16);
  });
});

describe("CDFW By The Numbers", () => {
  it("transcribes totals that add up", () => {
    for (const r of LANDINGS) {
      const sum = r.portAreas.reduce((s, a) => s + a.dollars, 0);
      expect(Math.abs(sum - r.portTotal.dollars)).toBeLessThan(5);
      const lb = r.portAreas.reduce((s, a) => s + a.pounds, 0);
      expect(Math.abs(lb - r.portTotal.pounds)).toBeLessThan(5);
    }
    // CDFW's 2024 total, as printed (a transposed 117,684,528 was here before v6)
    expect(LANDINGS.find((r) => r.year === 2024)!.portTotal.pounds).toBe(177_684_524);
  });
  it("gives a species only its own fishery's value, never a neighbour's", () => {
    expect(speciesValues("tunas")).toEqual([]); // bluefin and bigeye are not albacore
    expect(speciesValues("groundfish")).toEqual([]); // sablefish and petrale are not nearshore groundfish
    expect(speciesValues("halibut").map((v) => v.fishery)).toEqual(["California Halibut", "California Halibut"]);
  });
  it("finds a port area and the reader's species, with the arithmetic labelled per pound", () => {
    const sf = portAreaValues("SAN FRANCISCO");
    expect(sf.map((x) => x.year)).toEqual([2024, 2023]);
    expect(sf[0].rank).toBe(3);
    const crab = speciesValues("dungeness");
    expect(crab[0]).toMatchObject({ year: 2024, rank: 2 });
    expect(crab[0].perLb).toBeCloseTo(49_744_974 / 14_173_811, 6);
    expect(speciesValues("salmon")).toEqual([]);
    expect(portAreaValues("NOWHERE")).toEqual([]);
  });
});

describe("the reader's sea limits against NWS's words", () => {
  it("reads the highest wind and seas the forecaster states", () => {
    expect(marineMaxima("SE wind around 5 kt. Seas 4 to 6 ft. SW swell 5 ft at 17 seconds.")).toEqual({ windKt: 5, seasFt: 6 });
    expect(marineMaxima("NW wind 10 to 15 kt, rising to 20 to 25 kt with gusts up to 30 kt in the afternoon. Seas 7 to 9 ft.")).toEqual({ windKt: 30, seasFt: 9 });
    expect(marineMaxima("W wind 5 kt. Wind waves 2 ft. NW swell 6 ft at 12 seconds.")).toEqual({ windKt: 5, seasFt: 6 });
    expect(marineMaxima("Patchy fog.")).toEqual({ windKt: null, seasFt: null });
  });
  it("keeps only sensible limits and compares", () => {
    expect(parseLimits('{"windKt":15,"seasFt":6}')).toEqual({ windKt: 15, seasFt: 6 });
    expect(parseLimits('{"windKt":-1,"seasFt":"x"}')).toEqual({ windKt: null, seasFt: null });
    expect(parseLimits("not json")).toEqual({ windKt: null, seasFt: null });
    expect(overLimits({ windKt: 20, seasFt: 6 }, { windKt: 15, seasFt: 6 })).toEqual({ wind: true, seas: false });
    expect(overLimits({ windKt: null, seasFt: 9 }, { windKt: 15, seasFt: null })).toEqual({ wind: false, seas: false });
  });
});

describe("the NWS period for the trip day, by its own start", () => {
  const periods = [
    { name: "Tonight", text: "NW wind 10 kt. Seas 4 ft." },
    { name: "Tue", text: "NW wind 15 kt. Seas 5 ft." },
    { name: "Tue Night", text: "NW wind 20 kt. Seas 6 ft." },
    { name: "Wed", text: "NW wind 25 kt. Seas 8 ft." },
  ];
  // issued Monday Oct 12, 2026, 8 PM PDT (Oct 13 03:00 UTC)
  const c = { marine: { status: "ok", data: { periods, updated: "2026-10-13T03:00:00Z" } } };
  it("dates each period from the issue time and its name", () => {
    expect(datePeriods(periods, "2026-10-13T03:00:00Z").map((p) => [p.date, p.night])).toEqual([
      ["2026-10-12", true],
      ["2026-10-13", false],
      ["2026-10-13", true],
      ["2026-10-14", false],
    ]);
    expect(datePeriods(periods, null).every((p) => p.date === null)).toBe(true);
  });
  it("matches the trip day by date, daytime first, and never falls back to the first period", () => {
    expect(periodFor(c, "2026-10-12", "2026-10-12")?.name).toBe("Tonight");
    expect(periodFor(c, "2026-10-13", "2026-10-12")?.name).toBe("Tue");
    expect(periodFor(c, "2026-10-14", "2026-10-12")?.name).toBe("Wed");
    // a saved copy read three days later: nothing it holds starts on that day
    expect(periodFor(c, "2026-10-15", "2026-10-15")).toBeNull();
    expect(periodFor(c, null, "2026-10-16")).toBeNull();
    expect(periodFor({ marine: { status: "error" } }, "2026-10-13", "2026-10-13")).toBeNull();
  });
});
