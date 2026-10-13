import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OfficialDataset } from "@/generated/official";
import { speciesById, speciesPhrase, recordNamesSpecies } from "@/content/species";
import { associationsAt } from "@/lib/applicability";
import { nextSteps, type StepInput } from "@/lib/nextSteps";
import type { Conditions } from "@/lib/conditions";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const FAILED = path.resolve(__dirname, "../fixture-data-failed/v1");
const load = (dir: string): OfficialDataset => {
  const m = JSON.parse(readFileSync(path.join(dir, "manifest.json"), "utf8")) as { official_url: string };
  return JSON.parse(readFileSync(path.join(dir, m.official_url), "utf8"));
};
const official = load(FIX);
const MONTEREY = { lat: 36.60024, lon: -121.89469 };
const CRESCENT = { lat: 41.74901, lon: -124.20398 };

function input(over: Partial<StepInput> & { at?: { lat: number; lon: number }; ds?: OfficialDataset }): StepInput {
  const at = over.at ?? MONTEREY;
  const ds = over.ds ?? official;
  return {
    official: ds,
    verification: { state: "unverified", reasons: [], reviewedAt: null, reviewedBy: null, reviewAgeDays: null, lastCheckedAt: null },
    assocs: associationsAt(ds, "x", at.lat, at.lon),
    audience: "all",
    species: speciesById("any"),
    lat: at.lat,
    conditions: null,
    lastToxin: { station: "Monterey Wharf", km: 0.6, date: "2022-08-31", ageDays: 1503 },
    fmt: (s) => s,
    ...over,
  };
}

describe("next steps", () => {
  it("every step has a source to open, and none says waters are safe, open or good for fishing", () => {
    for (const sp of ["any", "anchovy-sardine", "razor-clam", "dungeness", "mussel"])
      for (const audience of ["all", "commercial", "recreational", "shellfish"] as const)
        for (const at of [MONTEREY, CRESCENT]) {
          const steps = nextSteps(input({ species: speciesById(sp), audience, at }));
          expect(steps.length).toBeGreaterThan(2);
          for (const s of steps) {
            expect("url" in s.source ? s.source.url : s.source.tel).toMatch(/^(https:\/\/|\+1)/);
            const text = `${s.title} ${s.reason}`;
            const bad = /\b(safe|open|all clear|good fishing|productive)\b/i.test(text) && !/\b(not|no|never)\b/i.test(text);
            expect(bad, text).toBe(false);
          }
        }
  });

  it("one step points to the notices that apply, for the chosen species, with the limit nearby", () => {
    const steps = nextSteps(input({ species: speciesById("anchovy-sardine"), audience: "recreational" }));
    const read = steps.find((s) => s.id === "read-notices")!;
    expect(read.title).toBe("Read the 2 notices that apply here, in the agencies' own words");
    expect(read.reason).toMatch(/south limit .* 8\.\d km south/);
  });

  it("a commercial fisher sees the commercial statement, a recreational one the recreational", () => {
    const com = nextSteps(input({ species: speciesById("dungeness"), audience: "commercial", at: CRESCENT })).filter((s) => s.id.startsWith("statement-"));
    const rec = nextSteps(input({ species: speciesById("dungeness"), audience: "recreational", at: CRESCENT })).filter((s) => s.id.startsWith("statement-"));
    expect(com.map((s) => s.title).join(" ")).toMatch(/commercial Dungeness/);
    expect(com.map((s) => s.title).join(" ")).not.toMatch(/recreational Dungeness crab fishery due/);
    expect(rec.map((s) => s.title).join(" ")).toMatch(/recreational Dungeness/);
  });

  it("a notice an agency release appears to lift comes first, with the release", () => {
    const ds = load(FAILED);
    const steps = nextSteps(input({ ds, species: speciesById("bivalve"), audience: "shellfish" }));
    expect(steps[0].id).toBe("dispute-cdph-2026-sn26-018-monterey-bivalves");
    expect("url" in steps[0].source && steps[0].source.url).toBe("https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx");
    expect(steps.find((s) => s.id === "read-notices")?.reason ?? "").not.toMatch(/Monterey County/);
  });

  it("gaps are stated: no recent toxin sample, an unavailable marine forecast", () => {
    const conditions = {
      lat: 0,
      lon: 0,
      marine: { status: "unavailable", reason: "The NWS forecast did not load (HTTP 503).", source: "NWS" },
      alerts: { status: "unavailable", reason: "x", source: "NWS" },
      buoy: { status: "none", reason: "No NDBC station reports within 60 km.", source: "NDBC" },
      tides: { status: "none", reason: "x", source: "CO-OPS" },
    } as Conditions;
    const steps = nextSteps(input({ conditions }));
    expect(steps.find((s) => s.id === "toxin-gap")?.kind).toBe("gap");
    expect(steps.find((s) => s.id === "marine-gap")?.reason).toMatch(/HTTP 503/);
  });
});

describe("species", () => {
  it("match an agency's own species list, never a place name", () => {
    expect(recordNamesSpecies(["razor clam"], speciesById("razor-clam"))).toBe(true);
    expect(recordNamesSpecies(["mussels (all species)"], speciesById("mussel"))).toBe(true);
    expect(recordNamesSpecies(["rock crab (internal organs)"], speciesById("rock-crab"))).toBe(true);
    expect(recordNamesSpecies(["northern anchovy"], speciesById("dungeness"))).toBe(false);
    expect(speciesPhrase(speciesById("dungeness"))).toBe("Dungeness crab");
    expect(speciesPhrase(speciesById("razor-clam"))).toBe("razor clams");
  });
  it("every species link is an https agency page", () => {
    for (const id of ["dungeness", "rock-crab", "razor-clam", "mussel", "bivalve", "anchovy-sardine", "salmon", "groundfish", "halibut", "squid", "lobster"])
      expect(speciesById(id).link?.url).toMatch(/^https:\/\/wildlife\.ca\.gov\//);
  });
});

