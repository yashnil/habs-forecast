import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { SPECIES, speciesById } from "@/content/species";
import { SPECIES_EVIDENCE } from "@/content/speciesEvidence";
import { checkGraph } from "@/lib/evidenceCheck";
import { harborEvidence, pageClaims, PAGE_MAX, type EvidenceGraph } from "@/lib/harborEvidence";

/**
 * The species line: the agencies' own sentences, quoted exactly (checked against excerpts of the
 * pages read on 2026-10-16, tests/unit/fixtures/agency-pages), then what Neritic's readings can say
 * about the toxins that matter for that species. It never says a species is safe or affected.
 */
const PAGES: Record<string, string> = {
  "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx": "cdph-domoic-acid",
  "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx": "cdph-marine-biotoxin",
  "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories": "cdfw-health-advisories",
};
const page = (url: string) => readFileSync(path.join(__dirname, "fixtures/agency-pages", `${PAGES[url]}-2026-10-16.txt`), "utf8");
const brief = (id: string) =>
  harborEvidence({ place: { name: "Monterey", lat: 36.6, lon: -121.89 }, now: new Date("2026-10-13T06:00:00Z"), tripDate: null, measured: null, model: null, sat: null, conditions: null, official: { assocs: [], verification: null, audience: "all", species: speciesById(id), statements: [] } });

describe("species evidence", () => {
  it("quotes every agency sentence exactly as its page says it", () => {
    for (const [id, ev] of Object.entries(SPECIES_EVIDENCE)) {
      for (const f of ev!.facts) {
        expect(PAGES[f.source.url], `${id}: unrecorded page ${f.source.url}`).toBeDefined();
        expect(page(f.source.url), `${id}: “${f.quote}”`).toContain(f.quote);
      }
    }
  });

  it("says which toxins the agencies watch for, and that the readings are the water, not the animal", () => {
    const crab = brief("dungeness").claims.find((c) => c.id === "species")!;
    expect(crab.headline).toBe("For Dungeness crab, the agencies watch for domoic acid");
    expect(crab.quote).toMatch(/pre-season testing of Dungeness crab/);
    // the page shows the headline and what the readings can say; the agency's other sentences fold with the quote
    expect(crab.detail).toBe("Neritic's shore readings and NOAA's model are about domoic acid in the water, not in the crab.");
    expect(crab.more).toMatch(/viscera .* much more domoic acid than the meat/);
  });

  it("says plainly that Neritic carries no reading for paralytic shellfish toxins", () => {
    const mussels = brief("mussel").claims.find((c) => c.id === "species")!;
    expect(mussels.headline).toMatch(/paralytic shellfish toxins \(saxitoxins\) and domoic acid$/);
    expect(mussels.detail).toMatch(/Neritic carries no seawater reading of the algae behind paralytic shellfish toxins/);
  });

  it("is a gap for species no agency source names a biotoxin test for, and says nothing for “any species”", () => {
    const salmon = brief("salmon").claims.find((c) => c.id === "species")!;
    expect(salmon.kind).toBe("gap");
    expect(salmon.headline).toBe("Neritic's sources name no biotoxin test specific to salmon");
    expect(salmon.detail).toBe("The shore readings describe the water, not the fish.");
    expect(brief("any").claims.some((c) => c.id === "species")).toBe(false);
  });

  it("adds a page line only for what it says: a gap species line stays in the API, not on the page", () => {
    expect(pageClaims(brief("salmon")).some((c) => c.id === "species")).toBe(false);
    expect(brief("salmon").claims.some((c) => c.id === "species")).toBe(true);
    expect(pageClaims(brief("mussel")).some((c) => c.id === "species")).toBe(true);
  });

  it("never shows more than six lines; satellite gives way first, then the model, then the sea; official words and the measurement stay", () => {
    const line = (id: string, kind: "official" | "measured" | "model" | "satellite" | "conditions" | "gap" = "official") => ({ id, kind, tone: "note" as const, headline: id, focus: null, text: id, supports: [] });
    const g = (ids: [string, Parameters<typeof line>[1]][]): EvidenceGraph => ({ place: { name: "x", lat: 0, lon: 0 }, generatedAt: "", window: { from: "", to: "" }, nodes: [], gaps: [], limits: [], claims: ids.map(([id, k]) => line(id, k)) });
    const eight = g([["season", "official"], ["disputed-a", "official"], ["notices", "official"], ["hab", "gap"], ["species", "official"], ["model", "model"], ["satellite", "satellite"], ["conditions", "conditions"]]);
    const page = pageClaims(eight).map((c) => c.id);
    expect(page.length).toBe(PAGE_MAX);
    expect(page).toEqual(["season", "disputed-a", "notices", "hab", "species", "conditions"]);
    // at or under six, nothing gives way
    expect(pageClaims(g([["notices", "official"], ["hab", "gap"], ["model", "model"], ["satellite", "satellite"], ["conditions", "conditions"]])).length).toBe(5);
  });

  it("keeps every consistency rule and never speaks of safety for any species", () => {
    for (const sp of SPECIES) {
      const g = brief(sp.id);
      expect(checkGraph(g)).toEqual([]);
      const c = g.claims.find((x) => x.id === "species");
      if (c) expect(`${c.headline} ${c.detail} ${c.text}`).not.toMatch(/\b(safe|unsafe|clean|all clear)\b/i);
    }
  });
});
