import { describe, expect, it } from "vitest";
import type { SeasonGate, WhaleGate } from "@/nx/coast/brief";
import { tripFishery } from "@/nx/coast/fisheries";
import { breakEvenGross, catchStep, needsWords, nextWords, passSteps, priceStep, seasonWarning, whatIfRows, zoneCaption } from "@/nx/coast/pass";
import { emptyWorkspace, evaluateWorkspace, money, setField, type Workspace } from "@/nx/coast/workspace";

/** The v14 trip pass's own arithmetic and words (src/nx/coast/pass.ts), over the owner's case. */
const crab = tripFishery("dungeness_commercial")!;
const sport = tripFishery("dungeness_recreational")!;
const owner = (o: Partial<Workspace> = {}): Workspace => ({
  ...emptyWorkspace(crab),
  catch: [{ label: "Dungeness crab", ref: "dungeness", amount: "400", price: "9" }],
  fuelGal: "30",
  fuelPrice: "8.24",
  fuelPayer: "shared",
  costs: [{ label: "Bait and ice", usd: "100", payer: "shared" }],
  feesPct: "0",
  crew: "1",
  crewKind: "afterExpenses",
  share: "20",
  ...o,
});

describe("the regression case on the pass", () => {
  it("works out to the cent: gross, fuel, operating, crew, net, break-evens", () => {
    const r = evaluateWorkspace(owner(), crab);
    const v = (k: keyof typeof r.out) => (r.out[k].status === "value" ? money((r.out[k] as { value: number }).value) : r.out[k].status);
    expect([v("gross"), v("fuel"), v("operating"), v("crewPay"), v("net")]).toEqual(["$3,600.00", "$247.20", "$347.20", "$650.56", "$2,602.24"]);
    const be = breakEvenGross(owner(), crab, r)!;
    expect(be.gross).toBeCloseTo(347.2, 6);
    expect(be.words).toBe("39 lb, or $0.87/lb");
    expect(be.beyond).toBe(false);
  });
  it("reads in the equation's order: gross, fuel, bait and ice, fees, crew, net", () => {
    expect(passSteps(evaluateWorkspace(owner(), crab).steps).map((s) => s.id)).toEqual(["gross", "fuel", "cost-0", "fees", "crew", "net"]);
  });
});

describe("what if", () => {
  it("catch ±25 % and price ±$1, each end the trip worked again (the audited values)", () => {
    const rows = whatIfRows(owner(), crab, evaluateWorkspace(owner(), crab))!;
    expect(rows.map((r) => [r.k, r.lo.label, money(r.lo.net), r.hi.label, money(r.hi.net)])).toEqual([
      ["catch", "300 lb", "$1,882.24", "500 lb", "$3,322.24"],
      ["price", "$8.00/lb", "$2,282.24", "$10.00/lb", "$2,922.24"],
    ]);
  });
  it("an end is exactly what setting that input gives", () => {
    const rows = whatIfRows(owner(), crab, evaluateWorkspace(owner(), crab))!;
    const set = evaluateWorkspace(owner({ catch: [{ label: "Dungeness crab", ref: "dungeness", amount: String(rows[0].hi.v), price: "9" }] }), crab).out.net;
    expect(set.status === "value" && set.value).toBe(rows[0].hi.net);
  });
  it("the price step: $1 between $4 and $20 a pound, otherwise a tenth (to the cent)", () => {
    expect(priceStep(9)).toBe(1);
    expect(priceStep(4)).toBe(1);
    expect(priceStep(20)).toBe(1);
    expect(priceStep(99)).toBeCloseTo(9.9, 9);
    expect(priceStep(3.5)).toBeCloseTo(0.35, 9);
    expect(priceStep(0.2)).toBe(0.05);
    expect(priceStep(900, "ton")).toBe(90);
    expect(catchStep(400)).toBe(100);
    expect(catchStep(10)).toBe(5);
  });
  it("is not drawn while net waits, for a recreational trip, or with more than one catch line", () => {
    const blank = emptyWorkspace(crab);
    expect(whatIfRows(blank, crab, evaluateWorkspace(blank, crab))).toBeNull();
    const rec = { ...emptyWorkspace(sport), fuelGal: "30", fuelPrice: "8.24", costs: [{ label: "Bait and ice", usd: "100", payer: null }], anglers: "2" };
    expect(whatIfRows(rec, sport, evaluateWorkspace(rec, sport))).toBeNull();
    const two = owner({ catch: [...owner().catch, { label: "Rock crab", ref: null, amount: "50", price: "4" }] });
    expect(whatIfRows(two, crab, evaluateWorkspace(two, crab))).toBeNull();
  });
});

describe("the one button and the blanks", () => {
  it("walks the blanks in the page's order, and says what a blank still needs, never 0", () => {
    const ws = setField(emptyWorkspace(crab), "fuelGal", "");
    const r = evaluateWorkspace(ws, crab);
    expect(nextWords(r.waiting[0], ws, crab)).toBe("Add catch");
    expect(nextWords("fuelGal", ws, crab)).toBe("Add fuel");
    expect(nextWords("crewKind", ws, crab)).toBe("Choose how crew is paid");
    expect(nextWords("fuelPayer", ws, crab)).toBe("Choose who pays fuel");
    expect(needsWords(["fuelGal", "cost.0.usd", "feesPct", "crew"], ws, crab)).toBe("Needs fuel, bait and ice, fees and crew. A blank is a gap, never 0.");
    expect(needsWords(r.waiting, ws, crab)).toMatch(/^Needs .+ more\. A blank is a gap, never 0\.$/);
  });
});

describe("the record's words", () => {
  const season = (o: Partial<SeasonGate> = {}): SeasonGate => ({ text: "Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).", tone: "regulation", level: "delayed", conflict: false, seasonal: false, source: "CDFW", url: null, checked: null, verified: false, stale: false, differ: null, ...o });
  it("the strip's season line: the record's sentence, its cause left to the gates", () => {
    expect(seasonWarning(season())).toBe("Season delayed to at least Jan 1, 2027: a plan, not a trip you can make now");
    expect(seasonWarning(season({ level: "open" as SeasonGate["level"] }))).toBeNull();
    expect(seasonWarning(season({ conflict: true }))).toBeNull();
  });
  it("the whale zone tag's second line", () => {
    const whale = { zone: "4", area: "Pigeon Point to Lopez Point", measures: "Fleet Advisory. No depth limit or gear reduction on record.", next: null, assessed: null, url: null, verified: false } as WhaleGate;
    expect(zoneCaption(season(), whale)).toBe("Season delayed · Fleet Advisory");
    expect(zoneCaption(null, { ...whale, measures: "No depth limit on record." })).toBeNull();
  });
});
