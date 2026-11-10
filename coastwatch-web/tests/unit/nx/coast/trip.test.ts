import { describe, expect, it } from "vitest";
import { sanitizeDraft } from "@/lib/tripEconomicsStore";
import { coastFishery, defaultFishery, fuelReference, priceReference, shareReference } from "@/nx/coast/fisheries";
import { assumptions, EMPTY_FORM, evaluateForm, fromDraft, gridValue, GRID_COLS, GRID_ROWS, scenarioDomain, toDraft, tripName, waitingWords, type CoastForm, type Result } from "@/nx/coast/trip";

/**
 * My Coast's worksheet adapter over evaluateTrip. Every expected figure below is worked by hand from the
 * formulas in src/lib/tripEconomics.ts (R gross, τ fees, D shared costs, B boat-paid costs, s crew share):
 *   share after costs: N = R(1−τ) − D; crew = s·N; net = N − crew − B
 *   share of gross:    crew = s·R;     net = R(1−s−τ) − D − B
 *   day rate:          crew = people × $/day × days; net = R(1−τ) − D − B − crew
 */
const crab = coastFishery("dungeness_commercial")!;
const squid = coastFishery("market_squid")!;

/** 800 lb at $4.75, fees 5 %, 120 gal at $4.60, $180 bait and ice, 2 crew on a 30 % share after costs. */
const A: CoastForm = { ...EMPTY_FORM, catch: "800", price: "4.75", fees: "5", fuelGal: "120", fuelPrice: "4.60", supplies: "180", crew: "2", crewKind: "afterExpenses", share: "30", fuelPayer: "shared" };

const v = (r: Result) => {
  if (r.status !== "value") throw new Error(`not a value: ${JSON.stringify(r)}`);
  return r.value;
};

describe("hand-computed trips through the adapter", () => {
  it("share after costs, fuel off the top", () => {
    const r = evaluateForm(A, crab);
    // R = 800 × 4.75 = 3,800; fees = 190; fuel = 552; D = 552 + 180 = 732
    expect(v(r.out.gross)).toBeCloseTo(3800, 6);
    expect(v(r.out.fees)).toBeCloseTo(190, 6);
    expect(v(r.out.fuel)).toBeCloseTo(552, 6);
    expect(v(r.out.supplies)).toBeCloseTo(180, 6);
    expect(v(r.out.tripCost)).toBeCloseTo(732, 6);
    // N = 3,800 − 190 − 732 = 2,878; crew = 0.3 × 2,878 = 863.4; net = 2,014.6; per crew = 431.7
    expect(v(r.out.crewPay)).toBeCloseTo(863.4, 6);
    expect(v(r.out.net)).toBeCloseTo(2014.6, 6);
    expect(v(r.out.perCrew)).toBeCloseTo(431.7, 6);
    // break-even price: 800 p × 0.95 = 732 → p = 0.963158; catch: 4.75 q × 0.95 = 732 → q = 162.216
    expect(v(r.out.breakEvenPrice)).toBeCloseTo(732 / 760, 6);
    expect(v(r.out.breakEvenCatch)).toBeCloseTo(732 / 4.5125, 6);
    expect(r.waiting).toEqual([]);
  });

  it("share after costs, the boat pays fuel, bait and ice", () => {
    const r = evaluateForm({ ...A, fuelPayer: "boat" }, crab);
    // N = 3,800 − 190 = 3,610; crew = 1,083; net = 3,610 − 1,083 − 732 = 1,795
    expect(v(r.out.crewPay)).toBeCloseTo(1083, 6);
    expect(v(r.out.net)).toBeCloseTo(1795, 6);
    // 0.7 × 760 p − 732 = 0 → p = 732 / 532
    expect(v(r.out.breakEvenPrice)).toBeCloseTo(732 / 532, 6);
  });

  it("share of gross", () => {
    const r = evaluateForm({ ...A, crew: "3", crewKind: "ofGross", share: "25", fuelPayer: null }, crab);
    // crew = 0.25 × 3,800 = 950; net = 3,800 × 0.70 − 732 = 1,928; per crew 316.67
    expect(v(r.out.crewPay)).toBeCloseTo(950, 6);
    expect(v(r.out.net)).toBeCloseTo(1928, 6);
    expect(v(r.out.perCrew)).toBeCloseTo(950 / 3, 6);
    // who pays fuel does not move a share of gross, so it is not waited for
    expect(r.waiting).toEqual([]);
  });

  it("day rate uses the days at sea", () => {
    const r = evaluateForm({ ...A, crewKind: "dayRate", share: "", dayRate: "200", days: "2", fuelPayer: null }, crab);
    // crew = 2 × 200 × 2 = 800; net = 3,610 − 732 − 800 = 2,078; per crew = 400
    expect(v(r.out.crewPay)).toBeCloseTo(800, 6);
    expect(v(r.out.net)).toBeCloseTo(2078, 6);
    expect(v(r.out.perCrew)).toBeCloseTo(400, 6);
  });

  it("market squid in short tons, no hired crew", () => {
    const r = evaluateForm({ ...EMPTY_FORM, catch: "30", price: "1000", fees: "0", fuelGal: "400", fuelPrice: "5", supplies: "500", crew: "0" }, squid);
    // 30 ton = 60,000 lb at $0.50/lb: R = 30,000; costs 2,000 + 500; net = 27,500
    expect(v(r.out.gross)).toBeCloseTo(30000, 6);
    expect(v(r.out.crewPay)).toBe(0);
    expect(r.out.perCrew.status).toBe("na");
    expect(v(r.out.net)).toBeCloseTo(27500, 6);
    // break-evens come back in the worksheet's unit: $83.33/ton, 2.5 ton
    expect(v(r.out.breakEvenPrice)).toBeCloseTo(2500 / 30, 6);
    expect(v(r.out.breakEvenCatch)).toBeCloseTo(2.5, 6);
  });

  it("0 gallons needs no fuel price", () => {
    const r = evaluateForm({ ...A, fuelGal: "0", fuelPrice: "" }, crab);
    expect(v(r.out.fuel)).toBe(0);
    // N = 3,610 − 180 = 3,430; net = 0.7 × 3,430 = 2,401
    expect(v(r.out.net)).toBeCloseTo(2401, 6);
  });

  it("costs over gross: crew gets nothing and net is negative", () => {
    const r = evaluateForm({ ...A, catch: "100" }, crab);
    // R = 475; N = 475 − 23.75 − 732 = −280.75 → crew 0, net −280.75
    expect(v(r.out.crewPay)).toBe(0);
    expect(v(r.out.net)).toBeCloseTo(-280.75, 6);
  });
});

describe("nothing is assumed", () => {
  it("an empty worksheet waits for every field, and says which", () => {
    const r = evaluateForm(EMPTY_FORM, crab);
    expect(r.out.gross).toEqual({ status: "waiting", fields: ["catch", "price"] });
    expect(r.out.net.status).toBe("waiting");
    expect(r.waiting).toEqual(["catch", "price", "fuelGal", "fuelPrice", "supplies", "fees", "crew"]);
    expect(r.grid).toBeNull();
    expect(r.steps.every((s) => s.amount === null && s.waiting?.startsWith("waiting for"))).toBe(true);
  });

  it("a blank fee or bait line is a gap, never 0", () => {
    const r = evaluateForm({ ...A, fees: "" }, crab);
    expect(r.out.fees).toEqual({ status: "waiting", fields: ["fees"] });
    expect(r.out.net).toEqual({ status: "waiting", fields: ["fees"] });
    expect(r.out.gross.status).toBe("value");
    const s = evaluateForm({ ...A, supplies: "" }, crab);
    expect(s.out.net).toEqual({ status: "waiting", fields: ["supplies"] });
    expect(waitingWords(s.out.net, crab)).toBe("waiting for bait and ice");
  });

  it("who pays fuel is a required choice for a share after costs, with no default", () => {
    const r = evaluateForm({ ...A, fuelPayer: null }, crab);
    expect(r.out.crewPay).toEqual({ status: "waiting", fields: ["fuelPayer"] });
    expect(r.out.net).toEqual({ status: "waiting", fields: ["fuelPayer"] });
    expect(waitingWords(r.out.net, crab)).toBe("waiting for who pays fuel");
    // gross, fees and fuel are worked out meanwhile
    expect(v(r.out.tripCost)).toBeCloseTo(732, 6);
  });

  it("how crew is paid is asked only when someone else is paid", () => {
    expect(evaluateForm({ ...A, crewKind: null }, crab).out.net).toEqual({ status: "waiting", fields: ["crewKind"] });
    expect(evaluateForm({ ...A, crew: "", crewKind: null }, crab).out.net).toEqual({ status: "waiting", fields: ["crew"] });
    expect(evaluateForm({ ...A, crew: "0", crewKind: null, share: "", fuelPayer: null }, crab).out.net.status).toBe("value");
  });

  it("a word in a number field is an error and a gap", () => {
    const r = evaluateForm({ ...A, price: "four" }, crab);
    expect(r.errors.price).toBe("Enter a number");
    expect(r.out.gross).toEqual({ status: "waiting", fields: ["price"] });
  });

  it("names many gaps briefly", () => {
    expect(waitingWords({ status: "waiting", fields: ["catch", "price", "fees", "crew"] }, crab)).toBe("waiting for catch, price and 2 more");
  });
});

describe("the sensitivity grid previews the waterfall", () => {
  it("a cell's net is the engine's grid, and the previewed scenario agrees with it", () => {
    const r = evaluateForm(A, crab);
    // catch +30 %, price −20 %: R = 1,040 × 3.80 = 3,952; N = 3,952 − 197.6 − 732 = 3,022.4; net = 0.7 N = 2,115.68
    expect(gridValue(r.grid, 1.3, 0.8)).toBeCloseTo(2115.68, 6);
    for (const row of GRID_ROWS)
      for (const col of GRID_COLS) {
        const preview = evaluateForm(A, crab, { catch: row.factor, price: col.factor });
        expect(v(preview.out.net)).toBeCloseTo(gridValue(r.grid, row.factor, col.factor)!, 6);
      }
    expect(gridValue(r.grid, 1, 1)).toBeCloseTo(2014.6, 6);
  });

  it("keeps one money scale across every scenario", () => {
    const [lo, hi] = scenarioDomain(A, crab);
    expect(lo).toBe(0);
    // the largest gross on the grid: 1,040 lb × $5.70 = 5,928
    expect(hi).toBeCloseTo(5928, 6);
  });
});

describe("references enter only on a tap, labelled", () => {
  it("offers CDFW's past average in the fishery's unit, and none for a species without one", () => {
    const p = priceReference(crab)!;
    expect(p.value).toBe("3.51");
    expect(p.words).toMatch(/2024 statewide average/);
    expect(p.taken).toMatch(/not today's dock price/);
    expect(priceReference(squid)!.words).toMatch(/\/ton/);
    expect(priceReference(coastFishery("salmon_commercial"))).toBeNull();
  });

  it("marks the diesel week old after 14 days, and offers NOAA's share only where the survey has one", () => {
    expect(fuelReference(new Date("2026-11-10T12:00:00Z")).stale).toBe(true);
    expect(fuelReference(new Date("2026-10-25T12:00:00Z")).stale).toBe(false);
    expect(shareReference(crab)!.value).toBe("28.1");
    expect(shareReference(squid)).toBeNull();
  });

  it("chooses no fishery for the reader unless they follow one", () => {
    expect(defaultFishery([])).toBeNull();
    expect(defaultFishery(["bivalve_shellfish", "market_squid"])).toBe("market_squid");
  });
});

describe("saved on this device", () => {
  it("round-trips through the trip store's draft, squid tons and references included", () => {
    const form: CoastForm = { ...A, fuelPrice: "4.6", taken: { price: true, share: true } };
    const back = fromDraft(sanitizeDraft(JSON.parse(JSON.stringify(toDraft(form, crab)))), "dungeness_commercial");
    expect(back).toEqual(form);
    const sq: CoastForm = { ...EMPTY_FORM, catch: "30", price: "1000", crew: "0" };
    const sqBack = fromDraft(sanitizeDraft(toDraft(sq, squid)), "market_squid");
    expect(sqBack.catch).toBe("30");
    expect(sqBack.price).toBe("1000");
    expect(sqBack.crew).toBe("0");
    expect(sqBack.fuelPayer).toBeNull();
  });

  it("names a trip by its own numbers", () => {
    expect(tripName(A, crab)).toBe("Dungeness crab, 800 lb at $4.75/lb");
    expect(tripName(EMPTY_FORM, crab)).toBe("Dungeness crab");
  });
});

describe("assumptions in plain words", () => {
  it("states what net leaves out and how fuel is paid, without repeating a figure", () => {
    const a = assumptions(A, crab).join(" ");
    expect(a).toMatch(/not profit or take-home pay/);
    expect(a).toMatch(/come off the top/);
    expect(a).not.toMatch(/\d/);
    expect(assumptions({ ...A, fuelPayer: null }, crab).join(" ")).toMatch(/nothing is worked out until you choose/);
  });
});
