import { describe, expect, it } from "vitest";
import { sanitizeDraft } from "@/lib/tripEconomicsStore";
import { tripFishery } from "@/nx/coast/fisheries";
import {
  addCatch,
  addCost,
  assumptions,
  breakEvenCatchText,
  breakEvenFuelText,
  breakEvenPriceText,
  emptyWorkspace,
  evaluateWorkspace,
  fromDraft,
  gridValue,
  hasEntry,
  money,
  moneyExtent,
  NET_IS_NOT,
  roundUp,
  setCatch,
  setCost,
  setField,
  setShareBasis,
  sweep,
  switchFishery,
  takeReference,
  toDraft,
  toScenario,
  waitingWords,
  type Result,
  type Workspace,
} from "@/nx/coast/workspace";

/**
 * The v13 trip workspace model over evaluateTrip. Every expected figure is worked by hand from the formulas in
 * src/lib/tripEconomics.ts (R gross, τ fees % of gross, u fees per lb, Q lb, D shared costs, B boat-paid costs,
 * s the crew's total share):
 *   share after costs: N = R(1−τ) − uQ − D; crew = s·N (0 if N ≤ 0); net = N − crew − B
 *   share of gross:    crew = s·R;          net = R(1−s−τ) − uQ − D − B
 *   day rate:          crew = people × $/day × days; net = R(1−τ) − uQ − D − B − crew
 * The base trip is the owner's hands-on test (v13 brief): 400 lb at $9, 30 gal at $8.24, bait and ice $100.
 */
const crab = tripFishery("dungeness_commercial")!;
const squid = tripFishery("market_squid")!;
const sport = tripFishery("dungeness_recreational")!;

const base = (o: Partial<Workspace> = {}): Workspace => ({
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

const v = (r: Result) => {
  if (r.status !== "value") throw new Error(`not a value: ${JSON.stringify(r)}`);
  return r.value;
};
const figures = (ws: Workspace, f = crab) => {
  const { out } = evaluateWorkspace(ws, f);
  return { gross: v(out.gross), fees: v(out.fees), fuel: v(out.fuel), operating: v(out.operating), crew: v(out.crewPay), net: v(out.net), beCatch: v(out.breakEvenCatch), bePrice: v(out.breakEvenPrice) };
};

describe("the owner's test, to the cent", () => {
  const r = evaluateWorkspace(base(), crab);
  it("gross 3,600.00, fuel 247.20, operating 347.20, crew 650.56, net to boat 2,602.24", () => {
    expect(money(v(r.out.gross))).toBe("$3,600.00");
    expect(money(v(r.out.fuel))).toBe("$247.20");
    expect(money(v(r.out.operating))).toBe("$347.20");
    expect(money(v(r.out.crewPay))).toBe("$650.56");
    expect(money(v(r.out.net))).toBe("$2,602.24");
    expect(money(v(r.out.perCrew))).toBe("$650.56");
  });
  it("break-even catch 39 lb and price $0.87/lb, both rounded up", () => {
    expect(v(r.out.breakEvenCatch)).toBeCloseTo(347.2 / 9, 9); // 38.58
    expect(breakEvenCatchText(v(r.out.breakEvenCatch), "lb")).toBe("39 lb");
    expect(v(r.out.breakEvenPrice)).toBeCloseTo(0.868, 9);
    expect(breakEvenPriceText(v(r.out.breakEvenPrice), "lb")).toBe("$0.87/lb");
  });
  it("carries the engine's worked sum for each figure and the waterfall's steps", () => {
    const net = r.out.net;
    expect(net.status === "value" && net.formula).toContain("Net to boat $2,602");
    expect(r.steps.map((s) => [s.id, s.amount === null ? null : Math.round(s.amount * 100) / 100])).toEqual([
      ["gross", 3600],
      ["fees", 0],
      ["fuel", 247.2],
      ["cost-0", 100],
      ["crew", 650.56],
      ["net", 2602.24],
    ]);
    expect(r.waiting).toEqual([]);
    expect(r.warnings).toEqual([]);
  });
});

describe("every pay model, without and with fees (audit B, section 2)", () => {
  const fees = { feesPct: "5", feesPerUnit: "0.10" };
  const boat: Partial<Workspace> = { fuelPayer: "boat", costs: [{ label: "Bait and ice", usd: "100", payer: "boat" }] };
  const rows: Array<[string, Partial<Workspace>, { fees: number; crew: number; net: number; beCatch: number; bePrice: number }]> = [
    ["A share after costs, shared off the top", {}, { fees: 0, crew: 650.56, net: 2602.24, beCatch: 347.2 / 9, bePrice: 0.868 }],
    ["A share after costs, the boat pays fuel and bait", boat, { fees: 0, crew: 720, net: 2532.8, beCatch: 347.2 / 7.2, bePrice: 1.085 }],
    ["B share of gross", { crewKind: "ofGross" }, { fees: 0, crew: 720, net: 2532.8, beCatch: 347.2 / 7.2, bePrice: 1.085 }],
    ["C day rate, 1 × $200 × 1 day", { crewKind: "dayRate", dayRate: "200", days: "1" }, { fees: 0, crew: 200, net: 3052.8, beCatch: 547.2 / 9, bePrice: 1.368 }],
    ["C day rate, 2 × $200 × 2 days", { crew: "2", crewKind: "dayRate", dayRate: "200", days: "2" }, { fees: 0, crew: 800, net: 2452.8, beCatch: 1147.2 / 9, bePrice: 2.868 }],
    ["nobody else paid", { crew: "0" }, { fees: 0, crew: 0, net: 3252.8, beCatch: 347.2 / 9, bePrice: 0.868 }],
    // fees 5 % of gross (180) + $0.10/lb (40) = 220
    ["A with fees", fees, { fees: 220, crew: 606.56, net: 2426.24, beCatch: 347.2 / 8.45, bePrice: 387.2 / 380 }],
    ["A, the boat pays, with fees", { ...fees, ...boat }, { fees: 220, crew: 676, net: 2356.8, beCatch: 347.2 / (0.8 * 8.45), bePrice: (347.2 / 0.8 + 40) / 380 }],
    ["B with fees", { ...fees, crewKind: "ofGross" }, { fees: 220, crew: 720, net: 2312.8, beCatch: 347.2 / 6.65, bePrice: 387.2 / 300 }],
    ["C with fees", { ...fees, crewKind: "dayRate", dayRate: "200", days: "1" }, { fees: 220, crew: 200, net: 2832.8, beCatch: 547.2 / 8.45, bePrice: 587.2 / 380 }],
    ["nobody else paid, with fees", { ...fees, crew: "0" }, { fees: 220, crew: 0, net: 3032.8, beCatch: 347.2 / 8.45, bePrice: 387.2 / 380 }],
    ["A with a 5 % fee only", { feesPct: "5" }, { fees: 180, crew: 614.56, net: 2458.24, beCatch: 347.2 / 8.55, bePrice: 347.2 / 380 }],
  ];
  it.each(rows)("%s", (_name, o, want) => {
    const got = figures(base(o));
    expect(got.gross).toBeCloseTo(3600, 6);
    expect(got.fuel).toBeCloseTo(247.2, 6);
    expect(got.operating).toBeCloseTo(347.2, 6);
    expect(got.fees).toBeCloseTo(want.fees, 6);
    expect(got.crew).toBeCloseTo(want.crew, 6);
    expect(got.net).toBeCloseTo(want.net, 6);
    expect(got.beCatch).toBeCloseTo(want.beCatch, 6);
    expect(got.bePrice).toBeCloseTo(want.bePrice, 6);
  });

  it("the day rate break-even is 128 lb, not 127: at 127 lb the boat is still under water", () => {
    const ws = base({ crew: "2", crewKind: "dayRate", dayRate: "200", days: "2" });
    const r = evaluateWorkspace(ws, crab);
    expect(v(r.out.breakEvenCatch)).toBeCloseTo(127.4667, 3);
    expect(breakEvenCatchText(v(r.out.breakEvenCatch), "lb")).toBe("128 lb");
    expect(breakEvenPriceText(v(r.out.breakEvenPrice), "lb")).toBe("$2.87/lb");
    expect(v(evaluateWorkspace(setCatch(ws, 0, { amount: "127" }), crab).out.net)).toBeLessThan(0);
    expect(v(evaluateWorkspace(setCatch(ws, 0, { amount: "128" }), crab).out.net)).toBeGreaterThan(0);
  });

  it("who pays is chosen per line: fuel off the top, bait from the boat's share", () => {
    // D = 247.20, B = 100: N = 3352.80; crew 670.56; net 3352.80 − 670.56 − 100 = 2582.24
    const got = figures(base({ costs: [{ label: "Bait and ice", usd: "100", payer: "boat" }] }));
    expect(got.crew).toBeCloseTo(670.56, 6);
    expect(got.net).toBeCloseTo(2582.24, 6);
  });

  it("several catch lines and cost lines add up; break-evens are for the first line", () => {
    let ws = addCatch(base());
    ws = setCatch(ws, 1, { label: "Rock crab", amount: "100", price: "2" });
    ws = setCost(addCost(ws), 1, { label: "Food", usd: "50", payer: "shared" });
    const r = evaluateWorkspace(ws, crab);
    // R = 3800, D = 397.20, N = 3402.80, crew 680.56, net 2722.24
    expect(v(r.out.gross)).toBeCloseTo(3800, 6);
    expect(v(r.out.operating)).toBeCloseTo(397.2, 6);
    expect(v(r.out.net)).toBeCloseTo(2722.24, 6);
    // first line alone must cover 397.20 − 200 of other gross: 197.2 / 9 = 21.9 lb
    expect(breakEvenCatchText(v(r.out.breakEvenCatch), "lb")).toBe("22 lb");
    expect(r.steps.map((s) => s.id)).toEqual(["gross", "fees", "fuel", "cost-0", "cost-1", "crew", "net"]);
    expect(assumptions(ws, crab).join(" ")).toContain("first catch line");
  });

  it("fuel worked out from distance, speed and burn gives the same trip as its gallons", () => {
    // 20 nm ÷ 8 kt × 6 gal/h = 15 gal, + 5 h × 3 gal/h = 15 gal → 30 gal
    const ws = base({ fuelRoute: "distance", fuelGal: "", roundTripNm: "20", cruiseKt: "8", steamGph: "6", fishingHours: "5", fishingGph: "3" });
    const r = evaluateWorkspace(ws, crab);
    expect(v(r.out.gallons)).toBeCloseTo(30, 9);
    expect(v(r.out.net)).toBeCloseTo(2602.24, 6);
    // a blank leg is a gap, never nothing; a leg set to 0 needs nothing more
    const blank = evaluateWorkspace({ ...ws, fishingHours: "" }, crab);
    expect(waitingWords(blank.out.net, ws, crab)).toBe("waiting for hours fishing");
    expect(v(evaluateWorkspace({ ...ws, fishingHours: "0", fishingGph: "" }, crab).out.gallons)).toBeCloseTo(15, 9);
  });

  it("market squid in short tons, fee per ton", () => {
    // 20 ton at $1,000, fee $10/ton = 200; fuel 100 gal at $5 = 500; ice 300; no crew: 20,000 − 200 − 800 = 19,000
    const ws: Workspace = { ...emptyWorkspace(squid), catch: [{ label: "Market squid", ref: "squid", amount: "20", price: "1000" }], fuelGal: "100", fuelPrice: "5", costs: [{ label: "Ice", usd: "300", payer: null }], feesPct: "0", feesPerUnit: "10", crew: "0" };
    const r = evaluateWorkspace(ws, squid);
    expect(v(r.out.fees)).toBeCloseTo(200, 6);
    expect(v(r.out.net)).toBeCloseTo(19000, 6);
    // 800 / (1000 − 10) = 0.808 ton → 0.81; (800 + 200) / 20 = $50/ton
    expect(breakEvenCatchText(v(r.out.breakEvenCatch), "ton")).toBe("0.81 ton");
    expect(breakEvenPriceText(v(r.out.breakEvenPrice), "ton")).toBe("$50/ton");
  });
});

describe("the crew share, total or each", () => {
  it("10 % each for 2 crew is the same trip as 20 % in total", () => {
    const total = evaluateWorkspace(base({ crew: "2" }), crab);
    const each = evaluateWorkspace(base({ crew: "2", share: "10", shareBasis: "each" }), crab);
    expect(v(each.out.crewPay)).toBeCloseTo(v(total.out.crewPay), 9);
    expect(v(each.out.net)).toBeCloseTo(2602.24, 6);
    expect(v(each.out.perCrew)).toBeCloseTo(325.28, 6);
    expect(total.share).toMatchObject({ basis: "total", crew: 2, totalPct: 20, eachPct: 10, of: "after", words: "20 % in total, split 2 ways: 10 % each." });
    expect(total.share!.eachUsd).toBeCloseTo(325.28, 6);
    expect(each.share!.words).toBe("10 % each for 2 crew: 20 % in total.");
  });
  it("20 % each for 2 crew is 40 % of what is left, not 20 %", () => {
    const r = evaluateWorkspace(base({ crew: "2", shareBasis: "each" }), crab);
    expect(v(r.out.crewPay)).toBeCloseTo(0.4 * 3252.8, 6);
    expect(v(r.out.perCrew)).toBeCloseTo(650.56, 6);
  });
  it("switching the basis keeps the money where it was", () => {
    const ws = setShareBasis(base({ crew: "2" }), crab, "each");
    expect(ws.share).toBe("10");
    expect(v(evaluateWorkspace(ws, crab).out.net)).toBeCloseTo(2602.24, 6);
    expect(setShareBasis(ws, crab, "total").share).toBe("20");
  });
  it("each × crew over 100 % is an error, and crew comes whole", () => {
    const over = evaluateWorkspace(base({ crew: "2", share: "60", shareBasis: "each" }), crab);
    expect(over.errors.share).toBe("60 % each is more than 100 % between 2 crew");
    expect(over.out.net.status).toBe("waiting");
    const half = evaluateWorkspace(base({ crew: "1.5" }), crab);
    expect(half.errors.crew).toBe("Whole people only");
    expect(waitingWords(half.out.net, base(), crab)).toBe("waiting for crew");
    expect(evaluateWorkspace(base({ crew: "1" }), crab).share!.words).toBe("20 % to your 1 crew.");
  });
});

describe("nothing is assumed", () => {
  it("an empty workspace waits for every field and has no entry", () => {
    const ws = emptyWorkspace(crab);
    const r = evaluateWorkspace(ws, crab);
    expect(hasEntry(ws)).toBe(false);
    expect(r.out.net.status).toBe("waiting");
    expect(waitingWords(r.out.gross, ws, crab)).toBe("waiting for catch and price");
    expect(r.grid).toBeNull();
    expect(r.waiting).toEqual(["catch.0.amount", "catch.0.price", "fuelGal", "fuelPrice", "cost.0.usd", "feesPct", "crew"]);
  });
  it("a payer is asked for only where it moves the money", () => {
    const open = base({ fuelPayer: null, costs: [{ label: "Bait and ice", usd: "100", payer: null }] });
    expect(waitingWords(evaluateWorkspace(open, crab).out.net, open, crab)).toBe("waiting for who pays fuel and who pays bait and ice");
    expect(assumptions(open, crab).join(" ")).toContain("Who pays fuel and bait and ice changes the crew's share");
    // share of gross, a day rate or no crew: the boat's net is the same whoever is said to pay
    expect(v(evaluateWorkspace({ ...open, crewKind: "ofGross" }, crab).out.net)).toBeCloseTo(2532.8, 6);
    expect(v(evaluateWorkspace({ ...open, crew: "0" }, crab).out.net)).toBeCloseTo(3252.8, 6);
    // no fuel burned and a $0 line need no payer
    const none = { ...open, fuelGal: "0", fuelPrice: "", costs: [{ label: "Bait and ice", usd: "0", payer: null }] };
    expect(v(evaluateWorkspace(none, crab).out.net)).toBeCloseTo(0.8 * 3600, 6);
  });
  it("an added line left blank is a gap: fee per pound, a second cost, a second species", () => {
    const fee = base({ feesPerUnit: "" });
    expect(waitingWords(evaluateWorkspace(fee, crab).out.net, fee, crab)).toBe("waiting for fees per pound");
    const cost = setCost(addCost(base()), 1, { label: "Food" });
    expect(waitingWords(evaluateWorkspace(cost, crab).out.net, cost, crab)).toBe("waiting for food and who pays food");
    const two = setCatch(addCatch(base()), 1, { label: "Rock crab" });
    expect(waitingWords(evaluateWorkspace(two, crab).out.gross, two, crab)).toBe("waiting for catch of rock crab and price of rock crab");
  });
  it("typing never labels a value as a reference's; tapping the reference does, and typing over it drops the label", () => {
    const typed = setField(base({ fuelPrice: "" }), "fuelPrice", "8.24");
    expect(typed.taken).toEqual({});
    const tapped = takeReference(typed, "fuelPrice", "8.24");
    expect(tapped.taken).toEqual({ fuelPrice: true });
    expect(setField(tapped, "fuelPrice", "8.30").taken).toEqual({});
    expect(setCatch(takeReference(base(), "price", "3.51"), 0, { price: "4" }).taken).toEqual({});
  });
});

describe("zero, impossible and inconsistent inputs", () => {
  it("a price of $0 says so, instead of blaming fees and crew pay", () => {
    const r = evaluateWorkspace(base({ catch: [{ label: "Dungeness crab", ref: null, amount: "400", price: "0" }] }), crab);
    expect(r.out.breakEvenCatch).toEqual({ status: "none", reason: "No catch covers the costs at a price of $0." });
    expect(v(r.out.net)).toBeCloseTo(-347.2, 6);
  });
  it("0 lb landed: no price covers the costs; the catch needed is still 39 lb", () => {
    const r = evaluateWorkspace(base({ catch: [{ label: "Dungeness crab", ref: null, amount: "0", price: "9" }] }), crab);
    expect(r.out.breakEvenPrice.status).toBe("none");
    expect(breakEvenCatchText(v(r.out.breakEvenCatch), "lb")).toBe("39 lb");
  });
  it("the engine's warnings reach the page", () => {
    const r = evaluateWorkspace(base({ crewKind: "ofGross", share: "100" }), crab);
    expect(r.warnings).toEqual(["Fees and crew share together take all of the gross: no catch covers the costs."]);
    expect(v(r.out.net)).toBeCloseTo(-347.2, 6);
  });
  it("costs over gross: the crew's share after costs is $0 and net is negative", () => {
    const r = evaluateWorkspace(base({ catch: [{ label: "Dungeness crab", ref: null, amount: "20", price: "9" }] }), crab);
    expect(v(r.out.crewPay)).toBe(0);
    expect(money(v(r.out.net))).toBe("−$167.20");
  });
  it("negatives and words are errors and gaps; over-large values are clamped and said", () => {
    const neg = evaluateWorkspace(base({ fuelGal: "-5" }), crab);
    expect(neg.errors.fuelGal).toBe("Can't be negative");
    expect(neg.out.fuel.status).toBe("waiting");
    const word = evaluateWorkspace(setCatch(base(), 0, { price: "abc" }), crab);
    expect(word.errors["catch.0.price"]).toBe("Enter a number");
    expect(evaluateWorkspace(base({ share: "150", crewKind: "ofGross" }), crab).clamped).toEqual(["share"]);
    expect(evaluateWorkspace(base({ fuelRoute: "distance", roundTripNm: "20", cruiseKt: "0", steamGph: "6", fishingHours: "0" }), crab).errors.cruiseKt).toBe("Above 0");
  });
  it("rounds up through float dust, never past the next step", () => {
    expect(roundUp(38.999999999999, 1)).toBe(39);
    expect(roundUp(39, 1)).toBe(39);
    expect(roundUp(39.0001, 1)).toBe(40);
    expect(breakEvenPriceText(0.87, "lb")).toBe("$0.87/lb");
    expect(breakEvenPriceText(0.8701, "lb")).toBe("$0.88/lb");
    // the most fuel can cost is rounded down: a cent more would lose money
    expect(breakEvenFuelText(12.349)).toBe("$12.34/gal");
    expect(breakEvenFuelText(116.67)).toBe("above $50.00/gal");
    expect(money(-0.001)).toBe("$0.00");
  });
});

describe("a recreational trip is cost only", () => {
  const ws: Workspace = { ...emptyWorkspace(sport), fuelGal: "30", fuelPrice: "8.24", costs: [{ label: "Bait and ice", usd: "100", payer: null }], anglers: "2" };
  const r = evaluateWorkspace(ws, sport);
  it("has a trip cost and a split, and no price, gross, crew pay or net anywhere", () => {
    expect(r.audience).toBe("recreational");
    expect(money(v(r.out.operating))).toBe("$347.20");
    expect(money(v(r.out.perPerson))).toBe("$173.60");
    for (const k of ["gross", "fees", "crewPay", "perCrew", "net", "breakEvenCatch", "breakEvenPrice", "breakEvenFuel"] as const) expect(r.out[k]).toEqual({ status: "na" });
    expect(r.grid).toBeNull();
    expect(r.share).toBeNull();
    expect(r.steps.map((s) => s.label)).toEqual(["Fuel", "Bait and ice", "Trip cost"]);
    expect(r.fuelSwing).toMatchObject({ measure: "tripCost" });
    expect(r.fuelSwing!.up).toBeCloseTo(247.2 * 1.2 + 100, 6);
  });
  it("hands the engine no catch, price, fee or crew, even if some were typed before", () => {
    const dirty: Workspace = { ...ws, catch: [{ label: "Dungeness crab", ref: null, amount: "400", price: "9" }], feesPct: "5", crew: "1", crewKind: "ofGross", share: "20" };
    const { scenario, audience } = toScenario(dirty, sport);
    expect(audience).toBe("recreational");
    expect(scenario.catch).toEqual([]);
    expect(scenario.feesPctOfGross).toBeNull();
    expect(scenario.crew).toEqual({ kind: "none" });
    expect(evaluateWorkspace(dirty, sport).out.gross).toEqual({ status: "na" });
    expect(toDraft(dirty, sport).inputs.catch).toEqual([]);
  });
  it("waits for the people sharing only for the split, and says what it leaves out", () => {
    const alone = evaluateWorkspace({ ...ws, anglers: "" }, sport);
    expect(v(alone.out.operating)).toBeCloseTo(347.2, 6);
    expect(waitingWords(alone.out.perPerson, ws, sport)).toBe("waiting for people sharing the cost");
    expect(assumptions(ws, sport)[0]).toContain("not sold");
    expect(assumptions(ws, sport).join(" ")).not.toContain("Net to boat");
    expect(sweep(ws, sport, "catch")).toBeNull();
  });
});

describe("sensitivity", () => {
  it("the grid is the engine's, and a previewed cell agrees with it", () => {
    const r = evaluateWorkspace(base(), crab);
    // catch +30 %, price −20 %: R = 3744, N = 3396.80, net 2717.44
    expect(gridValue(r.grid, 1.3, 0.8)).toBeCloseTo(2717.44, 6);
    expect(gridValue(r.grid, 0.7, 1.2)).toBeCloseTo(2141.44, 6);
    expect(gridValue(r.grid, 1, 1)).toBeCloseTo(2602.24, 6);
    expect(v(evaluateWorkspace(base(), crab, { catch: 1.3, price: 0.8 }).out.net)).toBeCloseTo(2717.44, 6);
    const [lo, hi] = moneyExtent(base(), crab);
    expect(lo).toBe(0);
    expect(hi).toBeCloseTo(3600 * 1.3 * 1.2, 6);
  });
  it("one thing changes: net against catch, price and fuel price, each point the engine's", () => {
    const c = sweep(base(), crab, "catch")!;
    expect(c.unit).toBe("lb");
    expect(c.entered).toBe(400);
    expect(c.points).toHaveLength(25);
    expect(c.points[0].x).toBe(0);
    expect(c.points[0].net).toBeCloseTo(-347.2, 6);
    expect(c.points[12].x).toBe(400);
    expect(c.points[12].net).toBeCloseTo(2602.24, 6);
    expect(c.points[24].net).toBeCloseTo(0.8 * (7200 - 347.2), 6);
    expect(c.breakEven).toBeCloseTo(38.5778, 3);
    const p = sweep(base(), crab, "price")!;
    expect(p.unit).toBe("$/lb");
    expect(p.breakEven).toBeCloseTo(0.868, 6);
    const f = sweep(base(), crab, "fuelPrice")!;
    // fuel at $16.48: N = 3600 − 100 − 494.40 = 3005.60 → net 2404.48; break-even ($116.67) is off the range
    expect(f.points[24].net).toBeCloseTo(2404.48, 6);
    expect(f.breakEven).toBeNull();
    expect(sweep(emptyWorkspace(crab), crab, "catch")).toBeNull();
  });
});

describe("the store's draft", () => {
  const round = (ws: Workspace, f = crab) => fromDraft(sanitizeDraft(JSON.parse(JSON.stringify(toDraft(ws, f)))), f.id);
  it("round-trips what the reader typed", () => {
    const cases: Array<[Workspace, typeof crab]> = [
      [base(), crab],
      [base({ crew: "2", share: "10", shareBasis: "each", feesPerUnit: "0.1", taken: { fuelPrice: true } }), crab],
      [base({ fuelRoute: "distance", fuelGal: "", roundTripNm: "20", cruiseKt: "8", steamGph: "6", fishingHours: "5", fishingGph: "3" }), crab],
      [base({ fuelPayer: null, costs: [{ label: "Bait and ice", usd: "100", payer: null }, { label: "Food", usd: "50", payer: "boat" }] }), crab],
      [setCatch(addCatch(base()), 1, { label: "Rock crab", amount: "100", price: "2" }), crab],
      [base({ crew: "0", crewKind: null, share: "" }), crab],
      [{ ...emptyWorkspace(squid), catch: [{ label: "Market squid", ref: "squid", amount: "20", price: "1000" }], feesPerUnit: "10" }, squid],
      [{ ...emptyWorkspace(sport), fuelGal: "30", fuelPrice: "8.24", anglers: "2" }, sport],
      [emptyWorkspace(crab), crab],
    ];
    for (const [ws, f] of cases) expect(round(ws, f)).toEqual(ws);
  });
  it("stores the share as the crew's total, with how it was typed beside it", () => {
    const d = toDraft(base({ crew: "2", share: "10", shareBasis: "each" }), crab);
    expect(d.inputs.crew).toMatchObject({ kind: "afterExpenses", pct: 20, crew: 2 });
    expect(d.view).toEqual({ shareBasis: "each" });
    // a per-person share typed before the crew count stays as typed
    expect(round(base({ crew: "", share: "10", shareBasis: "each" })).share).toBe("10");
  });
});

describe("another fishery", () => {
  it("keeps the reader's numbers, drops what no longer means the same, and says what it dropped", () => {
    const from = takeReference(base(), "price", "3.51");
    const same = switchFishery(from, crab, tripFishery("rock_crab"));
    expect(same.cleared).toEqual(["price"]);
    expect(same.workspace.catch[0]).toMatchObject({ label: "Rock crab", amount: "400", price: "", ref: "rock-crab" });
    expect(same.workspace.fuelGal).toBe("30");
    const tons = switchFishery(base({ feesPerUnit: "0.1" }), crab, squid);
    expect(tons.cleared).toEqual(["price", "catch", "fees per pound"]);
    expect(tons.workspace.costs[0].label).toBe("Ice");
    expect(tons.workspace.share).toBe("20");
    expect(switchFishery(base(), crab, crab).cleared).toEqual([]);
  });
  it("says what net to boat is not, wherever it is shown", () => {
    expect(NET_IS_NOT).toContain("not profit or take-home pay");
    expect(assumptions(base(), crab)[0]).toBe(NET_IS_NOT);
    expect(assumptions(base(), crab).join(" ")).toContain("Fuel and bait and ice come off the top");
  });
});
