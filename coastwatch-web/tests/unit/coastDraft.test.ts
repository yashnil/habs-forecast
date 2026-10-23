import { describe, expect, it } from "vitest";
import { applyReference, presetById, presetReferences } from "@/content/fisheryPresets";
import { evaluateTrip } from "@/lib/tripEvaluation";
import { draftInputs } from "@/lib/tripEconomicsStore";
import { initialDraft, switchFishery } from "@/v4/coast/economics/draft";
import { econAudience, hasInput, needLabel } from "@/v4/coast/economics/model";

/** v10 worksheet drafts: nothing numeric is filled in (D1), and how crew is paid starts unchosen (D2). */
describe("a fresh worksheet", () => {
  it("fills no burn rate, share, catch or price, and leaves the crew model unchosen", () => {
    for (const sp of ["salmon", "dungeness", "groundfish", "squid", "rock-crab", "halibut"]) {
      const d = initialDraft(sp);
      expect(d.inputs.steamGph ?? null, sp).toBeNull();
      expect(d.inputs.fishingGph ?? null, sp).toBeNull();
      expect(d.inputs.feesPctOfGross ?? null, sp).toBeNull();
      expect(d.inputs.crew?.kind ?? null, sp).toBeNull();
      expect(d.inputs.crew?.pct ?? null, sp).toBeNull();
      expect(d.basis, sp).toEqual({});
      for (const c of d.inputs.catch ?? []) expect([c.lb, c.usdPerLb], sp).toEqual([null, null]);
      expect(hasInput(d), sp).toBe(false);
    }
    expect(initialDraft("salmon").inputs.catch![0].species).toBe("Salmon");
  });
  it("offers the fishery's references beside the fields, copied in only by applyReference", () => {
    const d = initialDraft("dungeness");
    const refs = presetReferences(presetById(d.preset));
    expect(refs.map((r) => r.field)).toContain("steamGph");
    const used = applyReference(d.inputs, refs.find((r) => r.field === "steamGph")!);
    expect(used.steamGph).toBe(7.8);
    expect(d.inputs.steamGph ?? null).toBeNull();
  });
  it("switching fishery clears the previous fishery's tapped references and fills nothing new", () => {
    const d = initialDraft("dungeness");
    d.inputs.steamGph = 7.8;
    d.basis.steamGph = "preset";
    d.inputs.fuelUsdPerGal = 5;
    switchFishery(d, presetById("albacore"));
    expect(d.preset).toBe("albacore");
    expect(d.inputs.steamGph).toBeNull();
    expect(d.inputs.fuelUsdPerGal).toBe(5);
    expect(d.inputs.catch![0].species).toBe("Albacore");
  });
  it("the Santa Cruz salmon state waits for exactly fuel used and how crew is paid", () => {
    const d = initialDraft("salmon");
    d.inputs.catch![0].lb = 400;
    d.inputs.catch![0].usdPerLb = 9;
    d.inputs.fuelUsdPerGal = 8.242;
    const e = evaluateTrip(draftInputs(d), { audience: econAudience("commercial") });
    expect(e.gross).toMatchObject({ status: "value", value: 3600 });
    expect(e.missingFields).toEqual(["fuelGal", "crew.kind"]);
    expect(e.missingFields.map((k) => needLabel(k, d))).toEqual(["fuel used", "how crew is paid"]);
  });
  it("recreational and shellfish trips are worked out as cost only", () => {
    expect(econAudience("recreational")).toBe("recreational");
    expect(econAudience("shellfish")).toBe("recreational");
    expect(econAudience("commercial")).toBe("commercial");
    expect(econAudience("all")).toBe("commercial");
  });
});
