/**
 * Rule-based guidance keyed off regional algae tier (from project grid / ML export).
 * Not a stock assessment — operational heuristics for fishermen.
 */

export type Tier = "Lower" | "Typical" | "Higher" | string;

export function fishTargetsForTier(tier: Tier): string[] {
  const t = (tier || "").toLowerCase();
  if (t.includes("lower")) {
    return [
      "Favor usual nearshore finfish and crab programs when seasons allow — surface chlorophyll is relatively subdued vs other water on this map.",
      "Shellfish growers: still run **CDPH/OEHHA** checks; low chlorophyll does **not** mean low toxin risk.",
    ];
  }
  if (t.includes("higher")) {
    return [
      "Consider **diversifying target species** (e.g., more finfish vs. shellfish focus) if your buyers are sensitive to bloom news.",
      "Talk to your buyer **before** shifting effort — some markets discount certain ports during bloom press cycles even when landings are legal.",
      "If you operate both nearshore and slightly offshore blocks, compare **relative** colors on the map; avoid over-interpreting a single pixel.",
    ];
  }
  return [
    "Conditions look **middle-of-the-pack** for this snapshot vs other ocean water on the map — use your normal port intel plus official notices.",
  ];
}

export function economicRiskBullets(tier: Tier): string[] {
  const t = (tier || "").toLowerCase();
  const base = [
    "**No landing-price model here** — economic risk depends on species, grade, and contracts; this app only shows algae proxies.",
    "Reduce revenue volatility by **splitting trips** across species/ports when regulations allow, and by locking verbal buyer commitments when possible.",
  ];
  if (t.includes("higher")) {
    return [
      ...base,
      "During elevated chlorophyll periods, **media and wholesale narratives** can move faster than biology — keep screenshots of official state bulletins to share with buyers.",
      "If you depend on live tanks or brailer boats, plan **pump/filter maintenance** when nearshore water is thick with algae — operational downtime is a real cost.",
    ];
  }
  return base;
}

export function officialChecklist(): string[] {
  return [
    "CDPH — marine biotoxin monitoring",
    "OEHHA — domoic acid / fish consumption",
    "CDFW — seasons, bag limits, closures",
    "NOAA WC — harmful algal bloom bulletins",
  ];
}
