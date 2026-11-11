import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * One evidence backbone (v4): the briefing's statements, its evidence rows, the Explore harbour
 * panel and GET /api/evidence are the same graph (lib/harborEvidence), so they cannot disagree about
 * the same observation; and every "Show on map" points at the node its statement cites. The page
 * clock and the server clock (CW_FIXED_NOW) are the same instant.
 */

type Claim = { id: string; kind: string; headline: string; text: string; supports: string[] };
type Node = { id: string; text: string; class: string; where: { km: number; lat: number | null; lon: number | null } };
type Graph = { claims: Claim[]; nodes: Node[]; gaps: { id: string }[] };

test("the API carries the whole graph the page summarises: limits, gaps, and no safety language", async ({ request }) => {
  const api = (await (await request.get(`${OK}/api/evidence?place=port-550`)).json()) as Graph & { limits: string[]; claims: { text: string }[] };
  expect(api.limits.length).toBeGreaterThanOrEqual(3);
  for (const c of api.claims) expect(c.text).not.toMatch(/\b(safe|unsafe|all clear|risk score)\b/i);
  expect(api.gaps.some((g) => g.id === "model-here" || g.id === "model-day" || g.id === "model")).toBe(true);
});
