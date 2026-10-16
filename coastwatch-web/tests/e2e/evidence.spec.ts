import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * One evidence backbone: the briefing's lines, the evidence list, the map's thread and
 * GET /api/evidence are all the same graph (lib/harborEvidence), so they cannot disagree about the
 * same observation. The page clock and the server clock (CW_FIXED_NOW) are the same instant.
 */
const NOW = "2026-10-13T06:00:00Z";
const PLACES = ["port-550", "port-593", "port-201", "port-215"];

type Claim = { id: string; kind: string; headline: string; text: string; supports: string[] };
type Node = { id: string; text: string; where: { km: number } };
type Graph = { claims: Claim[]; nodes: Node[]; gaps: { id: string }[] };

for (const place of PLACES) {
  test(`${place}: the page's lines, evidence rows and map thread match /api/evidence`, async ({ browser, request }) => {
    // Coastal mode (the trip briefing), the clock at the server's fixed instant
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] }] } });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=${place}`);
    await expect(page.getByTestId("matter-hab")).not.toContainText("Reading");
    await expect(page.getByTestId("evidence-list")).toBeVisible();
    // the page and the API read the same trip day (today, Pacific) and the same data
    const api = (await (await request.get(`${OK}/api/evidence?place=${place}`)).json()) as Graph;
    const items = await page.locator('[data-testid^="matter-"][data-kind]').evaluateAll((els) =>
      els.map((e) => ({ id: e.getAttribute("data-testid")!.replace(/^matter-/, ""), kind: e.getAttribute("data-kind"), headline: e.querySelector("p")!.textContent!.trim(), supports: (e.getAttribute("data-supports") ?? "").split(" ").filter(Boolean) })),
    );
    expect(items.length).toBeGreaterThanOrEqual(2);
    for (const it of items) {
      const c = api.claims.find((x) => x.id === it.id);
      expect(c, `API has claim ${it.id}`).toBeDefined();
      expect(it.headline).toBe(c!.headline);
      expect(it.kind).toBe(c!.kind);
      expect(it.supports).toEqual(c!.supports);
      for (const s of it.supports) expect(api.nodes.some((n) => n.id === s), `node ${s} exists`).toBe(true);
    }
    // every evidence row is a node of the API's graph with the same value text
    const rows = await page.locator("[data-node]").evaluateAll((els) => els.map((e) => ({ id: e.getAttribute("data-node"), text: e.querySelector("span")!.textContent!.trim() })));
    for (const r of rows) {
      const n = api.nodes.find((x) => x.id === r.id);
      expect(n, `API has node ${r.id}`).toBeDefined();
      expect(r.text).toBe(n!.text);
    }
    // the HAB detail's summary paragraph (behind "More detail") is the graph's own sentences for the
    // HAB, model and satellite lines, folded as the page folds them: no second wording anywhere
    await page.getByTestId("coast-details").locator("summary").click();
    const hab = api.claims.find((c) => c.id === "hab");
    const folded = api.claims.filter((c) => (c.id === "hab" || c.id === "model" || c.id === "satellite") && !(c.id === "model" && hab && (hab.kind === "measured" || hab.kind === "model")));
    await expect(page.getByTestId("hab-summary")).toHaveText(folded.map((c) => c.text).join(" "));
    // the map's thread to the nearest model cell carries the same distance the graph's node does
    await page.setViewportSize({ width: 1440, height: 900 });
    await expect(page.getByTestId("coast-map-desktop")).toBeVisible();
    const charm = api.nodes.find((n) => n.id.startsWith("charm:offshore:"));
    if (charm) {
      await page.waitForFunction(() => {
        const m = (window as unknown as { __cwMap?: { getSource: (id: string) => { serialize: () => { data?: { features?: unknown[] } } } | undefined } }).__cwMap;
        const d = m?.getSource("coast-overlay")?.serialize().data;
        return !!d && (d.features ?? []).some((f) => (f as { properties: { kind: string; to: string } }).properties.kind === "end" && (f as { properties: { kind: string; to: string } }).properties.to === "cell");
      });
      const km = await page.evaluate(() => {
        const m = (window as unknown as { __cwMap: { getSource: (id: string) => { serialize: () => { data: { features: { properties: Record<string, unknown> }[] } } } } }).__cwMap;
        const f = m.getSource("coast-overlay").serialize().data.features.find((x) => x.properties.kind === "end" && x.properties.to === "cell");
        return f?.properties.km as number;
      });
      expect(km).toBe(+charm.where.km.toFixed(1));
    }
    await ctx.close();
  });
}

test("the API carries the whole graph the page summarises: limits, gaps, and no safety language", async ({ request }) => {
  const api = (await (await request.get(`${OK}/api/evidence?place=port-550`)).json()) as Graph & { limits: string[]; claims: { text: string }[] };
  expect(api.limits.length).toBeGreaterThanOrEqual(3);
  for (const c of api.claims) expect(c.text).not.toMatch(/\b(safe|unsafe|all clear|risk score)\b/i);
  expect(api.gaps.some((g) => g.id === "model-here" || g.id === "model-day" || g.id === "model")).toBe(true);
});
