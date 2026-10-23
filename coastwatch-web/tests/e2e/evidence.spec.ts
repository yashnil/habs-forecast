import { expect, test } from "@playwright/test";
import { OK } from "./ports";
import { openEvidence } from "./helpers";

/**
 * One evidence backbone (v4): the briefing's statements, its evidence rows, the Explore harbour
 * panel and GET /api/evidence are the same graph (lib/harborEvidence), so they cannot disagree about
 * the same observation; and every "Show on map" points at the node its statement cites. The page
 * clock and the server clock (CW_FIXED_NOW) are the same instant.
 */
const NOW = "2026-10-13T06:00:00Z";
const PLACES = ["port-550", "port-593", "port-201", "port-215"];

type Claim = { id: string; kind: string; headline: string; text: string; supports: string[] };
type Node = { id: string; text: string; class: string; where: { km: number; lat: number | null; lon: number | null } };
type Graph = { claims: Claim[]; nodes: Node[]; gaps: { id: string }[] };

async function claimsOn(page: import("@playwright/test").Page, scope: string) {
  return page.locator(`${scope} [data-testid^="claim-"][data-kind]`).evaluateAll((els) =>
    els.map((e) => ({
      id: e.getAttribute("data-testid")!.replace(/^claim-/, ""),
      kind: e.getAttribute("data-kind"),
      headline: e.querySelector("[data-headline]")!.textContent!.trim(),
      supports: (e.getAttribute("data-supports") ?? "").split(" ").filter(Boolean),
    })),
  );
}

for (const place of PLACES) {
  test(`${place}: the briefing's statements and evidence rows match /api/evidence`, async ({ browser, request }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=${place}`);
    await expect(page.getByTestId("matters").locator("li").first()).toBeVisible();
    // the conditions arrive after the first statements: wait until the briefing reports them read
    await expect.poll(async () => page.locator('[data-testid="matters"] [data-kind="conditions"]').count(), { timeout: 15000 }).toBeGreaterThan(0);
    const api = (await (await request.get(`${OK}/api/evidence?place=${place}`)).json()) as Graph;
    const items = await claimsOn(page, '[data-testid="matters"]');
    expect(items.length).toBeGreaterThanOrEqual(2);
    for (const it of items) {
      const c = api.claims.find((x) => x.id === it.id);
      expect(c, `API has claim ${it.id}`).toBeDefined();
      expect(it.headline).toBe(c!.headline);
      expect(it.kind).toBe(c!.kind);
      expect(it.supports).toEqual(c!.supports);
      for (const s of it.supports) expect(api.nodes.some((n) => n.id === s), `node ${s} exists`).toBe(true);
    }
    const rows = await page.locator("[data-node]").evaluateAll((els) => els.map((e) => ({ id: e.getAttribute("data-node"), text: e.querySelector("[data-value]")!.textContent!.trim() })));
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) {
      const n = api.nodes.find((x) => x.id === r.id);
      expect(n, `API has node ${r.id}`).toBeDefined();
      expect(r.text).toBe(n!.text);
    }
    await ctx.close();
  });

  test(`${place}: Explore's harbour panel states the same claims, and each "Show on map" targets its node`, async ({ browser, request }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?sel=harbor:${place.slice(5)}&layer=forecast`);
    const panel = page.getByTestId("harbor-claims");
    await expect(panel.locator("li").first()).toBeVisible();
    const api = (await (await request.get(`${OK}/api/evidence?place=${place}`)).json()) as Graph;
    const items = await claimsOn(page, '[data-testid="harbor-claims"]');
    for (const it of items) {
      const c = api.claims.find((x) => x.id === it.id)!;
      expect(c, `API has claim ${it.id}`).toBeDefined();
      expect(it.headline).toBe(c.headline);
    }
    // a model claim's "Show on map" selects the cell its node is in
    const model = items.find((i) => i.kind === "model" || i.kind === "gap");
    const node = model?.supports.map((s) => api.nodes.find((n) => n.id === s)).find((n) => n?.class === "published_forecast" && n.where?.lat != null);
    if (model && node) {
      await openEvidence(page);
      await page.getByTestId(`claim-show-${model.id}`).click();
      await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "cell");
      const sel = new URL(page.url()).searchParams.get("sel")!;
      const [lat, lon] = sel.slice(5).split(",").map(Number);
      expect(lat).toBeCloseTo(node.where.lat!, 3);
      expect(lon).toBeCloseTo(node.where.lon!, 3);
      // and Back returns to the harbour
      await page.getByTestId("inspect-back").click();
      await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
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
