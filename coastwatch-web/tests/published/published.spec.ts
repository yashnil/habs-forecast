import { expect, test } from "@playwright/test";

const BASE = process.env.PUBLISHED_DATA_URL ?? "https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data/v1";
const APP = "http://localhost:3300";

test("app renders the published dataset: forecast image, grids and values load cross-origin", async ({ page, request }) => {
  const manifest = await (await request.get(`${BASE}/manifest.json`)).json();
  const report = await (await request.get(`${BASE}/verification/charm-points.json`)).json();
  const failed: string[] = [];
  page.on("requestfailed", (r) => failed.push(`${r.failure()?.errorText} ${r.url()}`));
  page.on("console", (m) => m.type() === "error" && failed.push(m.text()));

  await page.goto(`${APP}/?var=particulate_domoic&lead=1&inspect=36.80,-121.95`);
  await expect(page.getByTestId("data-unavailable")).toHaveCount(0);
  await expect(page.getByTestId("fixture-banner")).toHaveCount(0);
  await expect(page.getByTestId("forecast-panel")).toContainText("C-HARM v3.1");

  // the map's image source points at the published, content-addressed asset
  await page.waitForFunction(() => {
    const m = (window as unknown as { __cwMap?: { getSource: (id: string) => unknown; loaded: () => boolean } }).__cwMap;
    return !!m && !!m.getSource("forecast") && m.loaded();
  });
  const src = await page.evaluate(() => (window as unknown as { __cwMap: { getSource: (id: string) => { url: string } } }).__cwMap.getSource("forecast").url);
  const layer = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_particulate_domoic_lead1");
  expect(src).toBe(`${BASE}/${layer.image.url}`);

  // inspector values decoded in the browser from the published grids equal the values
  // the pipeline verified against ERDDAP before publishing
  const rows = report.rows.filter((r: { point: string; layer_id: string }) => r.point === "Monterey Bay (mid-bay)" && r.layer_id.endsWith("lead1"));
  for (const r of rows) {
    const v = r.layer_id.replace(/^charm_/, "").replace(/_lead\d$/, "");
    await expect(page.getByTestId(`inspect-value-${v}`).first()).toHaveText(`${(r.grid_value * 100).toFixed(0)}%`);
  }
  expect(failed.filter((f) => f.includes(BASE))).toEqual([]);
});

test("M2 data loads from the published URL: official notices, overlay and port summaries", async ({ page, request }) => {
  const manifest = await (await request.get(`${BASE}/manifest.json`)).json();
  test.skip(!manifest.official_url || !manifest.port_intel_url, "published dataset predates M2 (no official/port files)");
  const official = await (await request.get(`${BASE}/${manifest.official_url}`)).json();
  const intel = await (await request.get(`${BASE}/${manifest.port_intel_url}`)).json();
  const failed: string[] = [];
  page.on("requestfailed", (r) => failed.push(`${r.failure()?.errorText} ${r.url()}`));

  await page.goto(`${APP}/?region=monterey_bay&port=550&var=particulate_domoic&lead=1`);
  const rail = page.getByTestId("official-status");
  const expected = official.registry.review.status === "human_verified" ? /verified|aging|unverified/ : /unverified/;
  await expect(rail.getByTestId("official-verification")).toHaveAttribute("data-state", expected);
  const p = intel.ports.find((x: { port_code: number }) => x.port_code === 550);
  const panel = page.getByTestId("port-panel");
  for (const rel of p.official_relations) await expect(panel.getByTestId(`notice-${rel.record_id}`)).toBeVisible();
  const lead1 = p.charm.leads.find((l: { lead_days: number }) => l.lead_days === 1);
  await expect(panel.getByTestId("port-median-particulate_domoic")).toHaveText(`${Math.round(lead1.variables.particulate_domoic.median * 100)}%`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { getSource: (s: string) => unknown } }).__cwMap?.getSource("official"));
  expect(failed.filter((f) => f.includes(BASE))).toEqual([]);
});
