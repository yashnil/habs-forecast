import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** County reference layer: drawn quietly, switchable, sourced, and never presented as a marine boundary. */
async function ready(page: Page, url: string) {
  await page.goto(url);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
  await page.waitForFunction(() => {
    const m = (window as unknown as { __cwMap: { getSource: (id: string) => unknown; isSourceLoaded: (id: string) => boolean } }).__cwMap;
    return !!m.getSource("counties") && m.isSourceLoaded("counties");
  });
}
const rendered = (page: Page, layer: string) =>
  page.evaluate((l) => (window as unknown as { __cwMap: { queryRenderedFeatures: (o: object) => { properties: { name?: string } }[] } }).__cwMap.queryRenderedFeatures({ layers: [l] }).map((f) => f.properties.name ?? "line"), layer);

test.describe("counties", () => {
  test("borders and names draw at the region view, with the source credited", async ({ page }) => {
    await ready(page, `${OK}/?region=mendocino_sonoma`);
    await expect.poll(() => rendered(page, "county-borders").then((x) => x.length), { timeout: 15000 }).toBeGreaterThan(0);
    await expect.poll(() => rendered(page, "county-labels"), { timeout: 15000 }).toEqual(expect.arrayContaining(["Sonoma"]));
    await expect(page.locator(".maplibregl-ctrl-attrib")).toContainText("CA Dept. of Technology", { useInnerText: false });
  });

  test("the control switches lines and names separately, says what they are not, and remembers", async ({ page }) => {
    await ready(page, `${OK}/?region=mendocino_sonoma`);
    await page.getByTestId("counties-toggle").click();
    await expect(page.getByTestId("counties-panel")).toBeVisible();
    await expect(page.getByTestId("counties-note")).toContainText("do not define fishery closures");
    await page.getByTestId("counties-labels").uncheck();
    await expect.poll(() => rendered(page, "county-labels")).toEqual([]);
    await expect.poll(() => rendered(page, "county-borders").then((x) => x.length)).toBeGreaterThan(0);
    await page.getByTestId("counties-lines").uncheck();
    await expect.poll(() => rendered(page, "county-borders")).toEqual([]);
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("counties-panel")).toHaveCount(0);
    await ready(page, `${OK}/?region=mendocino_sonoma`);
    await page.waitForTimeout(500);
    expect(await rendered(page, "county-borders")).toEqual([]);
    await page.getByTestId("counties-toggle").click();
    await expect(page.getByTestId("counties-lines")).not.toBeChecked();
  });

  test("axe: the open control has no violations", async ({ page }) => {
    await ready(page, `${OK}/?region=mendocino_sonoma`);
    await page.getByTestId("counties-toggle").click();
    const r = await new AxeBuilder({ page }).include(".maplibregl-ctrl-bottom-right").analyze();
    expect(r.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  });
});
