import { expect, test } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, newContext, SIZES } from "./helpers";

/**
 * Test J (DIRECTION §17): the pages v15 did not set out to change still work. Each opens at Moss Landing, its primary
 * element renders with its content, and nothing throws. Run at a desk and a phone size.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const PAGES: Array<{ name: string; path: string; root: string; primary: (p: import("@playwright/test").Page) => import("@playwright/test").Locator }> = [
  { name: "Today", path: `/today?place=${MOSS}`, root: "nx-margin", primary: (p) => p.getByRole("heading", { level: 1 }).first() },
  { name: "Explore", path: `/explore?place=${MOSS}`, root: "nx-explore", primary: (p) => p.getByTestId("nx-explore") },
  { name: "My Coast", path: `/coast?place=${MOSS}`, root: "nx-coast", primary: (p) => p.getByRole("heading", { level: 1 }).first() },
  { name: "My Coast, trip worksheet", path: `/coast?place=${MOSS}&view=trip`, root: "nx-coast", primary: (p) => p.getByTestId("trip-inputs") },
  { name: "Research", path: `/research?station=HABs-SantaCruzWharf&var=pDA&range=3y&day=2026-05-27`, root: "nx-research", primary: (p) => p.getByTestId("research-plot") },
  { name: "Alerts", path: `/alerts?place=${MOSS}`, root: "nx-alerts", primary: (p) => p.getByTestId("alerts-follow").or(p.getByRole("heading", { level: 1 })).first() },
  { name: "Notices", path: `/notices?place=${MOSS}`, root: "nx-margin", primary: (p) => p.getByRole("heading", { level: 1 }).first() },
];

for (const [label, size] of [
  ["desk", SIZES.desk],
  ["phone", SIZES.phone],
] as const) {
  test.describe(`J: existing pages at ${label} size`, () => {
    for (const pg of PAGES) {
      test(`${pg.name} loads and its primary element renders`, async ({ browser }) => {
        const { ctx, page, errors } = await newContext(browser, size);
        const res = await page.goto(`${OK}${pg.path}`);
        expect(res?.status()).toBeLessThan(400);
        await expect(page.getByTestId(pg.root).first()).toBeAttached();
        await expect(pg.primary(page)).toBeVisible({ timeout: 20_000 });
        expect(errors).toEqual([]);
        await ctx.close();
      });
    }
  });
}

test("J: the chart still draws on My Coast (one persistent chart)", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/coast?place=${MOSS}&view=trip`);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", /full|reduced|static/);
  await ctx.close();
});
