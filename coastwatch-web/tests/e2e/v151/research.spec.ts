import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Research coverage (v15.1, Agent C): the stations near a place, with distances, what each measures and the latest
 * date per variable; one press to a latest sample; the place kept across station and variable; the statewide map.
 * Fixture clock 2026-10-13. Crescent City is sparse (nearest station 77 km), Monterey dense, Fort Bragg has nothing
 * within 60 km of any kind.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const params = (page: Page) => Object.fromEntries(new URL(page.url()).searchParams);

async function open(page: Page, q: string) {
  await page.addInitScript(() => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("cw-intro-dismissed-v1", "1");
  });
  await page.goto(`${OK}/research?${q}`);
  await expect(page.getByTestId("research-record")).toBeVisible();
}

test("Crescent City (sparse): opens on the stations near it, never on a distant pier as its own water", async ({ page }) => {
  await open(page, "place=port-201");
  const nearby = page.getByTestId("research-nearby");
  await expect(nearby).toBeVisible();
  await expect(page.getByTestId("research-nearby-lead")).toContainText("Nothing samples within 60 km of Crescent City");
  await expect(page.getByTestId("research-nearby-lead")).toContainText("Trinidad Pier, 77 km south");
  // stations with distance and direction, nearest first
  const rows = page.getByTestId("research-nearby-stations").locator(":scope > li");
  await expect(rows).toHaveCount(3);
  await expect(rows.first()).toHaveAttribute("data-testid", "nearby-HABs-TrinidadPier");
  await expect(page.getByTestId("nearby-where-HABs-TrinidadPier")).toContainText("77 km south · beyond 60 km");
  // which variables it measures, the latest date per variable, its state; what it does not collect
  await expect(page.getByTestId("nearby-latest-HABs-TrinidadPier-pDA")).toContainText("Apr 17, 2026");
  await expect(page.getByTestId("nearby-latest-HABs-TrinidadPier-pDA")).toContainText("Old but valid");
  await expect(page.getByTestId("nearby-not-HABs-TrinidadPier")).toContainText("Not collected here: seriata, delicatissima, chlorophyll, temperature");
  // variables no listed station measures: the nearest that does, far and said so
  await expect(page.getByTestId("nearby-reach-pn_seriata")).toContainText("The nearest station measuring Pseudo-nitzschia (seriata) is Bodega Marine Lab");
  await expect(page.getByTestId("nearby-reach-pn_seriata")).toContainText("nothing nearer samples it");
  // the statewide map keeps the place
  await expect(page.getByTestId("research-statewide")).toHaveAttribute("href", "/explore?view=coast&place=port-201&st=1&from=research");
  // never a zero for a variable a station does not collect
  await expect(nearby).not.toContainText(/\b0 cells|0 ng\/mL/);
});

test("Crescent City: one press reaches the latest sample; the place stays; the open station's distance is said", async ({ page }) => {
  await open(page, "place=port-201");
  await page.getByTestId("nearby-latest-HABs-TrinidadPier-tDA").click();
  await expect(page.getByTestId("research-nearby")).toHaveCount(0);
  await expect.poll(() => params(page)).toMatchObject({ place: "port-201", station: "HABs-TrinidadPier", var: "tDA", day: "2026-04-17" });
  await expect(page.getByTestId("research-status")).toContainText("tDA last measured Apr 17, 2026");
  await expect(page.getByTestId("inspector-value")).toBeVisible();
  // a farther station on another variable: the place is kept and the distance stated
  await page.getByTestId("research-nearby-open").click();
  await page.getByTestId("nearby-reach-pn_seriata").getByRole("button").click();
  await expect.poll(() => params(page)).toMatchObject({ place: "port-201", station: "HABs-BodegaMarineLab", var: "pn_seriata" });
  await expect(page.getByTestId("research-away")).toContainText("Bodega Marine Lab is");
  await expect(page.getByTestId("research-away")).toContainText("south of Crescent City: its record is of that water, not Crescent City's.");
  // the whole historical record is one press away
  await page.getByTestId("range-all").click();
  await expect(page.getByTestId("range-all")).toHaveAttribute("aria-checked", "true");
});

test("Monterey (dense): opens on the record; the nearby stations are one press away and current ones say so", async ({ page }) => {
  await open(page, "place=port-550");
  await expect(page.getByTestId("research-nearby")).toHaveCount(0);
  await expect(page.getByTestId("research-plot").or(page.getByTestId("research-empty"))).toBeVisible();
  await page.getByTestId("research-nearby-open").click();
  await expect(page.getByTestId("research-nearby-lead")).toContainText("within 60 km of Monterey");
  await expect(page.getByTestId("nearby-latest-HABs-MontereyWharf-temp")).toContainText("Regular samples");
  await expect(page.getByTestId("nearby-latest-HABs-MontereyWharf-temp")).toContainText("weekly");
  await expect(page.getByTestId("nearby-latest-HABs-MontereyWharf-pDA")).toContainText("Aug 31, 2022");
  await expect(page.getByTestId("nearby-latest-HABs-MontereyWharf-pDA")).toContainText("Old but valid");
  await expect(page.getByTestId("nearby-HABs-SantaCruzWharf")).toBeVisible();
  // the model, kept apart and labelled
  await expect(page.getByTestId("research-nearby-model")).toContainText("Modelled, not sampled");
  // changing station and variable from here keeps the place
  await page.getByTestId("nearby-HABs-SantaCruzWharf-chl_extracted").getByRole("button").first().click();
  await expect.poll(() => params(page)).toMatchObject({ place: "port-550", station: "HABs-SantaCruzWharf", var: "chl_extracted" });
  await expect(page.getByTestId("research-away")).toContainText("Santa Cruz Wharf");
});

test("Fort Bragg (nothing near): every listed station is beyond 60 km and says so", async ({ page }) => {
  await open(page, "place=port-223");
  await expect(page.getByTestId("research-nearby")).toHaveAttribute("data-near", "0");
  await expect(page.getByTestId("research-nearby-lead")).toContainText("Nothing samples within 60 km of Fort Bragg");
  const wheres = page.locator('[data-testid^="nearby-where-"]');
  await expect(wheres).toHaveCount(3);
  for (const t of await wheres.allTextContents()) expect(t).toContain("beyond 60 km");
  // the arrival line still says the record is not this water's
  await expect(page.getByTestId("research-arrived")).toContainText("No station is near Fort Bragg");
});

test("phone: the stations near a sparse place read first in the sheet", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "place=port-201");
  await expect(page.getByTestId("research-nearby-panel")).toHaveAttribute("data-first", "true");
  await page.getByTestId("nearby-latest-HABs-TrinidadPier-pDA").scrollIntoViewIfNeeded();
  await page.getByTestId("nearby-latest-HABs-TrinidadPier-pDA").click();
  await expect.poll(() => params(page)).toMatchObject({ station: "HABs-TrinidadPier", var: "pDA", day: "2026-04-17" });
  const sw = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(sw).toBeLessThanOrEqual(390);
  await ctx.close();
});
