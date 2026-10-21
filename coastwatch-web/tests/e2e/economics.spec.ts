import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * v6 trip economics inside My Coast (src/v4/coast/economics): the fisherman's own numbers worked
 * through, kept on the device. Presets show their source; catch and price are never pre-filled; CDFW
 * and EIA references sit beside the fields and are copied in only on request.
 */
const NOW = "2026-10-13T06:00:00Z";
const URL = `${OK}/coast?place=port-452&for=commercial&species=dungeness`;
const BANNED = /\b(safe|go\/no-go|good day|hotspot|expected catch|price forecast)\b/i;

async function openWorksheet(page: Page) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(URL);
  const s = page.getByTestId("brief-trip-economics");
  await expect(s).toBeVisible();
  await expect(s.getByTestId("econ-kicker")).toHaveText("Scenario from your inputs, not a forecast.");
  await s.getByTestId("econ-toggle").click();
  await expect(s.getByTestId("econ-worksheet")).toBeVisible();
  return s;
}

async function fillTrip(page: Page) {
  const s = page.getByTestId("brief-trip-economics");
  await s.getByTestId("econ-nm").fill("20");
  await s.getByTestId("econ-kt").fill("8");
  await s.getByTestId("econ-hours").fill("8");
  await s.getByTestId("econ-fuel-price").fill("5");
  await s.getByTestId("econ-cost-0-usd").fill("200");
  await s.getByTestId("econ-catch-0-lb").fill("1000");
  await s.getByTestId("econ-catch-0-price").fill("3.50");
}

test("Dungeness preset: sourced defaults, nothing pre-filled for catch or price, live results", async ({ page }) => {
  const s = await openWorksheet(page);
  // sourced, editable defaults
  await expect(s.getByTestId("econ-fishery")).toHaveValue("dungeness");
  await expect(s.getByTestId("econ-fish-gph")).toHaveValue("4.8");
  await expect(s.getByTestId("econ-steam-gph")).toHaveValue("7.8");
  await expect(s.getByTestId("econ-crew-pct")).toHaveValue("28.1");
  await expect(s.getByTestId("econ-g-vessel")).toContainText("NOAA NWFSC-121, Table 31 (2008 survey)");
  await expect(s.getByTestId("econ-g-vessel")).toContainText("Fleet average then, not a norm");
  // never pre-filled
  await expect(s.getByTestId("econ-catch-0-lb")).toHaveValue("");
  await expect(s.getByTestId("econ-catch-0-price")).toHaveValue("");
  await expect(s.getByTestId("econ-fuel-price")).toHaveValue("");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("Past average, not today’s dock price");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("$3.51/lb");
  await expect(s.getByTestId("econ-diesel-ref")).toContainText("$8.242");

  await fillTrip(page);
  // fuel 19.5 + 38.4 = 57.9 gal × $5 = $289.50; + $200 → $490 trip cost
  await expect(s.getByTestId("econ-r-cost")).toHaveText("$490");
  await expect(s.getByTestId("econ-r-gross")).toHaveText("$3,500");
  // N = 3500 − 489.5; crew 28.1 % of N; net = N − crew
  await expect(s.getByTestId("econ-r-crew")).toHaveText("$846");
  await expect(s.getByTestId("econ-r-net")).toHaveText("+$2,165");
  await expect(s.getByTestId("econ-r-be-lb")).toHaveText("140 lb");
  await expect(s.getByTestId("econ-sens-0-0")).toHaveText("+$2,165");
  await expect(s.getByTestId("econ-summary").getByTestId("econ-sum-net")).toHaveText("+$2,165");

  // a negative is refused where it is typed
  await s.getByTestId("econ-nm").fill("-4");
  await expect(s.getByText("Can't be negative")).toBeVisible();
  expect(await s.innerText()).not.toMatch(BANNED);
});

test("a reference is copied in only on request, with its basis", async ({ page }) => {
  const s = await openWorksheet(page);
  // albacore has no species-level reference (never a neighbouring species' price)
  await s.getByTestId("econ-fishery").selectOption("albacore");
  await expect(s.getByTestId("econ-catch-0-name")).toHaveValue("Albacore");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("No species-level reference");
  await expect(s.getByTestId("econ-fish-gph")).toHaveValue("");
  await s.getByTestId("econ-fishery").selectOption("dungeness");
  await s.getByTestId("econ-catch-0-use").click();
  await expect(s.getByTestId("econ-catch-0-price")).toHaveValue("3.51");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("basis: CDFW history");
  await s.getByTestId("econ-use-diesel").click();
  await expect(s.getByTestId("econ-fuel-price")).toHaveValue("8.242");
  await expect(s.getByTestId("econ-diesel-ref")).toContainText("basis: EIA");
});

test("scenarios: save, duplicate, compare only the differing inputs; kept across reloads", async ({ page }) => {
  const s = await openWorksheet(page);
  await fillTrip(page);
  await s.getByTestId("econ-scen-name").fill("Opener");
  await s.getByTestId("econ-scen-save").click();
  await s.getByTestId("econ-scen-dup").click();
  await s.getByTestId("econ-catch-0-lb").fill("800");
  await s.getByTestId("econ-scen-save").click();
  await s.getByTestId("econ-cmp-a").selectOption({ label: "Opener" });
  await s.getByTestId("econ-cmp-b").selectOption({ label: "Opener (copy)" });
  const diff = s.getByTestId("econ-cmp-diff");
  await expect(diff).toHaveCount(1);
  await expect(diff).toContainText("Dungeness crab, lb");
  await page.reload();
  const s2 = page.getByTestId("brief-trip-economics");
  await expect(s2.getByTestId("econ-sum-cost")).toHaveText("$490");
  await s2.getByTestId("econ-toggle").click();
  await expect(s2.getByTestId("econ-scenarios")).toContainText("2 of 20");
});

test("a scenario link carries numbers only and opens unsaved", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const s = await openWorksheet(page);
  await fillTrip(page);
  await s.getByTestId("econ-cost-0").getByRole("textbox").first().fill("Skipper Bob's bait");
  await s.getByTestId("econ-share-copy").click();
  const link = await s.getByTestId("econ-share-url").inputValue();
  expect(link).toContain("#trip=");
  const payload = Buffer.from(link.split("#trip=")[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
  expect(payload).not.toMatch(/Bob|Princeton|Half Moon/);

  const other = await context.newPage();
  await other.clock.setFixedTime(new Date(NOW));
  await other.goto(link);
  const o = other.getByTestId("brief-trip-economics");
  await expect(o.getByTestId("econ-shared")).toContainText("From a shared link, not saved");
  await expect(o.getByTestId("econ-r-gross")).toHaveText("$3,500");
  await o.getByTestId("econ-shared-save").click();
  await expect(o.getByTestId("econ-shared")).toHaveCount(0);
});

test("damaged trip data is kept and reported; delete all clears only trip data", async ({ page }) => {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("seeded")) {
      localStorage.setItem("nx-trip-v1:scenarios", "{broken");
      sessionStorage.setItem("seeded", "1");
    }
  });
  const s = await openWorksheet(page);
  await expect(s.getByTestId("econ-store-status")).toContainText("could not be read");
  await s.getByTestId("econ-data").locator("summary").click();
  await s.getByTestId("econ-delete-all").click();
  await s.getByTestId("econ-delete-confirm").click();
  await expect(s.getByTestId("econ-data-msg")).toContainText("deleted");
  const keys = await page.evaluate(() => Object.keys(localStorage).filter((k) => k.startsWith("nx-trip-v1:")));
  expect(keys).toEqual([]);
  expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("research");
});

for (const width of [390, 320]) {
  test(`phone ${width}: results bar above the dock, no sideways scroll, 44 px targets, accessible`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    const s = await openWorksheet(page);
    await fillTrip(page);
    await s.getByTestId("econ-g-catch").scrollIntoViewIfNeeded();
    const bar = page.getByTestId("econ-bar");
    await expect(bar).toBeVisible();
    await expect(bar).toContainText("+$2,165");
    // fixed just above the dock, on screen (not pinned inside a transformed or contained ancestor)
    const box = (await bar.boundingBox())!;
    const dock = (await page.getByTestId("nav-dock").boundingBox())!;
    expect(Math.abs(box.y + box.height - dock.y)).toBeLessThanOrEqual(2);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    const small = await s.locator("button:visible, select:visible, input:visible:not([type=checkbox]):not([type=file])").evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).filter((r) => r.height < 44).length);
    expect(small).toBe(0);
    const axe = await new AxeBuilder({ page }).include("#trip-economics").analyze();
    expect(axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
  });
}
