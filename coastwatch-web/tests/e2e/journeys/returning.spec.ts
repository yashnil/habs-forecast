import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { M2, OK } from "../ports";

/**
 * Journey 2 (docs/coastwatch/v12/design-system/journeys.md): a returning fisherman checking the home port.
 * `/` renders Today directly (no opening, no Home), the change line is computed from the last visit and
 * says so in one line when nothing changed, and every value opens its provenance. Also: a dataset without
 * the pilot sidecars (production today) gets worded states, not blanks. Video and stills go to
 * docs/coastwatch/v12/implementation/today/.
 */
// review media go to the docs only on request (NX_MEDIA=1); an ordinary run keeps them with its own results
const OUT = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/today") : path.resolve(__dirname, "../../../test-results/journey-media/today");
test.use({ storageState: { cookies: [], origins: [] } });

const RECORD = (fisheries: string[]) => JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries, vessel: null });

async function seed(ctx: BrowserContext, items: Record<string, string>) {
  await ctx.addInitScript((kv) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, items);
}
const h1 = (page: Page) => page.getByTestId("nx-today").getByRole("heading", { level: 1 });

test("Journey 2: Today at / at once, the change line from the last visit, provenance one tap away", async ({ browser }) => {
  mkdirSync(OUT, { recursive: true });
  const size = { width: 1440, height: 900 };
  const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] }, recordVideo: { dir: path.join(test.info().outputDir, "video"), size } });
  await seed(ctx, { "nx-port-v1": RECORD(["dungeness_commercial"]), "nx-tour-v1": "dismissed" });
  await ctx.addInitScript(() => {
    const w = window as unknown as { __homeSeen: boolean };
    w.__homeSeen = false;
    const look = () => {
      const el = document.querySelector("[data-nx-home]");
      if (el && getComputedStyle(el).display !== "none" && (el as HTMLElement).offsetHeight > 0) w.__homeSeen = true;
    };
    new MutationObserver(look).observe(document, { childList: true, subtree: true });
  });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  // the first visit on this device: no change line (nothing is invented)
  await page.goto(`${OK}/`);
  await expect(h1(page)).toHaveText("Season delayed to at least Jan 1, 2027");
  await expect(page).toHaveURL(`${OK}/`);
  expect(await page.evaluate(() => document.documentElement.getAttribute("data-nx-opening"))).toBeNull();
  expect(await page.evaluate(() => (window as unknown as { __homeSeen: boolean }).__homeSeen)).toBe(false);
  await expect(page.locator("[data-nx-sounding=buoy]")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("nx-change")).toHaveCount(0);

  // the next visit: nothing changed, said in one line
  await page.reload();
  await expect(page.getByTestId("nx-change")).toHaveText(/^Since \d{1,2}:\d\d (AM|PM) PT: no change in season status, tests, notices or models\.$/);

  // a visit after a change: the stored last visit had another status; the line names it
  await page.evaluate(() => {
    const k = "nx-today-seen-v1:port-592";
    const s = JSON.parse(localStorage.getItem(k)!);
    s.at = "2026-10-08T17:00:00Z";
    s.facts = s.facts.map((f: { id: string; value: string; headline: string }) => (f.id === "status:dungeness_commercial" ? { ...f, value: "No restriction on record", headline: "Dungeness crab (commercial): No closure or restriction on record here." } : f));
    localStorage.setItem(k, JSON.stringify(s));
  });
  await page.reload();
  const change = page.getByTestId("nx-change");
  await expect(change).toContainText("Since Oct 8: Dungeness crab (commercial) is now delayed.");
  await page.waitForTimeout(800);
  await change.click();
  await expect(page.getByText("Before: Dungeness crab (commercial): No closure or restriction on record here.")).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "journey-2-changed-1440x900.png") });

  // the status's source: the stamp opens the provenance sheet, which says "not verified" in words
  await page.getByRole("button", { name: /Official · CDFW · checked .* · not verified\. Open where this comes from/ }).first().click();
  const sheet = page.getByRole("dialog");
  await expect(sheet).toContainText("Not verified. Transcribed from the agency's page by machine");
  await expect(sheet.getByRole("link", { name: "Open the source" })).toHaveAttribute("href", "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries");
  await page.waitForTimeout(900);
  await page.screenshot({ path: path.join(OUT, "journey-2-provenance-1440x900.png") });
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);

  // a sounding opens its own provenance
  await page.locator("[data-nx-sounding=test] button").click();
  await expect(page.getByRole("dialog")).toContainText("CDPH test");
  await expect(page.getByRole("dialog")).toContainText("Sampled");
  await page.keyboard.press("Escape");

  // a row expands in place; the models say past run and their mandatory words
  await page.getByRole("button", { name: /^Models/ }).click();
  await expect(page.getByText("C-HARM, past run")).toBeVisible();
  await expect(page.getByText(/probability of conditions in the water, not toxin in seafood/)).toBeVisible();

  // the next steps carry the port
  const today = page.getByTestId("nx-today");
  await expect(today.getByRole("link", { name: "Read CDFW’s notice" })).toHaveAttribute("href", "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries");
  await expect(today.getByRole("link", { name: "Plan a trip" })).toHaveAttribute("href", "/coast?place=port-592");
  await expect(today.getByRole("link", { name: "Alerts" })).toHaveAttribute("href", "/account?place=port-592#alerts");
  await page.waitForTimeout(600);
  const video = page.video();
  await ctx.close();
  if (video) copyFileSync(await video.path(), path.join(OUT, "journey-2-returning-1440.webm"));
  expect(errors).toEqual([]);
});

test("a deep link to /today with a port and no fishery asks, and the chart is on the port", async ({ page }) => {
  await page.goto(`${OK}/today?place=port-452`);
  const chooser = page.getByTestId("nx-fishery-chooser");
  await expect(chooser.getByRole("heading", { level: 1 })).toHaveText("Princeton / Half Moon Bay");
  await chooser.getByRole("button", { name: "Groundfish" }).click();
  await expect(h1(page)).toBeVisible();
  // the fishery can be changed from the other fisheries
  await page.getByRole("button", { name: /^Others/ }).click();
  await page.getByRole("button", { name: /Lead with dungeness crab$/ }).first().click();
  await expect(h1(page)).toHaveText(/Season delayed/);
});

test("without the pilot sidecars (production data), every missing source is worded", async ({ page, context }) => {
  await seed(context, { "nx-port-v1": RECORD(["dungeness_commercial"]), "nx-tour-v1": "dismissed", "nx-opening-seen-v1": "1" });
  await page.goto(`${M2}/today?place=port-592`);
  const today = page.getByTestId("nx-today");
  await expect(today).toContainText(/isn.t available|could not be read/, { timeout: 15_000 });
  await expect(today).not.toContainText("undefined");
  await expect(today).not.toContainText("NaN");
});

test("no port: Today says what to do", async ({ page, context }) => {
  await seed(context, { "nx-opening-seen-v1": "1" });
  await page.goto(`${OK}/today`);
  await expect(page.getByTestId("nx-today")).toContainText("Choose your port to see its season status");
  await expect(page.getByTestId("nx-today").getByRole("button", { name: "Find your port" })).toBeVisible();
});

test("reduced motion: Today is whole at once, soundings in place", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce", viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
  await seed(ctx, { "nx-port-v1": RECORD(["dungeness_commercial"]) });
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await expect(h1(page)).toBeVisible();
  await expect(page.locator("[data-nx-sounding=test]")).toBeVisible({ timeout: 10_000 });
  await ctx.close();
});

/** Review stills at the widths the product is used at, and in Dusk and Night (NX_MEDIA=1). */
test.describe("media", () => {
  test.skip(!process.env.NX_MEDIA, "media only on request (NX_MEDIA=1)");
  for (const [name, w, h, mode] of [
    ["today-1440x900", 1440, 900, "day"],
    ["today-390x844", 390, 844, "day"],
    ["today-320x640", 320, 640, "day"],
    ["today-844x390", 844, 390, "day"],
    ["today-dusk-1440x900", 1440, 900, "dusk"],
    ["today-dusk-390x844", 390, 844, "dusk"],
    ["today-night-1440x900", 1440, 900, "night"],
    ["today-night-390x844", 390, 844, "night"],
  ] as const) {
    test(name, async ({ browser }) => {
      mkdirSync(OUT, { recursive: true });
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, storageState: { cookies: [], origins: [] } });
      await seed(ctx, { "nx-port-v1": RECORD(["dungeness_commercial"]), "nx-chart-mode-v1": mode, "nx-tour-v1": "dismissed" });
      const page = await ctx.newPage();
      await page.goto(`${OK}/`);
      await expect(h1(page)).toBeVisible();
      await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
      await page.waitForTimeout(2200);
      await page.screenshot({ path: path.join(OUT, `${name}.png`) });
      await ctx.close();
    });
  }
  for (const [name, w, h] of [
    ["home-1440x900", 1440, 900],
    ["home-390x844", 390, 844],
    ["home-320x640", 320, 640],
    ["home-844x390", 844, 390],
  ] as const) {
    test(name, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, storageState: { cookies: [], origins: [] } });
      await seed(ctx, { "nx-opening-seen-v1": "1" });
      const page = await ctx.newPage();
      await page.goto(`${OK}/`);
      await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
      await page.waitForTimeout(1200);
      await page.screenshot({ path: path.join(OUT, `${name}.png`) });
      if (w === 1440 || w === 390) {
        await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
        await page.getByTestId("nx-home-finder").getByRole("combobox").fill("mo");
        await page.waitForTimeout(500);
        await page.screenshot({ path: path.join(OUT, `home-finder-${w}x${h}.png`) });
        await page.getByTestId("nx-home-finder").getByRole("option", { name: /Moss Landing/ }).click();
        await expect(page.getByTestId("nx-fishery-chooser")).toContainText("Suggested");
        await page.waitForTimeout(900);
        await page.screenshot({ path: path.join(OUT, `chooser-${w}x${h}.png`) });
      }
      await ctx.close();
    });
  }
});
