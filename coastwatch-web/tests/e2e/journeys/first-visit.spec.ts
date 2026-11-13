import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Journey 1 (docs/coastwatch/v12/design-system/journeys.md): a fisherman evaluating Neritic for the first
 * time. A fresh browser: the opening, then Home, Find your port, a fishery, and Today assembled in place at
 * `/`, in three taps. Then the same device offline: Today still opens. Recorded on video into
 * docs/coastwatch/v12/implementation/today/.
 */
// review media go to the docs only on request (NX_MEDIA=1); an ordinary run keeps them with its own results
const OUT = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/today") : path.resolve(__dirname, "../../../test-results/journey-media/today");
test.use({ storageState: { cookies: [], origins: [] } });
test.describe.configure({ mode: "serial" });

const opening = (page: Page) => page.evaluate(() => document.documentElement.getAttribute("data-nx-opening"));
const scene = (page: Page) => page.evaluate(() => (window as unknown as { __nxChart?: { scene: string | null } }).__nxChart?.scene ?? null);

async function recorded(browser: import("@playwright/test").Browser, name: string, size: { width: number; height: number }, body: (page: Page, ctx: import("@playwright/test").BrowserContext) => Promise<void>) {
  mkdirSync(OUT, { recursive: true });
  const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] }, recordVideo: { dir: path.join(test.info().outputDir, "video"), size } });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  try {
    await body(page, ctx);
  } finally {
    const video = page.video();
    await ctx.close();
    if (video) copyFileSync(await video.path(), path.join(OUT, `${name}.webm`));
  }
  expect(errors, "no page errors (hydration included)").toEqual([]);
}

test("Journey 1, desktop: opening → Find your port → fishery → Today, three taps, then offline", async ({ browser }) => {
  // a recorded journey with an offline step (the saved copy waits for an idle moment): generous on a loaded machine
  test.setTimeout(180_000);
  await recorded(browser, "journey-1-first-visit-1440", { width: 1440, height: 900 }, async (page, ctx) => {
    const res = await page.goto(`${OK}/`, { waitUntil: "commit" });
    const html = await res!.text();
    // the purpose is legible at frame one: the statement is server HTML
    expect(html).toContain("Season status, crab tests, whale measures and the next decision date for your port.");
    await expect(page.getByTestId("nx-still")).toBeVisible();
    expect(["on", "run"]).toContain(await opening(page));
    await expect.poll(() => opening(page), { timeout: 7_000 }).toBeNull();
    const home = page.getByTestId("nx-home");
    await expect(home).toBeVisible();
    // no statewide numbers before a port is chosen
    expect(await home.innerText()).not.toMatch(/\d/);
    await expect(page.locator("[data-nx-sounding]")).toHaveCount(0);
    await expect(home.getByRole("navigation", { name: "Other ways in" }).getByRole("link")).toHaveCount(3);
    await page.waitForTimeout(600);

    // tap 1: Find your port; type, as a stranger would
    await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
    const finder = page.getByTestId("nx-home-finder");
    await expect(finder.getByRole("combobox")).toBeFocused();
    await finder.getByRole("combobox").pressSequentially("moss", { delay: 90 });
    await page.waitForTimeout(300);
    // the typed name is the best match, selected: one ranked list, no region groups
    await expect(finder.getByRole("option").first()).toContainText("Moss Landing");
    await expect(finder.getByRole("option", { selected: true })).toContainText("Moss Landing");
    // tap 2: Enter opens what was typed
    await page.keyboard.press("Enter");
    const chooser = page.getByTestId("nx-fishery-chooser");
    await expect(chooser.getByRole("heading", { level: 1, name: "Moss Landing" })).toBeVisible();
    await expect(chooser).toContainText("Suggested: Neritic has records for these here");
    // none preselected: nothing is pressed, Today has not assembled
    await expect(page.getByRole("heading", { level: 1, name: /Season delayed/ })).toHaveCount(0);
    await page.waitForTimeout(700);
    // tap 3: the fishery
    await chooser.getByRole("button", { name: "Dungeness crab (commercial)" }).click();

    // Today assembles in place at `/`, the chart pans to the port and the soundings register
    await expect(page).toHaveURL(`${OK}/`);
    const today = page.getByTestId("nx-today");
    await expect(today.getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027");
    await expect(today).toContainText("not verified");
    await expect(today).toContainText("Dec 15–18");
    await expect.poll(() => scene(page)).toBe("today:port-592");
    // the readings register at their points: the buoy under Sea now (on), the CDPH test under Toxin tests
    await expect(page.locator("[data-nx-sounding=buoy]")).toBeVisible({ timeout: 10_000 });
    await page.locator('[data-testid="nx-layers"] [data-layer="toxin"]').click();
    await expect(page.locator("[data-nx-sounding=test]")).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("[data-nx-sounding=test]")).toContainText("of 30 ppm");
    const rec = await page.evaluate(() => JSON.parse(localStorage.getItem("nx-port-v1") ?? "{}"));
    expect(rec.lastPort).toBe("port-592");
    expect(rec.fisheries).toEqual(["dungeness_commercial"]);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, "journey-1-today-1440x900.png") });

    // offline after the first visit: the copy this device kept answers `/`
    await expect.poll(() => page.evaluate(() => localStorage.getItem("nx-today-saved-v1") ?? ""), { timeout: 25_000 }).toContain("port-592");
    await ctx.setOffline(true);
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-today").getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027", { timeout: 15_000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUT, "journey-1-offline-1440x900.png") });
    await ctx.setOffline(false);
  });
});

test("Journey 1, phone: the same three taps; the sheet rests on the status and the date", async ({ browser }) => {
  await recorded(browser, "journey-1-first-visit-390", { width: 390, height: 844 }, async (page) => {
    await page.goto(`${OK}/`);
    await expect.poll(() => opening(page), { timeout: 7_000 }).toBeNull();
    await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
    await page.getByTestId("nx-home-finder").getByRole("combobox").pressSequentially("moss", { delay: 90 });
    await page.keyboard.press("Enter");
    await page.getByTestId("nx-fishery-chooser").getByRole("button", { name: "Dungeness crab (commercial)" }).click();
    const h1 = page.getByTestId("nx-today").getByRole("heading", { level: 1 });
    await expect(h1).toHaveText("Season delayed to at least Jan 1, 2027");
    // at rest, the status and the date are inside the visible sheet, above the tab bar
    const tabs = await page.getByTestId("nx-tabs").boundingBox();
    // v14: the ticket's Status carries the next date at rest, above the tab bar
    const date = await page.getByTestId("nx-decision").boundingBox();
    expect(date!.y + date!.height).toBeLessThanOrEqual(tabs!.y);
    // a phone keeps the buoy's reading on the strip; the test is in the ticket's Status
    await expect(page.getByTestId("nx-test-cell")).toContainText("of 30 ppm");
    await page.waitForTimeout(1500);
  });
});

test("Find your port ranks by name: eureka → Eureka, zzz → no match said in words", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
  await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
  const finder = page.getByTestId("nx-home-finder");
  const input = finder.getByRole("combobox");
  await input.pressSequentially("zzz", { delay: 40 });
  await expect(finder.getByRole("option")).toHaveCount(0);
  await expect(finder.getByTestId("nx-port-finder-empty")).toContainText("No CDFW landing port matches “zzz”");
  // Enter with no match opens nothing
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("nx-fishery-chooser")).toHaveCount(0);
  await input.fill("");
  await input.pressSequentially("eureka", { delay: 40 });
  await expect(finder.getByTestId("nx-port-finder-empty")).toHaveCount(0);
  await expect(finder.getByRole("option", { selected: true })).toContainText("Eureka");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("nx-fishery-chooser").getByRole("heading", { level: 1, name: "Eureka" })).toBeVisible();
  await ctx.close();
});

test("Show me around: four steps, one sentence each, gone for good", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
  await ctx.addInitScript(() => {
    if (sessionStorage.getItem("__s")) return;
    sessionStorage.setItem("__s", "1");
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-port-v1", JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null }));
  });
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await expect(page.locator("[data-nx-sounding=buoy]")).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "Show me around", exact: true }).click();
  const tour = page.getByTestId("nx-tour");
  for (const [i, words] of [
    [1, /season status/],
    [2, /where they were measured/],
    [3, /What decides a trip, in parts/],
    [4, /source, the date/],
  ] as const) {
    await expect(tour).toContainText(`${i} of 4`);
    await expect(tour).toContainText(words);
    if (i === 3) await page.screenshot({ path: path.join(OUT, "tour-step-3-1440x900.png") });
    await tour.getByRole("button", { name: i < 4 ? "Next" : "Done" }).click();
  }
  await expect(tour).toHaveCount(0);
  await page.reload();
  await expect(page.getByTestId("nx-today").getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByRole("button", { name: "Show me around", exact: true })).toHaveCount(0);
  await ctx.close();
});
