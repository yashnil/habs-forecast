import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Today as the three gates (v13 lead decision 6; audit-C acceptance tests 1–8), in the v14 synthesis layout: the
 * ticket's Status tab says why, the next date (once) and the domoic-acid test; its Record tab holds the three gates in
 * detail; the sea dock holds the NWS advisories; readings sit on the chart under their layer chips (the CDPH test
 * under "Toxin tests", off by default on Today). A returning Moss Landing
 * commercial Dungeness reader, on the fixture; the sea and the page clock are changed per test by answering the
 * page's own two reads (/today/data, /api/conditions), never by adding a fact to the product.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const RECORD = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const SIZES = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 }, narrow: { width: 320, height: 640 } };

async function reader(ctx: BrowserContext) {
  await ctx.addInitScript((rec) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    localStorage.setItem("nx-port-v1", rec);
    localStorage.setItem("nx-tour-v1", "dismissed");
    localStorage.setItem("nx-opening-seen-v1", "1");
  }, RECORD);
}

/** An NWS advisory added to the recorded sea, as NWS words one. */
async function withAdvisory(page: Page) {
  await page.route("**/api/conditions**", async (route) => {
    const res = await route.fetch();
    const body = (await res.json()) as { alerts?: { status: string; data: unknown[] } };
    if (body.alerts?.status === "ok") body.alerts.data = [{ event: "Small Craft Advisory", headline: "Small Craft Advisory in effect from Thursday morning through Friday morning", url: "https://www.weather.gov/mtr/", ends: "2026-10-14T16:00:00Z", severity: "Minor" }];
    await route.fulfill({ response: res, json: body });
  });
}

/** The same published records read a month later: the Sep 28 test is past its 30-day limit. */
async function monthLater(page: Page) {
  await page.route("**/today/data**", async (route) => {
    const res = await route.fetch();
    const body = (await res.json()) as { inputs: { now: string } };
    body.inputs.now = "2026-11-12T18:00:00Z";
    await route.fulfill({ response: res, json: body });
  });
}

const gates = (page: Page) => page.getByTestId("gate-list");
/** the ticket's Record tab: the source record and the three gates in detail */
const record = (page: Page) => page.getByTestId("nx-today").getByRole("tab", { name: "Record" }).click();
/** the "Toxin tests" chip: the CDPH test's figure on the chart */
const toxinLayer = (page: Page) => page.locator('[data-testid="nx-layers"] [data-layer="toxin"]').click();
const inViewport = async (page: Page, testid: string) => {
  const b = await page.getByTestId(testid).first().boundingBox();
  const v = page.viewportSize()!;
  return !!b && b.y >= 0 && b.y + b.height <= v.height && b.x >= 0 && b.x + b.width <= v.width;
};

test("the gates, in order: whale measures with the next assessment, then domoic acid, then the sea", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: SIZES.desktop, storageState: { cookies: [], origins: [] } });
  await reader(ctx);
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await expect(page.getByTestId("nx-today").getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027");
  // the next date that can change it is said once on the page (the ticket's Status), above the detail
  const today = page.getByTestId("nx-today");
  await expect(today).toContainText("Next whale assessment, subject to change");
  await expect(today).toContainText("Dec 15–18");
  expect((await today.innerText()).match(/Next whale assessment/g)).toHaveLength(1);
  await record(page);
  await expect(gates(page)).toBeVisible();
  expect(await gates(page).locator("[data-gate]").evaluateAll((els) => els.map((e) => e.getAttribute("data-gate")))).toEqual(["whale", "domoic", "sea"]);
  const whale = page.getByTestId("gate-whale");
  await expect(whale).toContainText("Whale measures");
  await expect(whale).toContainText("CDFW RAMP");
  await expect(whale).toContainText("Zone 4: Fleet Advisory");
  await expect(whale).toContainText("This is what delays the season: a whale-entanglement decision by CDFW, not a toxin result.");
  await expect(whale).toContainText("CDFW, for commercial crabbers: “Commercial season delayed");
  await expect(whale).not.toContainText(/recreational/i);
  const domoic = page.getByTestId("gate-domoic");
  await expect(domoic).toContainText("Domoic acid");
  await expect(domoic).toContainText("CDPH tests");
  await expect(domoic).toContainText("Dungeness test, Monterey Bay, Sep 28: highest 9.8 of 30 ppm, 6 samples");
  await expect(domoic).toContainText("A test result, at any level, does not lift the whale delay");
  await expect(page.getByTestId("gate-sea")).toContainText("NWS and NDBC");
  await expect(page.getByTestId("gate-sea")).toContainText("The sea and the season are separate");
  // the fresh test (14 days) is still a figure on the chart, under Toxin tests, with the buoy under Sea now
  await expect(page.locator("[data-nx-sounding=buoy]")).toBeVisible({ timeout: 10_000 });
  await toxinLayer(page);
  await expect(page.locator("[data-nx-sounding=test]")).toBeVisible({ timeout: 10_000 });
  // small words (audit T-4, T-6)
  const text = await page.getByTestId("nx-today").innerText();
  expect(text).not.toMatch(/NWS (mon|tue|wed|thu|fri|sat|sun)\b/);
  expect(text).not.toMatch(/\b[NSEW]{1,3} 0 kt/);
  expect(text).not.toMatch(/^Dungeness crab: /m);
  expect(text).not.toMatch(/\b(safe|all clear|good to fish|recommended)\b/i);
  await ctx.close();
});

test("a test older than 30 days is history: off the chart, worded as old in its gate, and it never lifts the whale delay", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: SIZES.desktop, storageState: { cookies: [], origins: [] } });
  await reader(ctx);
  const page = await ctx.newPage();
  await monthLater(page);
  await page.goto(`${OK}/`);
  // the ticket's Status says it is old (half-filled mark and the words)
  await expect(page.getByTestId("nx-test-cell")).toHaveAttribute("data-state", "old");
  await expect(page.getByTestId("nx-test-cell")).toContainText("(old)");
  await record(page);
  const domoic = page.getByTestId("gate-domoic");
  await expect(domoic).toHaveAttribute("data-state", "old");
  await expect(domoic.getByTestId("nx-gate-domoic-headline")).toContainText("highest 9.8 of 30 ppm");
  await expect(domoic).toContainText("old");
  await expect(domoic).toContainText("This test is older than 30 days: it is history, not a reading of the water now.");
  await expect(domoic).toContainText("does not lift the whale delay");
  // the buoy registers on the chart; the old test never does, not even with Toxin tests on
  await expect(page.locator("[data-nx-sounding=buoy]")).toBeVisible({ timeout: 10_000 });
  await toxinLayer(page);
  await page.waitForTimeout(600);
  await expect(page.locator("[data-nx-sounding=test]")).toHaveCount(0);
  // the port's other test site is in the gate too, old as well
  await expect(domoic).toContainText(/rock crab.*highest 20 ppm of 30/);
  await ctx.close();
});

for (const [name, size] of Object.entries({ desktop: SIZES.desktop, phone: SIZES.phone })) {
  test(`an NWS advisory is in the first screen at ${size.width} px (the sea dock; the ticket's strip on a phone), stamped with NWS and its issue time`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] } });
    await reader(ctx);
    const page = await ctx.newPage();
    await withAdvisory(page);
    await page.goto(`${OK}/`);
    const adv = page.getByTestId("advisories");
    await expect(adv).toHaveCount(1);
    await expect(adv).toContainText("Small Craft Advisory");
    await expect(adv).toContainText("in effect from Thursday morning through Friday morning");
    await expect(adv).toContainText(/Advisory · NWS · issued/i);
    await expect(adv).not.toContainText("not verified");
    await page.waitForTimeout(600);
    // in the first screen at every width: the dock along the chart's foot on a desktop, the ticket's strip under the
    // status on a phone (above the tabs)
    expect(await inViewport(page, "advisories"), `advisory block inside the first viewport at ${name}`).toBe(true);
    // and its span over the forecast bars
    await expect(page.getByTestId("nx-sea-dock")).toContainText(/Small craft/i);
    // it also leads the sea gate
    await record(page);
    await expect(page.getByTestId("nx-gate-sea-headline")).toHaveAttribute("data-role", "advisory");
    await ctx.close();
  });
}

for (const size of [SIZES.phone, SIZES.narrow]) {
  test(`at ${size.width} px the measured waves are in the sea gate (the chart has room for little)`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] } });
    await reader(ctx);
    const page = await ctx.newPage();
    await page.goto(`${OK}/`);
    await record(page);
    await expect(page.getByTestId("gate-sea")).toContainText(/Waves \d+\.\d ft at \d+ s at buoy 46236/, { timeout: 15_000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await ctx.close();
  });
}

test("each gate's next steps keep the port and are never dead ends; a line of fact opens where it comes from", async ({ browser }) => {
  test.setTimeout(120_000);
  const ctx = await browser.newContext({ viewport: SIZES.desktop, storageState: { cookies: [], origins: [] } });
  await reader(ctx);
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await record(page);
  await expect(page.getByTestId("gate-sea")).toContainText("Waves", { timeout: 15_000 });
  const hrefs = await gates(page).locator("a[href]").evaluateAll((as) => as.map((a) => a.getAttribute("href")!));
  const inApp = hrefs.filter((h) => h.startsWith("/"));
  expect(inApp).toEqual(expect.arrayContaining(["/alerts?place=port-592", "/research?place=port-592&var=pDA", "/explore?place=port-592&layer=sst", "/coast?place=port-592&view=trip"]));
  for (const h of inApp) {
    expect(h).toContain("place=port-592");
    const r = await page.request.get(`${OK}${h}`);
    expect(r.status(), h).toBeLessThan(400);
  }
  expect(hrefs.some((h) => /^https:\/\/wildlife\.ca\.gov\//.test(h))).toBe(true);
  // a fact line opens its provenance
  await page.getByTestId("gate-sea").getByRole("button", { name: /NDBC.*Open where this comes from/ }).first().click();
  await expect(page.getByRole("dialog")).toContainText("NDBC");
  await page.keyboard.press("Escape");
  // and the sea gate leads to Explore's conditions with the port kept
  await page.getByTestId("gate-sea").getByRole("link", { name: "Conditions on the chart in Explore" }).click();
  await expect(page).toHaveURL(/\/explore\?place=port-592&layer=sst/);
  await expect(page.getByTestId("nx-legend")).toHaveAttribute("data-product", "sst");
  await ctx.close();
});

test("first visit: what Neritic is, what choosing a port leads to, and the one action", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: SIZES.desktop, storageState: { cookies: [], origins: [] } });
  await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  const home = page.getByTestId("nx-home");
  await expect(home.getByRole("heading", { level: 1 })).toContainText("Season status, crab tests, whale measures and the next decision date for your port.");
  await expect(page.getByTestId("nx-home-next")).toContainText("Choose your port and what you fish.");
  await expect(page.getByTestId("nx-home-next")).toContainText("each with its source and its date");
  await expect(page.getByTestId("nx-find-port")).toBeVisible();
  await expect(page.getByTestId("gate-list")).toHaveCount(0);
  await ctx.close();
});

test("what decides a trip is in the first screen: why and the next date, the domoic-acid test, the sea; all at 1440, the status at 390 at rest", async ({ browser }) => {
  for (const [size, need] of [[SIZES.desktop, 3], [SIZES.phone, 1]] as const) {
    const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] } });
    await reader(ctx);
    const page = await ctx.newPage();
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-sea-dock")).toContainText("NWS PZZ535", { timeout: 15_000 });
    await page.waitForTimeout(800);
    // whale: the delay's cause and the next date it can change, said once; domoic acid: the test and its age; the sea
    const parts = [page.getByTestId("nx-decision"), page.getByTestId("nx-test-cell"), page.getByTestId("nx-sea-dock")];
    await expect(parts[0]).toHaveText(/Next whale assessment, subject to change · Dec 15–18/);
    await expect(page.getByTestId("nx-today")).toContainText("Humpback whale entanglements");
    await expect(parts[1]).toContainText("Dungeness, highest 9.8 of 30 ppm");
    await expect(parts[1]).toContainText("14 days ago");
    await expect(parts[2]).toContainText(/ft · NWS forecast/i);
    for (let i = 0; i < need; i++) await expect(parts[i], `${size.width}: part ${i + 1} in the first screen`).toBeInViewport({ ratio: size.width === 1440 ? 1 : 0.5 });
    await ctx.close();
  }
});

test("nothing jumps when the port's records arrive: the placeholder stands at the page's own geometry", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: SIZES.phone, storageState: { cookies: [], origins: [] } });
  await reader(ctx);
  await ctx.addInitScript(() => {
    const w = window as unknown as { __cls: number };
    w.__cls = 0;
    new PerformanceObserver((list) => {
      for (const e of list.getEntries() as unknown as Array<{ value: number; hadRecentInput: boolean }>) if (!e.hadRecentInput) w.__cls += e.value;
    }).observe({ type: "layout-shift", buffered: true });
  });
  const page = await ctx.newPage();
  // the records arrive late, so the placeholder is on screen first
  await page.route("**/today/data**", async (route) => {
    await new Promise((r) => setTimeout(r, 900));
    await route.continue();
  });
  await page.goto(`${OK}/`);
  await expect(page.getByTestId("skeleton").first()).toBeVisible();
  await expect(page.getByTestId("nx-decision")).toBeVisible({ timeout: 15_000 });
  await page.waitForTimeout(1500);
  expect(await page.evaluate(() => (window as unknown as { __cls: number }).__cls)).toBeLessThan(0.1);
  await ctx.close();
});
