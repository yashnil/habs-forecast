import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Journey: planning a trip on My Coast (docs/coastwatch/v12/design-system/journeys.md, "Planning a trip").
 * Gates with their sources, the sea window chosen by the reader, then the worksheet from the reader's own
 * numbers: nothing filled in, "waiting for …" while blank, who pays fuel chosen explicitly, the waterfall
 * previewing a grid scenario and returning, a trip saved on this device and reopened.
 * Run with tests/e2e/journeys/coast.config.ts. NX_MEDIA=1 writes the video and stills for review.
 */
const OUT = path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/coast");
const MEDIA = !!process.env.NX_MEDIA;

async function seed(ctx: BrowserContext, mode = "day") {
  await ctx.addInitScript((m) => {
    if (sessionStorage.getItem("seeded")) return;
    sessionStorage.setItem("seeded", "1");
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-chart-mode-v1", m);
    localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
  }, mode);
}

const net = (p: Page) => p.getByTestId("ledger-net");
const gross = (p: Page) => p.getByTestId("ledger-gross");

async function fill(p: Page, slow = 0) {
  const type = async (label: string, v: string) => {
    const f = p.getByLabel(label, { exact: true });
    await f.click();
    if (slow) await f.pressSequentially(v, { delay: slow });
    else await f.fill(v);
  };
  await type("Catch", "800");
  await type("Price", "4.75");
  await type("Fuel", "120");
  await type("Fuel price", "4.60");
  await type("Bait and ice", "180");
  await type("Fees", "5");
  await type("Crew", "2");
  await p.getByRole("radio", { name: "Share after costs" }).click();
  await type("Crew share", "30");
}

async function open(p: Page) {
  await p.goto(`${OK}/coast?place=port-592`);
  await expect(p.getByTestId("coast-worksheet")).toBeVisible();
  await expect(p.getByText(/delayed to at least Jan 1, 2027/)).toBeVisible({ timeout: 15_000 });
}

test("a trip from Moss Landing: gates, the sea, the worksheet, a scenario, saved and reopened", async ({ browser }) => {
  const dir = path.join(test.info().outputDir, "video");
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, recordVideo: { dir, size: { width: 1440, height: 900 } } });
  await seed(ctx);
  const p = await ctx.newPage();
  await open(p);
  const scroll = (to: string) => p.locator(to).first().evaluate((el) => el.scrollIntoView({ block: "start", behavior: "smooth" }));

  // gates: CDFW's season and whale measures, CDPH's test, each with source, date and "not verified"
  const gates = p.getByRole("list", { name: "Gates for Dungeness crab" });
  await expect(gates).toContainText("Zone 4, Pigeon Point to Lopez Point");
  await expect(gates).toContainText("Next assessment Dec 15–18, then Jan 13–15, subject to change");
  await expect(gates).toContainText("Monterey Bay Dungeness crab, Sep 28: highest 9.8 ppm");
  await expect(gates.getByRole("button", { name: /Official · CDFW · checked .* · not verified/ })).toBeVisible();
  await expect(gates.getByRole("button", { name: /Measured · CDPH · sampled Sep 28/ })).toBeVisible();
  await p.waitForTimeout(MEDIA ? 1200 : 0);

  // the sea window: the reader chooses the day; NWS's own sentence opens in place
  await scroll("#coast-sea");
  await expect(p.getByRole("radiogroup", { name: "Day to read" })).toBeVisible({ timeout: 15_000 });
  const wed = p.getByTestId("sea-day-2026-10-14");
  await wed.click();
  await expect(wed).toHaveAttribute("aria-checked", "true");
  const dayRow = p.getByRole("button", { name: /^Day/ });
  await expect(dayRow).toContainText("NE 5–10 kt");
  await dayRow.click();
  await expect(p.getByText("NE wind 5 to 10 kt, backing to NW in the afternoon. Seas around 3 ft.", { exact: false })).toBeVisible();
  await expect(p.getByText(/Tide/).first()).toBeVisible();
  await p.waitForTimeout(MEDIA ? 1000 : 0);

  // the worksheet: nothing filled in, every result waiting and saying for what
  await scroll("#coast-sheet");
  for (const label of ["Catch", "Price", "Fuel", "Fuel price", "Bait and ice", "Fees", "Crew"]) await expect(p.getByLabel(label, { exact: true })).toHaveValue("");
  await expect(gross(p)).toContainText("waiting for catch and price");
  await expect(net(p)).toContainText("waiting for");
  await expect(p.getByRole("button", { name: /Use \$3\.51\/lb, a 2024 statewide average \(CDFW\)/ })).toBeVisible();
  await fill(p, MEDIA ? 70 : 0);
  await expect(gross(p)).toContainText("$3,800", { timeout: 5000 });
  // who pays fuel has no default: net waits for it
  await expect(p.getByRole("radio", { name: "Shared, off the top" })).toHaveAttribute("aria-checked", "false");
  await expect(p.getByRole("radio", { name: "The boat" })).toHaveAttribute("aria-checked", "false");
  await expect(net(p)).toContainText("waiting for who pays fuel");
  await p.getByRole("radio", { name: "Shared, off the top" }).click();
  await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
  await expect(p.getByText("Hypothetical, from your numbers.")).toBeVisible();
  await expect(p.locator("dl").getByText("$0.96/lb")).toBeVisible();
  await expect(p.locator("dl").getByText("162 lb")).toBeVisible();

  // the grid previews a scenario in the waterfall, then returns to the reader's numbers
  await scroll("[data-testid=ledger-gross]");
  await p.waitForTimeout(MEDIA ? 900 : 300);
  const cell = p.getByRole("button", { name: /^\+30 % catch, \+20 % price: \$3,430$/ });
  await cell.hover();
  await expect(p.getByText("Showing catch +30 %, price +20 %.")).toBeVisible();
  await expect(gross(p)).toContainText("$5,928", { timeout: 5000 });
  await expect(net(p)).toContainText("$3,430", { timeout: 5000 });
  await p.waitForTimeout(MEDIA ? 900 : 0);
  await p.getByRole("button", { name: /^−30 % catch, −20 % price/ }).hover();
  await expect(net(p)).toContainText("$903", { timeout: 5000 });
  await p.waitForTimeout(MEDIA ? 900 : 0);
  await p.getByRole("heading", { name: "Worksheet" }).hover();
  await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
  await expect(gross(p)).toContainText("$3,800");

  // saved on this device, reopened after a reload
  await p.getByRole("button", { name: "Save this trip" }).click();
  await expect(p.getByText("Saved on this device.")).toBeVisible();
  await p.reload();
  await expect(p.getByTestId("coast-worksheet")).toBeVisible();
  await expect(p.getByLabel("Catch", { exact: true })).toHaveValue("");
  const saved = p.getByTestId("coast-saved-trip");
  await expect(saved).toContainText("Dungeness crab, 800 lb at $4.75/lb");
  await saved.scrollIntoViewIfNeeded();
  await p.waitForTimeout(MEDIA ? 700 : 0);
  await p.getByRole("button", { name: "Open Dungeness crab, 800 lb at $4.75/lb" }).click();
  await expect(p.getByLabel("Catch", { exact: true })).toHaveValue("800");
  await expect(p.getByRole("radio", { name: "Shared, off the top" })).toHaveAttribute("aria-checked", "true");
  await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
  await p.waitForTimeout(MEDIA ? 1200 : 0);
  const video = p.video();
  await ctx.close();
  if (MEDIA && video) {
    mkdirSync(OUT, { recursive: true });
    copyFileSync(await video.path(), path.join(OUT, "journey-1440.webm"));
  }
});

test("on a phone: the gates rest in the sheet; one thumb fills the worksheet; a tap previews a scenario and a second tap returns", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  await seed(ctx);
  const p = await ctx.newPage();
  await open(p);
  // at rest the sheet shows the season line
  const season = p.getByText(/delayed to at least Jan 1, 2027/);
  const box = await season.boundingBox();
  expect(box!.y + box!.height).toBeLessThan(844 - 56);
  // numeric keyboards and 48 px targets
  await expect(p.getByLabel("Catch", { exact: true })).toHaveAttribute("inputmode", "decimal");
  await expect(p.getByLabel("Crew", { exact: true })).toHaveAttribute("inputmode", "numeric");
  expect((await p.getByLabel("Catch", { exact: true }).boundingBox())!.height).toBeGreaterThanOrEqual(48);
  for (let i = 0; i < 2; i++) await p.getByRole("button", { name: "Show more of this page" }).tap();
  await fill(p);
  await p.getByRole("radio", { name: "Shared, off the top" }).tap();
  await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
  for (const r of await p.getByRole("radio").all()) if (await r.isVisible()) expect((await r.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  const cell = p.getByRole("button", { name: /^\+30 % catch, \+20 % price/ });
  await cell.scrollIntoViewIfNeeded();
  await cell.tap();
  await expect(net(p)).toContainText("$3,430", { timeout: 5000 });
  await p.waitForTimeout(400);
  await expect(net(p)).toContainText("$3,430");
  await cell.tap();
  await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
  await ctx.close();
});

test("reduced motion: figures and bars cut to their values", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  await seed(ctx);
  const p = await ctx.newPage();
  await open(p);
  await fill(p);
  await p.getByRole("radio", { name: "Shared, off the top" }).click();
  await p.getByRole("button", { name: /^\+30 % catch, \+20 % price/ }).hover();
  // no spring: the new value is there at once
  await expect(net(p)).toContainText("$3,430", { timeout: 300 });
  await ctx.close();
});

test("media: stills at the widths and modes the page is used at", async ({ browser }) => {
  test.skip(!MEDIA, "media only on request (NX_MEDIA=1)");
  test.setTimeout(240_000);
  mkdirSync(OUT, { recursive: true });
  const sizes: Array<[number, number]> = [
    [1440, 900],
    [390, 844],
    [320, 640],
    [844, 390],
  ];
  const shoot = async (w: number, h: number, mode: string, filled: boolean, name: string, scrollTo?: string, hover = false) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: w < 721, deviceScaleFactor: 1 });
    await seed(ctx, mode);
    const p = await ctx.newPage();
    await open(p);
    await p.waitForTimeout(1200);
    if (filled) {
      await fill(p);
      await p.getByRole("radio", { name: "Shared, off the top" }).click();
      await expect(net(p)).toContainText("$2,015", { timeout: 5000 });
    }
    if (scrollTo) {
      if (w < 721) for (let i = 0; i < 2; i++) await p.getByRole("button", { name: "Show more of this page" }).click();
      await p.locator(scrollTo).first().evaluate((el) => el.scrollIntoView({ block: "start" }));
    }
    if (hover) await p.getByRole("button", { name: /^\+30 % catch, \+20 % price/ }).hover();
    await p.waitForTimeout(900);
    await p.screenshot({ path: path.join(OUT, `${name}.png`) });
    await ctx.close();
  };
  for (const [w, h] of sizes) {
    await shoot(w, h, "day", false, `coast-empty-${w}x${h}`);
    await shoot(w, h, "day", true, `coast-worksheet-${w}x${h}`, "#coast-sheet");
    await shoot(w, h, "day", true, `coast-results-${w}x${h}`, "[data-testid=ledger-gross]");
  }
  await shoot(1440, 900, "day", true, "coast-scenario-1440x900", "[data-testid=ledger-gross]", true);
  await shoot(1440, 900, "day", false, "coast-sea-1440x900", "#coast-sea");
  for (const mode of ["dusk", "night"]) {
    await shoot(1440, 900, mode, false, `coast-empty-${mode}-1440x900`);
    await shoot(1440, 900, mode, true, `coast-results-${mode}-1440x900`, "[data-testid=ledger-gross]");
    await shoot(390, 844, mode, true, `coast-results-${mode}-390x844`, "[data-testid=ledger-gross]");
  }
});
