import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { exact, field, inView, MOSS, savedCount, seed } from "../v13/trip";

/**
 * Journey: planning a trip on My Coast (docs/coastwatch/v12/design-system/journeys.md, "Planning a trip"; the
 * v13 desk page and trip workspace). The brief: gates with their sources, the sea window chosen by the reader.
 * Then the money (`?view=trip`): the reader's own numbers, nothing filled in, "waiting for …" while blank, who
 * pays chosen explicitly, a grid scenario shown in the result and returned from, a trip saved on this device
 * and still there, still open, after a reload. The season line stays in view the whole way.
 * Run with tests/e2e/journeys/coast.config.ts. NX_MEDIA=1 writes the video and stills for review.
 */
// review media go to the docs only on request (NX_MEDIA=1); an ordinary run keeps them with its own results
const OUT = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v13/implementation/coast") : path.resolve(__dirname, "../../../test-results/journey-media/coast");
const MEDIA = !!process.env.NX_MEDIA;

/** 800 lb at $4.75, fees 5 %, 120 gal at $4.60, $180 bait and ice, 2 crew on 30 % after costs. */
async function fill(p: Page, slow = 0) {
  const put = async (label: string, v: string) => {
    const f = field(p, label);
    await f.scrollIntoViewIfNeeded();
    await f.click();
    if (slow) await f.pressSequentially(v, { delay: slow });
    else await f.fill(v);
  };
  await put("Catch", "800");
  await put("Price", "4.75");
  await put("Fuel", "120");
  await put("Fuel price", "4.60");
  await put("Bait and ice", "180");
  await put("Fees", "5");
  await put("Crew", "2");
  await p.getByRole("radio", { name: "After costs" }).click();
  await put("Crew share", "30");
}
const payers = async (p: Page) => {
  await p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await p.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
};

const status = (p: Page) => p.getByTestId("coast-status");
async function open(p: Page, query = "") {
  await p.goto(`${OK}/coast?place=${MOSS}${query}`);
  await expect(p.getByTestId("nx-coast")).toBeVisible();
  await expect(status(p)).toContainText(/delayed to at least Jan 1, 2027/, { timeout: 15_000 });
}

test("a trip from Moss Landing: gates, the sea, the money, a scenario, saved and still open after a reload", async ({ browser }) => {
  const dir = path.join(test.info().outputDir, "video");
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, recordVideo: { dir, size: { width: 1440, height: 900 } } });
  await seed(ctx);
  const p = await ctx.newPage();
  await open(p);
  const scroll = (to: string) => p.locator(to).first().evaluate((el) => el.scrollIntoView({ block: "start", behavior: "smooth" }));
  /** the official season line, pinned in the header: in view wherever the page has been scrolled to */
  const seasonInView = async (where: string) => expect(await inView(p, status(p)), `season line in view ${where}`).toBe(true);
  await seasonInView("on opening");

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
  await seasonInView("at the sea window");
  await p.waitForTimeout(MEDIA ? 1000 : 0);

  // the money: its own state of the page, one press away, and Back returns to the brief
  await p.getByTestId("coast-to-trip-2").click();
  await expect(p).toHaveURL(/view=trip/);
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await p.goBack();
  await expect(p.getByRole("heading", { name: "Sea window" })).toBeVisible();
  await p.goForward();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(p.locator("[data-nx-desk-header]")).toContainText("Hypothetical trip · Commercial");

  // nothing filled in, every result waiting and saying for what
  for (const label of ["Catch", "Price", "Fuel", "Fuel price", "Bait and ice", "Fees", "Crew"]) await expect(field(p, label)).toHaveValue("");
  await expect(p.getByTestId("trip-gross")).toContainText("waiting for catch and price");
  await expect(p.getByTestId("trip-net")).toContainText("waiting for");
  await expect(p.getByRole("button", { name: /Use \$3\.51\/lb, a 2024 statewide average \(CDFW\)/ })).toBeVisible();
  await fill(p, MEDIA ? 70 : 0);
  await expect(exact(p, "trip-gross")).toHaveText("$3,800.00");
  // who pays has no default: the groups look unset and net waits for them
  await expect(p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" })).toHaveAttribute("aria-checked", "false");
  await expect(p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Boat" })).toHaveAttribute("aria-checked", "false");
  await expect(p.getByTestId("trip-net")).toContainText("waiting for who pays fuel and who pays bait and ice");
  await payers(p);
  // R = 3800, fees 190, D = 552 + 180 = 732, N = 2878, crew 863.40, net 2014.60
  await expect(exact(p, "trip-net")).toHaveText("$2,014.60");
  await expect(exact(p, "trip-crew")).toHaveText("$863.40");
  await expect(p.getByTestId("trip-crew")).toContainText("$431.70 each of 2");
  await expect(p.getByText("Hypothetical, from your numbers.")).toBeVisible();
  // 732 / (800 × 0.95) = $0.9632 → $0.97; 732 / (4.75 × 0.95) = 162.2 → 163 lb: rounded up
  await expect(exact(p, "trip-be-price")).toHaveText("$0.97/lb");
  await expect(exact(p, "trip-be-catch")).toHaveText("163 lb");
  await seasonInView("with the trip filled in");

  // the grid shows a scenario in the result and the waterfall, then returns to the reader's numbers
  await p.getByTestId("sensitivity-grid").scrollIntoViewIfNeeded();
  await p.waitForTimeout(MEDIA ? 900 : 300);
  await p.getByTestId("sens-0-2").hover();
  await expect(p.getByText("Showing catch +30 %, price +20 %.")).toBeVisible();
  await expect(exact(p, "trip-gross")).toHaveText("$5,928.00");
  // R = 5928, fees 296.40, N = 4899.60, net 70 % of it
  await expect(exact(p, "trip-net")).toHaveText("$3,429.72");
  expect(await inView(p, p.getByTestId("trip-net"))).toBe(true);
  await p.waitForTimeout(MEDIA ? 900 : 0);
  await p.getByTestId("sens-2-0").hover();
  await expect(exact(p, "trip-net")).toHaveText("$902.72");
  await p.waitForTimeout(MEDIA ? 900 : 0);
  await p.getByRole("heading", { name: "If things move" }).hover();
  await expect(exact(p, "trip-net")).toHaveText("$2,014.60");
  await expect(exact(p, "trip-gross")).toHaveText("$3,800.00");
  await seasonInView("at the sensitivity grid");

  // saved on this device; after a reload the same trip is still on the worksheet and still the one Update changes
  await p.getByTestId("trip-save-new").click();
  await expect(p.getByTestId("trip-said")).toContainText("Saved as “Dungeness crab, 800 lb at $4.75, 2 crew 30 % after costs,");
  await p.reload();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(field(p, "Catch")).toHaveValue("800");
  await expect(p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" })).toHaveAttribute("aria-checked", "true");
  await expect(exact(p, "trip-net")).toHaveText("$2,014.60");
  await expect(p.getByTestId("trip-state")).toContainText(/^Saved /);
  await p.getByTestId("trip-clear").click();
  await expect(field(p, "Catch")).toHaveValue("");
  await p.getByTestId("trip-saved-open").click();
  const row = p.getByRole("dialog", { name: "Saved trips" }).getByTestId("trip-saved-row");
  await expect(row).toContainText("Dungeness crab, 800 lb at $4.75, 2 crew 30 % after costs");
  await p.waitForTimeout(MEDIA ? 700 : 0);
  await row.getByRole("button", { name: /^Open / }).click();
  await expect(field(p, "Catch")).toHaveValue("800");
  await expect(exact(p, "trip-net")).toHaveText("$2,014.60");
  expect(await savedCount(p)).toBe(1);
  await p.waitForTimeout(MEDIA ? 1200 : 0);
  const video = p.video();
  await ctx.close();
  if (MEDIA && video) {
    mkdirSync(OUT, { recursive: true });
    copyFileSync(await video.path(), path.join(OUT, "journey-1440.webm"));
  }
});

test("on a phone: the page scrolls under the chart strip; the season line and the result stay in reach; a tap holds a scenario and a second tap returns", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  await seed(ctx);
  const p = await ctx.newPage();
  await open(p, "&view=trip");
  // v13: My Coast is a desk page. On a phone it is the page itself under a chart strip (no sheet to raise)
  await expect(p.getByRole("button", { name: "Show more of this page" })).toBeHidden();
  // the season line is whole, in view and clear of both bars, before and after the page is scrolled
  const whole = async (where: string) => {
    const box = (await status(p).boundingBox())!;
    expect(box.y, where).toBeGreaterThanOrEqual(52);
    expect(box.y + box.height, where).toBeLessThanOrEqual(844 - 56);
    expect(box.x + box.width, where).toBeLessThanOrEqual(390);
  };
  await status(p).scrollIntoViewIfNeeded();
  await whole("on opening");
  // numeric keyboards and thumb-sized targets
  await expect(field(p, "Price")).toHaveAttribute("inputmode", "decimal");
  await expect(field(p, "Crew")).toHaveAttribute("inputmode", "numeric");
  expect((await field(p, "Catch").boundingBox())!.height).toBeGreaterThanOrEqual(44);
  await fill(p);
  await payers(p);
  const bar = p.getByTestId("trip-netbar");
  await expect(bar).toContainText("$2,014.60");
  await expect(bar).toContainText("Break-even 163 lb");
  await whole("with the inputs filled");
  for (const r of await p.getByRole("radio").all()) if (await r.isVisible()) expect((await r.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  const cell = p.getByTestId("sens-0-2");
  await cell.scrollIntoViewIfNeeded();
  await cell.tap();
  await expect(bar).toContainText("$3,429.72");
  await expect(bar).toContainText("With catch +30 %, price +20 %");
  await p.waitForTimeout(400);
  await expect(bar).toContainText("$3,429.72");
  await whole("at the grid");
  await cell.tap();
  await expect(bar).toContainText("$2,014.60");
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
  ];
  const shoot = async (w: number, h: number, mode: string, view: "brief" | "trip", filled: boolean, name: string, scrollTo?: string, hover = false) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: w < 721, deviceScaleFactor: 1 });
    await seed(ctx, { mode });
    const p = await ctx.newPage();
    await open(p, view === "trip" ? "&view=trip" : "");
    await p.waitForTimeout(1200);
    if (filled) {
      await fill(p);
      await payers(p);
      await expect(exact(p, "trip-net")).toHaveText("$2,014.60");
    }
    if (scrollTo) await p.locator(scrollTo).first().evaluate((el) => el.scrollIntoView({ block: "start" }));
    if (hover) await p.getByTestId("sens-0-2").hover();
    await p.waitForTimeout(900);
    await p.screenshot({ path: path.join(OUT, `${name}.png`) });
    await ctx.close();
  };
  for (const [w, h] of sizes) {
    await shoot(w, h, "day", "brief", false, `coast-brief-${w}x${h}`);
    await shoot(w, h, "day", "trip", false, `coast-trip-empty-${w}x${h}`);
    await shoot(w, h, "day", "trip", true, `coast-trip-inputs-${w}x${h}`);
    await shoot(w, h, "day", "trip", true, `coast-trip-results-${w}x${h}`, "[data-testid=trip-results]");
  }
  await shoot(1440, 900, "day", "trip", true, "coast-trip-scenario-1440x900", "[data-testid=sensitivity-grid]", true);
  for (const mode of ["dusk", "night"]) {
    await shoot(1440, 900, mode, "trip", true, `coast-trip-results-${mode}-1440x900`, "[data-testid=trip-results]");
    await shoot(390, 844, mode, "trip", true, `coast-trip-results-${mode}-390x844`, "[data-testid=trip-results]");
  }
});
