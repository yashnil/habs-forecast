import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { chromium, expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { choosePayers, exact, field, fillOwner, inView, MOSS, openTrip, savedCount, seed, type } from "./trip";

/**
 * The trip workspace on My Coast (`/coast?view=trip`): the acceptance tests of the v13 audit (audit B, §5), held in the
 * v14 synthesis layout (the trip is one pass along the chart's foot: strip, equation, terms, waterfall and what-if in
 * its main part, under its fold the rest of the inputs and the worked figures; the result and the one button in its
 * stub. On a phone the pass is a block in the sheet, with the result and the button in a bar at its foot).
 * The model's arithmetic is held by tests/unit/nx/coast/*; here it is the page: the owner's case to the cent, the
 * context that never scrolls away, a result that stays in view, saved trips without accidental copies, and work
 * that survives leaving, reloading, restarting and losing the network.
 * Run with tests/e2e/journeys/coast.config.ts (one production server, the main fixture).
 */

const SIZES: Array<[number, number]> = [
  [1440, 900],
  [390, 844],
  [320, 640],
];

const page = async (browser: import("@playwright/test").Browser, w = 1440, h = 900, o: { fishery?: string; reducedMotion?: "reduce" } = {}) => {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: w <= 720, isMobile: w <= 720, reducedMotion: o.reducedMotion });
  await seed(ctx, { fishery: o.fishery });
  return { ctx, p: await ctx.newPage() };
};

const said = (p: Page) => p.getByTestId("trip-said");

/**
 * The pass's front on a wide screen: the result, the whole waterfall and the what-if in view, and the pass leaves the
 * chart at least 45 % of the screen's height (v14 design issue 2).
 */
async function firstScreen(p: Page, when: string) {
  await p.getByTestId("trip-pass").locator("[data-testid=trip-inputs]").evaluate((el) => el.closest("[class*=scroll]")?.scrollTo(0, 0));
  await p.waitForTimeout(250);
  const fall = (await p.getByTestId("trip-waterfall").boundingBox())!;
  const vh = p.viewportSize()!.height;
  expect(fall.width, when).toBeGreaterThanOrEqual(400);
  expect(fall.height, when).toBeGreaterThanOrEqual(120);
  expect(await inView(p, p.getByTestId("trip-waterfall")), `the waterfall is whole ${when}`).toBe(true);
  expect(await inView(p, p.getByTestId("trip-whatif")), `the what-if is whole ${when}`).toBe(true);
  expect(await inView(p, p.getByTestId("trip-net")), `the result is in view ${when}`).toBe(true);
  expect((await p.getByTestId("trip-pass").boundingBox())!.y, `the chart keeps 45 % of the height ${when}`).toBeGreaterThanOrEqual(vh * 0.45);
}
const phoneSize = (p: Page) => p.viewportSize()!.width <= 640;
const result = (p: Page) => (phoneSize(p) ? p.getByTestId("trip-netbar-value") : p.getByTestId("trip-net"));
/** The result is in reach: the stub's figure in view, or on a phone the bar at the sheet's foot saying it. */
async function resultInReach(p: Page): Promise<boolean> {
  if (await inView(p, p.getByTestId("trip-net"))) return true;
  return phoneSize(p) && (await inView(p, p.getByTestId("trip-netbar"))) && (await p.getByTestId("trip-netbar").locator("[data-quiet]").count()) === 0;
}

test("the owner's test, to the cent, with port, fishery, 'hypothetical' and the season status in view at every step", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  const context = async (step: string) => {
    for (const [what, l] of [
      ["port", p.getByTestId("coast-title")],
      ["fishery", p.getByTestId("coast-fishery")],
      ["status", p.getByTestId("coast-status")],
      ["hypothetical", p.locator("[data-nx-desk-header]").getByText("Hypothetical trip", { exact: true })],
      ["result", p.getByTestId("trip-net")],
    ] as const)
      expect(await inView(p, l), `${what} in view after ${step}`).toBe(true);
  };
  await expect(p.getByTestId("coast-title")).toHaveText("A trip from Moss Landing");
  await expect(p.getByTestId("coast-fishery")).toHaveValue("dungeness_commercial");
  // nothing is filled in, and nothing reads as 0
  for (const label of ["Catch", "Price", "Fuel", "Fuel price", "Bait and ice", "Fees", "Crew"]) await expect(field(p, label)).toHaveValue("");
  await expect(p.getByTestId("trip-net")).toContainText("waiting for catch, price and 5 more");
  // no figure reads as $0 while it waits (the waterfall's axis has a $0 tick; that is a scale, not a figure)
  for (const id of ["trip-hero", "trip-figures"]) expect(await p.getByTestId(id).innerText()).not.toMatch(/\$0\b/);
  await context("opening");

  const steps: Array<[string, string]> = [
    ["Catch", "400"],
    ["Price", "9"],
    ["Fuel", "30"],
    ["Fuel price", "8.24"],
    ["Bait and ice", "100"],
    ["Fees", "0"],
    ["Crew", "1"],
  ];
  for (const [label, v] of steps) {
    await type(p, label, v);
    await context(label);
  }
  await expect(exact(p, "trip-gross")).toHaveText("$3,600.00");
  await expect(exact(p, "trip-operating")).toHaveText("$347.20");
  // F-03: with some fields still blank the chart says what it waits for once, under it; a bar's place holds one word
  await expect(p.getByTestId("trip-waterfall-waiting")).toHaveText("2 bars are not drawn yet: still waiting for how crew is paid.");
  const labels = await p.getByTestId("trip-waterfall").locator("svg text").allInnerTexts();
  expect(labels.filter((t) => /waiting for/.test(t))).toEqual([]);
  await p.getByRole("radio", { name: "After costs" }).click();
  await type(p, "Crew share", "20");
  await context("Crew share");

  // who pays is not chosen for the reader: both groups look unset, and net says what it waits for
  for (const name of ["Who pays fuel", "Who pays bait and ice"]) {
    const g = p.getByRole("radiogroup", { name });
    await expect(g.getByRole("radio", { name: "Shared" })).toHaveAttribute("aria-checked", "false");
    await expect(g.getByRole("radio", { name: "Boat" })).toHaveAttribute("aria-checked", "false");
  }
  await expect(p.getByText("Who pays fuel · Choose one")).toBeVisible();
  await expect(p.getByTestId("trip-net")).toContainText("waiting for who pays fuel and who pays bait and ice");
  await choosePayers(p);
  await context("who pays");

  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  await expect(exact(p, "trip-gross")).toHaveText("$3,600.00");
  await expect(exact(p, "trip-operating")).toHaveText("$347.20");
  await expect(exact(p, "trip-crew")).toHaveText("$650.56");
  await expect(exact(p, "trip-be-catch")).toHaveText("39 lb");
  await expect(exact(p, "trip-be-price")).toHaveText("$0.87/lb");
  await expect(p.getByTestId("trip-note")).toContainText("Not profit or take-home.");
  // every control reads on one line, with what it means said under it
  await expect(p.getByTestId("trip-crew-words")).toHaveText("A share of what is left after fees and the shared costs.");
  await expect(p.getByTestId("trip-payer-words")).toContainText("Shared: taken off the top, before the crew's share is worked out.");
  for (const r of await p.getByTestId("trip-inputs").getByRole("radio").all()) expect((await r.boundingBox())!.height, await r.innerText()).toBeLessThanOrEqual(50);
  // this trip clears at any fuel price the worksheet takes ($116.67/gal breaks even): no figure is made of that
  await expect(p.getByTestId("trip-be-fuel")).toHaveCount(0);

  // one screen at 1440 × 900: the result, the whole waterfall and the what-if, with the equation above them
  await firstScreen(p, "with the owner's case entered");
  // the inputs are in view and whole
  expect(await inView(p, p.getByRole("region", { name: "Trip inputs" }))).toBe(true);

  // F-02: saving does not grow the pass or push the waterfall off the first screen, and the name stays on one line
  const passBefore = (await p.getByTestId("trip-pass").boundingBox())!;
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as “Dungeness crab, 400 lb at $9, 1 crew 20 % after costs,");
  await firstScreen(p, "after Save as new");
  const passAfter = (await p.getByTestId("trip-pass").boundingBox())!;
  expect(passAfter.height, "the pass is no taller after saving").toBeLessThanOrEqual(passBefore.height + 1);
  expect((await p.getByTestId("trip-name").boundingBox())!.height, "the saved name is on one line (its rename control is 44 px)").toBeLessThanOrEqual(48);
  // the saving controls are in the stub, in view: the confirmed state on the one button, Save as new and Saved beside it
  await expect(p.getByTestId("trip-update")).toContainText("Saved on this device");
  const hero = (await p.getByTestId("trip-hero").boundingBox())!;
  for (const id of ["trip-update", "trip-save-new", "trip-saved-open"]) {
    const b = (await p.getByTestId(id).boundingBox())!;
    expect(b.x, `${id} is in the stub`).toBeGreaterThanOrEqual(hero.x - 0.5);
    expect(await inView(p, p.getByTestId(id)), `${id} in view`).toBe(true);
  }
  // the confirmation is longer than its line: it is cut with an ellipsis (its whole text in the title), not clipped
  const cut = await said(p).evaluate((el) => ({ cs: getComputedStyle(el), over: el.scrollWidth > el.clientWidth, title: el.getAttribute("title") ?? "" }));
  expect([cut.cs.textOverflow, cut.cs.whiteSpace, cut.cs.overflowX, cut.cs.display]).toEqual(["ellipsis", "nowrap", "hidden", "block"]);
  expect(cut.title).toContain("Saved as “Dungeness crab, 400 lb at $9, 1 crew 20 % after costs,");
  // the confirmation is said and then goes; the line it had stays, so nothing moves
  await expect(said(p)).toHaveText("", { timeout: 9000 });
  expect((await p.getByTestId("trip-pass").boundingBox())!.height).toBe(passAfter.height);
  await firstScreen(p, "once the confirmation has gone");
  await field(p, "Catch").scrollIntoViewIfNeeded();
  await expect(p.getByTestId("trip-share-note")).toHaveText("20 % to your 1 crew.");
  // the fuel payment is the fuel bar's own worked sum
  await p.getByTestId("trip-waterfall").scrollIntoViewIfNeeded();
  await p.locator('[data-step="fuel"]').hover();
  await expect(p.getByTestId("waterfall-callout")).toContainText("Fuel −$247.20");
  await expect(p.getByTestId("waterfall-callout")).toContainText("30 gal × $8.24/gal = $247.20");
  await p.locator('[data-step="crew"]').hover();
  await expect(p.getByTestId("waterfall-callout")).toContainText("20 % × ($3,600.00 gross − $0.00 fees − $347.20 shared costs) = $650.56");
  await p.locator('[data-step="net"]').hover();
  await expect(p.getByTestId("waterfall-callout")).toContainText("$3,600.00 gross − $0.00 fees − $347.20 trip costs − $650.56 crew = $2,602.24");
  await ctx.close();
});

test("the season is delayed per an unverified record: the pass says the trip is hypothetical and cannot be made now", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await expect(p.getByTestId("coast-status")).toContainText("Season delayed to at least Jan 1, 2027: a plan, not a trip you can make now");
  expect(await inView(p, p.getByTestId("coast-status"))).toBe(true);
  await expect(p.getByTestId("coast-status")).toContainText("CDFW · not verified");
  // the status in the header opens the gates, with their sources
  await p.getByTestId("coast-status").click();
  const gates = p.getByRole("dialog").getByRole("list", { name: "Gates for Dungeness crab" });
  await expect(gates).toContainText("Zone 4, Pigeon Point to Lopez Point");
  await expect(gates.getByRole("button", { name: /Official · CDFW · checked .* · not verified/ })).toBeVisible();
  await ctx.close();
});

test("every pay model through the page; Total and Each give the same trip; a break-even rounds up", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await fillOwner(p);
  // the boat pays both: N = 3600, crew 720, net 3600 − 720 − 347.20
  await p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Boat" }).click();
  await p.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Boat" }).click();
  await expect(exact(p, "trip-net")).toHaveText("$2,532.80");
  await expect(exact(p, "trip-be-catch")).toHaveText("49 lb"); // 48.2, up
  // share of gross: the payers no longer move the money, so they are not asked
  await p.getByRole("radio", { name: "Of gross" }).click();
  await expect(p.getByRole("radiogroup", { name: "Who pays fuel" })).toHaveCount(0);
  await expect(exact(p, "trip-net")).toHaveText("$2,532.80");
  await expect(exact(p, "trip-crew")).toHaveText("$720.00");
  // a day rate: 2 × $200 × 2 days; 127.47 lb breaks even, shown as 128
  await type(p, "Crew", "2");
  await p.getByRole("radio", { name: "Day rate" }).click();
  await type(p, "Day rate", "200");
  await type(p, "Days at sea", "2");
  await expect(exact(p, "trip-net")).toHaveText("$2,452.80");
  await expect(exact(p, "trip-crew")).toHaveText("$800.00");
  await expect(p.getByTestId("trip-crew")).toContainText("$400.00 each");
  await expect(exact(p, "trip-be-catch")).toHaveText("128 lb");
  await expect(exact(p, "trip-be-price")).toHaveText("$2.87/lb");
  // a share for 2 crew: 20 % in total is 10 % each, and switching keeps the money where it was
  await p.getByRole("radio", { name: "After costs" }).click();
  await choosePayers(p);
  await expect(field(p, "Crew share")).toHaveValue("20");
  await expect(p.getByTestId("trip-share-note")).toHaveText("20 % in total, split 2 ways: 10 % each. $325.28 each.");
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  await p.getByRole("radio", { name: "Each" }).click();
  await expect(field(p, "Crew share")).toHaveValue("10");
  await expect(p.getByTestId("trip-share-note")).toHaveText("10 % each for 2 crew: 20 % in total. $325.28 each.");
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  // fees per pound, added as its own line: 400 × $0.10 = $40 → N = 3212.80 → net 2570.24
  await p.getByRole("button", { name: "Add a fee per pound" }).click();
  await expect(p.getByTestId("trip-net")).toContainText("waiting for fees per pound");
  await type(p, "Fee per pound", "0.10");
  await expect(exact(p, "trip-fees")).toHaveText("$40.00");
  await expect(exact(p, "trip-net")).toHaveText("$2,570.24");
  // nobody else paid
  await type(p, "Crew", "0");
  await expect(p.getByRole("radiogroup", { name: "How crew is paid" })).toHaveCount(0);
  await expect(exact(p, "trip-net")).toHaveText("$3,212.80");
  await expect(p.getByTestId("trip-crew")).toContainText("Nobody else is paid");
  // fuel from distance and burn: 20 nm ÷ 8 kt × 6 gal/h + 5 h × 3 gal/h = 30 gal, the same trip
  await p.getByRole("radio", { name: "Distance" }).click();
  await expect(p.getByTestId("trip-net")).toContainText("waiting for");
  await type(p, "Round trip", "20");
  await type(p, "Cruising speed", "8");
  await type(p, "Steaming burn", "6");
  await type(p, "Hours fishing", "5");
  await type(p, "Fishing burn", "3");
  await expect(p.getByTestId("trip-gallons")).toHaveText("30 gal for the trip.");
  await expect(exact(p, "trip-net")).toHaveText("$3,212.80");
  await ctx.close();
});

test("impossible and inconsistent inputs are said, never worked through", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await fillOwner(p);
  await type(p, "Crew", "1.5");
  await expect(p.getByText("Whole people only")).toBeVisible();
  await expect(p.getByTestId("trip-net")).toContainText("waiting for crew");
  await type(p, "Crew", "1");
  await type(p, "Catch", "-5");
  await expect(p.getByText("Can't be negative")).toBeVisible();
  await expect(p.getByTestId("trip-gross")).toContainText("waiting for catch");
  await type(p, "Catch", "400");
  await type(p, "Price", "0");
  await expect(p.getByTestId("trip-be-catch")).toContainText("No catch covers the costs at a price of $0.");
  await expect(exact(p, "trip-net")).toHaveText("−$347.20");
  await type(p, "Price", "9");
  // a fuel break-even within reach is shown, plainly: 100 gal → (3,600 − 100) ÷ 100 = $35.00/gal
  await type(p, "Fuel", "100");
  await expect(p.getByTestId("trip-be-fuel")).toContainText("Break-even fuel price");
  await expect(exact(p, "trip-be-fuel")).toHaveText("$35.00/gal");
  await type(p, "Fuel", "30");
  // the engine's warning reaches the page
  await p.getByRole("radio", { name: "Of gross" }).click();
  await type(p, "Crew share", "100");
  await expect(p.getByTestId("trip-warning")).toHaveText("Fees and crew share together take all of the gross: no catch covers the costs.");
  await ctx.close();
});

test("a grid scenario moves the result and the waterfall where they are read; one thing at a time marks the break-even", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await fillOwner(p);
  await p.getByTestId("sensitivity-grid").scrollIntoViewIfNeeded();
  // catch +30 %, price +20 %: R = 5616, N = 5268.80, net 4215.04
  await p.getByTestId("sens-0-2").hover();
  await expect(p.getByTestId("trip-showing")).toHaveText("Showing catch +30 %, price +20 %.");
  await expect(exact(p, "trip-net")).toHaveText("$4,215.04");
  expect(await inView(p, p.getByTestId("trip-net")), "the result is in view while the grid is pointed at").toBe(true);
  await p.getByRole("heading", { name: "If things move" }).hover();
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  // the keyboard: one tab stop, arrows move and preview, Enter holds, leaving returns unless held
  await p.getByTestId("sens-1-1").focus();
  await p.keyboard.press("ArrowDown");
  await p.keyboard.press("ArrowLeft");
  await expect(p.getByTestId("trip-showing")).toHaveText("Showing catch −30 %, price −20 %.");
  // R = 280 × 7.20 = 2016; N = 1668.80; net 80 % of it
  await expect(exact(p, "trip-net")).toHaveText("$1,335.04");
  await p.keyboard.press("Enter");
  await field(p, "Catch").focus();
  await expect(exact(p, "trip-net")).toHaveText("$1,335.04");
  await p.getByRole("button", { name: "Back to your numbers" }).click();
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");

  await p.getByRole("radio", { name: "One at a time" }).click();
  await expect(p.getByTestId("trip-sweep-words")).toContainText("Net to boat reaches $0 at 39 lb (rounded up).");
  await expect(p.getByTestId("trip-sweep-words")).toContainText("not a forecast");
  await p.getByTestId("trip-sweep").getByRole("radio", { name: "Price", exact: true }).click();
  await expect(p.getByTestId("trip-sweep-words")).toContainText("Net to boat reaches $0 at $0.87/lb (rounded up).");
  await p.getByTestId("trip-sweep").getByRole("radio", { name: "Fuel price" }).click();
  await expect(p.getByTestId("trip-sweep-words")).toContainText("Net to boat stays above $0 at every fuel price in this range.");
  await ctx.close();
});

test("Save as new, Update, Duplicate, Rename, Delete and Undo: no accidental copies", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await expect(p.getByTestId("trip-update")).toHaveCount(0);
  await fillOwner(p);
  await expect(p.getByTestId("trip-state")).toHaveText("Not saved");
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText(/^Saved as “Dungeness crab, 400 lb at \$9, 1 crew 20 % after costs, .+”\. Kept on this device only\.$/);
  await expect(p.getByTestId("trip-state")).toContainText(/^Saved /);
  expect(await savedCount(p)).toBe(1);

  // the same numbers again: refused, and the trip that holds them is named
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("These exact numbers are already saved as “Dungeness crab, 400 lb at $9, 1 crew 20 % after costs,");
  expect(await savedCount(p)).toBe(1);
  // Clear, type the same again, save: still one (v12 made a second here)
  await p.getByTestId("trip-clear").click();
  await expect(field(p, "Catch")).toHaveValue("");
  await fillOwner(p);
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("These exact numbers are already saved as");
  expect(await savedCount(p)).toBe(1);
  await said(p).getByRole("button", { name: "Open it" }).click();
  await expect(p.getByTestId("trip-state")).toContainText(/^Saved /);

  // an edit: Update is offered, and changes the one trip
  await expect(p.getByTestId("trip-update")).toBeDisabled();
  await type(p, "Catch", "450");
  await expect(p.getByTestId("trip-state")).toHaveText("Changes not saved");
  await p.getByTestId("trip-update").click();
  // F-11: the name Neritic wrote follows the numbers
  await expect(said(p)).toContainText("Updated “Dungeness crab, 450 lb at $9, 1 crew 20 % after costs,");
  await expect(p.getByTestId("trip-name")).toContainText("Dungeness crab, 450 lb at $9");
  expect(await savedCount(p)).toBe(1);
  expect(await p.evaluate(() => JSON.parse(localStorage.getItem("nx-trip-v1:scenarios")!)[0].draft.inputs.catch[0].lb)).toBe(450);

  // a second trip, then a deliberate copy
  await type(p, "Price", "10");
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as “Dungeness crab, 450 lb at $10,");
  await p.getByTestId("trip-duplicate").click();
  await expect(said(p)).toContainText("(copy)”.");
  expect(await savedCount(p)).toBe(3);

  // the saved list: every trip, the true count, rename, delete, undo
  await p.getByTestId("trip-saved-open").click();
  const dialog = p.getByRole("dialog", { name: "Saved trips" });
  await expect(dialog).toContainText("3 of 20 trips saved on this device, across every port.");
  const rows = dialog.getByTestId("trip-saved-row");
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(1)).toContainText("Moss Landing · Dungeness crab · 450 lb at $10, 1 crew 20 % after costs · Net to boat");
  await rows.first().getByRole("button", { name: /^Rename / }).click();
  await dialog.getByRole("textbox", { name: /^New name for / }).fill("Opener");
  await dialog.getByRole("button", { name: "Save name" }).click();
  await expect(rows.filter({ hasText: "Opener" })).toHaveCount(1);
  await dialog.getByRole("button", { name: "Delete Opener" }).click();
  await expect(rows).toHaveCount(2);
  await expect(dialog.getByTestId("trip-undo-drawer")).toContainText("Deleted “Opener”.");
  await dialog.getByTestId("trip-undo-drawer").getByRole("button", { name: "Undo" }).click();
  await expect(rows).toHaveCount(3);
  await expect(rows.filter({ hasText: "Opener" })).toHaveCount(1);
  expect(await savedCount(p)).toBe(3);
  await ctx.close();
});

test("compare two saved trips: only what differs, and the change in each result", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await openTrip(p);
  await fillOwner(p);
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as");
  await type(p, "Price", "10");
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("400 lb at $10");
  await p.getByTestId("trip-saved-open").click();
  const dialog = p.getByRole("dialog", { name: "Saved trips" });
  // newest first: A is the $10 trip, B the $9 one
  await dialog.getByRole("checkbox").nth(0).check();
  await dialog.getByRole("checkbox").nth(1).check();
  const cmp = dialog.getByTestId("trip-compare");
  await expect(cmp.locator('tr[data-kind="input"]')).toHaveCount(1);
  await expect(cmp.locator('tr[data-kind="input"]')).toContainText("Price per pound, line 1$10$9");
  await expect(cmp.locator("tr[data-net]")).toContainText("Net to boat$2,922.24$2,602.24−$320.00");
  await expect(cmp.getByRole("row", { name: /^Crew pay/ })).toContainText("−$80.00");
  await expect(cmp.getByRole("row", { name: /^Break-even catch/ })).toContainText("35 lb39 lb+4 lb");
  await ctx.close();
});

for (const mode of ["day", "dusk", "night"]) {
  test(`Saved trips, ${mode}: an unticked Compare box is an empty box, a ticked one is ink (F-12)`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, colorScheme: mode === "day" ? "light" : "dark" });
    await seed(ctx, { mode });
    const p = await ctx.newPage();
    await openTrip(p);
    for (const price of ["9", "10", "11"]) {
      await type(p, "Catch", "400");
      await type(p, "Price", price);
      await p.getByTestId("trip-save-new").tap();
      await expect(said(p)).toContainText(`400 lb at $${price}`);
    }
    await p.getByTestId("trip-saved-open").tap();
    const dialog = p.getByRole("dialog", { name: "Saved trips" });
    const boxes = dialog.getByRole("checkbox");
    await expect(boxes).toHaveCount(3);
    // the mode's own paper and ink, read where the drawer is drawn
    const tokens = await dialog.evaluate((el) => {
      const probe = (v: string) => {
        const d = document.createElement("i");
        d.style.backgroundColor = `var(${v})`;
        el.appendChild(d);
        const c = getComputedStyle(d).backgroundColor;
        d.remove();
        return c;
      };
      return { paper: probe("--paper"), ink: probe("--ink") };
    });
    expect(tokens.paper).not.toBe(tokens.ink);
    const look = (i: number) => boxes.nth(i).evaluate((el) => ({ bg: getComputedStyle(el).backgroundColor, border: getComputedStyle(el).borderTopStyle, appearance: getComputedStyle(el).appearance, tick: getComputedStyle(el, "::after").content }));
    for (const i of [0, 1, 2]) expect(await look(i), `box ${i} unticked`).toMatchObject({ bg: tokens.paper, border: "solid", appearance: "none", tick: "none" });
    await boxes.nth(0).check();
    expect(await look(0), "ticked").toMatchObject({ bg: tokens.ink });
    expect((await look(0)).tick).not.toBe("none");
    expect(await look(1), "its neighbour stays empty").toMatchObject({ bg: tokens.paper });
    // two picked: the third cannot be, and still reads as empty (dashed), never as ticked
    await boxes.nth(1).check();
    await expect(boxes.nth(2)).toBeDisabled();
    expect(await look(2), "disabled").toMatchObject({ bg: tokens.paper, border: "dashed", tick: "none" });
    await boxes.nth(0).uncheck();
    expect(await look(0), "unticked again").toMatchObject({ bg: tokens.paper, tick: "none" });
    await ctx.close();
  });
}

test("the fishery's whole name is shown in the trip's strip at 320 px, the strip stays in view while the equation is worked, and the way back is reached (R-4)", async ({ browser }) => {
  const { ctx, p } = await page(browser, 320, 640);
  await openTrip(p);
  const select = p.getByTestId("coast-fishery");
  const whole = async (when: string) => {
    // the select is wide enough for its chosen option's words: nothing is cut
    const fits = await select.evaluate((el: HTMLSelectElement) => {
      const cs = getComputedStyle(el);
      const c = document.createElement("canvas").getContext("2d")!;
      c.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      const words = c.measureText(el.selectedOptions[0].text).width;
      return { words, room: el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) };
    });
    expect(fits.words, `${when}: "Dungeness crab, commercial" fits its select`).toBeLessThanOrEqual(fits.room + 0.5);
  };
  await whole("on opening");
  await type(p, "Catch", "400");
  await field(p, "Fuel").scrollIntoViewIfNeeded();
  // the strip (port, fishery, hypothetical, season line) is pinned at the top of the sheet while the equation is worked
  for (const id of ["coast-title", "coast-fishery", "coast-status"]) expect(await inView(p, p.getByTestId(id)), `${id} in view`).toBe(true);
  await whole("pinned");
  // with much wider text the fishery's name still fits and the season line is still whole
  await p.addStyleTag({ content: "[data-nx-desk-header], [data-nx-desk-header] * { letter-spacing: 0.08em !important; word-spacing: 0.2em !important; }" });
  await p.waitForTimeout(300);
  await whole("with much wider text");
  const st = (await p.getByTestId("coast-status").boundingBox())!;
  expect(st.x + st.width, "the season line is not cut").toBeLessThanOrEqual(320.5);
  // the way back to the gates and the sea is under the fold, reached by scrolling
  const back = p.getByTestId("coast-to-brief");
  await back.scrollIntoViewIfNeeded();
  expect(await inView(p, back)).toBe(true);
  await ctx.close();
});

test("unsaved work and the open trip survive leaving the page, a reload and a browser restart", async () => {
  const dir = mkdtempSync(path.join(tmpdir(), "nx-trip-"));
  const launch = async () => {
    const ctx = await chromium.launchPersistentContext(dir, { channel: process.env.CI ? undefined : "chrome", viewport: { width: 1440, height: 900 } });
    return { ctx, p: ctx.pages()[0] ?? (await ctx.newPage()) };
  };
  let { ctx, p } = await launch();
  await p.goto(`${OK}/coast?place=${MOSS}`);
  await p.evaluate(() => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
  });
  await openTrip(p);
  await fillOwner(p);
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as");
  await type(p, "Catch", "450");
  await expect(p.getByTestId("trip-state")).toHaveText("Changes not saved");
  const unchanged = async (how: string) => {
    await expect(field(p, "Catch"), how).toHaveValue("450");
    await expect(field(p, "Crew share"), how).toHaveValue("20");
    await expect(p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }), how).toHaveAttribute("aria-checked", "true");
    await expect(p.getByTestId("trip-state"), how).toHaveText("Changes not saved");
    await expect(p.getByTestId("trip-update"), how).toBeEnabled();
    // 450 × 9 = 4050; N = 3702.80; net 2962.24
    await expect(exact(p, "trip-net"), how).toHaveText("$2,962.24");
  };

  // to Today and back, by the app's own links
  await p.getByRole("link", { name: "Today", exact: true }).first().click();
  await expect(p).toHaveURL(/\/today/);
  await p.goBack();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await unchanged("after leaving for Today and coming back");
  await p.reload();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await unchanged("after a reload");
  await ctx.close();
  ({ ctx, p } = await launch());
  await openTrip(p);
  await unchanged("after closing and reopening the browser");
  // and Update still means update: one trip, not two
  await p.getByTestId("trip-update").click();
  await expect(said(p)).toContainText("Updated “");
  expect(await savedCount(p)).toBe(1);
  await ctx.close();
});

test("offline after a restart: the saved field pack opens the workspace, the saved trips and the open work, and it still computes", async () => {
  test.setTimeout(180_000);
  const dir = mkdtempSync(path.join(tmpdir(), "nx-trip-off-"));
  const launch = async (offline: boolean) => {
    const ctx = await chromium.launchPersistentContext(dir, { channel: process.env.CI ? undefined : "chrome", viewport: { width: 1440, height: 900 }, offline });
    return { ctx, p: ctx.pages()[0] ?? (await ctx.newPage()) };
  };
  let { ctx, p } = await launch(false);
  await p.goto(`${OK}/coast?place=${MOSS}`);
  await p.evaluate(() => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
  });
  await openTrip(p);
  await fillOwner(p);
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as");
  await type(p, "Catch", "450");
  // the port's field pack (Menu): My Coast with its gates is one of the pages it keeps
  await p.getByTestId("nx-menu-button").click();
  await p.getByTestId("nx-pack-save").click();
  await expect(p.getByTestId("nx-pack-status")).toContainText("Moss Landing saved", { timeout: 120_000 });
  await ctx.close();

  ({ ctx, p } = await launch(true));
  await p.goto(`${OK}/coast?place=${MOSS}`);
  await expect(p.getByTestId("coast-status")).toContainText(/delayed to at least Jan 1, 2027/, { timeout: 20_000 });
  await p.getByTestId("coast-to-trip").click();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(field(p, "Catch")).toHaveValue("450");
  await expect(p.getByTestId("trip-state")).toHaveText("Changes not saved");
  await expect(exact(p, "trip-net")).toHaveText("$2,962.24");
  // it is the page working, not a picture of it: an edit recomputes with no network
  await type(p, "Catch", "400");
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  await expect(p.getByTestId("trip-saved-open")).toHaveAccessibleName("Saved trips (1)");
  await ctx.close();
});

test("a recreational trip is cost only: no price, gross, crew or net anywhere", async ({ browser }) => {
  const { ctx, p } = await page(browser, 1440, 900, { fishery: "dungeness_recreational" });
  await p.goto(`${OK}/coast?place=${MOSS}&view=trip`);
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(p.locator("[data-nx-desk-header]")).toContainText("Hypothetical trip · cost only");
  for (const label of ["Catch", "Price", "Fees", "Crew", "Crew share"]) await expect(field(p, label)).toHaveCount(0);
  await expect(p.getByRole("radiogroup", { name: "How crew is paid" })).toHaveCount(0);
  await type(p, "Fuel", "30");
  await type(p, "Fuel price", "8.24");
  await type(p, "Bait and ice", "100");
  await expect(p.getByTestId("trip-net")).toContainText("Trip cost");
  await expect(exact(p, "trip-net")).toHaveText("$347.20");
  await expect(p.getByTestId("trip-per-person")).toContainText("waiting for people sharing the cost");
  await type(p, "People sharing the cost", "2");
  await expect(exact(p, "trip-per-person")).toHaveText("$173.60");
  const text = await p.getByTestId("nx-coast").innerText();
  expect(text).not.toMatch(/Net to boat|Gross|Crew pay|Break-even|Price\b/);
  await expect(p.getByTestId("trip-note")).toContainText("A recreational catch is not sold");
  await expect(p.getByTestId("sensitivity-grid")).toHaveCount(0);
  // saved like any trip, and named for what it is
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as “Dungeness crab, recreational, 30 gal, 2 people,");
  await ctx.close();
});

test("the numbers-only link and the export carry nothing about the reader; the point and date from Explore are shown, not used", async ({ browser }) => {
  const { ctx, p } = await page(browser);
  await ctx.grantPermissions(["clipboard-read", "clipboard-write"]);
  await openTrip(p, "&at=36.8,-121.9&date=2026-10-14");
  await expect(p.getByTestId("coast-carried")).toHaveText("Point from Explore: 36°48.0′ N, 121°54.0′ W · Date from Explore: Oct 14");
  await fillOwner(p);
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24"); // the carried point changes no figure
  await p.getByTestId("trip-save-new").click();
  await expect(said(p)).toContainText("Saved as");
  await p.getByTestId("trip-copy-link").click();
  await expect(said(p)).toContainText("It carries the numbers only");
  const url = await p.evaluate(() => navigator.clipboard.readText());
  expect(url).toMatch(/\/coast\?view=trip&fishery=dungeness_commercial#trip=[A-Za-z0-9_-]+$/);
  const payload = Buffer.from(url.split("#trip=")[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
  for (const secret of [MOSS, "Moss", "Dungeness", "Bait", "36.8", "121.9"]) expect(url + payload, secret).not.toContain(secret);

  // someone else opens it, at their own port: the same numbers, nothing of the sender's
  const other = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await seed(other, { fishery: "rock_crab" });
  const q = await other.newPage();
  await q.goto(url.replace(/^https?:\/\/[^/]+/, OK).replace("/coast?", "/coast?place=port-550&"));
  await expect(q.getByTestId("coast-fishery")).toHaveValue("dungeness_commercial");
  await expect(exact(q, "trip-net")).toHaveText("$2,602.24");
  await expect(field(q, "Catch")).toHaveValue("400");
  expect(q.url()).not.toContain("#trip=");
  await other.close();

  // the export: the reader's saved trips and vessels, and nothing else on the device
  await p.getByTestId("trip-saved-open").click();
  const [download] = await Promise.all([p.waitForEvent("download"), p.getByTestId("trip-export").click()]);
  const file = JSON.parse(readFileSync((await download.path())!, "utf8"));
  expect(Object.keys(file).sort()).toEqual(["app", "exportedAt", "scenarios", "schema", "vessels"]);
  expect(file.scenarios).toHaveLength(1);
  expect(JSON.stringify(file)).not.toContain("36.8");
  await ctx.close();
});

for (const [w, h] of SIZES) {
  test(`${w}×${h}: the work has the room, the result stays in reach, nothing scrolls sideways, and axe finds nothing`, async ({ browser }) => {
    const { ctx, p } = await page(browser, w, h);
    await openTrip(p);
    // on opening, before anything is scrolled: the first input is there to type in, and so is the result
    // (a 320 × 640 screen is too short for the chart strip, the header and an input at once: there it is one swipe away)
    if (w >= 1100) expect(await inView(p, field(p, "Catch")), "the first input is in view on opening").toBe(true);
    expect(await inView(p, p.getByTestId("trip-net")), "the result is in view on opening").toBe(true);
    await fillOwner(p);
    await expect(result(p)).toContainText("$2,602.24");
    const vw = p.viewportSize()!.width;
    if (w >= 1100) {
      // the pass runs the width between the gutters; the chart keeps at least 45 % of the height above it
      const pass = (await p.getByTestId("trip-pass").boundingBox())!;
      expect(pass.width / vw).toBeGreaterThanOrEqual(0.9);
      expect(pass.y / h).toBeGreaterThanOrEqual(0.45);
      // the one result is far larger than a field of the equation
      const big = await p.getByTestId("trip-net-figure").evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
      const body = await field(p, "Catch").evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
      expect(big / body).toBeGreaterThanOrEqual(2.5);
    } else {
      // a phone: the bar above the destinations holds the result and the break-even, and Save can be reached
      await expect(p.getByTestId("trip-netbar")).toContainText("Break-even 39 lb");
      expect(await inView(p, p.getByTestId("trip-netbar"))).toBe(true);
      for (const r of await p.getByRole("radio").all()) if (await r.isVisible()) expect((await r.boundingBox())!.height).toBeGreaterThanOrEqual(44);
      expect((await field(p, "Catch").boundingBox())!.height).toBeGreaterThanOrEqual(44);
    }
    // after any input change the result is in view (the hero beside the inputs, or the phone's bar)
    for (const [label, v] of [
      ["Catch", "410"],
      ["Crew share", "25"],
      ["Fees", "2"],
    ] as const) {
      await type(p, label, v);
      expect(await resultInReach(p), `result in reach after ${label}`).toBe(true);
      expect(await inView(p, p.getByTestId("coast-status")), `status in view after ${label}`).toBe(true);
    }
    await type(p, "Catch", "400");
    await type(p, "Crew share", "20");
    await type(p, "Fees", "0");
    // a grid scenario is seen where it is chosen: beside the grid on a wide screen, in the bar on a phone
    const cell = p.getByTestId("sens-0-2");
    await cell.scrollIntoViewIfNeeded();
    if (w <= 640) await cell.tap();
    else await cell.hover();
    await expect(result(p)).toContainText("$4,215.04");
    expect(await resultInReach(p)).toBe(true);
    if (w <= 640) {
      await expect(p.getByTestId("trip-netbar")).toContainText("With catch +30 %, price +20 %");
      await cell.tap();
      await expect(result(p)).toContainText("$2,602.24");
      // Save is reachable with a thumb
      const save = p.getByTestId("trip-save-new");
      await save.scrollIntoViewIfNeeded();
      expect(await inView(p, save)).toBe(true);
      await save.tap();
      await expect(said(p)).toContainText("Saved as");
    }
    expect(await p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), "no sideways scroll").toBeLessThanOrEqual(0);
    // axe reads colours as they are painted: let the figures and the message finish moving first
    await p.waitForTimeout(700);
    const r = await new AxeBuilder({ page: p }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
    expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
    await ctx.close();
  });
}

test("reduced motion: figures cut to their values", async ({ browser }) => {
  const { ctx, p } = await page(browser, 1440, 900, { reducedMotion: "reduce" });
  await openTrip(p);
  await fillOwner(p);
  await p.getByTestId("sens-0-2").hover();
  // no spring: the visible digits are the new value at once
  await expect(p.getByTestId("trip-net-figure")).toContainText("$4,215.04", { timeout: 400 });
  await ctx.close();
});
