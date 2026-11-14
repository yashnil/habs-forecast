import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { NS, newContext, seedTrip } from "./helpers";

/**
 * Tests E and F (DIRECTION §6, §11, §17).
 *
 *  E  Departure windows: two periods compared, with a change of wind direction between them; a limit set in the plan
 *     makes a period cross it (and says by what stated number); a half-day the forecast skips is shown as no forecast,
 *     never as a calm period.
 *  F  Completing the trip: actual catch, price, fuel, supplies and crew pay give the planned-vs-actual figures through
 *     the engine (planned net to boat $2,602.24 → actual $2,725.00, +$122.76), attributed to catch, price, fuel,
 *     supplies and crew.
 */
test.use({ storageState: { cookies: [], origins: [] } });

async function openWorkspace(page: Page, id: string) {
  await page.goto(`${OK}/coast/trips?id=${id}`);
  await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
}

const windowCell = (page: Page, key: string) => page.getByTestId(`window-${key}`);

test("E: two periods compared, a direction change, a limit crossing after setting it, and a missing interval", async ({ browser }) => {
  // no limits yet; the saved forecast skips Wednesday night (a half-day NWS's answer did not give)
  const seeded = await seedTrip({
    id: "e-trip",
    limits: { windKt: null, seasFt: null },
    editSea: (s) => {
      s.sea.lanes = s.sea.lanes.filter((l) => l.name !== "Wed Night");
    },
  });
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage);
  await openWorkspace(page, "e-trip");
  const timeline = page.getByTestId("departure-timeline").first();
  await expect(timeline).toBeVisible();

  // the missing half-day is its own state: no forecast, never a period without crossings
  await expect(windowCell(page, "2026-10-14|night")).toHaveAttribute("data-state", "unavailable");
  await expect(windowCell(page, "2026-10-14|night")).toHaveAttribute("aria-label", /no NWS forecast/);

  // no limit set: nothing crosses
  await expect(windowCell(page, "2026-10-13|night")).toHaveAttribute("data-crossings", "0");

  // compare Tuesday (NE, backing to W) with Wednesday (NE, backing to NW)
  await windowCell(page, "2026-10-13|day").click();
  await windowCell(page, "2026-10-14|day").click();
  // one comparison, not one per component that could draw it
  await expect(page.getByTestId("window-compare")).toHaveCount(1);
  const cmp = page.getByTestId("window-compare");
  await expect(cmp).toBeVisible();
  const dir = cmp.locator('[data-field="windDir"]');
  await expect(dir).toHaveCount(1);
  await expect(dir).toContainText("W");
  await expect(dir).toContainText("NW");

  // set a wind limit of 12 kt in the plan: Tuesday night's stated 15 kt now crosses it
  await page.getByTestId("trip-edit").click();
  const dialog = page.getByRole("dialog", { name: "Edit plan" });
  await dialog.getByRole("button", { name: /Your limits and area/ }).click();
  const wind = dialog.getByRole("textbox", { name: "Wind", exact: true });
  await wind.fill("12");
  await page.getByTestId("trip-edit-save").click();
  await expect(windowCell(page, "2026-10-13|night")).toHaveAttribute("data-crossings", "1");
  await expect(windowCell(page, "2026-10-13|night")).toHaveAttribute("aria-label", /Wind stated up to 15 kt; your limit 12 kt/);
  const trip = JSON.parse((await page.evaluate((k) => localStorage.getItem(k), `${NS}trip:e-trip`)) ?? "{}");
  expect(trip.limits.windKt).toBe(12);
  expect(errors).toEqual([]);
  await ctx.close();
});

test("F: the recorded outcome gives planned against actual through the engine, attributed to its parts", async ({ browser }) => {
  const seeded = await seedTrip({ id: "f-trip" });
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage);
  await openWorkspace(page, "f-trip");
  await page.getByTestId("trip-more").click();
  await page.getByTestId("trip-complete").click();
  const form = page.getByTestId("outcome-form");
  await expect(form).toBeVisible();
  const field = (name: string) => form.getByLabel(name, { exact: true }).first();
  await field("Species").fill("Dungeness crab");
  await field("Weight").fill("450");
  await field("Price received").fill("8.50");
  await field("Fuel used").fill("35");
  await field("Fuel price").fill("8.00");
  await field("What").fill("Bait and ice");
  await field("Paid").fill("120");
  await field("Crew paid, in total").fill("700");
  await form.getByTestId("outcome-save").click();

  const pv = page.getByTestId("planned-vs-actual");
  await expect(pv).toBeVisible();
  await expect(page.getByTestId("trip-workspace")).toHaveAttribute("data-state", "completed");
  await expect(pv.getByTestId("pv-difference")).toContainText("+$122.76 net to boat against plan");
  const parts = pv.getByRole("list", { name: "Where the difference came from" }).getByRole("listitem");
  for (const [label, value] of [
    ["Catch landed", "+$450.00"],
    ["Price received", "−$225.00"],
    ["Fuel", "−$32.80"],
    ["Bait, ice and other", "−$20.00"],
    ["Crew pay", "−$49.44"],
  ])
    await expect(parts.filter({ hasText: label })).toContainText(value);
  // "net to boat", never profit
  expect(await pv.innerText()).not.toMatch(/profit/i);
  const trip = JSON.parse((await page.evaluate((k) => localStorage.getItem(k), `${NS}trip:f-trip`)) ?? "{}");
  expect(trip.state).toBe("completed");
  expect(trip.outcome.landed[0]).toMatchObject({ lb: 450, usdPerLb: 8.5 });
  expect(errors).toEqual([]);
  await ctx.close();
});
