import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { NS, newContext, routeGates, routeSea, seedTrip, strongerWind, VERDICT } from "./helpers";

/**
 * Tests B, C and D (DIRECTION §8, §9, §17): reopening a trip and rechecking it against changed answers.
 *
 *  B  The forecast changed for the departure day: the feed shows exactly that change (stronger winds, the planned and
 *     the new range), nothing else; "What Neritic showed you" still shows the planning snapshot, which is unchanged.
 *  C  CDFW's record was revised and a new whale measure published: each change links to the agency's own page, and the
 *     feed says nothing that reads as permission.
 *  D  The sea endpoint fails (503) and CDPH's tests did not load: the recheck is incomplete, names what could not be
 *     checked, and never says no changes were found.
 *
 * The planning snapshot is the fixture's own answers (seedTrip); the recheck goes to /coast/data, answered through
 * page.route with a changed copy of the same answer.
 */
test.use({ storageState: { cookies: [], origins: [] } });

async function openWorkspace(page: Page, id: string) {
  await page.goto(`${OK}/coast/trips?id=${id}`);
  await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
}

async function recheck(page: Page) {
  const feed = page.getByTestId("trip-change-feed");
  // the feed's own control, or the primary when the next action is itself a recheck (one control, never both)
  const own = feed.getByTestId("trip-recheck");
  if (await own.count()) await own.click();
  else await page.getByTestId("trip-next").click();
  await expect(feed).not.toHaveAttribute("data-state", /checking|unchecked/, { timeout: 20_000 });
  return feed;
}

const stored = (page: Page, key: string) => page.evaluate((k) => localStorage.getItem(k), key);

test("B: a changed forecast for the departure day is the one change found; the planning snapshot is kept as it was", async ({ browser }) => {
  const seeded = await seedTrip({ id: "b-trip" });
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage, { routed: true });
  // NWS raises Wednesday's (the departure day's) wind; the direction and seas stay as they were
  await routeSea(page, (s) => strongerWind(s, "Wed", "NE wind 20 to 25 kt, backing to NW in the afternoon. Seas around 3 ft. Wave Detail: NW 3 ft at 7 seconds and SW 2 ft at 15 seconds."));
  await openWorkspace(page, "b-trip");
  const before = await stored(page, `${NS}snap:b-trip-plan`);
  const feed = await recheck(page);
  const items = feed.getByTestId("trip-changes").getByRole("listitem");
  await expect(items).toHaveCount(1);
  await expect(items.first()).toContainText(/Marine forecast updated since you planned: stronger winds/);
  await expect(items.first()).toContainText("5–10 kt");
  await expect(items.first()).toContainText("20–25 kt");
  await expect(feed).not.toContainText(/No changes found/);
  // the original is untouched, and the workspace still shows it on demand
  expect(await stored(page, `${NS}snap:b-trip-plan`)).toBe(before);
  await page.getByTestId("trip-open-snapshot").click();
  const snap = page.getByTestId("trip-snapshot");
  await expect(snap).toBeVisible();
  await expect(page.getByRole("dialog", { name: "What Neritic showed you" })).toBeVisible();
  await expect(snap).toContainText("Commercial season delayed to at least Jan 1, 2027");
  // the trip records the recheck beside its planning snapshot
  const trip = JSON.parse((await stored(page, `${NS}trip:b-trip`)) ?? "{}");
  expect(trip.snapshots.planning).toBe("b-trip-plan");
  expect(trip.snapshots.rechecks).toHaveLength(1);
  expect(errors).toEqual([]);
  await ctx.close();
});

test("C: a revised CDFW record and a new whale measure are found, each linked to the agency's page, with no permission words", async ({ browser }) => {
  const seeded = await seedTrip({ id: "c-trip" });
  const { ctx, page } = await newContext(browser, undefined, seeded.storage, { routed: true });
  const SEASON_URL = "https://wildlife.ca.gov/Conservation/Marine/Invertebrates/Crabs/Dungeness/2026-27-season";
  await routeGates(page, (g) => {
    g.season!.text = "Commercial season delayed to at least Jan 15, 2027 (humpback whale entanglements).";
    g.season!.url = SEASON_URL;
    g.season!.checked = "2026-10-13";
    g.whale!.measures = "Depth constraint: 30 fathoms. Fleet Advisory.";
  });
  await openWorkspace(page, "c-trip");
  const feed = await recheck(page);
  const items = feed.getByTestId("trip-changes").getByRole("listitem");
  const agency = items.filter({ hasText: "Agency information changed; review the original CDFW page." });
  await expect(agency).toHaveCount(1);
  await expect(agency.getByRole("link", { name: /Source/ })).toHaveAttribute("href", SEASON_URL);
  const whale = items.filter({ hasText: /CDFW whale measures changed/ });
  await expect(whale).toHaveCount(1);
  await expect(whale.getByRole("link", { name: /Source/ })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
  expect(await feed.innerText()).not.toMatch(VERDICT);
  await ctx.close();
});

test("D: the sea endpoint fails and CDPH's tests did not load: an incomplete recheck that names what it could not check", async ({ browser }) => {
  const seeded = await seedTrip({ id: "d-trip" });
  const { ctx, page } = await newContext(browser, undefined, seeded.storage, { routed: true });
  await routeSea(page, { status: 503 });
  await routeGates(page, (g) => {
    g.test = null;
    g.testsError = true;
  });
  await openWorkspace(page, "d-trip");
  const feed = await recheck(page);
  await expect(feed).toHaveAttribute("data-state", "incomplete");
  await expect(feed.getByRole("heading", { level: 2 })).toContainText(/could not be checked/);
  const unchecked = feed.getByTestId("trip-unchecked");
  await expect(unchecked).toBeVisible();
  for (const what of [/NWS/, /CDPH/, /buoy|NDBC/i, /tide/i]) await expect(unchecked).toContainText(what);
  await expect(feed).not.toContainText(/No changes found/i);
  expect(await feed.innerText()).not.toMatch(VERDICT);
  await ctx.close();
});
