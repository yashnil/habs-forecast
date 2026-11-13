import { expect, test, type BrowserContext, type Page, type Route } from "@playwright/test";
import type { CoastSea } from "@/nx/coast/brief";
import { OK } from "../ports";
import { newContext, seedTrip, strongerWind } from "./helpers";

/**
 * Test K (review blocker, DIRECTION §8, §9, §16): a failed or offline check never replaces evidence the device already
 * holds. NWS issues a small craft advisory and raises the departure day's wind to 15–25 kt; a recheck finds both. Then
 * the sea endpoint fails (503) and a second recheck runs; then the device goes offline and a third runs. At every step
 * the workspace's departure windows and the departure briefing still show the advisory and 15–25 kt, with how old
 * they are once they no longer come from the latest check, and the feed never says there is nothing new.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const SCA_URL = "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.e2e-sca";
const DEPART = "2026-10-14|day";

function changedSea(s: CoastSea) {
  strongerWind(s, "Wed", "N wind 15 to 25 kt, backing to NW in the afternoon. Seas around 3 ft. Wave Detail: NW 3 ft at 7 seconds and SW 2 ft at 15 seconds.");
  s.sea.alerts = [{ event: "Small Craft Advisory", headline: "Small Craft Advisory until Thursday 3 AM PDT", url: SCA_URL, ends: "2026-10-15T10:00:00Z" }];
}

/** The sea endpoint, switchable: the changed answer, or a 503. */
async function seaSwitch(ctx: BrowserContext) {
  let mode: "changed" | "down" = "changed";
  await ctx.route(
    (u) => u.pathname === "/coast/data" && u.searchParams.get("part") === "sea",
    async (route: Route) => {
      if (mode === "down") return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ ok: false, error: "Service unavailable" }) });
      const res = await route.fetch();
      const j = (await res.json()) as CoastSea;
      changedSea(j);
      await route.fulfill({ response: res, json: j });
    },
  );
  return (m: "changed" | "down") => {
    mode = m;
  };
}

async function recheck(page: Page) {
  const feed = page.getByTestId("trip-change-feed");
  // the feed's own control, or the primary when the next action is itself a recheck (one control, never both)
  const own = feed.getByTestId("trip-recheck");
  if (await own.count()) await own.click();
  else await page.getByTestId("trip-next").click();
  await expect(feed).not.toHaveAttribute("data-state", /checking/, { timeout: 20_000 });
  await expect(feed).not.toContainText(/nothing new/i);
  return feed;
}

/** The workspace's departure day still carries the advisory and the 15–25 kt the device knows. */
async function workspaceKnows(page: Page) {
  const cell = page.getByTestId(`window-${DEPART}`).first();
  await expect(cell).toHaveAttribute("aria-label", /15.{1,4}25 kt/);
  await expect(cell).toHaveAttribute("aria-label", /Small Craft Advisory/);
}

/** The briefing still shows them; `aged`: with how old they are, since they are not from the latest check. */
async function briefingKnows(page: Page, id: string, aged: boolean) {
  await page.goto(`${OK}/coast/trips?id=${id}&view=briefing`);
  const b = page.getByTestId("trip-briefing");
  await expect(b).toBeVisible();
  await expect(b).not.toContainText(/could not be read/);
  await expect(b.getByTestId("briefing-period").filter({ hasText: "Wednesday" }).or(b.getByTestId("briefing-period").filter({ hasText: /^Wed/ })).first()).toContainText(/15.{1,4}25 kt/);
  await expect(b.getByTestId("briefing-official").filter({ hasText: "Small Craft Advisory" })).toHaveCount(1);
  if (aged) await expect(b.getByTestId("briefing-known-age").first()).toBeVisible();
  await page.goto(`${OK}/coast/trips?id=${id}`);
  await expect(page.getByTestId("trip-workspace")).toBeVisible();
}

test("K: a failed and an offline recheck keep the advisory and the 15–25 kt the device already had, with their age", async ({ browser }) => {
  test.setTimeout(120_000);
  const seeded = await seedTrip({ id: "k-known" });
  // the service worker stays on (as on a real device): the context's route answers its requests too
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage);
  const sea = await seaSwitch(ctx);
  await page.goto(`${OK}/coast/trips?id=k-known`);
  await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });

  // 1. NWS changed: the recheck finds the advisory and the stronger wind
  sea("changed");
  const feed = await recheck(page);
  await expect(feed.getByTestId("trip-changes")).toContainText("Small Craft Advisory");
  await expect(feed.getByTestId("trip-changes")).toContainText(/15.{1,4}25 kt/);
  await workspaceKnows(page);
  await briefingKnows(page, "k-known", false);

  // 2. the sea endpoint fails: incomplete, and what the device knew stays, with its age
  sea("down");
  const failed = await recheck(page);
  await expect(failed).toHaveAttribute("data-state", "incomplete");
  await workspaceKnows(page);
  await briefingKnows(page, "k-known", true);

  // 3. offline: compared with the saved copy, never a live check, and still the same evidence
  await ctx.setOffline(true);
  const offline = await recheck(page);
  await expect(offline).not.toContainText(/No changes found/);
  await workspaceKnows(page);
  await ctx.setOffline(false);
  await briefingKnows(page, "k-known", true);
  expect(errors).toEqual([]);
  await ctx.close();
});
