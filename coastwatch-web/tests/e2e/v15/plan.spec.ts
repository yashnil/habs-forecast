import { expect, test, type Locator, type Page } from "@playwright/test";
import { OK } from "../ports";
import { exact, fillOwner, MOSS, seed } from "../v13/trip";

/**
 * v15 acceptance A and I (agent A): plan and save a trip from My Coast, land on its workspace; the lifecycle on a
 * phone at 390 and 320 px with nothing overlapping and nothing scrolling sideways.
 * Moss Landing is port-592 (tests/e2e/v13/trip.ts). The device's own date picks the departure (tomorrow), so the
 * lifecycle words do not depend on the fixture's clock.
 */

const tomorrow = () => {
  const d = new Date(Date.now() + 86_400_000);
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit" }).format(d);
};

async function brief(p: Page) {
  await p.goto(`${OK}/coast?place=${MOSS}`);
  await expect(p.getByTestId("coast-status").first()).toContainText(/delayed to at least Jan 1, 2027/, { timeout: 15_000 });
}

const trips = (p: Page) => p.evaluate(() => (JSON.parse(localStorage.getItem("nx-trips-v1:index") ?? "{}") as { ids?: string[] }).ids ?? []);

test("A: Moss Landing, commercial Dungeness: port, fishery, date, the regression economics, Save trip → the trip's workspace with $2,602.24 net to boat", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await seed(ctx);
  const p = await ctx.newPage();
  const errors: string[] = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await brief(p);
  await expect(p.getByTestId("coast-title").first()).toContainText("Moss Landing");
  await p.getByTestId("coast-fishery").first().selectOption("dungeness_commercial");
  const date = tomorrow();
  await p.getByTestId("plan-date").fill(date);
  // the reader's own limits, said as theirs
  await p.getByTestId("plan-more").getByRole("button").click();
  await expect(p.getByTestId("plan-more")).toContainText("Your own thresholds");
  await p.getByRole("textbox", { name: "Wind", exact: true }).fill("20");
  await p.getByRole("textbox", { name: "Seas", exact: true }).fill("8");

  // the optional expenses step: the worksheet, with the date kept in its strip
  await p.getByTestId("coast-to-trip").click();
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(p.getByTestId("plan-date-strip")).toHaveValue(date);
  await fillOwner(p);
  await expect(exact(p, "trip-net")).toHaveText("$2,602.24");
  expect(await trips(p)).toEqual([]);

  await p.getByTestId("trip-plan-save").click();
  await expect(p).toHaveURL(/\/coast\/trips\?id=/);
  const ws = p.getByTestId("trip-workspace");
  await expect(ws).toBeVisible();
  await expect(p.getByTestId("trip-heading")).toContainText("Dungeness crab from Moss Landing");
  await expect(p.getByTestId("trip-heading")).toContainText("Moss Landing");
  await expect(p.getByTestId("trip-lifecycle")).toHaveText("Planned");
  await expect(p.getByTestId("trip-money-figure")).toHaveText("$2,602.24");
  await expect(p.getByTestId("trip-since")).toContainText("Since you planned this trip");
  await expect(p.getByTestId("trip-next")).toHaveCount(1);
  await expect(p.getByTestId("trip-local")).toContainText("Kept on this device only");
  await expect(p.getByTestId("trip-departures")).toBeVisible();

  // stored once, with its planning snapshot and the reader's limits
  const ids = await trips(p);
  expect(ids).toHaveLength(1);
  const stored = await p.evaluate((id) => JSON.parse(localStorage.getItem(`nx-trips-v1:trip:${id}`)!), ids[0]);
  expect(stored).toMatchObject({ state: "planned", placeKey: MOSS, fishery: "dungeness_commercial", activity: "commercial", limits: { windKt: 20, seasFt: 8 }, depart: { date } });
  expect(stored.snapshots.planning).toBeTruthy();
  const snap = await p.evaluate((id) => JSON.parse(localStorage.getItem(`nx-trips-v1:snap:${id}`)!), stored.snapshots.planning);
  expect(snap.kind).toBe("planning");
  expect(snap.evidence.some((e: { kind: string }) => e.kind === "season")).toBe(true);

  // a recheck keeps the workspace on screen (the feed shows its checking state); the page placeholder never returns
  await p.getByTestId("trip-change-feed").waitFor();
  const recheck = p.getByTestId("trip-recheck");
  if (await recheck.count()) await recheck.click();
  else await p.getByTestId("trip-next").click();
  for (let i = 0; i < 15; i++) {
    expect(await p.getByText("Reading the trips kept on this device").count(), "the placeholder replaced the workspace during a recheck").toBe(0);
    await p.waitForTimeout(200);
  }
  await expect(ws).toBeVisible();

  // a reload keeps it; the history lists it
  await p.reload();
  await expect(p.getByTestId("trip-lifecycle")).toHaveText("Planned");
  await p.goto(`${OK}/coast/trips`);
  await expect(p.getByTestId("history")).toContainText("Dungeness crab from Moss Landing");
  expect(errors).toEqual([]);
  await ctx.close();
});

test("A2: saving with no economics is a trip of conditions and official information only; no date keeps a draft", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await seed(ctx);
  const p = await ctx.newPage();
  await brief(p);
  await expect(p.getByTestId("coast-save-note")).toContainText("No date yet: kept as a draft.");
  await p.getByTestId("coast-save-trip").click();
  await expect(p).toHaveURL(/\/coast\/trips\?id=/);
  await expect(p.getByTestId("trip-lifecycle")).toHaveText("Draft");
  await expect(p.getByTestId("trip-next")).toHaveText(/Plan this trip/);
  await expect(p.getByTestId("trip-workspace")).not.toContainText("$2,602.24");
  // Plan this trip asks for the date, and planning it makes it Planned
  await p.getByTestId("trip-next").click();
  const dialog = p.getByRole("dialog", { name: "Edit plan" });
  await dialog.getByTestId("plan-date").fill(tomorrow());
  await dialog.getByTestId("trip-edit-save").click();
  await expect(p.getByTestId("trip-lifecycle")).toHaveText("Planned");
  await ctx.close();
});

test("an offline recheck updates in place: the address and the workspace stay, and it says it compared the saved copy", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
  await seed(ctx);
  const p = await ctx.newPage();
  await brief(p);
  await p.getByTestId("plan-date").fill(tomorrow());
  await p.getByTestId("coast-save-trip").click();
  await expect(p).toHaveURL(/\/coast\/trips\?id=/);
  await expect(p.getByTestId("trip-change-feed")).toBeVisible();
  const url = p.url();
  let navigated = 0;
  p.on("framenavigated", (f) => f === p.mainFrame() && (navigated += 1));
  await ctx.setOffline(true);
  const recheck = p.getByTestId("trip-recheck");
  if (await recheck.count()) await recheck.click();
  else await p.getByTestId("trip-next").click();
  for (let i = 0; i < 15; i++) {
    expect(p.url()).toBe(url);
    expect(await p.getByText("Reading the trips kept on this device").count()).toBe(0);
    await p.waitForTimeout(200);
  }
  expect(navigated, "the recheck navigated the document").toBe(0);
  await expect(p.getByTestId("trip-workspace")).toBeVisible();
  await expect(p.getByTestId("trip-change-feed")).not.toHaveAttribute("data-state", /checking|unchecked|complete$/);
  await ctx.close();
});

test("320 px: the pinned primary never covers the last change, and the trip's name wraps instead of truncating", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true, serviceWorkers: "block" });
  await seed(ctx);
  // a changed sea answer: stronger wind on every period and an advisory, so the feed has several changes
  await ctx.route(
    (u) => u.pathname === "/coast/data" && u.searchParams.get("part") === "sea",
    async (route) => {
      const res = await route.fetch();
      const j = await res.json();
      if (j.sea?.lanes && (globalThis as { __changed?: boolean }).__changed) {
        j.sea.lanes = j.sea.lanes.map((l: { text: string }) => ({ ...l, wind: "NW 20 to 30 kt", text: "NW wind 20 to 30 kt. Seas 8 to 10 ft.", seasLo: 8, seasHi: 10 }));
        j.sea.alerts = [{ event: "Small Craft Advisory", headline: "Small Craft Advisory in effect", url: "https://www.weather.gov/", ends: null }];
      }
      await route.fulfill({ response: res, json: j });
    },
  );
  const p = await ctx.newPage();
  await brief(p);
  await p.getByTestId("plan-date").scrollIntoViewIfNeeded();
  await p.getByTestId("plan-date").fill(tomorrow());
  await p.getByTestId("coast-save-trip").tap();
  await expect(p).toHaveURL(/\/coast\/trips\?id=/);
  (globalThis as { __changed?: boolean }).__changed = true;
  await p.getByTestId("trip-recheck").tap();
  const items = p.getByTestId("trip-changes").getByRole("listitem");
  await expect(items.first()).toBeVisible({ timeout: 20_000 });
  (globalThis as { __changed?: boolean }).__changed = false;

  // the name: whole, on more than one line if it needs them
  const t = await p.getByTestId("trip-title").locator("button").evaluate((el) => ({ sw: el.scrollWidth, cw: el.clientWidth, ws: getComputedStyle(el).whiteSpace }));
  expect(t.ws).not.toBe("nowrap");
  expect(t.sw).toBeLessThanOrEqual(t.cw + 1);

  // scroll the sheet to its foot, then bring the last change in: it stops above the pinned bar
  await p.evaluate(() => {
    const sc = document.querySelector<HTMLElement>(".n12-margin-scroll");
    if (sc) sc.scrollTop = sc.scrollHeight;
  });
  const last = items.last();
  await last.evaluate((el) => el.scrollIntoView({ block: "nearest" }));
  await p.waitForTimeout(300);
  const a = (await last.boundingBox())!;
  const bar = (await p.getByTestId("trip-next").boundingBox())!;
  expect(a.y + a.height <= bar.y + 0.5 || a.y >= bar.y + bar.height - 0.5, "the last change sits under the pinned primary").toBe(true);
  await ctx.close();
});

test("the empty history: one line and one action into the planner", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await seed(ctx);
  const p = await ctx.newPage();
  await p.goto(`${OK}/coast/trips`);
  await expect(p.getByTestId("trips-empty")).toBeVisible();
  await expect(p.getByTestId("trips-plan")).toHaveAttribute("href", "/coast?view=trip");
  await expect(p.getByTestId("trips-local")).toContainText("Kept on this device only");
  await ctx.close();
});

/** No page-level sideways scroll, in the document or the phone sheet. */
async function noSideways(p: Page, how: string) {
  const over = await p.evaluate(() => {
    const els = [document.scrollingElement, ...document.querySelectorAll<HTMLElement>(".n12-margin-scroll, [data-testid='nx-sheet']")].filter(Boolean) as Element[];
    return els.map((e) => e.scrollWidth - e.clientWidth).filter((d) => d > 1);
  });
  expect(over, `${how}: something scrolls sideways`).toEqual([]);
}

async function box(l: Locator) {
  await l.scrollIntoViewIfNeeded();
  return (await l.boundingBox())!;
}
const overlap = (a: { x: number; y: number; width: number; height: number }, b: typeof a) => a.x < b.x + b.width - 0.5 && b.x < a.x + a.width - 0.5 && a.y < b.y + b.height - 0.5 && b.y < a.y + a.height - 0.5;

/** The primary controls are inside the width and none overlaps another (each measured in its own scroll position). */
async function noOverlap(p: Page, ids: string[], how: string) {
  const w = p.viewportSize()!.width;
  const boxes: { id: string; b: Awaited<ReturnType<typeof box>> }[] = [];
  for (const id of ids) {
    const l = p.getByTestId(id).first();
    if (!(await l.count()) || !(await l.isVisible())) continue;
    const b = await box(l);
    expect(b.x, `${how}: ${id} starts off screen`).toBeGreaterThanOrEqual(-0.5);
    expect(b.x + b.width, `${how}: ${id} runs past ${w}px`).toBeLessThanOrEqual(w + 0.5);
    boxes.push({ id, b: { ...b, y: b.y + (await p.evaluate(() => (document.querySelector(".n12-margin-scroll")?.scrollTop ?? 0) + window.scrollY)) } });
  }
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) expect(overlap(boxes[i].b, boxes[j].b), `${how}: ${boxes[i].id} overlaps ${boxes[j].id}`).toBe(false);
}

/** The pinned primary: wholly on screen, above the tab bar, and its button not covered by anything. */
async function pinnedInView(p: Page, how: string) {
  const btn = p.getByTestId("trip-next");
  await btn.scrollIntoViewIfNeeded();
  const b = (await btn.boundingBox())!;
  const v = p.viewportSize()!;
  expect(b.x >= -0.5 && b.x + b.width <= v.width + 0.5, `${how}: the primary runs off the side`).toBe(true);
  expect(b.y + b.height, `${how}: the primary sits under the tab bar`).toBeLessThanOrEqual(v.height - 56 + 0.5);
  const top = await p.evaluate(([x, y]) => document.elementFromPoint(x, y)?.closest('[data-testid="trip-next"]') !== null, [b.x + b.width / 2, b.y + b.height / 2]);
  expect(top, `${how}: something covers the primary`).toBe(true);
}

for (const width of [390, 320]) {
  test(`I: the lifecycle at ${width} px: plan, save, cancel, reopen, complete; nothing overlaps or scrolls sideways`, async ({ browser }) => {
    test.setTimeout(90_000);
    const ctx = await browser.newContext({ viewport: { width, height: width === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });
    await seed(ctx);
    const p = await ctx.newPage();
    await brief(p);
    await p.getByTestId("plan-date").scrollIntoViewIfNeeded();
    await p.getByTestId("plan-date").fill(tomorrow());
    await noOverlap(p, ["plan-date", "plan-time", "plan-more", "coast-to-trip", "coast-save-trip"], "planner");
    await noSideways(p, "planner");
    await p.getByTestId("coast-save-trip").tap();
    await expect(p).toHaveURL(/\/coast\/trips\?id=/);
    await expect(p.getByTestId("trip-lifecycle")).toHaveText("Planned");

    // the next action is pinned at the foot of the sheet's ticket (it rides over the feed by design): it is checked on its own
    await noOverlap(p, ["trip-heading", "trip-since", "trip-departures", "trip-status", "trip-money"], "planned");
    await pinnedInView(p, "planned");
    await noSideways(p, "planned");

    await p.getByTestId("trip-cancel").scrollIntoViewIfNeeded();
    await p.getByTestId("trip-cancel").tap();
    await expect(p.getByTestId("trip-lifecycle")).toHaveText("Cancelled");
    await expect(p.getByTestId("trip-next")).toContainText("Reopen this trip");
    await noOverlap(p, ["trip-heading", "trip-cancelled"], "cancelled");
    await pinnedInView(p, "cancelled");
    await noSideways(p, "cancelled");
    await p.getByTestId("trip-next").tap();
    await expect(p.getByTestId("trip-lifecycle")).toHaveText("Planned");

    await p.getByTestId("trip-complete").scrollIntoViewIfNeeded();
    await p.getByTestId("trip-complete").tap();
    const dialog = p.getByRole("dialog", { name: "Record what happened" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("textbox", { name: "Weight" }).fill("380");
    await dialog.getByRole("textbox", { name: "Price received" }).fill("9");
    await dialog.getByTestId("outcome-save").tap();
    await expect(p.getByTestId("trip-lifecycle")).toHaveText("Completed");
    await expect(p.getByTestId("planned-vs-actual")).toBeVisible();
    await noOverlap(p, ["trip-heading", "planned-vs-actual", "trip-departures"], "completed");
    await noSideways(p, "completed");

    await p.goto(`${OK}/coast/trips`);
    await expect(p.getByTestId("trips-history")).toBeVisible();
    await noSideways(p, "history");
    await ctx.close();
  });
}
