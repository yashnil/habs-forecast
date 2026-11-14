import { expect, test } from "@playwright/test";
import { OK } from "../ports";
import { NS, newContext, seedTrip, tripKeys, watchRequests } from "./helpers";

/**
 * Test G (DIRECTION §16, §17 G): trips are device-local by default. The trip lives only in this browser context's
 * localStorage; another context (another person, another profile) sees none of it; deleting removes the trip and its
 * snapshots; and no request the app makes carries the trip's own data (its id, name, notes, area, catch or money).
 */
test.use({ storageState: { cookies: [], origins: [] } });

const NOTE = "e2e-private-note-7f3a";
const AREA = "e2e-private-area-outside-the-bay";

test("G: a trip stays in this context's storage, and no request carries its data", async ({ browser }) => {
  const seeded = await seedTrip({ id: "g-trip", edit: (t) => ({ ...t, notes: NOTE, area: AREA }) });
  const { ctx, page, errors } = await newContext(browser, undefined, seeded.storage);
  // the trip's id is in the page's own address (?id=), so it reaches this site's server as any address does; it is an
  // opaque key. What the trip holds (its notes, area, name and money) must never leave the device.
  const seen = watchRequests(ctx, [NOTE, AREA, "2602.24", "2,602.24", "Moss Landing, Dungeness crab", '"economics"']);
  await page.goto(`${OK}/coast/trips?id=g-trip`);
  await page.waitForLoadState("networkidle");
  await page.goto(`${OK}/coast/trips?id=g-trip&view=briefing`);
  await expect(page.getByTestId("trip-briefing")).toBeVisible();
  await page.waitForLoadState("networkidle");
  await page.goto(`${OK}/coast/trips`);
  await page.waitForLoadState("networkidle");
  expect(seen.hits, "no request may carry the trip's data").toEqual([]);
  // the data is in this context's localStorage, under the trips namespace only
  const keys = await tripKeys(page);
  expect(keys).toEqual(expect.arrayContaining([`${NS}trip:g-trip`, `${NS}snap:g-trip-plan`]));
  // nothing of it in cookies or session storage
  expect((await ctx.cookies()).filter((c) => c.value.includes("g-trip"))).toEqual([]);
  expect(await page.evaluate(() => Object.values(sessionStorage).join("\n"))).not.toContain("g-trip");

  // another context (another person on another device) has no trips and cannot open this one
  const other = await newContext(browser);
  await other.page.goto(`${OK}/coast/trips?id=g-trip&view=briefing`);
  expect(await tripKeys(other.page)).not.toContain(`${NS}trip:g-trip`);
  await expect(other.page.locator("body")).not.toContainText(NOTE);
  await expect(other.page.locator("body")).not.toContainText("$2,602.24");
  await other.ctx.close();
  expect(errors).toEqual([]);
  await ctx.close();
});

test("G: Delete everything on this device removes the trips, their snapshots and the kept trips page", async ({ browser }) => {
  const seeded = await seedTrip({ id: "g-del" });
  const { ctx, page } = await newContext(browser, undefined, seeded.storage);
  await page.goto(`${OK}/coast/trips?id=g-del`);
  await expect.poll(() => page.evaluate(() => caches.has("neritic-trips-v1")), { timeout: 30_000 }).toBe(true);
  await page.goto(`${OK}/account`);
  await page.getByTestId("account-delete").scrollIntoViewIfNeeded();
  await page.getByTestId("account-delete").click();
  await page.getByTestId("account-delete-yes").click();
  // the page may reload once everything is gone: read the device again after it settles
  const read = async () => {
    try {
      return { keys: await tripKeys(page), cache: await page.evaluate(() => caches.has("neritic-trips-v1")) };
    } catch {
      return null;
    }
  };
  await expect.poll(read, { timeout: 15_000 }).toEqual({ keys: [], cache: false });
  await ctx.close();
});

test("G: deleting one trip from the history removes the trip and every snapshot of it, and nothing else", async ({ browser }) => {
  const one = await seedTrip({ id: "g-one" });
  const two = await seedTrip({ id: "g-two", edit: (t) => ({ ...t, name: "Second trip" }) });
  // both trips in one index, as the store writes them
  const index = JSON.stringify({ ...JSON.parse(one.storage[`${NS}index`]), ids: ["g-two", "g-one"] });
  const { ctx, page } = await newContext(browser, undefined, { ...one.storage, ...two.storage, [`${NS}index`]: index });
  await page.goto(`${OK}/coast/trips`);
  const row = page.getByTestId("history-row").filter({ hasText: "Moss Landing, Dungeness crab" });
  await expect(row).toHaveCount(1);
  await row.getByRole("button", { name: /More for/ }).click();
  await page.getByTestId("history-delete").click();
  await expect(page.getByTestId("history-row").filter({ hasText: "Moss Landing, Dungeness crab" })).toHaveCount(0);
  const keys = await tripKeys(page);
  expect(keys).not.toContain(`${NS}trip:g-one`);
  expect(keys).not.toContain(`${NS}snap:g-one-plan`);
  expect(keys).toEqual(expect.arrayContaining([`${NS}trip:g-two`, `${NS}snap:g-two-plan`]));
  await ctx.close();
});

test("G: a trip id that is not on this device shows a clean not-found state, never another device's or a server's trip", async ({ browser }) => {
  // another context has a trip with this id; this one does not: the id alone opens nothing
  const seeded = await seedTrip({ id: "g-elsewhere", edit: (t) => ({ ...t, notes: NOTE }) });
  const owner = await newContext(browser, undefined, seeded.storage);
  await owner.page.goto(`${OK}/coast/trips?id=g-elsewhere`);
  await expect(owner.page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });

  const { ctx, page, errors } = await newContext(browser);
  const trips: string[] = [];
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (u.origin === new URL(OK).origin && !u.pathname.startsWith("/_next/") && u.pathname !== "/coast/trips" && /trip/i.test(u.pathname + u.search)) trips.push(r.url());
  });
  for (const id of ["g-elsewhere", "not-a-trip", "%3Cscript%3E"]) {
    await page.goto(`${OK}/coast/trips?id=${id}`);
    await expect(page.getByRole("heading", { name: "This trip is not on this device." })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("trip-workspace")).toHaveCount(0);
    await expect(page.locator("body")).not.toContainText(NOTE);
    await expect(page.getByRole("link", { name: "All trips on this device" })).toBeVisible();
  }
  expect(trips, "nothing asks a server for a trip").toEqual([]);
  expect(await tripKeys(page)).toEqual(expect.not.arrayContaining([`${NS}trip:g-elsewhere`]));
  expect(errors).toEqual([]);
  await ctx.close();
  await owner.ctx.close();
});
