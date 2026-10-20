import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Loading and lifecycle states (v4): a slow provider is said for that part only while the rest of the
 * page works; the device going offline and coming back is said and recovered from; a saved field pack
 * ages, survives a service-worker update, and when the browser evicts it, the page says it is gone.
 */
const NOW = "2026-10-08T20:00:00Z";
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const status = async (page: Page, id: string) => {
  await page.getByTestId("status-center").click();
  const s = await page.getByTestId("status-panel").getByTestId(`source-${id}`).getAttribute("data-state");
  await page.keyboard.press("Escape");
  return s;
};

test("a slow conditions service: the briefing says it is reading, the rest is there, then the statements arrive", async ({ page }) => {
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.route("**/api/conditions**", async (r) => {
    await delay(4000);
    await r.continue();
  });
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("matters-reading")).toContainText("Reading the notices");
  // the page around it is already usable
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  await expect(page.getByTestId("brief-official")).toBeVisible();
  expect(await status(page, "evidence")).toBe("loading");
  expect(await status(page, "conditions")).toBe("loading");
  await expect(page.getByTestId("matters").locator(":scope > li").first()).toBeVisible({ timeout: 15000 });
  expect(await status(page, "evidence")).not.toBe("loading");
});

test("a slow model grid: the status centre says the forecast is loading, and Ready once it is drawn", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  let held = true;
  await page.route(/\/charm\/.*\.(bin|gz|json|png)/, async (r) => {
    while (held) await delay(100);
    await r.continue();
  });
  await page.goto(`${OK}/?layer=forecast`);
  await expect(page.getByTestId("notices-button")).toBeVisible();
  await expect.poll(() => status(page, "model")).toBe("loading");
  held = false;
  await expect.poll(() => status(page, "model"), { timeout: 20000 }).toBe("ready");
});

test("going offline is said at once and persistently; coming back clears it and the page works again", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await expect(page.getByTestId("harbor-claims").locator("[data-kind]").first()).toBeVisible();
  await ctx.setOffline(true);
  await expect(page.getByTestId("offline-banner")).toContainText("Offline");
  await expect(page.getByTestId("status-center")).toHaveAttribute("data-state", "offline");
  await ctx.setOffline(false);
  await expect(page.getByTestId("offline-banner")).toHaveCount(0);
  await expect(page.getByTestId("status-center")).not.toHaveAttribute("data-state", "offline");
  // a new request after recovery is answered
  await page.goto(`${OK}/?layer=forecast&sel=harbor:593`);
  await expect(page.getByTestId("harbor-claims").locator("[data-kind]").first()).toBeVisible();
  await ctx.close();
});

test("a saved pack: older than a day it is marked stale; it survives a service-worker update; evicted, it is said to be gone", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`${OK}/coast?place=port-550`);
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await page.getByTestId("pack-open").click();
  const go = page.getByTestId("pack-download");
  await expect(go).toBeEnabled({ timeout: 60_000 });
  await go.click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });

  // a day and a half later
  await page.clock.setFixedTime(new Date(Date.now() + 36 * 3_600_000));
  await page.reload();
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-status")).toContainText("More than a day old");
  await page.keyboard.press("Escape");
  await page.reload();
  expect(await status(page, "pack")).toBe("stale");

  // a new service worker version: an older version's static cache is removed, the field pack is kept.
  // (Playwright cannot serve a changed worker script, so the update is a new script URL, which the
  // browser installs and activates as an update of the same scope.)
  await page.evaluate(() => caches.open("neritic-static-v0").then((c) => c.put("/old", new Response("old"))));
  const scripts = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.register("/sw.js?v=update-test", { scope: "/" });
    await new Promise<void>((done) => {
      const w = reg.installing ?? reg.waiting;
      if (!w) return done();
      w.addEventListener("statechange", () => w.state === "activated" && done());
    });
    return reg.active?.scriptURL ?? "";
  });
  expect(scripts).toContain("v=update-test");
  await expect.poll(() => page.evaluate(() => caches.has("neritic-static-v0")), { timeout: 20000 }).toBe(false);
  expect(await page.evaluate(() => caches.has("neritic-pack-monterey"))).toBe(true);
  await page.reload();
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true");
  await page.keyboard.press("Escape");

  // the browser evicts the storage
  await page.evaluate(() => caches.delete("neritic-pack-monterey"));
  await page.reload();
  expect(await status(page, "pack")).toBe("unavailable");
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-evicted")).toContainText("the browser removed it");
  await expect(page.getByTestId("pack-download")).toBeVisible({ timeout: 60_000 });
  await ctx.close();
});
