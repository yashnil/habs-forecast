import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * v12 moved Explore to /explore and made / the front door. Packs saved by earlier builds are not rewritten:
 * the service worker (public/sw.js, neritic-sw-5) answers the new addresses from them.
 */
test.use({ storageState: { cookies: [], origins: [] } });

test("saved packs open at the new addresses, and an old Explore link still finds its copy", async ({ page, context }) => {
  await page.goto(`${OK}/about`);
  await page.evaluate(async () => {
    await navigator.serviceWorker.register("/sw.js?b=nx-shell-test", { scope: "/" });
    await navigator.serviceWorker.ready;
    const html = (marker: string) => new Response(`<!doctype html><html><head></head><body><p id="saved">${marker}</p></body></html>`, { headers: { "Content-Type": "text/html; charset=utf-8", "x-neritic-saved-at": "2026-10-12T20:00:00Z", "x-neritic-saved-label": "Moss%20Landing" } });
    // a field pack and a Today save as an earlier build wrote them
    await (await caches.open("neritic-pack-monterey")).put("/__pack__/page/explore", html("saved explore"));
    await (await caches.open("neritic-pack-today")).put("/today?place=port-592", html("saved today at Moss Landing"));
  });
  await page.reload();
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  await context.setOffline(true);
  const open = async (path: string) => {
    await page.goto(`${OK}${path}`);
    return page.locator("#saved").textContent();
  };
  expect(await open("/explore")).toBe("saved explore");
  expect(await open("/?view=map")).toBe("saved explore");
  expect(await open("/")).toBe("saved today at Moss Landing");
  expect(await open("/today?place=port-592")).toBe("saved today at Moss Landing");
  // the copy says it is a saved copy, for the address it was saved for
  expect(await page.evaluate(() => (window as unknown as { __NERITIC_SAVED__?: { url: string } }).__NERITIC_SAVED__?.url)).toBe("/today?place=port-592");
  await context.setOffline(false);
});
