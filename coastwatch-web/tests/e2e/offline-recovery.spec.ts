import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Offline field pack (v4): what happens when things go wrong, and that the page says the truth.
 * - storage runs out part way through a download (a real quota, set through the browser's DevTools
 *   protocol): the panel says storage is full, nothing claims to be saved or verified, what was saved
 *   is kept and said to be unfinished, and after space is freed the download finishes;
 * - a pack saved by an older version of the app, a damaged record, a record with no files behind it,
 *   files with no record: each is said as it is, and one download repairs it;
 * - a new service worker (its code cache renamed, the old one removed) while the app's code files have
 *   new names: the saved pages still open with no network, on the code the pack holds.
 * "No network" here: every request, the service worker's own included, is refused (context routing),
 * and the browser reports itself offline. The proof with the app server stopped and the browser
 * restarted is scripts/offline-proof.mjs.
 */
const PACK = "neritic-pack-monterey";
const RECORD = "/__pack__/record.json";

async function openPanel(page: Page) {
  await page.goto(`${OK}/coast?place=port-550`);
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
}
async function save(page: Page) {
  await openPanel(page);
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
}
const record = (page: Page) =>
  page.evaluate(
    async ([name, key]) => {
      const r = await (await caches.open(name)).match(key);
      return r ? ((await r.json()) as { version: number; complete: boolean; items: { url: string }[]; verified: { missing: number } }) : null;
    },
    [PACK, RECORD],
  );
async function reopen(page: Page) {
  await page.reload();
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
}
async function closePanel(page: Page) {
  await page.getByTestId("offline-panel").getByRole("button", { name: "Close" }).click();
  await expect(page.getByTestId("offline-panel")).toHaveCount(0);
}
/** Every request refused while "offline" (what the pages or the service worker asked of the network). */
const refused: string[] = [];
async function goOffline(ctx: BrowserContext) {
  await ctx.route("**/*", (r) => {
    refused.push(r.request().url());
    return r.abort("internetdisconnected");
  });
  await ctx.setOffline(true);
}
async function goOnline(ctx: BrowserContext) {
  await ctx.unrouteAll({ behavior: "ignoreErrors" });
  await ctx.setOffline(false);
}

test("storage runs out during a download: said as full, nothing claimed saved, the part kept, finished after space is freed", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await openPanel(page);
  // a quota a little above what the site already uses: the basemap (about 7 MB) cannot fit
  const cdp = await ctx.newCDPSession(page);
  const used = await page.evaluate(async () => (await navigator.storage.estimate()).usage ?? 0);
  await cdp.send("Storage.overrideQuotaForOrigin", { origin: OK, quotaSize: used + 2_500_000 });
  await page.getByTestId("pack-download").click();

  const panel = page.getByTestId("offline-panel");
  await expect(page.getByTestId("pack-quota")).toContainText("storage for this site is full", { timeout: 90_000 });
  await expect(page.getByTestId("pack-quota")).toContainText("not complete");
  // nothing claims to be saved or verified
  const status = page.getByTestId("pack-status");
  // the status re-reads the stored record after the quota error: a slow runner can take a few seconds
  await expect(status).toHaveAttribute("data-complete", "false", { timeout: 20_000 });
  await expect(status).toContainText("Incomplete");
  await expect(status).not.toContainText("verified");
  await expect(panel).not.toContainText("Saved and verified");
  await expect(page.getByTestId("pack-download")).toContainText("Finish the download");
  // the part that fitted is kept, and its record says it is unfinished
  const partial = await record(page);
  expect(partial?.complete).toBe(false);
  expect(partial!.items.length).toBeGreaterThan(0);
  expect(partial!.verified.missing).toBeGreaterThan(0);
  // the status centre says incomplete, never ready
  await closePanel(page);
  await page.getByTestId("status-center").click();
  await expect(page.getByTestId("status-panel").getByTestId("source-pack")).toHaveAttribute("data-state", "incomplete");
  await page.keyboard.press("Escape");

  // space is freed: the download finishes, keeping what was already saved
  await cdp.send("Storage.overrideQuotaForOrigin", { origin: OK });
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toContainText("Finish the download", { timeout: 60_000 });
  await page.getByTestId("pack-download").click();
  await expect(status).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
  await expect(status).toContainText("verified");
  await expect(page.getByTestId("pack-quota")).toHaveCount(0);
  const done = await record(page);
  expect(done).toMatchObject({ complete: true, verified: { missing: 0 } });
  expect(errors).toEqual([]);
  await ctx.close();
});

test("an older pack format, a damaged record, missing files and an unfinished download are each said, and one download repairs them", async ({ browser }) => {
  test.setTimeout(240_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await save(page);
  expect((await record(page))?.version).toBe(2);

  // (b) a pack saved by an older version: kept and usable, said to be older, and downloading again updates it
  await page.evaluate(
    async ([name, key]) => {
      const c = await caches.open(name);
      const r = await (await c.match(key))!.json();
      await c.put(key, new Response(JSON.stringify({ ...r, version: 1 }), { headers: { "content-type": "application/json" } }));
    },
    [PACK, RECORD],
  );
  await reopen(page);
  await expect(page.getByTestId("pack-older")).toContainText("earlier version of Neritic");
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true");
  await expect(page.getByTestId("pack-download")).toContainText("Download again");
  // offline, the older pack still opens, and the banner says it is the older pack
  await page.keyboard.press("Escape");
  await goOffline(ctx);
  await page.goto(`${OK}/?sel=harbor:550&layer=forecast`);
  await expect(page.getByTestId("offline-banner")).toContainText("older pack");
  await expect.poll(() => page.evaluate(() => (window as unknown as { __cwMap?: { getStyle(): { sources: { omt?: { url?: string } } } } }).__cwMap?.getStyle().sources.omt?.url ?? ""), { timeout: 20_000 }).toMatch(/^pmtiles:/);
  await goOnline(ctx);
  await page.goto(`${OK}/coast?place=port-550`);
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toContainText("verified", { timeout: 90_000 });
  await expect(page.getByTestId("pack-older")).toHaveCount(0);
  expect((await record(page))?.version).toBe(2);

  // (c) a file the record lists is gone: not complete, said, and finished by one download
  const gone = await page.evaluate(
    async ([name, key]) => {
      const c = await caches.open(name);
      const r = (await (await c.match(key))!.json()) as { items: { url: string }[] };
      const u = r.items.find((i) => i.url.includes("/currents/"))!.url;
      await c.delete(u);
      return u;
    },
    [PACK, RECORD],
  );
  expect(gone).toContain("/currents/");
  await reopen(page);
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "false");
  await expect(page.getByTestId("pack-status")).toContainText("1 of");
  await expect(page.getByTestId("pack-missing")).toBeVisible();
  await expect(page.getByTestId("pack-download")).toContainText("Finish the download");
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
  expect(await page.evaluate(([name, u]) => caches.open(name).then((c) => c.match(u)).then((r) => !!r), [PACK, gone])).toBe(true);

  // (c) a damaged record: not counted as saved, said, and replaced by a download
  await page.evaluate(([name, key]) => caches.open(name).then((c) => c.put(key, new Response("{not json", { headers: { "content-type": "application/json" } }))), [PACK, RECORD]);
  await reopen(page);
  await expect(page.getByTestId("pack-problem")).toHaveAttribute("data-problem", "damaged-record");
  await expect(page.getByTestId("pack-status")).toHaveCount(0);
  await closePanel(page);
  await page.getByTestId("status-center").click();
  await expect(page.getByTestId("status-panel").getByTestId("source-pack")).toHaveAttribute("data-state", "incomplete");
  await page.keyboard.press("Escape");
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });

  // files with no record (a download cut off before it was written): an unfinished download
  await page.evaluate(([name, key]) => caches.open(name).then((c) => c.delete(key)), [PACK, RECORD]);
  await reopen(page);
  await expect(page.getByTestId("pack-problem")).toHaveAttribute("data-problem", "unfinished");
  await expect(page.getByTestId("pack-download")).toContainText("Finish the download");
  await expect(page.getByTestId("pack-evicted")).toHaveCount(0);
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
  await ctx.close();
});

test("a new service worker with a renamed code cache, and code files renamed by a deploy: the saved pages still open offline", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await save(page);
  // the app registers its worker as /sw.js?b=<build>, and the code cache is named for that build
  const staticCaches = () => page.evaluate(() => caches.keys().then((k) => k.filter((x) => x.startsWith("neritic-static-"))));
  const script = await page.evaluate(() => navigator.serviceWorker.ready.then((r) => r.active?.scriptURL ?? ""));
  const build = new URL(script).searchParams.get("b");
  expect(build).toBeTruthy();
  await expect.poll(staticCaches).toContain(`neritic-static-${build}`);

  // (a) a worker update (a new script URL: Playwright cannot serve a changed worker script). A version
  // with a renamed code cache removes the old one on activation (states.spec: neritic-static-v0 goes);
  // here its result: the old code cache is gone and the new one starts empty.
  const active = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.register("/sw.js?b=next-build&v=2", { scope: "/" });
    await new Promise<void>((done) => {
      const w = reg.installing ?? reg.waiting;
      if (!w) return done();
      w.addEventListener("statechange", () => w.state === "activated" && done());
    });
    return reg.active?.scriptURL ?? "";
  });
  expect(active).toContain("v=2");
  // the new build's worker dropped the old build's code cache; its own starts empty
  await expect.poll(staticCaches).not.toContain(`neritic-static-${build}`);
  await page.evaluate(() => caches.keys().then((k) => Promise.all(k.filter((x) => x.startsWith("neritic-static-")).map((x) => caches.delete(x)))));
  expect(await page.evaluate(() => caches.has("neritic-pack-monterey"))).toBe(true);

  // (d) a deploy renamed every code file while the device was away: the server no longer has the saved
  // pages' code, and the new code cache is empty. The saved pages run on the pack's own copy.
  await goOffline(ctx);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey", { timeout: 20_000 });
  await expect(page.getByTestId("offline-banner")).toContainText(/Offline · Monterey Bay saved/);
  // the briefing settles first (its worker update check and link prefetches fail offline): a navigation
  // started while those are in flight can be cancelled on a slow runner (net::ERR_ABORTED), not by the app
  await page.waitForLoadState("networkidle");
  await page.goto(`${OK}/?sel=harbor:550&layer=forecast`);
  await expect(page.getByTestId("inspect-title")).toHaveText("Monterey", { timeout: 20_000 });
  await expect.poll(() => page.evaluate(() => !!(window as unknown as { __cwMap?: { getSource(id: string): unknown } }).__cwMap?.getSource("forecast")), { timeout: 20_000 }).toBe(true);
  // every code file came from the pack: none was kept by the new worker's code cache
  // no code file was asked of the network: each came from the pack
  expect(refused.filter((u) => u.includes("/_next/static/"))).toEqual([]);
  expect(errors).toEqual([]);
  await goOnline(ctx);
  await ctx.close();
});

test("offline, the saved briefing is dated, never 'now'; its age is said once outdated; a place no pack holds says so; no hydration error", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await save(page);
  // online, the briefing is live
  await page.reload();
  await expect(page.getByTestId("matters-heading")).toHaveText("What matters now");

  await goOffline(ctx);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  // three days later, another trip of the same harbour: the pack's saved page answers it
  await page.clock.setFixedTime(new Date(Date.now() + 3 * 86_400_000 + 3_600_000));
  await page.goto(`${OK}/coast?place=port-550&for=commercial`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey", { timeout: 20_000 });
  await expect(page.getByTestId("matters-heading")).toContainText(/What mattered when saved · \w{3} \d{1,2}, \d{1,2}:\d{2} [AP]M/);
  await expect(page.getByTestId("matters-heading")).not.toContainText("now");
  await expect(page.getByTestId("offline-banner")).toContainText(/Offline · Monterey Bay saved .* · 3 days old/);
  // the pack saved the briefing for all species: another trip is told so, never that it was saved for it
  await expect(page.getByTestId("offline-banner")).toContainText("saved for all species");
  // the address bar's own trip, after hydrating the saved HTML
  await expect(page.getByTestId("trip-summary")).toContainText("Commercial fishing");

  // a harbour no pack holds: said, with the way to the saved one; never Monterey's banner over it
  await page.goto(`${OK}/coast?place=port-201`);
  await expect(page.getByTestId("coast-title")).toHaveText("Crescent City", { timeout: 20_000 });
  await expect(page.getByTestId("offline-banner")).toContainText("Crescent City is not saved for offline use");
  await expect(page.getByTestId("offline-banner")).not.toContainText("Monterey Bay saved");
  await expect(page.getByTestId("matters-heading")).toHaveText("What matters now");
  await page.getByTestId("offline-outside").getByRole("link", { name: /Open Monterey Bay/ }).click();
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  expect(errors.filter((e) => /#418|[Hh]ydrat/.test(e))).toEqual([]);
  await goOnline(ctx);
  await ctx.close();
});

test("the pack dialog: focus moves in and stays, Escape closes it and returns focus; a running download survives closing and shows its progress when reopened", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await openPanel(page);
  const dialog = page.getByTestId("pack-dialog");
  await expect(dialog).toBeFocused();
  // Tab never leaves the dialog
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press("Tab");
    expect(await dialog.evaluate((d) => d.contains(document.activeElement))).toBe(true);
  }
  const selBefore = new URL(page.url()).searchParams.get("sel");
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.getByTestId("pack-open")).toBeFocused();
  // Escape closed the dialog only: the selection behind it is still open
  expect(new URL(page.url()).searchParams.get("sel")).toBe(selBefore);

  // slow the download so it is still running when the dialog closes
  await ctx.route(/\/charm\/|\/satellite\//, async (r) => {
    await new Promise((d) => setTimeout(d, 150));
    await r.continue();
  });
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-progress")).toBeVisible();
  // a tap on the backdrop does not close it while downloading
  await page.mouse.click(10, 10);
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-progress")).toBeVisible();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 120_000 });
  await ctx.unrouteAll({ behavior: "ignoreErrors" });
  await ctx.close();
});

test("the pack holds every lazily loaded chunk of the briefing: a section that loads its code on demand (trip economics) opens offline", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  const t0 = Date.now();
  await save(page);
  // planning reads the pages off-screen until they go quiet (no fixed 6 s waits)
  expect(Date.now() - t0).toBeLessThan(90_000);
  const rec = await record(page);
  const saved = new Set(rec!.items.map((i) => i.url));
  // every chunk the briefing's code names (the loaders of its lazy sections, followed to the end)
  const named = await page.evaluate(async () => {
    const seen = new Set<string>();
    let next = performance
      .getEntriesByType("resource")
      .map((e) => new URL(e.name))
      .filter((u) => u.origin === location.origin && /^\/_next\/static\/chunks\/.+\.js$/.test(u.pathname))
      .map((u) => u.pathname);
    while (next.length) {
      const found: string[] = [];
      for (const js of next) {
        if (seen.has(js)) continue;
        seen.add(js);
        const txt = await (await fetch(js)).text();
        for (const m of txt.matchAll(/["'](?:\/_next\/)?(static\/chunks\/[^"'\\\s]+\.js)["']/g)) found.push(`/_next/${m[1]}`);
      }
      next = found.filter((x) => !seen.has(x));
    }
    return [...seen];
  });
  expect(named.length).toBeGreaterThan(3);
  expect(named.filter((u) => !saved.has(u))).toEqual([]);

  await goOffline(ctx);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  refused.length = 0;
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey", { timeout: 20_000 });
  // trip economics (My Coast) loads its worksheet on demand: opened offline, its code comes from the pack
  const econ = page.locator("#trip-economics");
  if (await econ.count()) {
    await econ.scrollIntoViewIfNeeded();
    const opener = econ.locator("summary, button[aria-expanded='false']").first();
    if (await opener.count()) await opener.click();
    await page.waitForTimeout(1500);
  }
  // and every lazily named chunk answers offline (from the pack), whichever section asks for it
  const failed = await page.evaluate(async (urls) => {
    const out: string[] = [];
    for (const u of urls) {
      try {
        if (!(await fetch(u)).ok) out.push(u);
      } catch {
        out.push(u);
      }
    }
    return out;
  }, named);
  expect(failed).toEqual([]);
  expect(refused.filter((u) => u.includes("/_next/static/"))).toEqual([]);
  expect(errors).toEqual([]);
  await goOnline(ctx);
  await ctx.close();
});

test("a slow network gives way to the pack after 5 s; a page the pack never held gets the short offline page, never the saved Explore", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await save(page);
  // a hung network (lie-fi): every request waits 30 s
  await ctx.route("**/*", async (r) => {
    await new Promise((d) => setTimeout(d, 30_000));
    await r.continue().catch(() => undefined);
  });
  let t = Date.now();
  await page.goto(`${OK}/coast?place=port-550`, { waitUntil: "domcontentloaded", timeout: 20_000 });
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey", { timeout: 15_000 });
  expect(Date.now() - t).toBeLessThan(15_000);
  t = Date.now();
  await page.goto(`${OK}/research`, { waitUntil: "domcontentloaded", timeout: 20_000 });
  await expect(page.locator("h1")).toHaveText("You are offline");
  expect(Date.now() - t).toBeLessThan(12_000);
  await ctx.unrouteAll({ behavior: "ignoreErrors" });
  // offline: a path the pack never saved (Fisheries) is the offline page at once, not the saved Explore
  await goOffline(ctx);
  t = Date.now();
  await page.goto(`${OK}/fisheries`, { waitUntil: "domcontentloaded" });
  await expect(page.locator("h1")).toHaveText("You are offline");
  await expect(page.getByTestId("explore")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Open the saved briefing" })).toBeVisible();
  expect(Date.now() - t).toBeLessThan(5_000);
  await goOnline(ctx);
  await ctx.close();
});
