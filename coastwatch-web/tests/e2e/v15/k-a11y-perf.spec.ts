import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, newContext, seedTrip, serious, SIZES } from "./helpers";

/**
 * Accessibility and performance of the trips pages (DIRECTION §18). axe-core (WCAG 2.1 A/AA) finds no serious or
 * critical violation on the history, a trip's workspace and its briefing, at a desk and a phone size. Performance:
 * JavaScript transferred for /coast/trips against /coast (the trips page must not cost much more than My Coast) and
 * the largest contentful paint, measured with a PerformanceObserver; the figures are written to
 * test-results/v15-perf.json for docs/coastwatch/v15/qa/README.md.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const STATES: Array<[string, string, string]> = [
  ["history", "/coast/trips", "nx-trips"],
  ["workspace", "/coast/trips?id=k-trip", "trip-workspace"],
  ["briefing", "/coast/trips?id=k-trip&view=briefing", "trip-briefing"],
];

for (const [label, size] of [
  ["desk", SIZES.desk],
  ["phone", SIZES.phone],
] as const) {
  for (const [name, path, root] of STATES) {
    test(`a11y: ${name} at ${label} size has no serious or critical violation`, async ({ browser }) => {
      const seeded = await seedTrip({ id: "k-trip" });
      const { ctx, page } = await newContext(browser, size, seeded.storage);
      await page.goto(`${OK}${path}`);
      await expect(page.getByTestId(root).first()).toBeVisible({ timeout: 20_000 });
      await page.waitForTimeout(800); // entrances settle
      expect(await serious(page)).toEqual([]);
      await ctx.close();
    });
  }
}

/** Bytes on the wire per script (the browser's own count, compressed as served), and the largest contentful paint. */
async function measure(page: Page, url: string) {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Network.enable");
  const urls = new Map<string, string>();
  let js = 0;
  let files = 0;
  cdp.on("Network.responseReceived", (e) => urls.set(e.requestId, e.response.url));
  cdp.on("Network.loadingFinished", (e) => {
    const u = urls.get(e.requestId) ?? "";
    if (u.includes("/_next/static/") && /\.js(\?|$)/.test(u)) {
      js += e.encodedDataLength;
      files += 1;
    }
  });
  await page.addInitScript(() => {
    const w = window as unknown as { __lcp: number };
    w.__lcp = 0;
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) w.__lcp = Math.max(w.__lcp, e.startTime);
    }).observe({ type: "largest-contentful-paint", buffered: true });
  });
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const lcpMs = await page.evaluate(() => Math.round((window as unknown as { __lcp: number }).__lcp));
  return { jsKB: Math.round(js / 1024), jsFiles: files, lcpMs };
}

test("perf: JavaScript and LCP of /coast/trips against /coast", async ({ browser }) => {
  test.setTimeout(90_000);
  const seeded = await seedTrip({ id: "k-trip" });
  const out: Record<string, unknown> = {};
  for (const [name, path] of [
    ["/coast (trip worksheet)", `/coast?place=${MOSS}&view=trip`],
    ["/coast/trips (history)", "/coast/trips"],
    ["/coast/trips?id= (workspace)", "/coast/trips?id=k-trip"],
    ["/coast/trips briefing", "/coast/trips?id=k-trip&view=briefing"],
  ] as const) {
    // a cold load each time: a new context, so nothing is cached
    const { ctx, page } = await newContext(browser, SIZES.desk, seeded.storage);
    out[name] = await measure(page, `${OK}${path}`);
    await ctx.close();
  }
  mkdirSync("test-results", { recursive: true });
  writeFileSync(join("test-results", "v15-perf.json"), JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, null, 2));
  const coast = out["/coast (trip worksheet)"] as { jsKB: number };
  const ws = out["/coast/trips?id= (workspace)"] as { jsKB: number; lcpMs: number };
  // budget: the workspace costs at most 35 % more script than the worksheet it replaces, and paints within 2.5 s locally
  expect(ws.jsKB).toBeLessThanOrEqual(Math.round(coast.jsKB * 1.35));
  expect(ws.lcpMs).toBeLessThan(2500);
});
