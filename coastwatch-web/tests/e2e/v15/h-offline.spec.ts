import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium, expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "../ports";
import { NOW, planTrip, returning, seedTrip, SIZES } from "./helpers";

/**
 * Test H (DIRECTION §16, §17 H): a trip saved online opens after the network is gone and the browser has restarted.
 * One online visit keeps the trips page and its code (src/nx/offline/trips.ts, public/sw.js "neritic-trips-v1"); the
 * browser is then closed and a new browser process opens the same profile with no network at all: the trip, its
 * briefing, the economics and the history still work from the device's storage, and nothing claims to be live.
 */
test.describe.configure({ mode: "serial" });

const channel = process.env.CI ? undefined : "chrome";

async function launch(dir: string, offline: boolean): Promise<{ ctx: BrowserContext; page: Page; requests: string[]; failed: string[] }> {
  const ctx = await chromium.launchPersistentContext(dir, { channel, viewport: SIZES.desk, offline, serviceWorkers: "allow" });
  const requests: string[] = [];
  const failed: string[] = [];
  ctx.on("request", (r) => requests.push(r.url()));
  ctx.on("requestfailed", (r) => failed.push(r.url()));
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  await page.clock.setFixedTime(new Date(NOW));
  return { ctx, page, requests, failed };
}

test("H: plan and save online, kill the network, restart the browser: the trip, its briefing and history open", async () => {
  test.setTimeout(120_000);
  const dir = mkdtempSync(join(tmpdir(), "nx-v15-h-"));
  try {
    let id = "";
    // 1. online: the reader plans and saves the trip on My Coast, landing on its workspace
    {
      const { ctx, page } = await launch(dir, false);
      await page.goto(`${OK}/about`);
      await page.evaluate((kv) => {
        for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
      }, returning);
      id = await planTrip(page);
      // the page keeps itself for offline use once it has drawn
      await expect
        .poll(() => page.evaluate(async () => (await caches.has("neritic-trips-v1")) && !!(await (await caches.open("neritic-trips-v1")).match("/coast/trips"))), { timeout: 30_000 })
        .toBe(true);
      await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller), { timeout: 15_000 }).toBe(true);
      await ctx.close();
    }
    // 2. a new browser process, the same profile, no network
    {
      const { ctx, page, requests, failed } = await launch(dir, true);
      await page.goto(`${OK}/coast/trips?id=${id}`);
      await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
      await expect(page.getByTestId("trip-money-figure")).toContainText("$2,602.24");
      await page.goto(`${OK}/coast/trips?id=${id}&view=briefing`);
      await expect(page.getByTestId("trip-briefing")).toBeVisible({ timeout: 20_000 });
      await expect(page.getByRole("heading", { level: 1, name: "Moss Landing" })).toBeVisible();
      await expect(page.getByTestId("briefing-economics")).toContainText("$2,602.24");
      await expect(page.getByTestId("briefing-basis")).toContainText("Planning snapshot");
      // the page says it is a saved copy, never live
      const mark = await page.evaluate(() => (window as unknown as { __NERITIC_SAVED__?: { at: string | null } }).__NERITIC_SAVED__ ?? null);
      expect(mark?.at, "the saved page carries the time it was kept").toBeTruthy();
      await expect(page.getByTestId("nx-offline-notice")).toBeVisible();
      await expect(page.getByTestId("nx-offline-notice")).not.toContainText(/live|current/i);
      // the history opens too (the same kept page, another address)
      await page.goto(`${OK}/coast/trips`);
      await expect(page.getByTestId("history")).toContainText("Moss Landing");
      // no request of the offline run left the device successfully: everything came from the cache or storage
      expect(failed.filter((u) => u.startsWith(OK) && u.includes("/_next/static/"))).toEqual([]);
      expect(requests.filter((u) => !u.startsWith(OK) && !u.startsWith("data:") && !u.startsWith("blob:"))).toEqual([]);
      await ctx.close();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("H: offline after a restart, the reader edits economics, records the outcome and sees it in history", async () => {
  test.setTimeout(150_000);
  const dir = mkdtempSync(join(tmpdir(), "nx-v15-h2-"));
  try {
    const seeded = await seedTrip({ id: "h2-trip" });
    {
      const { ctx, page } = await launch(dir, false);
      await page.goto(`${OK}/about`);
      await page.evaluate((kv) => {
        for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
      }, { ...returning, ...seeded.storage });
      await page.goto(`${OK}/coast/trips?id=h2-trip`);
      await expect(page.getByTestId("trip-workspace")).toBeVisible();
      await expect.poll(() => page.evaluate(async () => !!(await (await caches.open("neritic-trips-v1")).match("/coast/trips"))), { timeout: 30_000 }).toBe(true);
      await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller), { timeout: 15_000 }).toBe(true);
      await ctx.close();
    }
    const { ctx, page } = await launch(dir, true);
    await page.goto(`${OK}/coast/trips?id=h2-trip`);
    await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("trip-money-figure")).toContainText("$2,602.24");
    // economics recalculate offline, through the same engine: 500 lb at $9 → $3,322.24 net to boat
    await page.getByTestId("trip-open-money").click();
    await page.getByTestId("ledger-open").click();
    const sheet = page.getByTestId("worksheet");
    await sheet.getByRole("textbox", { name: "Catch", exact: true }).fill("500");
    await expect(page.getByTestId("worksheet-net")).toContainText("$3,322.24");
    await page.getByTestId("worksheet-save").click();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("trip-money-figure")).toContainText("$3,322.24");
    // record the outcome offline
    await page.getByTestId("trip-more").click();
  await page.getByTestId("trip-complete").click();
    const form = page.getByTestId("outcome-form");
    await form.getByLabel("Weight", { exact: true }).first().fill("450");
    await form.getByLabel("Price received", { exact: true }).first().fill("8.50");
    await form.getByTestId("outcome-save").click();
    await expect(page.getByTestId("planned-vs-actual")).toBeVisible();
    await expect(page.getByTestId("trip-workspace")).toHaveAttribute("data-state", "completed");
    // the history, offline: the trip is listed as completed
    await page.goto(`${OK}/coast/trips`);
    const row = page.getByTestId("history-row").filter({ hasText: "Moss Landing" });
    await expect(row).toHaveAttribute("data-state", "completed");
    // a second restart, still offline, keeps every edit (the device's storage, not the page)
    await ctx.close();
    const again = await launch(dir, true);
    await again.page.goto(`${OK}/coast/trips?id=h2-trip`);
    await expect(again.page.getByTestId("trip-workspace")).toHaveAttribute("data-state", "completed", { timeout: 20_000 });
    const stored = JSON.parse((await again.page.evaluate(() => localStorage.getItem("nx-trips-v1:trip:h2-trip"))) ?? "{}");
    expect(stored.outcome.landed[0]).toMatchObject({ lb: 450, usdPerLb: 8.5 });
    expect(stored.economics.inputs.catch[0].lb).toBe(500);
    await again.ctx.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
