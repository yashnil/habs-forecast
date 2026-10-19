import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Release hardening (2026-10-19): the official notices keep the agency's dates, Neritic's retrieval and
 * the review state apart; and the walkthrough's low-severity issues (L1–L9) stay fixed.
 */
const NOW = "2026-10-08T20:00:00Z";
type M = { loaded(): boolean; getLayer(id: string): unknown; getLayoutProperty(id: string, p: string): unknown };
const mapReady = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());
const phone = (width = 390) => ({ viewport: { width, height: width === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });

test("a notice keeps the agency's dates, Neritic's retrieval and the human review state apart", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await page.getByTestId("notices-button").click();
  const row = page.getByTestId("notice-cdph-2026-annual-mussel-quarantine");
  await row.locator("summary", { hasText: "The agency’s wording" }).click();
  const p = row.getByTestId("notice-provenance");
  await expect(p).toContainText("Took effect");
  await expect(p).toContainText("(the agency's date)");
  // the press release's own date, named by its release, never merged with when it took effect
  await expect(row.getByTestId("notice-published")).toContainText(/Sep 14(, 2026)? \(CDPH SN26-018/);
  await expect(row.getByTestId("notice-retrieved")).toContainText(/^Oct 8/);
  await expect(row.getByTestId("notice-review")).toHaveText("No. Transcribed by an AI assistant; not yet checked by a person against the agency's page");
});

test("L1 + L2: on a phone the search box and the evidence's links and controls are at least 44 px tall", async ({ browser }) => {
  const ctx = await browser.newContext(phone());
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  expect((await page.getByTestId("place-search-input").boundingBox())!.height).toBeGreaterThanOrEqual(44);
  await page.goto(`${OK}/coast?place=port-550`);
  await page.getByTestId("matters").waitFor();
  const small = await page.locator('[data-testid="matters"] a, [data-testid="matters"] button, [data-testid="matters"] summary').evaluateAll((els) =>
    els.filter((e) => (e as HTMLElement).offsetParent !== null && e.getBoundingClientRect().height < 44).map((e) => `${e.textContent?.trim().slice(0, 30)} ${Math.round(e.getBoundingClientRect().height)}`),
  );
  expect(small).toEqual([]);
  await ctx.close();
});

test("L3 + L5: a harbour's marker reserves its place among labels; a latitude limit is named along its line", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await mapReady(page);
  const got = await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: M }).__cwMap;
    return { collider: !!m.getLayer("ports-collider"), placement: m.getLayoutProperty("official-limit-label", "text-anchor") };
  });
  expect(got.collider).toBe(true);
  expect(got.placement).toBe("bottom-right"); // at the coast end, running out to sea
});

test("L4: a layer's subtitle (what it is and how new) is never cut off", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  const cut = await page.getByTestId("layer-list").locator("span.block.text-\\[13px\\]").evaluateAll((els) => els.filter((e) => e.scrollWidth > e.clientWidth + 1).map((e) => e.textContent));
  expect(cut).toEqual([]);
});

test("L6: offline at 320 px, the banner never covers the 'Notices not verified' disclosure", async ({ browser }) => {
  const ctx = await browser.newContext(phone(320));
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await mapReady(page);
  await ctx.setOffline(true);
  await expect(page.getByTestId("offline-banner")).toBeVisible();
  const warn = page.getByTestId("registry-disclosure").getByText("Notices not verified").filter({ visible: true });
  await expect(warn).toBeInViewport();
  // every line box of the warning is the top-most thing where it is drawn (it can wrap onto two lines,
  // and the centre of its bounding box then falls between them, on its own paragraph)
  const covered = await warn.evaluate((el) =>
    [...el.getClientRects()].filter((r) => {
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return !hit || !(el.contains(hit) || hit.contains(el));
    }).length,
  );
  expect(covered).toBe(0);
  // and the banner sits below it
  expect((await page.getByTestId("offline-banner").boundingBox())!.y).toBeGreaterThan((await warn.boundingBox())!.y);
  await ctx.setOffline(false);
  await ctx.close();
});

test("L8: on a phone, choosing a layer from the peeking sheet opens it, so its time control is in view", async ({ browser }) => {
  const ctx = await browser.newContext(phone());
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await mapReady(page);
  const card = page.getByTestId("active-layer");
  const peekTop = (await card.boundingBox())!.y;
  expect(peekTop).toBeGreaterThan(844 * 0.6); // the sheet peeks: only its first lines show
  await page.getByTestId("chip-currents").click();
  await expect(page).toHaveURL(/layer=currents/);
  // the sheet opened to half: the layer's time and what it shows are in view without a drag
  await expect.poll(async () => (await card.boundingBox())!.y).toBeLessThan(844 * 0.6);
  await expect(page.locator('[data-testid^="time-"][role="radio"]').last()).toBeInViewport();
  await ctx.close();
});
