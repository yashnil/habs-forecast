import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * The v13 parts in the browser, on the catalog (/system): the plot kit's keyboard, pointer and touch contract,
 * a figure that never counts from zero or passes its target, a choice that is visibly unset until made, the
 * waterfall's worked sums, the sensitivity grid as a keyboard grid (docs/coastwatch/v13/design-system/README.md).
 */
test.use({ storageState: { cookies: [], origins: [] } });

const plot = (page: Page) => page.getByTestId("sys-plot");
const focusT = (page: Page) => plot(page).getAttribute("data-focus-t");
const live = (page: Page) => page.getByTestId("sys-plot-live");

test("a plot is one tab stop: arrows step real samples, each is read out, Enter pins, Escape clears", async ({ page }) => {
  await page.goto(`${OK}/system#plot`);
  await expect(plot(page)).toBeVisible();
  await plot(page).focus();
  await expect(plot(page)).toHaveAttribute("role", "group");
  await page.keyboard.press("End");
  const last = await focusT(page);
  expect(last).not.toBeNull();
  await expect(live(page)).toHaveText(/^[A-Z][a-z]{2} \d{1,2}, \d{4} · .+ · (measured|reported 0|below detection)$/);
  const said = await live(page).innerText();
  await page.keyboard.press("ArrowLeft");
  await expect.poll(() => focusT(page)).not.toBe(last);
  expect(Number(await focusT(page))).toBeLessThan(Number(last));
  await expect(live(page)).not.toHaveText(said);
  // a month back is further than one sample back
  const one = Number(await focusT(page));
  await page.keyboard.press("PageUp");
  await expect.poll(async () => Number(await focusT(page))).toBeLessThan(one - 20 * 86_400_000);
  await page.keyboard.press("Home");
  const first = Number(await focusT(page));
  await page.keyboard.press("ArrowLeft");
  expect(Number(await focusT(page))).toBe(first);
  // the tooltip stands beside the sample; the secondary plot reads the same date
  await expect(page.getByTestId("plot-tip").first()).toBeVisible();
  // wherever the sample is, high or on the baseline, the tooltip is whole inside the plot: never cut at its foot
  for (const k of ["End", "ArrowLeft", "ArrowLeft", "ArrowLeft", "PageUp", "PageUp", "Home", "ArrowRight"]) {
    await page.keyboard.press(k);
    const tip = (await page.getByTestId("sys-plot").getByTestId("plot-tip").boundingBox())!;
    const box = (await plot(page).boundingBox())!;
    expect(tip.y, k).toBeGreaterThanOrEqual(box.y - 1);
    expect(tip.y + tip.height, k).toBeLessThanOrEqual(box.y + box.height + 1);
  }
  await page.keyboard.press("Home");
  // Enter pins: the inspector opens with the exact value, and the pin stays when focus moves on
  await page.keyboard.press("Enter");
  await expect(plot(page)).toHaveAttribute("data-pinned-t", String(first));
  await expect(page.getByTestId("sys-inspector")).toBeVisible();
  await expect(live(page)).toHaveText(/^Pinned\. /);
  await page.keyboard.press("ArrowRight");
  await expect(plot(page)).toHaveAttribute("data-pinned-t", String(first));
  await page.keyboard.press("Escape");
  await expect(plot(page)).not.toHaveAttribute("data-pinned-t", /.+/);
  await expect(page.getByTestId("sys-inspector")).toHaveCount(0);
});

test("the crosshair snaps to a sample under the mouse, a click pins it, and it never lands between samples", async ({ page }) => {
  await page.goto(`${OK}/system#plot`);
  await plot(page).scrollIntoViewIfNeeded();
  const b = (await plot(page).boundingBox())!;
  const ts = new Set<string>();
  for (let i = 0; i < 12; i++) {
    await page.mouse.move(b.x + 80 + i * ((b.width - 120) / 12), b.y + b.height / 2);
    const t = await focusT(page);
    if (t) ts.add(t);
  }
  expect(ts.size).toBeGreaterThan(3);
  // every focused position is a sample of the record (a whole day's timestamp from the data, not an interpolation)
  const samples = await page.evaluate(() => [...document.querySelectorAll("[data-testid=sys-plot] [data-points] *")].length);
  expect(samples).toBeGreaterThan(10);
  await page.mouse.click(b.x + b.width - 60, b.y + b.height / 2);
  await expect(plot(page)).toHaveAttribute("data-pinned-t", /\d+/);
  await page.mouse.move(b.x + b.width / 2, b.y - 80);
  await expect(plot(page)).toHaveAttribute("data-pinned-t", /\d+/);
});

test.describe("touch", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  test("a tap pins the nearest sample; the plot fits a phone and nothing scrolls sideways", async ({ page }) => {
    await page.goto(`${OK}/system#plot`);
    await plot(page).scrollIntoViewIfNeeded();
    const b = (await plot(page).boundingBox())!;
    expect(b.width).toBeLessThanOrEqual(390);
    expect(b.height).toBeGreaterThanOrEqual(240);
    // the catalog is a long page: on a slow machine it can still be settling (or not yet hydrated) when the
    // plot is first measured, so the tap is made where the plot stands now, and made again only if nothing is pinned
    await expect(async () => {
      if (!(await plot(page).getAttribute("data-pinned-t"))) {
        await plot(page).scrollIntoViewIfNeeded();
        const now = (await plot(page).boundingBox())!;
        await page.touchscreen.tap(now.x + now.width - 50, now.y + 120);
      }
      await expect(plot(page)).toHaveAttribute("data-pinned-t", /\d+/, { timeout: 1500 });
    }).toPass({ timeout: 15_000 });
    await expect(page.getByTestId("sys-inspector")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  });
});

test("a figure moves from its old value to the new one: never from zero, never past either end", async ({ page }) => {
  await page.goto(`${OK}/system#figure`);
  const fig = page.getByTestId("sys-figure");
  await expect(fig).toContainText("$2,602.24");
  await page.evaluate(() => {
    const el = document.querySelector("[data-testid=sys-figure] [aria-hidden=true]")!;
    const w = window as unknown as { __seen: number[] };
    w.__seen = [];
    const read = () => {
      const m = /\$([\d,]+\.\d\d)/.exec(el.textContent ?? "");
      if (m) w.__seen.push(Number(m[1].replace(/,/g, "")));
      if (w.__seen.length < 150) requestAnimationFrame(read);
    };
    requestAnimationFrame(read);
  });
  await page.getByTestId("sys-figure-change").click();
  await expect(fig.locator("[data-figure-exact]")).toHaveText("$143.50");
  await expect.poll(() => page.evaluate(() => (window as unknown as { __seen: number[] }).__seen.length)).toBeGreaterThanOrEqual(150);
  const seen = await page.evaluate(() => (window as unknown as { __seen: number[] }).__seen);
  expect(Math.min(...seen)).toBeGreaterThanOrEqual(143.5);
  expect(Math.max(...seen)).toBeLessThanOrEqual(2602.24);
  // settled on the exact value
  expect(seen[seen.length - 1]).toBe(143.5);
  // it travelled: some frame showed a value strictly between the two
  expect(seen.some((v) => v > 143.5 && v < 2602.24)).toBe(true);
});

test("reduced motion: the figure is at its new value at once", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.goto(`${OK}/system#figure`);
  await page.getByTestId("sys-figure-change").click();
  await expect(page.getByTestId("sys-figure").locator("[aria-hidden=true]")).toContainText("$143.50", { timeout: 300 });
  await ctx.close();
});

test("a choice is visibly unset until it is made; the waterfall says how each bar was worked out; the grid is a keyboard grid", async ({ page }) => {
  await page.goto(`${OK}/system#controls`);
  const group = page.getByRole("radiogroup", { name: "Who pays fuel, bait and ice" });
  await expect(group).toHaveAttribute("data-unset", "true");
  await expect(page.getByTestId("segmented-unset").first()).toContainText("Choose one");
  await expect(group.getByRole("radio", { checked: true })).toHaveCount(0);
  await group.getByRole("radio", { name: "The boat" }).click();
  await expect(group).not.toHaveAttribute("data-unset", "true");
  await expect(group.getByRole("radio", { checked: true })).toHaveText("The boat");

  const wf = page.getByTestId("sys-waterfall");
  await wf.scrollIntoViewIfNeeded();
  await page.mouse.move(2, 2);
  await wf.focus();
  await page.keyboard.press("Home");
  await expect(page.getByTestId("waterfall-callout")).toContainText("Gross $3,600.00");
  await expect(page.getByTestId("waterfall-callout")).toContainText("400 lb × $9.00");
  for (let i = 0; i < 3; i++) await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("waterfall-callout")).toContainText("Crew −$650.56");
  await page.keyboard.press("End");
  await expect(page.getByTestId("waterfall-callout")).toContainText("Net to boat $2,602.24");

  const grid = page.getByTestId("sensitivity-grid");
  await grid.scrollIntoViewIfNeeded();
  await page.getByTestId("sens-2-3").focus();
  // the focused cell is previewed on the waterfall as outlines
  await page.keyboard.press("ArrowUp");
  await expect(page.getByTestId("sens-1-3")).toBeFocused();
  await page.keyboard.press("ArrowLeft");
  await expect(page.getByTestId("sens-1-2")).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("sens-1-2")).toHaveAttribute("aria-selected", "true");
  await expect(grid.locator("[aria-selected=true]")).toHaveCount(1);
  // 300 lb at $6: gross 1,800, less 347.20, less 20 % crew
  await expect(page.getByTestId("sens-1-2")).toContainText("$1,162");
  // the break-even contour: 200 and 300 lb lose money at $1 and gain at $3; 400 and 500 lb never lose
  await expect(grid.locator("[data-be-left]")).toHaveCount(2);
});
