import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * v13 "Desk and Chart" (docs/coastwatch/v13/design-system/README.md §1): desk pages give the work the space and
 * make the chart a pane the reader sizes; chart pages keep the margin and the sheet; one tree at every width,
 * before and after hydration; one map instance throughout.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const RESEARCH = "/research?station=HABs-MontereyWharf&var=pDA";
const box = async (page: Page, id: string) => (await page.getByTestId(id).boundingBox())!;
const chart = (page: Page) => page.evaluate(() => (window as unknown as { __nxChart?: { inits: number } }).__nxChart?.inits ?? 0);
async function open(page: Page, url: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}${url}`);
}

test("desk pages: the work has at least 60 % of a 1440 px screen and the chart is a pane beside it", async ({ page }) => {
  for (const [url, id] of [
    [RESEARCH, "nx-research"],
    ["/coast?place=port-592", "nx-coast"],
  ]) {
    await open(page, url);
    await expect(page.getByTestId(id)).toBeVisible();
    const desk = await box(page, "nx-margin");
    const pane = await box(page, "nx-pane");
    expect(desk.width).toBeGreaterThanOrEqual(1440 * 0.6);
    expect(pane.width).toBeGreaterThanOrEqual(320);
    expect(pane.x).toBeGreaterThanOrEqual(desk.x + desk.width);
    // nothing of the page is laid over the chart
    await expect(page.getByTestId("nx-sheet")).toBeVisible();
    expect(await page.evaluate(() => getComputedStyle(document.querySelector(".n12-frame")!).getPropertyValue("--nx-cover-right"))).toBe("0px");
  }
});

test("chart pages keep the margin column beside a chart that takes the rest", async ({ page }) => {
  await open(page, "/explore?place=port-592");
  await expect(page.getByTestId("nx-explore")).toBeVisible();
  const margin = await box(page, "nx-margin");
  const pane = await box(page, "nx-pane");
  expect(margin.width).toBeLessThanOrEqual(372);
  expect(pane.width).toBeGreaterThanOrEqual(1440 - 372 - 1);
  await expect(page.getByTestId("nx-pane-divider")).toBeHidden();
});

test("the divider is a separator: arrows step the pane's sizes, Enter collapses to the rail, the size is kept, the map is never rebuilt", async ({ page }) => {
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  const work = page.getByTestId("nx-work");
  const divider = page.getByTestId("nx-pane-divider");
  await expect(divider).toHaveAttribute("role", "separator");
  await expect(work).toHaveAttribute("data-pane", "context");
  await divider.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(work).toHaveAttribute("data-pane", "split");
  await expect(divider).toHaveAttribute("aria-valuenow", "2");
  await expect.poll(async () => (await box(page, "nx-pane")).width).toBeGreaterThan(700);
  // the chart resized with its pane and refitted: the canvas is the frame's size
  await expect.poll(() => page.evaluate(() => Math.abs(document.querySelector(".maplibregl-canvas")!.clientWidth - document.querySelector(".n12-frame")!.clientWidth))).toBeLessThanOrEqual(1);
  await page.keyboard.press("Enter");
  await expect(work).toHaveAttribute("data-pane", "rail");
  expect((await box(page, "nx-pane")).width).toBeLessThanOrEqual(48);
  expect((await box(page, "nx-margin")).width).toBeGreaterThan(1300);
  await page.getByTestId("nx-pane-rail").click();
  await expect(work).toHaveAttribute("data-pane", "split");
  await page.reload();
  await expect(page.getByTestId("nx-work")).toHaveAttribute("data-pane", "split");
  // another route keeps its own size
  await page.getByTestId("nx-bar").getByRole("link", { name: "My Coast" }).click();
  await expect(page.getByTestId("nx-coast")).toBeVisible();
  await expect(page.getByTestId("nx-work")).toHaveAttribute("data-pane", "context");
  expect(await chart(page)).toBe(1);
});

test("dragging the divider never resizes the map until release", async ({ page }) => {
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  await page.evaluate(() => {
    const w = window as unknown as { __resizes: number; __nxChart: { map: { resize: (...a: unknown[]) => unknown } } };
    w.__resizes = 0;
    const m = w.__nxChart.map;
    const o = m.resize.bind(m);
    m.resize = (...a: unknown[]) => {
      w.__resizes += 1;
      return o(...a);
    };
  });
  const d = await box(page, "nx-pane-divider");
  await page.mouse.move(d.x + d.width / 2, d.y + 300);
  await page.mouse.down();
  for (let i = 1; i <= 12; i++) await page.mouse.move(d.x - i * 30, d.y + 300);
  expect(await page.evaluate(() => (window as unknown as { __resizes: number }).__resizes)).toBe(0);
  // the chart keeps its height while it is clipped
  expect((await box(page, "nx-sheet")).height).toBeGreaterThan(600);
  await page.mouse.up();
  await expect(page.getByTestId("nx-work")).toHaveAttribute("data-pane", "split");
  await expect.poll(() => page.evaluate(() => (window as unknown as { __resizes: number }).__resizes)).toBe(1);
});

test("a narrow screen gives the desk the width; the chart opens over it and Escape closes it", async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 700 });
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-research")).toBeVisible();
  expect((await box(page, "nx-margin")).width).toBeGreaterThanOrEqual(999);
  await expect(page.getByTestId("nx-pane")).toBeHidden();
  await page.getByTestId("nx-chart-toggle").click();
  await expect(page.getByTestId("nx-pane")).toBeVisible();
  const pane = await box(page, "nx-pane");
  expect(pane.width).toBeGreaterThanOrEqual(480);
  expect(pane.x + pane.width).toBeGreaterThanOrEqual(999);
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("nx-pane")).toBeHidden();
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

  test("the page's element is never replaced when the phone layout resolves, on a chart page and on a desk page", async ({ page }) => {
    for (const url of ["/explore?place=port-592", RESEARCH]) {
      await open(page, url);
      const margin = page.getByTestId("nx-margin");
      await margin.evaluate((el) => ((el as HTMLElement & { __kept?: boolean }).__kept = true));
      await page.waitForTimeout(1600);
      expect(await margin.evaluate((el) => (el as HTMLElement & { __kept?: boolean }).__kept === true)).toBe(true);
      await expect(page.getByTestId("nx-margin")).toHaveCount(1);
    }
  });

  test("a desk page scrolls as a page: the chart strip leaves a bar, 'Larger' grows it, and nothing scrolls sideways", async ({ page }) => {
    await open(page, "/coast?place=port-592");
    await expect(page.getByTestId("nx-coast")).toBeVisible();
    const work = page.getByTestId("nx-work");
    const strip = await box(page, "nx-pane");
    expect(strip.y).toBeGreaterThanOrEqual(51);
    await work.evaluate((el) => el.scrollTo(0, 600));
    await expect.poll(async () => (await box(page, "nx-pane")).y).toBeLessThan(0);
    const toggle = page.getByTestId("nx-pane-toggle");
    await expect(toggle).toBeInViewport();
    expect((await toggle.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    await toggle.tap();
    await expect(work).toHaveAttribute("data-pane-open", "");
    await expect.poll(async () => (await box(page, "nx-pane")).height).toBeGreaterThan(844 * 0.55);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    expect(await work.evaluate((el) => el.scrollWidth)).toBeLessThanOrEqual(390);
  });
});

test("the bar's Alerts link carries the port and is never a dead end; the menu says Your data", async ({ page }) => {
  await open(page, "/today?place=port-592");
  const alerts = page.getByTestId("nx-alerts-link");
  await expect(alerts).toHaveAttribute("href", "/alerts?place=port-592");
  // no count is shown while nothing is unread
  await expect(page.getByTestId("nx-unread")).toHaveCount(0);
  await page.getByTestId("nx-menu-button").click();
  await expect(page.getByRole("navigation", { name: "More" }).getByRole("link", { name: "Your data" })).toHaveAttribute("href", "/account");
  const res = await page.request.get(`${OK}/alerts?place=port-592`, { maxRedirects: 0 });
  expect(res.status()).toBeGreaterThanOrEqual(300);
  expect(res.status()).toBeLessThan(400);
  expect(res.headers().location).toContain("/account?place=port-592#alerts");
});

test("the chart's own buttons zoom and bring the page's view back; names keep clear of them", async ({ page }) => {
  await open(page, "/explore?place=port-592");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  const zoom = () => page.evaluate(() => (window as unknown as { __nxChart: { camera: { zoom: number } } }).__nxChart.camera.zoom);
  const controls = page.getByTestId("nx-chart-controls");
  await expect(controls).toBeVisible();
  for (const b of await controls.getByRole("button").all()) {
    const r = (await b.boundingBox())!;
    expect(r.width).toBeGreaterThanOrEqual(44);
    expect(r.height).toBeGreaterThanOrEqual(44);
  }
  const z0 = await zoom();
  await page.getByTestId("nx-zoom-in").click();
  await expect.poll(zoom).toBeGreaterThan(z0 + 0.9);
  // two quick presses are two steps, even while the camera is still travelling
  await page.getByTestId("nx-zoom-in").click();
  await expect.poll(zoom).toBeGreaterThan(z0 + 1.9);
  await page.getByTestId("nx-zoom-out").click();
  await expect.poll(zoom).toBeLessThan(z0 + 1.1);
  await page.getByRole("button", { name: "Back to Moss Landing" }).click();
  await expect.poll(async () => Math.abs((await zoom()) - z0)).toBeLessThan(0.05);
  // nothing written on the chart lies under the buttons
  const c = (await controls.boundingBox())!;
  const under = await page.evaluate(
    (c) =>
      [...document.querySelectorAll<HTMLElement>(".n12-mark")].filter((el) => {
        if (el.style.visibility !== "visible") return false;
        const r = el.getBoundingClientRect();
        return r.left < c.x + c.width && r.right > c.x && r.top < c.y + c.height && r.bottom > c.y;
      }).length,
    c,
  );
  expect(under).toBe(0);
  // one map instance through all of it
  expect(await chart(page)).toBe(1);
});
