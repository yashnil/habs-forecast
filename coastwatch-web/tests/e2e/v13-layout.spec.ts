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
  // (the columns travel to the new size)
  await expect.poll(async () => (await box(page, "nx-pane")).width).toBeLessThanOrEqual(48);
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
  // the link opens the alerts page itself, in the shell, at that port (tests/e2e/v13/alerts.spec.ts)
  await page.keyboard.press("Escape");
  await alerts.click();
  await expect(page).toHaveURL(/\/alerts\?place=port-592$/);
  await expect(page.getByTestId("nx-alerts")).toBeVisible();
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

test("a page change is sequenced: the old page is gone before the new one comes in, and the nav underline travels", async ({ page }) => {
  await open(page, "/today?place=port-592");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  // every view-transition animation of the page, recorded as it starts
  await page.evaluate(() => {
    const w = window as unknown as { __vt: Array<{ pseudo: string; delay: number; duration: number; name: string }> };
    w.__vt = [];
    const seen = new WeakSet<Animation>();
    const read = () => {
      for (const a of document.getAnimations()) {
        const e = a.effect as KeyframeEffect | null;
        if (!e || seen.has(a) || !e.pseudoElement?.startsWith("::view-transition")) continue;
        seen.add(a);
        const t = e.getComputedTiming();
        w.__vt.push({ pseudo: e.pseudoElement, delay: Number(t.delay), duration: Number(t.duration), name: (a as CSSAnimation).animationName ?? "" });
      }
      requestAnimationFrame(read);
    };
    requestAnimationFrame(read);
  });
  await page.getByTestId("nx-bar").getByRole("link", { name: "Explore" }).click();
  await expect(page.getByTestId("nx-explore")).toBeVisible();
  await expect(page.getByTestId("nx-bar").getByRole("link", { name: "Explore" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("nx-bar").locator(".n12-dest-on")).toHaveCount(1);
  const vt = await page.evaluate(() => (window as unknown as { __vt: Array<{ pseudo: string; delay: number; duration: number; name: string }> }).__vt);
  const fades = (kind: string) => vt.filter((a) => a.pseudo.startsWith(`::view-transition-${kind}(`) && /fade/.test(a.name));
  const out = fades("old"), incoming = fades("new");
  expect(out.length).toBeGreaterThan(0);
  expect(incoming.length).toBeGreaterThan(0);
  // no double exposure: the new page's fade starts no earlier than the old page's ends
  const outEnds = Math.max(...out.map((a) => a.delay + a.duration));
  const inStarts = Math.min(...incoming.map((a) => a.delay));
  expect(inStarts).toBeGreaterThanOrEqual(outEnds);
  expect(outEnds).toBeLessThanOrEqual(120);
  expect(inStarts + Math.max(...incoming.map((a) => a.duration))).toBeLessThanOrEqual(320);
});

test("changing the pane's size moves the columns without resizing the map more than once", async ({ page }) => {
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  await page.evaluate(() => {
    const w = window as unknown as { __resizes: number; __widths: number[]; __nxChart: { map: { resize: (...a: unknown[]) => unknown } } };
    w.__resizes = 0;
    w.__widths = [];
    const m = w.__nxChart.map;
    const o = m.resize.bind(m);
    m.resize = (...a: unknown[]) => {
      w.__resizes += 1;
      return o(...a);
    };
    const pane = document.querySelector(".n12-pane")!;
    const read = () => {
      w.__widths.push(Math.round(pane.getBoundingClientRect().width));
      if (w.__widths.length < 80) requestAnimationFrame(read);
    };
    requestAnimationFrame(read);
  });
  const before = (await box(page, "nx-pane")).width;
  await page.getByTestId("nx-pane-divider").focus();
  await page.keyboard.press("ArrowLeft");
  await expect.poll(async () => (await box(page, "nx-pane")).width).toBeGreaterThan(719);
  await expect(page.getByTestId("nx-work")).not.toHaveAttribute("data-pane-resizing", "");
  await expect.poll(() => page.evaluate(() => (window as unknown as { __widths: number[] }).__widths.length)).toBeGreaterThanOrEqual(60);
  const { widths, resizes } = await page.evaluate(() => {
    const w = window as unknown as { __resizes: number; __widths: number[] };
    return { widths: w.__widths, resizes: w.__resizes };
  });
  // the column travelled through widths between the two sizes (it did not cut), and the map resized once
  expect(widths.some((v) => v > before + 20 && v < 700)).toBe(true);
  expect(resizes).toBe(1);
  expect(await chart(page)).toBe(1);
});

test.describe("phone sheet requests", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  test("a page can ask for the peek detent, and for a place to be kept above the sheet where it stands", async ({ page }) => {
    await open(page, "/explore?place=port-592");
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
    const margin = page.getByTestId("nx-margin");
    const rest = (await margin.boundingBox())!.y;
    const coverBottom = () => page.evaluate(() => parseFloat(getComputedStyle(document.querySelector(".n12-frame")!).getPropertyValue("--nx-cover-bottom")));
    const restCover = await coverBottom();
    await page.evaluate(() => (window as unknown as { __nxShell: { openMargin: (d: string) => void } }).__nxShell.openMargin("peek"));
    await expect(margin).toHaveAttribute("data-detent", "peek");
    await expect.poll(async () => (await margin.boundingBox())!.y).toBeGreaterThan(rest + 40);
    // the chart's free area follows the lower sheet
    await expect.poll(coverBottom).toBeLessThan(restCover - 40);
    // "More" from the peek goes to rest
    await margin.getByRole("button", { name: "Show more of this page" }).click();
    await expect(margin).toHaveAttribute("data-detent", "0");
    // raise the sheet to half, then ask for a place low on the chart to be kept in view above it
    await margin.getByRole("button", { name: "Show more of this page" }).click();
    await expect(margin).toHaveAttribute("data-detent", "1");
    await expect.poll(async () => (await margin.boundingBox())!.y).toBeLessThan(rest - 100);
    // the chart has finished refitting to the sheet's changes before a place on it is named
    await page.waitForTimeout(1500);
    const sheetTop = (await margin.boundingBox())!.y;
    const at = await page.evaluate((y) => {
      const w = window as unknown as { __nxChart: { map: { unproject: (p: [number, number]) => { lng: number; lat: number } } } };
      const f = document.querySelector(".n12-frame")!.getBoundingClientRect();
      // a place that is on the chart but under the raised sheet
      const p = w.__nxChart.map.unproject([f.width / 2, y - f.top + 60]);
      return [p.lng, p.lat] as [number, number];
    }, sheetTop);
    const screenY = () =>
      page.evaluate(([lon, lat]) => {
        const w = window as unknown as { __nxChart: { map: { project: (p: [number, number]) => { x: number; y: number } } } };
        return w.__nxChart.map.project([lon, lat]).y + document.querySelector(".n12-frame")!.getBoundingClientRect().top;
      }, at);
    expect(await screenY()).toBeGreaterThan(sheetTop);
    await page.evaluate(([lon, lat]) => (window as unknown as { __nxChart: { keep: (lon: number, lat: number) => void } }).__nxChart.keep(lon, lat), at);
    await expect.poll(screenY).toBeLessThan(sheetTop - 40);
    // already clear: asking again moves nothing (once the first move has landed)
    await page.waitForTimeout(900);
    const y1 = await screenY();
    expect(y1).toBeLessThan(sheetTop - 40);
    await page.evaluate(([lon, lat]) => (window as unknown as { __nxChart: { keep: (lon: number, lat: number) => void } }).__nxChart.keep(lon, lat), at);
    await page.waitForTimeout(800);
    expect(Math.abs((await screenY()) - y1)).toBeLessThan(2);
  });
});

test("a desk header condenses once and stays: scrolling through its threshold never flips it back and forth or moves the page under the reader", async ({ page }) => {
  {
    await open(page, RESEARCH);
    const header = page.locator("[data-nx-desk-header]");
    await expect(header).toBeVisible();
    const scroller = page.locator(".n12-margin-scroll");
    const fullH = (await header.boundingBox())!.height;
    // watch every change of the header's state and every position the scroller takes
    await page.evaluate(() => {
      const w = window as unknown as { __flips: number; __tops: number[] };
      w.__flips = 0;
      w.__tops = [];
      const h = document.querySelector("[data-nx-desk-header]")!;
      new MutationObserver(() => (w.__flips += 1)).observe(h, { attributes: true, attributeFilter: ["data-condensed"] });
      const sc = document.querySelector(".n12-margin-scroll")!;
      sc.addEventListener("scroll", () => w.__tops.push(sc.scrollTop));
    });
    // the reader's own scroll, in small steps, to just past the threshold, then left alone
    for (let y = 8; y <= 96; y += 8) {
      await scroller.evaluate((el, top) => el.scrollTo(0, top), y);
      await page.waitForTimeout(40);
    }
    await page.waitForTimeout(700);
    await expect(header).toHaveAttribute("data-condensed", /^$/);
    const { flips, tops } = await page.evaluate(() => {
      const w = window as unknown as { __flips: number; __tops: number[] };
      return { flips: w.__flips, tops: w.__tops };
    });
    expect(flips).toBe(1);
    // the scroller is where the reader put it: the header's change of height did not throw it back toward the
    // top (at most a few px the first time, while the header's condensed height is still an estimate)
    expect(Math.abs(tops[tops.length - 1] - 96)).toBeLessThanOrEqual(8);
    expect(Math.min(...tops.slice(-3))).toBeGreaterThan(80);
    // the header is shorter, and keeps the room it had (what was under it has not jumped up)
    const small = (await header.boundingBox())!.height;
    expect(small).toBeLessThan(fullH - 20);
    expect(await header.evaluate((el) => parseFloat(getComputedStyle(el).marginBottom))).toBeGreaterThanOrEqual(fullH - small - 1);
    // back to the top, it opens again, once
    await scroller.evaluate((el) => el.scrollTo(0, 0));
    await expect(header).not.toHaveAttribute("data-condensed", /^$/);
    expect(await page.evaluate(() => (window as unknown as { __flips: number }).__flips)).toBe(2);
  }
});

test("the menu shows the chart mode as it is now, not as it was when the page loaded", async ({ page }) => {
  await open(page, "/today?place=port-592");
  await page.getByTestId("nx-menu-button").click();
  await expect(page.getByTestId("nx-mode-day")).toBeChecked();
  await page.keyboard.press("Escape");
  // changed elsewhere (Your data does this)
  await page.evaluate(() => document.documentElement.setAttribute("data-chart-mode", "dusk"));
  await page.getByTestId("nx-menu-button").click();
  await expect(page.getByTestId("nx-mode-dusk")).toBeChecked();
});

test.describe("phone desk header and sheet", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  test("a desk header's title has the width; its actions never squeeze it", async ({ page }) => {
    await open(page, RESEARCH);
    const title = page.getByTestId("research-title");
    await expect(title).toBeVisible();
    const header = (await page.locator("[data-nx-desk-header]").boundingBox())!;
    expect((await title.boundingBox())!.width).toBeGreaterThanOrEqual(header.width * 0.9);
  });
  test("a sheet brought back to rest shows the page from its first line", async ({ page }) => {
    await open(page, "/explore?place=port-592");
    const margin = page.getByTestId("nx-margin");
    const body = page.locator(".n12-margin-scroll");
    const more = margin.getByRole("button", { name: /Show (more|less) of this page/ });
    await more.click();
    await more.click();
    await expect(margin).toHaveAttribute("data-detent", "2");
    await body.evaluate((el) => el.scrollTo(0, 300));
    expect(await body.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
    await more.click();
    await expect(margin).toHaveAttribute("data-detent", "0");
    await expect.poll(() => body.evaluate((el) => el.scrollTop)).toBe(0);
    await expect(page.getByTestId("nx-explore-lead")).toBeInViewport();
  });
});

test.describe("phone chart", () => {
  for (const size of [{ width: 390, height: 844 }, { width: 320, height: 640 }]) {
    test(`at ${size.width} px the chart's buttons are one small column that covers neither the port's name nor the scale`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: size, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
      const page = await ctx.newPage();
      await open(page, "/explore?place=port-592");
      await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
      const controls = page.getByTestId("nx-chart-controls");
      await expect(controls).toBeVisible();
      // zoom is a pinch on a phone: its buttons are not shown; what is shown is 44 px wide and named
      await expect(page.getByTestId("nx-zoom-in")).toBeHidden();
      const c = (await controls.boundingBox())!;
      expect(c.width).toBeLessThanOrEqual(48);
      expect(c.height).toBeLessThanOrEqual(92);
      for (const b of await controls.getByRole("button").all()) if (await b.isVisible()) expect(await b.getAttribute("aria-label")).toBeTruthy();
      // above the resting sheet, inside the chart
      const sheetTop = (await page.getByTestId("nx-margin").boundingBox())!.y;
      expect(c.y + c.height).toBeLessThanOrEqual(sheetTop);
      const hit = (r: { x: number; y: number; width: number; height: number } | null) => !!r && r.x < c.x + c.width && r.x + r.width > c.x && r.y < c.y + c.height && r.y + r.height > c.y;
      expect(hit(await page.locator(".n12-port-name-focus").boundingBox())).toBe(false);
      expect(hit(await page.locator(".n12-scale").boundingBox())).toBe(false);
      await ctx.close();
    });
  }

  test("a desk page's chart strip names only what the page says its chart shows", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    await open(page, RESEARCH);
    const subject = page.getByTestId("nx-pane-subject");
    await expect(subject).toBeVisible();
    // never the saved port's name over a chart that shows a station
    await expect(subject).not.toContainText("Moss Landing");
    await expect(subject).toHaveText(/^Chart( · .+)?$/);
    await ctx.close();
  });
});

test("an unticked checkbox is an empty ruled box and a ticked one is filled, in day and night", async ({ page }) => {
  await open(page, "/today?place=port-592");
  for (const mode of ["day", "night"]) {
    const r = await page.evaluate((mode) => {
      document.documentElement.setAttribute("data-chart-mode", mode);
      const app = document.querySelector(".n12-app")!;
      const mk = (checked: boolean) => {
        const el = document.createElement("input");
        el.type = "checkbox";
        el.checked = checked;
        app.appendChild(el);
        const cs = getComputedStyle(el);
        const out = { bg: cs.backgroundColor, border: cs.borderTopColor, w: el.offsetWidth, appearance: cs.appearance };
        el.remove();
        return out;
      };
      const paper = getComputedStyle(app).backgroundColor;
      return { off: mk(false), on: mk(true), paper };
    }, mode);
    expect(r.off.appearance).toBe("none");
    expect(r.off.bg, mode).toBe(r.paper);
    expect(r.on.bg, mode).not.toBe(r.paper);
    expect(r.off.border, mode).toBe(r.on.bg);
    expect(r.off.w).toBe(20);
  }
});
