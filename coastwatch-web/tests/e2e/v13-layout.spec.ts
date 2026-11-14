import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * The shell's layout behaviour. v14 (docs/coastwatch/v14/design-system/README.md §2) replaced v13's "Desk and Chart"
 * with the synthesis composition: one full-bleed chart behind every page, the page's objects floating on it, a phone
 * sheet. The behaviours v13 held are held here in the new layout: one tree at every width, one map instance that is
 * never rebuilt or resized per frame, the page change sequenced, the phone sheet's detents and requests. A v13 desk
 * page (a floating panel until its page moves to the stage) keeps its own checks while it is one; each such check says
 * so and stands down once the page has moved.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const RESEARCH = "/research?station=HABs-MontereyWharf&var=pDA";
const box = async (page: Page, id: string) => (await page.getByTestId(id).boundingBox())!;
const chart = (page: Page) => page.evaluate(() => (window as unknown as { __nxChart?: { inits: number } }).__nxChart?.inits ?? 0);
const isDesk = async (page: Page) => (await page.locator('[data-nx-layout="desk"]').count()) > 0;
/** Counts the map's resizes from now on. */
const countResizes = (page: Page) =>
  page.evaluate(() => {
    const w = window as unknown as { __resizes: number; __nxChart: { map: { resize: (...a: unknown[]) => unknown } } };
    w.__resizes = 0;
    const m = w.__nxChart.map;
    const o = m.resize.bind(m);
    m.resize = (...a: unknown[]) => {
      w.__resizes += 1;
      return o(...a);
    };
  });
const resizes = (page: Page) => page.evaluate(() => (window as unknown as { __resizes: number }).__resizes);
async function open(page: Page, url: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}${url}`);
}

test("the chart is full-bleed behind every page: the page floats over it and the map fills the screen", async ({ page }) => {
  for (const [url, id] of [
    [RESEARCH, "nx-research"],
    ["/coast?place=port-592", "nx-coast"],
    ["/explore?place=port-592", "nx-explore"],
  ]) {
    await open(page, url);
    await expect(page.getByTestId(id)).toBeVisible();
    const pane = await box(page, "nx-pane");
    expect(pane.x).toBe(0);
    expect(pane.y).toBe(0);
    expect(pane.width).toBe(1440);
    expect(pane.height).toBeGreaterThanOrEqual(719);
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
    // the map's canvas is the whole sheet; the page stands inside the screen, over it
    await expect.poll(() => page.evaluate(() => document.querySelector(".maplibregl-canvas")!.clientWidth)).toBe(1440);
    const margin = await box(page, "nx-margin");
    expect(margin.x).toBeGreaterThanOrEqual(0);
    expect(margin.x + margin.width).toBeLessThanOrEqual(1440);
    // the v13 divider is gone
    await expect(page.getByTestId("nx-pane-divider")).toHaveCount(0);
  }
});

test("a v13 page not yet moved floats as a card or a panel, and the chart is fitted beside it", async ({ page }) => {
  await open(page, "/explore?place=port-592");
  await expect(page.getByTestId("nx-explore")).toBeVisible();
  if (await page.locator('[data-nx-layout="stage"]').count()) test.skip(true, "Explore has moved to the stage");
  const margin = await box(page, "nx-margin");
  expect(margin.width).toBeLessThanOrEqual(404);
  expect(margin.x + margin.width).toBeLessThanOrEqual(1440 - 23);
  // the chart's free area excludes the card (the camera fits beside it)
  await expect.poll(() => page.evaluate(() => parseFloat(getComputedStyle(document.querySelector(".n12-frame")!).getPropertyValue("--nx-cover-right")))).toBeGreaterThanOrEqual(404);
});

test("the map is built once and never resized while the reader moves between sections; the nav pill travels", async ({ page }) => {
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  await countResizes(page);
  const bar = page.getByTestId("nx-bar");
  for (const [name, id] of [
    ["My Coast", "nx-coast"],
    ["Today", "nx-today"],
    ["Research", "nx-research"],
  ]) {
    await bar.getByRole("link", { name }).click();
    await expect(page.getByTestId(id)).toBeVisible();
    await expect(bar.getByRole("link", { name })).toHaveAttribute("aria-current", "page");
    await expect(bar.locator(".n12-dest-on")).toHaveCount(1);
  }
  expect(await resizes(page)).toBe(0);
  expect(await chart(page)).toBe(1);
});

test("a narrow screen gives a v13 desk the width; the chart opens from its header and Escape gives the desk back", async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 700 });
  await open(page, RESEARCH);
  await expect(page.getByTestId("nx-research")).toBeVisible();
  if (!(await isDesk(page))) test.skip(true, "Research has moved to the stage");
  expect((await box(page, "nx-margin")).width).toBeGreaterThanOrEqual(1000 - 49);
  await page.getByTestId("nx-chart-toggle").click();
  await expect(page.getByTestId("nx-margin")).toBeHidden();
  await expect(page.getByTestId("nx-pane")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("nx-margin")).toBeVisible();
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

  test("a v13 desk page scrolls as a page: the chart strip leaves a bar, 'Larger' grows it, and nothing scrolls sideways", async ({ page }) => {
    await open(page, "/coast?place=port-592");
    await expect(page.getByTestId("nx-coast")).toBeVisible();
    if (!(await isDesk(page))) test.skip(true, "My Coast has moved to the stage");
    const work = page.getByTestId("nx-work");
    const strip = await box(page, "nx-pane");
    expect(strip.y).toBeGreaterThanOrEqual(0);
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
  // the prototype's 40 px buttons on a wide screen (a phone shows none: a pinch zooms)
  for (const b of await controls.getByRole("button").all()) {
    const r = (await b.boundingBox())!;
    expect(r.width).toBeGreaterThanOrEqual(40);
    expect(r.height).toBeGreaterThanOrEqual(40);
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

test("a page change is sequenced: the old page is gone before the new one is drawn, and the nav pill stays on its section", async ({ page }) => {
  await open(page, "/today?place=port-592");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  await expect(page.getByTestId("nx-today")).toBeVisible();
  // every frame, whether the two pages were ever on screen together (v14: the old object shrinks away on the ghost
  // layer, which is inert and unnamed; the new page arrives with the camera)
  await page.evaluate(() => {
    const w = window as unknown as { __both: number; __frames: number };
    w.__both = 0;
    w.__frames = 0;
    const read = () => {
      w.__frames += 1;
      if (document.querySelector('[data-testid="nx-today"]') && document.querySelector('[data-testid="nx-explore"]')) w.__both += 1;
      if (w.__frames < 240) requestAnimationFrame(read);
    };
    requestAnimationFrame(read);
  });
  await page.getByTestId("nx-menu-button").click();
  await page.getByRole("navigation", { name: "More" }).getByRole("link", { name: "Explore the water" }).click();
  await expect(page.getByTestId("nx-explore")).toBeVisible();
  // v15.1: Explore is its own section (Explore California Waters), its pill drawn once
  await expect(page.getByTestId("nx-nav-explore")).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("nx-bar").locator(".n12-dest-on")).toHaveCount(1);
  expect(await page.evaluate(() => (window as unknown as { __both: number }).__both)).toBe(0);
  // a leaving object's copy is inert and named nothing
  expect(await page.locator("#nx-stage-ghosts [data-testid]").count()).toBe(0);
});

test("a layer chip fades the depth contours in and out without resizing the map, and the chart says what it draws", async ({ page }) => {
  await open(page, "/today?place=port-592");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  await countResizes(page);
  const chip = page.getByRole("group", { name: "Map layers" }).locator('[data-layer="depth"]');
  const drawn = () => page.getByTestId("nx-sheet").getAttribute("data-layers");
  const was = (await chip.getAttribute("aria-pressed")) === "true";
  await expect.poll(async () => ((await drawn()) ?? "").split(" ").includes("depth")).toBe(was);
  await chip.click();
  await expect(chip).toHaveAttribute("aria-pressed", String(!was));
  await expect.poll(async () => ((await drawn()) ?? "").split(" ").includes("depth")).toBe(!was);
  await chip.click();
  await expect.poll(async () => ((await drawn()) ?? "").split(" ").includes("depth")).toBe(was);
  expect(await resizes(page)).toBe(0);
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
    await expect(page.getByTestId("nx-research")).toBeVisible();
    if (!(await isDesk(page))) test.skip(true, "Research has moved to the stage");
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
    // the scroller is where the reader put it: nothing adjusted it when the header changed height
    expect(tops[tops.length - 1]).toBe(96);
    expect(Math.max(...tops)).toBe(96);
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
    if (!(await isDesk(page))) test.skip(true, "Research has moved to the stage");
    const header = (await page.locator("[data-nx-desk-header]").boundingBox())!;
    expect((await title.boundingBox())!.width).toBeGreaterThanOrEqual(header.width * 0.9);
  });
  test("a sheet brought back to rest shows the page from its first line", async ({ page }) => {
    // Notices: a long page in a sheet (Explore's phone content is now shorter than one screen)
    await open(page, "/notices?place=port-592");
    const margin = page.getByTestId("nx-margin");
    const body = page.locator(".n12-margin-scroll");
    const first = page.getByRole("heading", { level: 1 });
    await expect(first).toBeInViewport();
    const more = margin.getByRole("button", { name: /Show (more|less) of this page/ });
    await more.click();
    await more.click();
    await expect(margin).toHaveAttribute("data-detent", "2");
    await body.evaluate((el) => el.scrollTo(0, 300));
    expect(await body.evaluate((el) => el.scrollTop)).toBe(300);
    await expect(first).not.toBeInViewport();
    await more.click();
    await expect(margin).toHaveAttribute("data-detent", "0");
    await expect.poll(() => body.evaluate((el) => el.scrollTop)).toBe(0);
    await expect(first).toBeInViewport();
    // at rest the scroller ends where the visible sheet ends: its last line can be brought above the destinations bar
    await body.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    const bodyBox = (await body.boundingBox())!;
    const tabs = (await page.getByTestId("nx-tabs").boundingBox())!;
    expect(bodyBox.y + bodyBox.height).toBeLessThanOrEqual(tabs.y + 0.5);
  });
});

test.describe("phone chart", () => {
  for (const size of [{ width: 390, height: 844 }, { width: 320, height: 640 }]) {
    test(`at ${size.width} px the chart strip shows no buttons and no scale (a pinch zooms), and the port is named above the sheet`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: size, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
      const page = await ctx.newPage();
      await open(page, "/explore?place=port-592");
      await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
      await expect(page.getByTestId("nx-zoom-in")).toBeHidden();
      await expect(page.locator(".n12-scale")).toBeHidden();
      // the layers are one button named "Layers"; the sections a tab bar
      await expect(page.getByRole("button", { name: "Layers", exact: true })).toBeVisible();
      await expect(page.getByTestId("nx-tabs")).toBeVisible();
      const sheetTop = (await page.getByTestId("nx-margin").boundingBox())!.y;
      const name = page.locator(".n12-port-name-focus");
      await expect(name).toBeVisible();
      const r = (await name.boundingBox())!;
      expect(r.y + r.height).toBeLessThanOrEqual(sheetTop);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(size.width);
      await ctx.close();
    });
  }

  test("a desk page's chart strip names only what the page says its chart shows", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    await open(page, RESEARCH);
    await expect(page.getByTestId("nx-research")).toBeVisible();
    if (!(await isDesk(page))) test.skip(true, "Research has moved to the stage");
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
      // v14: the box is paper (the objects' ground) on a chart that fills the screen
      const probe = document.createElement("div");
      probe.style.background = "var(--paper)";
      app.appendChild(probe);
      const paper = getComputedStyle(probe).backgroundColor;
      probe.remove();
      return { off: mk(false), on: mk(true), paper };
    }, mode);
    expect(r.off.appearance).toBe("none");
    expect(r.off.bg, mode).toBe(r.paper);
    expect(r.on.bg, mode).not.toBe(r.paper);
    // ruled when unticked, filled in the pressable colour when ticked
    expect(r.off.border, mode).not.toBe(r.paper);
    expect(r.off.w).toBe(20);
  }
});

test.describe("phone desk page, before the chart pane has arrived", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  test("the page starts under the strip's place whether or not the pane is in the document yet (no shift when it arrives)", async ({ page }) => {
    for (const url of [RESEARCH, "/coast?place=port-592"]) {
      await open(page, url);
      await expect(page.getByTestId("nx-pane")).toBeVisible();
      if (!(await isDesk(page))) continue; // a stage page's sheet is fixed over the chart: it never waits for the pane
      const tops = await page.evaluate(() => {
        const main = document.querySelector<HTMLElement>(".n12-margin")!;
        const pane = document.querySelector<HTMLElement>(".n12-pane")!;
        const work = pane.parentElement!;
        const withPane = main.getBoundingClientRect().top;
        // as a browser has it while the page's HTML is still arriving: the pane comes after the page in the document
        const next = pane.nextSibling;
        pane.remove();
        const without = main.getBoundingClientRect().top;
        work.insertBefore(pane, next);
        return { withPane, without, back: main.getBoundingClientRect().top };
      });
      expect(Math.abs(tops.without - tops.withPane), url).toBeLessThanOrEqual(2);
      expect(tops.back).toBe(tops.withPane);
    }
  });
});
