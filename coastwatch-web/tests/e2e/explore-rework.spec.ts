import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * v10 Explore rework (the independent review's "barely changed" verdict): the opening frames the coastal
 * ocean, not the state; with nothing current the instrument rests on the seafloor with one status line and
 * two dated choices; currents encode speed; the rail holds still while frames play; a stale value says so in
 * place; a phone's selection sheet leads with the place and its first line of evidence.
 */
const CURRENT = "2026-10-08T20:00:00Z"; // the fixture's run is current: the map opens on the forecast
const REST = "2026-10-14T20:00:00Z"; // nothing current: no layer drawn, the status and the dated choices
const STALE = "2026-10-12T20:00:00Z"; // the run is 4 days old: stale, still shown when chosen

type CwMap = { loaded(): boolean; project(p: [number, number]): { x: number; y: number }; unproject(p: [number, number]): { lng: number; lat: number }; getContainer(): HTMLElement };
type W = { __cwMap?: CwMap; __cwFlow?: { streaks(): { speed: number; px: number }[]; stats: { segments: number } } };

async function open(page: Page, url: string, at: string, size = { width: 1440, height: 900 }) {
  await page.setViewportSize(size);
  await page.clock.setFixedTime(new Date(at));
  await page.goto(`${OK}${url}`);
  await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
  // the opening camera is set when the map is ready; let a frame pass
  await page.waitForTimeout(300);
}

/** The coast's box on screen, and the share of the map that is inside Nevada (a rough polygon of the state). */
async function framing(page: Page) {
  return page.evaluate(() => {
    const m = (window as unknown as W).__cwMap!;
    const c = m.getContainer();
    const n = m.project([-124.2, 41.75]); // Crescent City
    const s = m.project([-117.13, 32.71]); // San Diego
    const nv: [number, number][] = [[-120, 42], [-114.04, 42], [-114.04, 35.0], [-114.6, 35.0], [-120, 39]];
    const inside = (x: number, y: number) => {
      let r = false;
      for (let i = 0, j = nv.length - 1; i < nv.length; j = i++) if (nv[i][1] > y !== nv[j][1] > y && x < ((nv[j][0] - nv[i][0]) * (y - nv[i][1])) / (nv[j][1] - nv[i][1]) + nv[i][0]) r = !r;
      return r;
    };
    let k = 0, t = 0;
    for (let y = 0; y < c.clientHeight; y += 20)
      for (let x = 0; x < c.clientWidth; x += 20) {
        const ll = m.unproject([x, y]);
        t++;
        if (inside(ll.lng, ll.lat)) k++;
      }
    return { w: c.clientWidth, h: c.clientHeight, north: n, south: s, nevada: k / t };
  });
}

test.describe("the opening frames the coastal ocean, not the state", () => {
  for (const [name, at] of [["with a current forecast", CURRENT], ["at rest, nothing current", REST]] as const)
    test(`desktop, ${name}: Crescent City to San Diego fill most of the height, in view, and Nevada is a sliver`, async ({ page }) => {
      await open(page, "/", at);
      const f = await framing(page);
      const rail = (await page.locator('section[aria-label="Map time"]').boundingBox())!;
      // the coast's own extent fills at least 70 % of the map's height
      expect(f.south.y - f.north.y).toBeGreaterThanOrEqual(0.7 * f.h);
      // both ends are in view, clear of the header and above the time rail
      expect(f.north.y).toBeGreaterThan(56);
      expect(f.south.y).toBeLessThan(rail.y);
      for (const p of [f.north, f.south]) {
        expect(p.x).toBeGreaterThan(280);
        expect(p.x).toBeLessThan(f.w - 60);
      }
      // the ocean is the subject: the interior of Nevada is nowhere near the majority of the frame
      expect(f.nevada).toBeLessThan(0.12);
    });

  test("phone: the coast and its ocean above the sheet, not the interior; deep links keep their own camera", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await open(page, "/", REST, { width: 390, height: 844 });
    const f = await framing(page);
    const sheetTop = await page.getByTestId("mobile-sheet").evaluate((el) => el.getBoundingClientRect().top);
    const canyon = await page.evaluate(() => (window as unknown as W).__cwMap!.project([-121.95, 36.75]));
    expect(canyon.x).toBeGreaterThan(0);
    expect(canyon.x).toBeLessThan(390);
    expect(canyon.y).toBeLessThan(sheetTop);
    expect(f.nevada).toBeLessThan(0.2);
    // a region link still frames its region
    await page.goto(`${OK}/?region=monterey_bay`);
    await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
    const z = await page.evaluate(() => (window as unknown as { __cwMap: { getZoom(): number } }).__cwMap.getZoom());
    expect(z).toBeGreaterThan(8);
    await ctx.close();
  });
});

test.describe("nothing current: the instrument at rest", () => {
  test("one status line in the time rail, two dated choices beside it, the caveat once, and the status chip is not Ready", async ({ page }) => {
    await open(page, "/", REST);
    const status = page.locator('[data-testid="rail-status"] [data-testid="nothing-current"]');
    await expect(status).toBeVisible();
    // one line, not a block of text in the legend
    await expect(page.getByTestId("map-legend").getByTestId("nothing-current")).toHaveCount(0);
    const line = page.getByTestId("nothing-current-line");
    await expect(line).toContainText("No current forecast or recent satellite view");
    await expect(line).toContainText(/newest: C-HARM Oct 8 · satellite \w{3} \d{1,2}/);
    const box = (await line.boundingBox())!;
    expect(box.height).toBeLessThanOrEqual(22);
    await expect(page.getByText(/does not mean conditions are normal/)).toHaveCount(1);
    // each choice carries its own date and age; neither is offered as current
    await expect(page.getByTestId("show-latest-satellite")).toHaveText(/^Show satellite chlorophyll · \w{3} \d{1,2} · \d+ days? old$/);
    await expect(page.getByTestId("show-latest-forecast")).toHaveText(/^Show the Oct 8 forecast · out of date, 6 days old$/);
    // the seafloor is drawn as an instrument: a depth legend, the isobaths labelled from the coastal view
    await expect(page.getByTestId("legend-none")).toContainText("Seafloor depth");
    expect(await page.evaluate(() => (window as unknown as { __cwMap: { getLayer(id: string): { minzoom?: number } } }).__cwMap.getLayer("isobath-labels")?.minzoom)).toBeLessThan(6);
    // the model and the satellite are reported old even with no layer drawn: the chip never reads Ready
    await expect(page.getByTestId("status-center")).not.toHaveAttribute("data-state", "ready");
    await expect(page.getByTestId("status-center")).toContainText("Some data is old");
  });

  test("phone: the line and both choices are in the sheet's peek, above the dock", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await open(page, "/", REST, { width: 320, height: 640 });
    await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "peek");
    const dock = (await page.getByTestId("nav-dock").boundingBox())!;
    for (const id of ["nothing-current-line", "show-latest-satellite", "show-latest-forecast"]) {
      const b = (await page.getByTestId(id).first().boundingBox())!;
      expect(b.y + b.height, id).toBeLessThanOrEqual(dock.y + 1);
    }
    await ctx.close();
  });
});

test.describe("currents: speed encoded, rail stable", () => {
  test.use({ serviceWorkers: "block" });
  test("a faster cell draws a longer streak (the layer's own streaks), and the legend gives the m/s scale", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay", CURRENT);
    await page.waitForFunction(() => ((window as unknown as W).__cwFlow?.streaks().length ?? 0) > 40, null, { timeout: 15000 });
    const s = await page.evaluate(() => (window as unknown as W).__cwFlow!.streaks());
    const sorted = [...s].sort((a, b) => a.speed - b.speed);
    const q = Math.max(5, Math.floor(sorted.length / 4));
    const mean = (xs: { px: number }[]) => xs.reduce((a, x) => a + x.px, 0) / xs.length;
    const slow = sorted.slice(0, q), fast = sorted.slice(-q);
    expect(fast[0].speed).toBeGreaterThan(slow[slow.length - 1].speed);
    expect(mean(fast)).toBeGreaterThan(mean(slow) * 1.5);
    const legend = page.getByTestId("legend-flow-scale").first();
    await expect(legend).toContainText("m/s");
    await expect(page.getByTestId("flow-scale-note").first()).toContainText("Screen speed is a scaled symbol");
  });

  test("playing the hours never moves the time rail (no layout shift)", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay", CURRENT);
    const rail = page.getByTestId("timebar");
    await expect(rail).toBeVisible();
    const y0 = (await rail.boundingBox())!.y;
    await page.getByTestId("time-play").click();
    const ys: number[] = [];
    for (let i = 0; i < 16; i++) {
      await page.waitForTimeout(300);
      ys.push((await rail.boundingBox())!.y);
    }
    await page.getByTestId("time-play").click();
    for (const y of ys) expect(Math.abs(y - y0)).toBeLessThanOrEqual(0.5);
  });
});

test.describe("a stale value says so in place", () => {
  test("a point's C-HARM and satellite rows carry their run or observation date and age beside the value", async ({ page }) => {
    await open(page, "/?layer=olci300&sel=point:36.85,-122.1", STALE);
    const model = page.getByTestId("point-forecast");
    await expect(model).toBeVisible();
    if (await model.getByTestId("point-value").count()) {
      await expect(model.getByTestId("point-stale")).toContainText("run Oct 8, 4 days old");
      await expect(model).toContainText("out-of-date run");
    }
    const sat = page.getByTestId("point-satellite");
    if (await sat.getByTestId("point-value").count()) await expect(sat.getByTestId("point-stale")).toContainText(/observed \w{3} \d{1,2}, \d+ days? old/);
  });

  test("a model cell's value carries the run's age", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&sel=cell:36.8,-121.95", STALE);
    await expect(page.getByTestId("cell-value")).toBeVisible();
    if ((await page.getByTestId("cell-value").innerText()) !== "No value") await expect(page.getByTestId("cell-stale")).toContainText("run Oct 8, 4 days old · out-of-date run");
  });

  test("while the run is current, nothing is marked", async ({ page }) => {
    await open(page, "/?layer=olci300&sel=point:36.85,-122.1", CURRENT);
    await expect(page.getByTestId("point-forecast")).toBeVisible();
    await expect(page.getByTestId("point-forecast").getByTestId("point-stale")).toHaveCount(0);
  });
});

test.describe("a selection on a phone", () => {
  test("320 px: no layer row in the sheet, the place and its water above the sheet, the first line of evidence in the sheet, the credits a chip", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await open(page, "/?layer=olci300&sel=harbor:593", CURRENT, { width: 320, height: 640 });
    await expect(page.getByTestId("inspect-title")).toHaveText("Santa Cruz");
    await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "half");
    await page.waitForTimeout(800); // the sheet and the camera settle
    // the sheet leads with the place: the layer row leaves it, and the layer stays one tap away as a chip on the map
    await expect(page.getByTestId("mobile-sheet").getByTestId("layer-capsule")).toHaveCount(0);
    await expect(page.getByTestId("layer-capsule")).toBeVisible();
    const sheetTop = await page.getByTestId("mobile-sheet").evaluate((el) => el.getBoundingClientRect().top);
    // the first line of evidence is inside the sheet as it opens, above the dock
    const first = page.getByTestId("harbor-claims").locator("li").first();
    await expect(first).toBeVisible();
    const dock = (await page.getByTestId("nav-dock").boundingBox())!;
    const fb = (await first.boundingBox())!;
    expect(fb.y).toBeLessThan(dock.y - 20);
    // the harbour is in the map strip above the sheet
    const at = await page.evaluate(() => (window as unknown as W).__cwMap!.project([-122.03081, 36.97412]));
    expect(at.y).toBeGreaterThan(90);
    expect(at.y).toBeLessThan(sheetTop - 10);
    expect(at.x).toBeGreaterThan(0);
    expect(at.x).toBeLessThan(320);
    // the credits fold to a chip that still names OpenStreetMap, and open in full on a tap
    const chip = page.getByTestId("credits-chip");
    await expect(chip).toContainText("© OpenStreetMap");
    await expect(page.getByTestId("attribution-short")).toHaveCount(0);
    await chip.click();
    const credit = page.getByTestId("attribution-short");
    await expect(credit).toBeVisible();
    await expect(credit.getByRole("link", { name: "© OpenStreetMap contributors" })).toHaveAttribute("href", "https://www.openstreetmap.org/copyright");
    await expect(credit).toContainText("not for navigation");
    // closing the place brings the layer row back
    await page.getByTestId("inspect-close").click();
    await expect(page.getByTestId("mobile-sheet").getByTestId("layer-capsule")).toBeVisible();
    await ctx.close();
  });

  test("desktop: with the inspector open the map tools lie in one row under it, clear of the coast beside it", async ({ page }) => {
    await open(page, "/?layer=olci300&sel=harbor:593", CURRENT);
    const aside = (await page.locator('aside[aria-label="Selected"]').boundingBox())!;
    const tools = page.getByTestId("map-controls");
    await expect(tools).toHaveAttribute("data-row", "true");
    const t = (await tools.boundingBox())!;
    expect(t.y).toBeGreaterThanOrEqual(aside.y + aside.height);
    expect(t.x).toBeGreaterThanOrEqual(aside.x - 1);
  });
});

test.describe("v10 fix 2: the phone's map tools are docked, off the coast", () => {
  for (const size of [{ width: 320, height: 640 }, { width: 390, height: 844 }]) {
    test(`${size.width} px: locate and More sit at the right edge just above the sheet; share and the field pack fold behind More`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: size, isMobile: true, hasTouch: true });
      const page = await ctx.newPage();
      await open(page, "/?layer=olci300&region=monterey_bay", CURRENT, size);
      const tools = page.getByTestId("map-controls");
      await expect(tools).toHaveAttribute("data-docked", "true");
      const t = (await tools.boundingBox())!;
      const sheet = (await page.getByTestId("mobile-sheet").boundingBox())!;
      // docked: its bottom just above the sheet's top, its right at the screen's edge, not up in the frame
      expect(t.y + t.height).toBeLessThanOrEqual(sheet.y + 1);
      expect(sheet.y - (t.y + t.height)).toBeLessThan(24);
      expect(size.width - (t.x + t.width)).toBeLessThan(16);
      // Santa Cruz's harbour mark is not under them
      const sc = await page.evaluate(() => (window as unknown as W).__cwMap!.project([-122.0, 36.96]));
      expect(sc.x >= t.x && sc.x <= t.x + t.width && sc.y >= t.y && sc.y <= t.y + t.height).toBe(false);
      // 56 px targets; share and the pack one tap behind More
      for (const b of await tools.locator("button").all()) expect((await b.boundingBox())!.height).toBeGreaterThanOrEqual(55.5);
      await expect(page.getByTestId("share-view")).toHaveCount(0);
      const more = page.getByTestId("map-tools-toggle");
      await expect(more).toHaveAttribute("aria-expanded", "false");
      await more.click();
      await expect(more).toHaveAttribute("aria-expanded", "true");
      await expect(page.getByTestId("share-view")).toBeVisible();
      await expect(page.getByTestId("pack-open")).toBeVisible();
      await page.getByTestId("pack-open").click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await ctx.close();
    });
  }
});
