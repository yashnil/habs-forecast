import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";
import { openLegend, openMoreTools } from "./helpers";

/**
 * Neritic v11 Explore (design.md §7.3, audit §4): the map is the product. Three chrome zones (top bar,
 * field dock with the time rail, tools), the legend as the map's title, a place sheet led by its name.
 * The honesty rules that came with moving everything behind disclosure are pinned here.
 */
const CURRENT = "2026-10-08T20:00:00Z"; // the fixture's run is current
const OLD = "2026-10-14T20:00:00Z"; // the run and the satellite are old by then
const NO_VIEW = "layer=olci300&sel=point:34.5,-123.9"; // open water the fixture satellite never saw

type M = { loaded(): boolean };
const mapReady = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded(), null, { timeout: 40_000 });
async function open(page: Page, q: string, at = CURRENT, size = { width: 1440, height: 900 }) {
  await page.setViewportSize(size);
  await page.clock.setFixedTime(new Date(at));
  await page.goto(`${OK}/?${q}`);
  await mapReady(page);
}
const phone = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

test.describe("the field dock: the legend is the map's title", () => {
  test("chlorophyll says what it is, its kind, its date and 'not toxin', beside its ramp and the 'no view' swatch", async ({ page }) => {
    await open(page, "layer=olci300");
    const head = page.getByTestId("dock-head");
    await expect(head).toContainText("Chlorophyll");
    await expect(page.getByTestId("dock-kind")).toHaveText("observed");
    await expect(page.getByTestId("dock-date")).toHaveText(/^\w{3} \d{1,2}/);
    await expect(page.getByTestId("dock-caveat")).toHaveText("not toxin");
    await expect(page.getByTestId("dock-ramp")).toBeVisible();
    const noView = page.getByTestId("no-view-key");
    await expect(noView).toContainText("no view");
    await expect(noView.locator(".nx-sw-gap")).toBeVisible();
    // nothing else floats on the map: no always-on layer list
    await expect(page.getByTestId("layer-deck")).toBeHidden();
  });

  test("C-HARM carries 'probability of conditions, not toxin in seafood' on the dock", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=1");
    await expect(page.getByTestId("dock-kind")).toContainText("model");
    await expect(page.getByTestId("dock-caveat")).toHaveText("probability of conditions, not toxin in seafood");
  });

  test("an old date is marked in place on the dock, in words; a run whose days have passed is a past run, not a forecast", async ({ page }) => {
    await open(page, "layer=olci300", OLD);
    const date = page.getByTestId("dock-date");
    await expect(date).toHaveAttribute("data-stale", "true");
    await expect(date).toContainText("(old)");
    await open(page, "layer=forecast&var=particulate_domoic&lead=1", OLD);
    await expect(page.getByTestId("dock-date")).toContainText("(old)");
    await expect(page.getByTestId("dock-kind")).toHaveText("model, past run");
  });

  test("the readout marker sits on the ramp at the point selected; a point with no view says so and never moves the marker to low", async ({ page }) => {
    await open(page, "layer=olci300&sel=point:36.85,-122.1");
    await expect(page.getByTestId("dock-readout")).toContainText("mg/m³");
    const pct = await page.getByTestId("dock-marker").evaluate((el) => parseFloat((el as HTMLElement).style.left));
    expect(pct).toBeGreaterThan(0);
    expect(pct).toBeLessThan(100);
    await open(page, NO_VIEW);
    await expect(page.getByTestId("dock-readout")).toContainText("no view here");
    await expect(page.getByTestId("dock-marker")).toHaveCount(0);
  });

  test("the layer list opens from the dock, grouped by kind, one line per layer; source, age and caveat on demand; Escape returns to the button", async ({ page }) => {
    await open(page, "layer=olci300");
    const capsule = page.getByTestId("layer-capsule");
    await expect(capsule).toHaveAttribute("aria-expanded", "false");
    await capsule.click();
    const deck = page.getByTestId("layer-deck");
    await expect(deck).toBeVisible();
    await expect(deck.locator(".nx-layer-group")).toHaveText(["Observed", "Analysis", "Model", "Outlook", "Reference"]);
    const note = page.getByTestId("layer-note");
    await expect(note).toContainText("Observed · satellite");
    await expect(note).toContainText("not toxin");
    // the arrow keys move in the order drawn: chlorophyll, then the currents
    await page.getByTestId("layer-satellite").focus();
    await page.keyboard.press("ArrowDown");
    await expect(page).toHaveURL(/layer=currents/);
    await expect(page.getByTestId("layer-currents")).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(deck).toBeHidden();
    await expect(capsule).toBeFocused();
    // the gap sentence, the bays and the state-scale hint stay one fold in
    await openLegend(page);
    await expect(page.getByTestId("state-scale-note")).toBeVisible();
    await expect(page.getByTestId("legend-detail").first()).toBeVisible();
  });
});

test.describe("the time rail", () => {
  test("observed days lie left of the now line and only model or outlook frames lie right of it; each frame is named with its own date and kind", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=1");
    const now = (await page.getByTestId("rail-now").boundingBox())!;
    await expect(page.getByTestId("rail-now-label")).toHaveText("now");
    const rows = await page.locator('[data-testid^="rail-ctx-sat-"]').all();
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) expect((await r.boundingBox())!.x).toBeLessThan(now.x);
    const radio = page.getByTestId("time-lead2");
    await expect(radio).toContainText(/Oct \d+ · (nowcast|forecast)/);
    await expect(page.getByTestId("rail-label-observed")).toContainText("observed");
    await expect(page.getByTestId("rail-label-model")).toContainText("C-HARM model");
  });

  test("Play steps real frames and pauses on the control; any other input stops it", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=0");
    const play = page.getByTestId("time-play");
    await expect(play).toHaveAttribute("data-playback-control", "true");
    await play.click();
    await expect(play).toHaveAttribute("aria-pressed", "true");
    await expect(page).toHaveURL(/lead=1/, { timeout: 2500 });
    await page.mouse.move(700, 400);
    await page.mouse.down();
    await page.mouse.up();
    await expect(play).toHaveAttribute("aria-pressed", "false");
    const url = page.url();
    await page.waitForTimeout(1200);
    expect(page.url().match(/lead=\d/)?.[0]).toBe(url.match(/lead=\d/)?.[0]);
  });
});

test.describe("the map is the product", () => {
  test("at 1440×900 the chrome is three zones and about 84 % of the viewport is open map at rest", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(() => window.localStorage.setItem("nx-explore-coach-v1", "1"));
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(CURRENT));
    await page.goto(`${OK}/?layer=olci300`);
    await mapReady(page);
    const clear = await page.evaluate(() => {
      const ids = ["field-dock", "timebar-deck", "map-controls", "map-margin", "brand-home", "nav-bar", "registry-disclosure", "notices-button", "more-button"];
      const rects = [...ids.map((i) => document.querySelector(`[data-testid="${i}"]`)), document.querySelector("#place-search")].filter((e): e is Element => !!e).map((e) => e.getBoundingClientRect());
      // sample the viewport (overlaps counted once)
      let hit = 0;
      for (let y = 2; y < innerHeight; y += 4) for (let x = 2; x < innerWidth; x += 4) if (rects.some((r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom)) hit++;
      const total = Math.ceil(innerWidth / 4) * Math.ceil(innerHeight / 4);
      return 1 - hit / total;
    });
    expect(clear).toBeGreaterThan(0.8);
    const margin = await page.getByTestId("map-margin").boundingBox();
    expect(margin!.height).toBeLessThanOrEqual(21);
    await expect(page.getByTestId("attribution-line").getByRole("link", { name: "© OpenStreetMap contributors" })).toBeVisible();
    await expect(page.locator('[data-testid="map"]')).toHaveClass(/nx-vt-stage/);
    await ctx.close();
  });

  test("the read hint is a one-time coach mark", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.clock.setFixedTime(new Date(CURRENT));
    await page.goto(`${OK}/?layer=forecast`);
    await mapReady(page);
    await expect(page.getByTestId("read-hint")).toBeVisible();
    await page.getByTestId("coach-dismiss").click();
    await expect(page.getByTestId("read-hint")).toHaveCount(0);
    await page.reload();
    await mapReady(page);
    await page.waitForTimeout(400);
    await expect(page.getByTestId("read-hint")).toHaveCount(0);
  });

  test("zoom and locate stay; share, the field pack and the area tool are behind one More; Escape returns to it", async ({ page }) => {
    await open(page, "layer=olci300");
    await expect(page.getByRole("button", { name: "Zoom in" })).toBeVisible();
    await expect(page.getByTestId("recenter")).toBeVisible();
    await expect(page.getByTestId("share-view")).toHaveCount(0);
    await openMoreTools(page);
    await expect(page.getByTestId("share-view")).toBeVisible();
    await expect(page.getByTestId("pack-open")).toBeVisible();
    await expect(page.getByTestId("area-tool")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("share-view")).toHaveCount(0);
    await expect(page.getByTestId("map-tools-toggle")).toBeFocused();
  });
});

test.describe("the phone peek", () => {
  for (const snap of ["peek", "half", "full"] as const)
    test(`${snap}: 'not toxin' is on screen with the chlorophyll ramp, whatever the sheet's height`, async ({ browser }) => {
      const ctx = await browser.newContext(phone);
      const page = await ctx.newPage();
      await page.clock.setFixedTime(new Date(CURRENT));
      await page.goto(`${OK}/?layer=olci300`);
      await mapReady(page);
      const sheet = page.getByTestId("mobile-sheet");
      const handle = page.getByTestId("sheet-handle");
      for (let i = 0; i < 2 && (await sheet.getAttribute("data-snap")) !== snap; i++) await handle.click();
      await expect(sheet).toHaveAttribute("data-snap", snap);
      await page.waitForTimeout(600); // the sheet settles
      const caveat = page.getByTestId("dock-caveat");
      await expect(caveat).toHaveText("not toxin");
      await expect(caveat).toBeInViewport();
      await expect(page.getByTestId("dock-ramp")).toBeInViewport();
      const nav = (await page.getByTestId("nav-dock").boundingBox())!;
      const b = (await caveat.boundingBox())!;
      expect(b.y + b.height).toBeLessThanOrEqual(nav.y);
      if (snap === "peek") {
        // the peek gives the map its room
        const top = (await sheet.boundingBox())!.y;
        expect(nav.y - top).toBeLessThanOrEqual(112);
      }
      await ctx.close();
    });

  test("a place open: the layer chip keeps its caveat over the map", async ({ browser }) => {
    const ctx = await browser.newContext(phone);
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(CURRENT));
    await page.goto(`${OK}/?layer=olci300&sel=harbor:593`);
    await mapReady(page);
    await expect(page.getByTestId("capsule-caveat")).toHaveText("not toxin");
    await expect(page.getByTestId("capsule-ramp")).toBeVisible();
    await ctx.close();
  });
});

test.describe("the place sheet", () => {
  test("led by the name, three statements in authority order, then Open today's briefing; the rest is folded", async ({ page }) => {
    await open(page, "layer=olci300&sel=harbor:592");
    await expect(page.getByTestId("inspect-title")).toHaveText("Moss Landing");
    const claims = page.getByTestId("harbor-claims").locator('> ul > li[data-testid^="claim-"]');
    await expect(claims).toHaveCount(3);
    const kinds = await claims.evaluateAll((els) => els.map((e) => (e as HTMLElement).dataset.kind));
    const rank = kinds.map((k) => ["official", "agency_page", "measured", "satellite", "conditions", "model", "gap"].indexOf(k!));
    expect([...rank].sort((a, b) => a - b)).toEqual(rank);
    const brief = page.getByTestId("open-briefing");
    await expect(brief).toHaveText("Open today's briefing");
    await expect(brief).toHaveAttribute("href", "/today?place=port-592");
    // the first official statement says it is not verified, in words
    await expect(claims.first().getByTestId("claim-not-verified")).toContainText("Not verified");
    await expect(page.getByTestId("inspect-evidence")).toBeVisible();
    await expect(page.getByTestId("inspect-evidence")).not.toHaveAttribute("open", "");
  });

  test("the disputed shellfish notice reads 'Status unclear', awaiting review", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.clock.setFixedTime(new Date("2026-10-12T20:00:00Z"));
    await page.goto(`${FAILED}/?layer=none&sel=harbor:550`);
    await page.waitForFunction(() => !!document.querySelector('[data-testid="inspect-panel"]'), null, { timeout: 30_000 });
    const claim = page.locator('[data-testid^="claim-disputed-"]').first();
    await expect(claim.locator("[data-headline]")).toContainText("Status unclear", { timeout: 30_000 });
    await expect(claim.getByTestId("claim-awaiting")).toHaveText("awaiting review");
  });
});
