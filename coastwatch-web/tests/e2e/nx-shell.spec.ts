import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Neritic v12: one entry, one chart (docs/coastwatch/v12/design-system/journeys.md, README §6 and §8).
 * Storage is empty unless a test sets it: these are first and returning visits.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const SAVED_PORT = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: [], vessel: null });

async function seed(ctx: BrowserContext, items: Record<string, string>) {
  await ctx.addInitScript((kv) => {
    // only on the first document of the context, so a test can watch what a later visit does to the keys
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, items);
}

const mapReady = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
const scene = (page: Page) => page.evaluate(() => (window as unknown as { __nxChart?: { scene: string | null } }).__nxChart?.scene ?? null);
const opening = (page: Page) => page.evaluate(() => document.documentElement.getAttribute("data-nx-opening"));

test.describe("one chart across the four destinations", () => {
  test("the same canvas, one MapLibre instance, the camera moving to each page's scene", async ({ page, context }) => {
    await seed(context, { "nx-opening-seen-v1": "1" });
    await page.goto(`${OK}/today?place=port-592`);
    await mapReady(page);
    await page.locator("[data-testid=nx-map] canvas").evaluate((c) => ((c as HTMLCanvasElement).dataset.nxTag = "first"));
    expect(await scene(page)).toBe("today:port-592");
    const bar = page.getByTestId("nx-bar");
    const visits: Array<[string, RegExp, string]> = [
      ["Explore", /\/explore\?place=port-592$/, "explore:port-592"],
      ["My Coast", /\/coast\?place=port-592$/, "coast:port-592"],
      ["Research", /\/research\?place=port-592$/, "research:port-592"],
      ["Today", /\/today\?place=port-592$/, "today:port-592"],
    ];
    for (const [label, url, id] of visits) {
      const before = await page.evaluate(() => (window as unknown as { __nxChart: { camera: { zoom: number } } }).__nxChart.camera.zoom);
      await bar.getByRole("link", { name: label, exact: true }).click();
      await expect(page).toHaveURL(url);
      await expect.poll(() => scene(page)).toBe(id);
      await expect(page.getByRole("link", { name: label, exact: true }).first()).toHaveAttribute("aria-current", "page");
      // the camera eases there (a different view for each page)
      await expect.poll(() => page.evaluate(() => (window as unknown as { __nxChart: { camera: { zoom: number } } }).__nxChart.camera.zoom)).not.toBeCloseTo(before, 1);
    }
    await page.getByRole("link", { name: "Neritic, home" }).click();
    await expect(page).toHaveURL(`${OK}/`);
    // the very same element: never re-created
    await expect(page.locator("canvas[data-nx-tag=first]")).toHaveCount(1);
    await expect(page.locator(".maplibregl-canvas")).toHaveCount(1);
    expect(await page.evaluate(() => (window as unknown as { __nxChart: { inits: number } }).__nxChart.inits)).toBe(1);
  });

  test("the port travels: tabs carry it, and a later visit without it keeps it", async ({ page, context }) => {
    await seed(context, { "nx-opening-seen-v1": "1" });
    await page.goto(`${OK}/coast?place=port-550`);
    await expect(page.getByTestId("nx-port-control")).toContainText("Monterey");
    await expect(page.getByTestId("nx-bar").getByRole("link", { name: "Research", exact: true })).toHaveAttribute("href", "/research?place=port-550");
    await page.goto(`${OK}/explore`);
    await expect(page.getByTestId("nx-port-control")).toContainText("Monterey");
    await expect.poll(() => scene(page)).toBe("explore:port-550");
  });

  test("choosing a port from the finder opens Today there and remembers it", async ({ page, context }) => {
    await seed(context, { "nx-opening-seen-v1": "1" });
    await page.goto(`${OK}/`);
    await mapReady(page);
    // a first visit stays at /: the finder opens in the margin, the port's name travels into the chooser, a
    // fishery is asked, and Today assembles in place (journeys.md Journey 1, three taps)
    await page.getByTestId("nx-find-port").click();
    const finder = page.getByTestId("nx-home-finder");
    await finder.getByRole("combobox").fill("moss");
    await finder.getByRole("option", { name: /Moss Landing/ }).click();
    const chooser = page.getByTestId("nx-fishery-chooser");
    await expect(chooser.getByRole("heading", { level: 1 })).toHaveText("Moss Landing");
    await expect(page).toHaveURL(`${OK}/`);
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem("nx-port-v1") ?? "{}").lastPort)).toBe("port-592");
    await chooser.getByRole("button", { name: "Dungeness crab (commercial)" }).click();
    await expect(page.getByTestId("nx-today")).toContainText("Moss Landing");
    await expect(page.getByTestId("nx-today").getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027");
    await expect(page).toHaveURL(`${OK}/`);
    await expect.poll(() => scene(page)).toBe("today:port-592");
    // the bar's port control changes the port from any page, and the page follows it
    await page.goto(`${OK}/explore`);
    await page.getByTestId("nx-port-control").click();
    const bar = page.getByTestId("nx-port-finder");
    await bar.getByRole("combobox").fill("monterey");
    await bar.getByRole("option", { name: /^Monterey/ }).first().click();
    await expect(page).toHaveURL(/\/explore\?place=port-550$/);
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem("nx-port-v1") ?? "{}").lastPort)).toBe("port-550");
  });
});

test.describe("one entry at /", () => {
  test("a first visit plays the opening once: statement from frame one, the still, then the chart", async ({ page }) => {
    const res = await page.goto(`${OK}/`, { waitUntil: "commit" });
    const html = await res!.text();
    expect(html).toContain("Season status, crab tests, whale measures and the next decision date for your port.");
    expect(html).toContain("Find your port");
    await expect(page.getByTestId("nx-still")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(["on", "run"]).toContain(await opening(page));
    await expect(page.getByTestId("nx-still")).toContainText("Depth exaggerated 4×");
    // at most 5 s (plus a little for a loaded machine), then the home screen, in place
    await expect.poll(() => opening(page), { timeout: 7_000 }).toBeNull();
    await expect(page.getByTestId("nx-still")).toBeHidden();
    await expect(page).toHaveURL(`${OK}/`);
    await expect(page.getByTestId("nx-home")).toBeVisible();
    // never again
    await page.reload();
    expect(await opening(page)).toBeNull();
    await expect(page.getByTestId("nx-still")).toBeHidden();
    await expect(page.getByTestId("nx-home")).toBeVisible();
  });

  test("Skip ends it at once on the final frame", async ({ page }) => {
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-skip-opening")).toBeVisible();
    const t = Date.now();
    // the button itself (a real pointer anywhere would already have skipped it: see the next test)
    await page.getByTestId("nx-skip-opening").dispatchEvent("click");
    await expect(page.getByTestId("nx-still")).toBeHidden({ timeout: 1_000 });
    expect(Date.now() - t).toBeLessThan(1_500);
    await expect(page.getByTestId("nx-skip-opening")).toBeHidden();
    await expect(page.getByTestId("nx-find-port")).toBeVisible();
    await mapReady(page);
    // the chart is whole: water filled, lines drawn
    await expect.poll(() => page.evaluate(() => (window as unknown as { __nxChart: { map: { getPaintProperty(l: string, p: string): unknown } } }).__nxChart.map.getPaintProperty("depth-10", "fill-opacity"))).toBe(1);
  });

  test("any key or pointer skips it, even before the app's script has loaded", async ({ page }) => {
    await page.route(/\/_next\/static\/chunks\//, (r) => new Promise((done) => setTimeout(() => done(r.continue()), 1500)));
    await page.goto(`${OK}/`, { waitUntil: "commit" });
    await expect(page.getByTestId("nx-still")).toBeVisible();
    await page.keyboard.press("Space");
    await expect(page.getByTestId("nx-still")).toBeHidden({ timeout: 1_000 });
    expect(await opening(page)).not.toBe("run");
  });

  test("a returning reader with a saved port gets Today at /, and home never paints", async ({ page, context }) => {
    await seed(context, { "nx-port-v1": SAVED_PORT });
    await context.addInitScript(() => {
      // watch every frame from the document's start: was the home statement ever rendered visible?
      const w = window as unknown as { __homeSeen: boolean };
      w.__homeSeen = false;
      const look = () => {
        const el = document.querySelector("[data-nx-home]");
        if (el && getComputedStyle(el).display !== "none" && (el as HTMLElement).offsetHeight > 0) w.__homeSeen = true;
      };
      new MutationObserver(look).observe(document, { childList: true, subtree: true });
      const tick = () => {
        look();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-today")).toContainText("Moss Landing");
    await expect(page).toHaveURL(`${OK}/`);
    expect(await opening(page)).toBeNull();
    await expect.poll(() => scene(page)).toBe("today:port-592");
    await page.waitForTimeout(500);
    expect(await page.evaluate(() => (window as unknown as { __homeSeen: boolean }).__homeSeen)).toBe(false);
    await expect(page.getByTestId("nx-bar").getByRole("link", { name: "Today", exact: true })).toHaveAttribute("aria-current", "page");
  });

  test("a v11 reader's old keys count as a saved port", async ({ page, context }) => {
    await seed(context, { "nx-today-place-v1": "port-593" });
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-today")).toContainText("Santa Cruz");
  });

  test("a deep link never gets the film", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-550`);
    expect(await opening(page)).toBeNull();
    await expect(page.getByTestId("nx-today")).toContainText("Monterey");
    await expect(page.getByTestId("nx-port-control")).toContainText("Monterey");
    // the opening is still owed to this device's first visit to /
    expect(await page.evaluate(() => localStorage.getItem("nx-opening-seen-v1"))).toBeNull();
  });
});

test.describe("addresses", () => {
  test("old addresses redirect to the one home of each view", async ({ request }) => {
    const go = async (path: string) => {
      const r = await request.get(`${OK}${path}`, { maxRedirects: 0 });
      return { status: r.status(), to: r.headers().location };
    };
    expect(await go("/?view=map")).toEqual({ status: 308, to: "/explore" });
    expect(await go("/?sel=harbor:592&layer=forecast")).toEqual({ status: 308, to: "/explore?sel=harbor%3A592&layer=forecast" });
    expect(await go("/bloom?station=SCW&var=pda")).toEqual({ status: 308, to: "/research?station=SCW&variable=pda" });
    expect(await go("/coast?mode=research&place=port-592")).toEqual({ status: 308, to: "/research?place=port-592" });
    expect((await go("/?utm_source=x")).status).toBe(200);
    const manifest = await (await request.get(`${OK}/manifest.webmanifest`)).json();
    expect(manifest.start_url).toBe("/");
  });

  test("deep links open their page; legacy pages keep working", async ({ page }) => {
    for (const [path, id] of [
      ["/explore?place=port-593", "nx-explore"],
      ["/research?place=port-452", "nx-research"],
      ["/coast?place=port-550", "nx-coast"],
    ] as const) {
      await page.goto(`${OK}${path}`);
      await expect(page.getByTestId(id)).toBeVisible();
    }
    for (const path of ["/about", "/privacy", "/terms", "/sources", "/ask", "/account", "/sms-terms", "/fisheries", "/coast/kept"]) {
      const r = await page.goto(`${OK}${path}`);
      expect(r!.status(), path).toBe(200);
      await expect(page.getByTestId("nx-app"), path).toHaveCount(0);
      await expect(page.locator("main, h1").first(), path).toBeVisible();
    }
  });
});

test.describe("motion and graphics tiers", () => {
  test("reduced motion: no opening, the final frame", async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: "reduce", storageState: { cookies: [], origins: [] }, viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    await page.goto(`${OK}/`);
    expect(await opening(page)).toBeNull();
    await expect(page.getByTestId("nx-still")).toBeHidden();
    await expect(page.getByTestId("nx-home")).toBeVisible();
    await mapReady(page);
    await ctx.close();
  });

  test("static tier: no MapLibre, the baked chart of the same view under the same names", async ({ page, context }) => {
    await context.addInitScript(() => {
      const get = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...rest: unknown[]) {
        return /webgl/.test(type) ? null : (get as (...a: unknown[]) => unknown).call(this, type, ...rest);
      } as typeof get;
      delete (window as unknown as { WebGL2RenderingContext?: unknown }).WebGL2RenderingContext;
    });
    const scripts: string[] = [];
    page.on("request", (r) => scripts.push(r.url()));
    await page.goto(`${OK}/today?place=port-592`);
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", "static");
    await expect(page.getByTestId("nx-poster").locator("img").first()).toBeVisible();
    await expect(page.locator(".maplibregl-canvas")).toHaveCount(0);
    await expect(page.locator(".n12-port-name-focus")).toHaveText("Moss Landing");
    await expect(page.locator(".n12-port-name-focus")).toBeVisible();
    await expect(page.getByTestId("nx-neatline")).toBeVisible();
    expect(scripts.filter((u) => /maplibre-gl-worker|maplibre/.test(u) && !/\.webp/.test(u))).toEqual([]);
    // the first visit on a static device gets the final frame too
    await page.goto(`${OK}/`);
    expect(await opening(page)).toBeNull();
  });

  test("day, dusk and night: the chart and the page take each mode's tokens", async ({ page, context }) => {
    await seed(context, { "nx-opening-seen-v1": "1" });
    await page.goto(`${OK}/today?place=port-592`);
    await mapReady(page);
    const paint = () =>
      page.evaluate(() => {
        const m = (window as unknown as { __nxChart: { map: { getPaintProperty(l: string, p: string): string } } }).__nxChart.map;
        return { bg: m.getPaintProperty("background", "background-color"), land: m.getPaintProperty("land", "fill-color"), page: getComputedStyle(document.querySelector("[data-testid=nx-app]")!).backgroundColor };
      });
    const day = await paint();
    expect(day.land).toBe("#dfd8ae");
    const seen = new Set([JSON.stringify(day)]);
    for (const mode of ["dusk", "night", "day"] as const) {
      await page.getByTestId("nx-menu-button").click();
      await page.getByTestId(`nx-mode-${mode}`).check();
      await expect(page.locator("html")).toHaveAttribute("data-chart-mode", mode);
      await page.keyboard.press("Escape");
      await expect.poll(async () => (await paint()).land).toBe({ day: "#dfd8ae", dusk: "#3d3a2e", night: "#15140f" }[mode]);
      seen.add(JSON.stringify(await paint()));
    }
    expect(seen.size).toBe(3);
    // kept for the next visit, before the first paint
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-chart-mode", "day");
  });

  test("a dark system starts in Dusk; Night is never automatic", async ({ browser }) => {
    const ctx = await browser.newContext({ colorScheme: "dark", storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    await page.goto(`${OK}/explore`);
    await expect(page.locator("html")).toHaveAttribute("data-chart-mode", "dusk");
    await ctx.close();
  });
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test("top bar with the port, four destinations in a 56 px bottom bar, the margin as a sheet", async ({ page, context }) => {
    await seed(context, { "nx-port-v1": SAVED_PORT });
    await page.goto(`${OK}/explore`);
    const tabs = page.getByTestId("nx-tabs");
    await expect(tabs).toBeVisible();
    expect((await tabs.boundingBox())!.height).toBeGreaterThanOrEqual(56);
    await expect(page.getByTestId("nx-bar").locator(".n12-dest")).toBeHidden();
    await expect(page.getByTestId("nx-port-control")).toContainText("Moss Landing");
    for (const a of await tabs.getByRole("link").all()) expect((await a.boundingBox())!.height).toBeGreaterThanOrEqual(48);
    await tabs.getByRole("link", { name: "Research" }).click();
    await expect(page).toHaveURL(/\/research\?place=port-592$/);
    // the margin is the ds MarginSheet: it rests with the page's statement above the tab bar, More raises it
    const margin = page.getByTestId("nx-margin");
    await expect(page.getByTestId("research-title")).toBeInViewport();
    const rest = (await margin.boundingBox())!.y;
    expect(rest).toBeGreaterThan(844 * 0.4);
    await margin.getByRole("button", { name: "Show more of this page" }).click();
    await expect.poll(async () => (await margin.boundingBox())!.y).toBeLessThan(rest - 100);
    // the chart frames its scene in the part the resting sheet leaves free
    const cover = await page.evaluate(() => getComputedStyle(document.querySelector(".n12-frame")!).getPropertyValue("--nx-cover-bottom"));
    expect(parseFloat(cover)).toBeGreaterThan(200);
  });
});
