import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Review media for the v12 shell (docs/coastwatch/v12/implementation/shell/): videos of the opening and of
 * page-to-page navigation, and stills at the widths the product is used at. Runs only with NX_MEDIA=1.
 */
const OUT = path.resolve(__dirname, "../../../docs/coastwatch/v12/implementation/shell");
test.skip(!process.env.NX_MEDIA, "media only on request (NX_MEDIA=1)");
test.describe.configure({ mode: "serial" });

const ready = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });

async function recorded(browser: Browser, name: string, size: { width: number; height: number }, body: (page: Page) => Promise<void>, init?: () => void) {
  mkdirSync(OUT, { recursive: true });
  const dir = path.join(test.info().outputDir, "video");
  const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] }, recordVideo: { dir, size } });
  if (init) await ctx.addInitScript(init);
  const page = await ctx.newPage();
  await body(page);
  const video = page.video();
  await ctx.close();
  if (video) copyFileSync(await video.path(), path.join(OUT, `${name}.webm`));
}

test("video: the opening on a first visit, settling into home", async ({ browser }) => {
  await recorded(browser, "opening-1440", { width: 1440, height: 900 }, async (page) => {
    await page.goto(`${OK}/`);
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUT, "opening-frame-0.8s-1440.png") });
    await page.waitForTimeout(1900);
    await page.screenshot({ path: path.join(OUT, "opening-frame-2.7s-1440.png") });
    await expect.poll(() => page.evaluate(() => document.documentElement.getAttribute("data-nx-opening")), { timeout: 7_000 }).toBeNull();
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUT, "home-1440.png") });
  });
});

test("video: the opening on a phone", async ({ browser }) => {
  await recorded(browser, "opening-390", { width: 390, height: 844 }, async (page) => {
    await page.goto(`${OK}/`);
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(OUT, "opening-frame-0.8s-390.png") });
    await expect.poll(() => page.evaluate(() => document.documentElement.getAttribute("data-nx-opening")), { timeout: 7_000 }).toBeNull();
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(OUT, "home-390.png") });
  });
});

test("video: page to page, one chart panning under the margin", async ({ browser }) => {
  await recorded(
    browser,
    "navigation-1440",
    { width: 1440, height: 900 },
    async (page) => {
      await page.goto(`${OK}/`);
      await ready(page);
      await page.waitForTimeout(600);
      await page.getByTestId("nx-find-port").click();
      await page.getByTestId("nx-port-finder").getByRole("textbox").pressSequentially("moss", { delay: 90 });
      await page.waitForTimeout(300);
      await page.getByTestId("nx-port-finder").getByRole("button", { name: /Moss Landing/ }).click();
      await page.waitForTimeout(1400);
      for (const label of ["Explore", "My Coast", "Research", "Today"]) {
        await page.getByTestId("nx-bar").getByRole("link", { name: label, exact: true }).click();
        await page.waitForTimeout(1500);
      }
      await page.getByRole("link", { name: "Neritic, home" }).click();
      await page.waitForTimeout(1400);
    },
    () => localStorage.setItem("nx-opening-seen-v1", "1"),
  );
});

test("video: page to page on a phone", async ({ browser }) => {
  await recorded(
    browser,
    "navigation-390",
    { width: 390, height: 844 },
    async (page) => {
      await page.goto(`${OK}/today?place=port-592`);
      await ready(page);
      await page.waitForTimeout(800);
      for (const label of ["Explore", "My Coast", "Research", "Today"]) {
        await page.getByTestId("nx-tabs").getByRole("link", { name: label }).click();
        await page.waitForTimeout(1400);
      }
    },
    () => localStorage.setItem("nx-opening-seen-v1", "1"),
  );
});

const SIZES = [
  [1440, 900],
  [390, 844],
  [320, 640],
  [844, 390],
] as const;

test("stills: every destination at every width, and day, dusk, night and the static tier", async ({ browser }) => {
  mkdirSync(OUT, { recursive: true });
  for (const [w, h] of SIZES) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, storageState: { cookies: [], origins: [] } });
    await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    const page = await ctx.newPage();
    for (const [name, url] of [
      ["home", "/"],
      ["today", "/today?place=port-592"],
      ["explore", "/explore?place=port-592"],
      ["coast", "/coast?place=port-592"],
      ["research", "/research?place=port-592"],
    ]) {
      await page.goto(`${OK}${url}`);
      await ready(page);
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(OUT, `${name}-${w}x${h}.png`) });
    }
    await ctx.close();
  }
  for (const mode of ["dusk", "night"]) {
    for (const [w, h] of [SIZES[0], SIZES[1]]) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, storageState: { cookies: [], origins: [] } });
      await ctx.addInitScript((m) => {
        localStorage.setItem("nx-opening-seen-v1", "1");
        localStorage.setItem("nx-chart-mode-v1", m);
      }, mode);
      const page = await ctx.newPage();
      await page.goto(`${OK}/today?place=port-592`);
      await ready(page);
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(OUT, `today-${mode}-${w}x${h}.png`) });
      await ctx.close();
    }
  }
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, storageState: { cookies: [], origins: [] } });
  await ctx.addInitScript(() => {
    const get = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...rest: unknown[]) {
      return /webgl/.test(type) ? null : (get as (...a: unknown[]) => unknown).call(this, type, ...rest);
    } as typeof get;
    delete (window as unknown as { WebGL2RenderingContext?: unknown }).WebGL2RenderingContext;
  });
  const page = await ctx.newPage();
  await page.goto(`${OK}/today?place=port-592`);
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", "static");
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "today-static-1440x900.png") });
  await page.goto(`${OK}/explore`);
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, "explore-coast-static-1440x900.png") });
  await ctx.close();
});
