import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The product is Neritic (formerly CoastWatch). NOAA CoastWatch, the NOAA program whose data
 * it uses, keeps its name everywhere it is credited. Stable identifiers (storage keys, links, the
 * data URL) are unchanged, so saved settings and shared links keep working.
 */
const ROUTES = ["/", "/coast", "/coast?place=port-550&for=commercial", "/coast?place=port-201&mode=research", "/bloom", "/fisheries", "/about", "/sources", "/coast/kept"];

// The only places the old name may appear on a page: NOAA's own program, and the one sentence that
// tells readers the product was renamed.
const ALLOWED = [/NOAA CoastWatch/g, /CoastWatch West Coast/g, /was called CoastWatch until October 2026/g];

const dismissed = { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }] };

test.describe("Neritic brand", () => {
  test("no public page uses the old product name; NOAA CoastWatch is still credited", async ({ browser }) => {
    for (const [w, h] of [
      [1440, 900],
      [390, 844],
    ]) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, storageState: dismissed });
      const page = await ctx.newPage();
      for (const r of ROUTES) {
        await page.goto(`${OK}${r}`);
        // the kept-briefings page is a standalone printable page without the masthead
        if (r === "/coast/kept") await expect(page.getByTestId("kept-page")).toBeVisible();
        else await expect(page.getByTestId("brand-home")).toHaveAttribute("aria-label", "Neritic home");
        // visible text and accessible labels both
        const text = await page.evaluate(() => [document.body.innerText, ...[...document.querySelectorAll("[aria-label],[title],[alt]")].map((e) => `${e.getAttribute("aria-label") ?? ""} ${e.getAttribute("title") ?? ""} ${e.getAttribute("alt") ?? ""}`)].join("\n"));
        const rest = ALLOWED.reduce((t, re) => t.replace(re, ""), text);
        expect(rest.match(/coast\s?watch/gi) ?? [], `${w}px ${r}`).toEqual([]);
        // the working name of the first rebrand draft never shipped
        expect(text, `${w}px ${r}`).not.toMatch(/Neritic Signal/i);
      }
      await ctx.close();
    }
    const page = await browser.newPage();
    await page.goto(`${OK}/sources`);
    await expect(page.locator("main")).toContainText("NOAA CoastWatch");
    await expect(page.getByTestId("sources-independence")).toContainText("not affiliated with or endorsed by NOAA, CDFW, or CDPH");
  });

  test("titles, sharing metadata, icons and the app manifest carry the new name", async ({ page, request }) => {
    await page.goto(`${OK}/`);
    expect(await page.title()).toMatch(/^Neritic \(research preview\)/);
    await expect(page.locator('meta[property="og:site_name"]')).toHaveAttribute("content", "Neritic");
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute("content", "Neritic — Coastal intelligence, grounded in evidence.");
    await expect(page.locator('meta[property="og:image"]').first()).toHaveAttribute("content", /opengraph-image/);
    await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute("content", "summary_large_image");
    await expect(page.locator('link[rel="icon"][type="image/svg+xml"]')).toHaveCount(1);
    await page.goto(`${OK}/about`);
    expect(await page.title()).toBe("About — Neritic");
    const m = await (await request.get(`${OK}/manifest.webmanifest`)).json();
    expect(m.short_name).toBe("Neritic");
    expect(m.icons.map((i: { src: string }) => i.src)).toEqual(expect.arrayContaining(["/icons/icon-192.png", "/icons/icon-512.png"]));
    for (const p of ["/icon.svg", "/favicon.ico", "/apple-icon.png", "/opengraph-image", "/icons/icon-512.png"]) {
      expect((await request.get(`${OK}${p}`)).status(), p).toBe(200);
    }
  });

  test("About says plainly who makes it, the old name, and that it is not NOAA's", async ({ page }) => {
    await page.goto(`${OK}/about`);
    const p = page.getByTestId("about-independence");
    await expect(p).toContainText("Neritic is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.");
    await expect(p).toContainText("was called CoastWatch until October 2026");
    await expect(p).toContainText("not NOAA CoastWatch");
    await expect(page.getByTestId("footer-independence")).toContainText("not affiliated with or endorsed by NOAA");
  });

  test("settings saved under the old name do not break anything, and old links still open", async ({ browser }) => {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "research" }, { name: "cw-home-port", value: "port-550" }] }] },
    });
    const page = await ctx.newPage();
    await page.goto(`${OK}/coast?place=port-550`);
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
    // shared links from before the rename (and before v4)
    for (const link of ["/?port=550", "/?region=monterey_bay&layer=forecast&var=pseudo_nitzschia&lead=1", "/coast?place=port-201&for=commercial&species=dungeness", "/bloom", "/fisheries"]) {
      const r = await page.goto(`${OK}${link}`);
      expect(r?.status(), link).toBe(200);
      await expect(page.getByTestId("brand-home")).toBeVisible();
    }
    // a v2 port link opens that harbour; a v2 Research-mode link opens the Research workspace
    await page.goto(`${OK}/?port=550`);
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
    await page.goto(`${OK}/coast?place=port-550&mode=research`);
    await expect(page).toHaveURL(/\/research\?place=port-550/);
    await expect(page.getByTestId("research-title")).toBeVisible();
    await ctx.close();
  });

  test("the wordmark fits the masthead at phone widths", async ({ browser }) => {
    for (const w of [320, 360, 390]) {
      const ctx = await browser.newContext({ viewport: { width: w, height: 760 }, isMobile: true, hasTouch: true, storageState: dismissed });
      const page = await ctx.newPage();
      await page.goto(`${OK}/bloom`);
      const fit = await page.evaluate(() => {
        const home = document.querySelector('[data-testid="brand-home"]')!.getBoundingClientRect();
        const pill = document.querySelector('[data-testid="notices-button"]')!.getBoundingClientRect();
        return { overlap: home.right > pill.left, sideScroll: document.documentElement.scrollWidth > innerWidth };
      });
      expect(fit, `${w}px`).toEqual({ overlap: false, sideScroll: false });
      await ctx.close();
    }
  });
});
