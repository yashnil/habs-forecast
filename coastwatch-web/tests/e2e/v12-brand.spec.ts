import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The name on the v12 pages (ported from brand.spec, whose masthead, contour mark and "research preview" title
 * belonged to the v4–v11 layouts): no page uses the old product name except to credit NOAA CoastWatch; titles,
 * sharing metadata and the app manifest say Neritic; settings saved under the old keys break nothing.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const ROUTES = ["/", "/today?place=port-592", "/explore", "/coast?place=port-550", "/research", "/bloom"];
const ALLOWED = [/NOAA CoastWatch/g, /CoastWatch West Coast/g, /was called CoastWatch until October 2026/g];

test("no v12 page uses the old product name; the wordmark says Neritic", async ({ browser }) => {
  for (const [w, h] of [
    [1440, 900],
    [390, 844],
  ]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    for (const r of ROUTES) {
      await page.goto(`${OK}${r}`);
      await expect(page.getByRole("link", { name: "Neritic, home" }), `${w}px ${r}`).toBeVisible();
      const text = await page.evaluate(() => [document.body.innerText, ...[...document.querySelectorAll("[aria-label],[title],[alt]")].map((e) => `${e.getAttribute("aria-label") ?? ""} ${e.getAttribute("title") ?? ""} ${e.getAttribute("alt") ?? ""}`)].join("\n"));
      const rest = ALLOWED.reduce((t, re) => t.replace(re, ""), text);
      expect(rest.match(/coast\s?watch/gi) ?? [], `${w}px ${r}`).toEqual([]);
      expect(text, `${w}px ${r}`).not.toMatch(/Neritic Signal/i);
    }
    await ctx.close();
  }
});

test("titles, sharing metadata and the app manifest carry the name", async ({ page, request }) => {
  await page.goto(`${OK}/explore`);
  expect(await page.title()).toBe("Explore · Neritic");
  await expect(page.locator('meta[property="og:site_name"]')).toHaveAttribute("content", "Neritic");
  await expect(page.locator('meta[property="og:image"]').first()).toHaveAttribute("content", /opengraph-image/);
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute("content", "summary_large_image");
  await page.goto(`${OK}/coast`);
  expect(await page.title()).toBe("My Coast · Neritic");
  await page.goto(`${OK}/`);
  expect(await page.title()).toMatch(/^Neritic: season status, crab tests and the next date for your port/);
  const m = await (await request.get(`${OK}/manifest.webmanifest`)).json();
  expect(m.short_name).toBe("Neritic");
  expect(m.start_url).toBe("/");
  for (const p of ["/favicon.ico", "/apple-icon.png", "/opengraph-image", "/icons/icon-192.png", "/icons/icon-512.png", "/icons/maskable-512.png"]) {
    expect((await request.get(`${OK}${p}`)).status(), p).toBe(200);
  }
});

test("settings saved under the old keys break nothing, and old links still open a v12 page", async ({ browser }) => {
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "research" }, { name: "cw-home-port", value: "port-550" }] }] },
  });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const [link, id] of [
    ["/coast?place=port-550", "nx-coast"],
    ["/?port=550", "nx-explore"],
    ["/?region=monterey_bay&layer=forecast&var=pseudo_nitzschia&lead=1", "nx-explore"],
    ["/bloom", "nx-research"],
  ] as const) {
    const r = await page.goto(`${OK}${link}`);
    expect(r?.status(), link).toBe(200);
    await expect(page.getByTestId(id), link).toBeVisible();
  }
  expect(errors).toEqual([]);
  await ctx.close();
});
