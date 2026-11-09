import { expect, test } from "@playwright/test";
import { OK, PILOT } from "./ports";

/**
 * The pilot's front door. With NERITIC_PILOT=1 a bare "/" opens Today at your port, every deep link
 * still opens Explore, and Explore has its own address (/?view=map) that the nav uses. Without the
 * flag "/" is Explore, exactly as on the public release. Both servers read the same fixture and build.
 */
test.describe("NERITIC_PILOT=1", () => {
  test("a bare / opens Today: the picker first, then the reader's own port", async ({ page }) => {
    await page.goto(`${PILOT}/`);
    await expect(page).toHaveURL(`${PILOT}/today`);
    await expect(page.getByTestId("today-picker")).toBeVisible();
    await page.getByTestId("pick-port-port-592").click();
    await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
    // the brand is the front door
    await page.getByTestId("brand-home").click();
    await expect(page).toHaveURL(`${PILOT}/today?place=port-592`);
    await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
  });

  test("the redirect is a real one (no script needed), and a campaign tag does not stop it", async ({ request }) => {
    const bare = await request.get(`${PILOT}/`, { maxRedirects: 0 });
    expect(bare.status()).toBe(307);
    expect(bare.headers().location).toBe("/today");
    expect((await request.get(`${PILOT}/?utm_source=newsletter`, { maxRedirects: 0 })).headers().location).toBe("/today");
  });

  test("existing deep links still open Explore", async ({ page, request }) => {
    for (const q of ["region=monterey_bay", "port=593", "region=monterey_bay&port=593", "var=particulate_domoic&lead=2", "layer=olci300", "inspect=36.80,-121.90", "sel=harbor:592", "at=36.8000,-121.9000,9.50"]) {
      const r = await request.get(`${PILOT}/?${q}`, { maxRedirects: 0 });
      expect(r.status(), q).toBe(200);
    }
    await page.goto(`${PILOT}/?region=monterey_bay&port=593`);
    await expect(page.getByTestId("explore")).toBeVisible();
    await expect(page.getByTestId("nav-bar").getByTestId("nav-explore")).toHaveAttribute("aria-current", "page");
  });

  test("Explore has its own address, the nav uses it, and a reload stays on the map", async ({ page }) => {
    await page.goto(`${PILOT}/today?place=port-592`);
    const explore = page.getByTestId("nav-bar").getByTestId("nav-explore");
    await expect(explore).toHaveAttribute("href", "/?view=map");
    await explore.click();
    await expect(page.getByTestId("explore")).toBeVisible();
    expect(new URL(page.url()).pathname).toBe("/");
    expect(new URL(page.url()).search).not.toBe("");
    await page.reload();
    await expect(page.getByTestId("explore")).toBeVisible();
    // Today's own link to the map names the harbour
    await page.goto(`${PILOT}/today?place=port-592`);
    await expect(page.getByTestId("link-explore")).toHaveAttribute("href", "/?view=map&sel=harbor:592");
    // and the About page's "open the map" goes to the map, not round to Today
    await page.goto(`${PILOT}/about`);
    await expect(page.getByTestId("about-tour")).toHaveAttribute("href", "/?view=map");
  });

  test("on a phone the dock's Explore tab opens the map", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(`${PILOT}/today?place=port-592`);
    await page.getByTestId("nav-dock").getByTestId("nav-explore").click();
    await expect(page.getByTestId("explore")).toBeVisible();
    await ctx.close();
  });
});

test.describe("without the flag (the public release)", () => {
  test("/ is Explore and the nav's Explore link is /", async ({ page, request }) => {
    const r = await request.get(`${OK}/`, { maxRedirects: 0 });
    expect(r.status()).toBe(200);
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("explore")).toBeVisible();
    await expect(page.getByTestId("nav-bar").getByTestId("nav-explore")).toHaveAttribute("href", "/");
    await page.goto(`${OK}/about`);
    await expect(page.getByTestId("about-tour")).toHaveAttribute("href", "/");
    // Today is still there, one tab away
    await expect(page.getByTestId("nav-bar").getByTestId("nav-today")).toHaveAttribute("href", "/today");
  });
});
