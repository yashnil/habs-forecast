import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser } from "@playwright/test";

/**
 * Portfolio-preview build only (NEXT_PUBLIC_CW_DEMO=1). Run with CW_E2E_DEMO=1 against a demo
 * build; skipped otherwise, since every other spec covers the full product.
 */
test.skip(process.env.CW_E2E_DEMO !== "1", "demo build only");

const OK = "http://localhost:3200";
const FAILED = "http://localhost:3201";
const SEEN = () => {
  try {
    localStorage.setItem("cw-demo-intro-seen-v1", "1");
  } catch {}
};

async function phone(browser: Browser, seen = true) {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  if (seen) await ctx.addInitScript(SEEN);
  return { ctx, page: await ctx.newPage() };
}

test("first visit: the introduction explains the layers and names the Del Norte notices", async ({ page }) => {
  await page.goto(OK);
  const intro = page.getByRole("dialog", { name: /Harmful algal blooms/ });
  await expect(intro).toBeVisible();
  await expect(intro).toContainText("Agency forecast");
  await expect(intro).toContainText("Observation");
  await expect(intro).toContainText("Not verified");
  await expect(intro).toContainText("Chlorophyll shows algae, not toxin");
  // the fixture registry lists both Del Norte records, so the intro says they are listed, unverified
  await expect(page.getByTestId("demo-del-norte")).toContainText("Both are in this list, unverified.");
  await expect(intro.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
  await expect(intro.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx");
  const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
  expect(r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id)).toEqual([]);

  await page.getByTestId("demo-intro-start").click();
  await expect(intro).toBeHidden();
  await page.reload();
  await expect(page.getByTestId("map")).toBeVisible();
  await expect(page.getByTestId("demo-intro")).toHaveCount(0);
  await page.getByRole("banner").getByTestId("demo-about").click();
  await expect(page.getByTestId("demo-intro")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("demo-intro")).toHaveCount(0);
});

test("the introduction's call to action is visible without scrolling on a laptop and a phone", async ({ browser }) => {
  for (const vp of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
    const ctx = await browser.newContext({ viewport: vp });
    const page = await ctx.newPage();
    await page.goto(OK);
    await expect(page.getByTestId("demo-intro-start")).toBeInViewport({ ratio: 1 });
    await ctx.close();
  }
});

test("one disclosure strip names the Del Norte closure and links to CDFW and CDPH", async ({ page }) => {
  await page.addInitScript(SEEN);
  await page.goto(OK);
  const d = page.getByTestId("regulatory-gap");
  await expect(d).toHaveCount(1);
  await expect(page.getByTestId("registry-disclosure")).toHaveCount(0);
  await expect(d).toHaveAttribute("data-del-norte", "listed");
  await expect(d).toContainText("Notices here are not verified and may be incomplete.");
  await expect(d).toContainText("Del Norte County");
  await expect(d.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
  await expect(d.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-020.aspx");
  await d.getByRole("button", { name: "See the list" }).click();
  const drawer = page.getByTestId("official-drawer");
  await expect(drawer.getByTestId("regulatory-gap-card")).toContainText("has not been checked by a person");
  await expect(drawer).toContainText("Recreational razor clam fishery closed in Del Norte County");
  await expect(drawer).toContainText("Not verified");
});

test("hidden pages redirect to the map and currents are not offered", async ({ page }) => {
  await page.addInitScript(SEEN);
  for (const p of ["/bloom", "/fisheries"]) {
    await page.goto(`${OK}${p}`);
    await expect(page).toHaveURL(new RegExp(`^${OK}/(\\?.*)?$`));
  }
  await expect(page.getByTestId("layer-dock")).toBeVisible();
  await expect(page.getByTestId("group-currents")).toHaveCount(0);
  await expect(page.getByTestId("demo-badge")).toHaveText("Research preview");
});

test("an outage of a hidden layer is not announced; other outages still are", async ({ page }) => {
  await page.addInitScript(SEEN);
  await page.goto(FAILED);
  const b = page.getByTestId("source-failure-banner");
  await expect(b).toContainText("C-HARM");
  await expect(b).not.toContainText("HF radar");
});

test("phone: the map keeps most of the screen and controls stay reachable", async ({ browser }) => {
  const { ctx, page } = await phone(browser);
  await page.goto(OK);
  const map = page.getByTestId("map");
  await expect(page.getByTestId("layer-dock")).toBeVisible();
  await expect(page.getByTestId("mobile-official")).toHaveCount(0);
  // visible map between the place button and the collapsed dock (the fixture forecast is out of
  // date, so its warning adds a line that the live map does not have)
  const top = (await page.getByTestId("place-button").boundingBox())!;
  const dock = (await page.getByTestId("layer-dock").boundingBox())!;
  expect(dock.y - (top.y + top.height)).toBeGreaterThan(240);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(0);
  // variable and day stay one tap away in the collapsed dock
  await expect(page.getByTestId("variable-select")).toBeVisible();
  await page.getByTestId("lead-3").click();
  await expect(page).toHaveURL(/lead=3/);
  await expect(page.getByTestId("lead-3")).toContainText("day 3");
  // the disclosure strip still names the closure on a phone
  await expect(page.getByTestId("regulatory-gap")).toContainText("Del Norte razor clams closed Oct 9");
  await expect(map).toBeVisible();
  await ctx.close();
});
