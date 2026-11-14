import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { NODATA, OK } from "../ports";

/**
 * v15.1 (docs/coastwatch/v15.1/routes.md): every reader-facing route renders inside the shell on the design system's
 * informational layout (src/nx/pages), with no v4 frame left; nothing overflows at 1440, 390 or 320; axe finds no
 * serious or critical violation; and the disclosures the old frames carried are still on the page.
 */
const ROUTES = ["/about", "/sources", "/privacy", "/terms", "/sms-terms", "/fisheries", "/coast/kept", "/no-such-page"] as const;
/** Classes and ids only the v4 frames drew (AppShell, PilotShell, KeptView, the v4 TopBar and trust pill). */
const LEGACY = [".env-chart", ".env-instrument", ".as-ground", ".as-footer", ".today-wrap", ".pilot-prose", ".nx-trust", '[data-testid="pilot-hero"]', '[data-testid="skip-link"]'];
const WIDTHS: Array<[number, number]> = [
  [1440, 900],
  [390, 844],
  [320, 640],
];

async function settle(page: Page, path: string) {
  const r = await page.goto(`${OK}${path}`);
  await expect(page.getByTestId("nx-info-page")).toBeVisible();
  return r;
}

test.describe("every legacy route is a page of the shell", () => {
  for (const path of ROUTES) {
    test(`${path}: the shell's bar and the ds page layout, no v4 frame`, async ({ page }) => {
      const r = await settle(page, path);
      expect(r!.status()).toBe(path === "/no-such-page" ? 404 : 200);
      await expect(page.getByTestId("nx-app")).toHaveCount(1);
      await expect(page.getByTestId("nx-bar")).toBeVisible();
      await expect(page.getByTestId("nx-menu-button")).toBeVisible();
      await expect(page.locator("[data-nx-info]")).toHaveCount(1);
      await expect(page.locator("h1")).toHaveCount(1);
      for (const sel of LEGACY) await expect(page.locator(sel), sel).toHaveCount(0);
    });
  }

  test("moving between them keeps the one chart (the group is not remounted)", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-592`);
    await expect(page.getByTestId("nx-app")).toBeVisible();
    await page.evaluate(() => ((window as unknown as { __nxMarker: unknown }).__nxMarker = document.querySelector('[data-testid="nx-app"]')));
    await page.getByTestId("nx-menu-button").click();
    await page.getByRole("link", { name: "About", exact: true }).click();
    await expect(page).toHaveURL(/\/about$/);
    await expect(page.getByTestId("nx-info-page")).toBeVisible();
    expect(await page.evaluate(() => (window as unknown as { __nxMarker: unknown }).__nxMarker === document.querySelector('[data-testid="nx-app"]'))).toBe(true);
    await page.getByTestId("footer-sources").click();
    await expect(page).toHaveURL(/\/sources$/);
    expect(await page.evaluate(() => (window as unknown as { __nxMarker: unknown }).__nxMarker === document.querySelector('[data-testid="nx-app"]'))).toBe(true);
  });
});

test.describe("no horizontal overflow", () => {
  for (const [width, height] of WIDTHS) {
    test(`at ${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      for (const path of ROUTES) {
        await settle(page, path);
        const over = await page.evaluate(() => {
          const scrollers = [document.documentElement, ...document.querySelectorAll<HTMLElement>(".n12-work, .n12-margin-scroll")];
          return scrollers.map((el) => el.scrollWidth - el.clientWidth).filter((d) => d > 1);
        });
        expect(over, `${path} @ ${width}`).toEqual([]);
      }
    });
  }
});

test.describe("axe", () => {
  for (const [width, height] of [WIDTHS[0], WIDTHS[1]]) {
    for (const path of ROUTES) {
      test(`${path} at ${width}: no serious or critical violations`, async ({ page }) => {
        await page.setViewportSize({ width, height });
        await settle(page, path);
        await page.waitForLoadState("networkidle");
        const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
        const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
      });
    }
  }
});

test.describe("the disclosures are still there", () => {
  test("every informational page: research preview, notices not verified, the agencies, and the independence statement", async ({ page }) => {
    for (const path of ["/about", "/sources", "/privacy", "/terms", "/sms-terms", "/fisheries"]) {
      await settle(page, path);
      const line = page.getByTestId("registry-disclosure");
      await expect(line, path).toContainText("Research preview", { useInnerText: true });
      await expect(line, path).toContainText("Not verified", { useInnerText: true });
      await expect(page.getByTestId("footer-independence"), path).toContainText("not affiliated with or endorsed by NOAA, CDFW, or CDPH");
    }
    await settle(page, "/sources");
    await page.getByTestId("notices-button").click();
    const panel = page.getByTestId("notices-panel");
    await expect(panel).toContainText("has not been checked by a person and may be incomplete");
    await expect(panel).toContainText("does not mean an area is open or that seafood is safe");
    await expect(panel.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
    await expect(panel.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx");
  });

  test("Privacy keeps its statements; Terms its clauses; SMS terms its consent and no-sharing promise", async ({ page }) => {
    await settle(page, "/privacy");
    const privacy = page.getByTestId("privacy-page");
    for (const t of ["you can use Neritic without an account", "We do not sell personal information", "never shared with or sold to third parties or affiliates for marketing", "Global Privacy Control", "not directed to children under 13", "Anthropic"])
      await expect(privacy).toContainText(t);
    await expect(page.getByTestId("legal-draft-banner")).toContainText("Draft — pending owner/legal review");
    await expect(page.getByTestId("legal-attribution")).toContainText("CC BY 4.0");
    await expect(privacy).toContainText("where you placed, sized or collapsed the panels");
    await expect(privacy).toContainText("Trips you save on My Coast");
    await expect(privacy).toContainText("your saved ports, trips and alerts are kept on your device only");
    await expect(privacy).toContainText("are not offered on the public site");
    await expect(page.getByTestId("legal-effective")).toContainText("Effective November 14, 2026 · version 2026-11-14");
    await expect(privacy).toContainText("they carry nothing about you, your ports or your trips");
    await expect(page.getByTestId("nx-info-page")).toContainText("Neritic research preview · legal");
    await settle(page, "/terms");
    const terms = page.getByTestId("terms-page");
    await expect(terms).not.toContainText("chlorophyll outlook");
    for (const t of ["gap-filled chlorophyll estimate (DINEOF), which is labelled as an estimate", "NOAA WCOFS ocean model guidance, which NOAA labels developmental", "None of it is a safety, legal or navigation", "AS IS", "Not an official source", "Not for navigation", "Satellite chlorophyll is not toxin", "AI answers may be wrong", "marked “Not verified” until a person has checked them", "VHF channel 16"]) await expect(terms).toContainText(t);
    await settle(page, "/sms-terms");
    await expect(page.getByTestId("sms-status")).toContainText("not being sent yet");
    await expect(page.getByTestId("sms-terms-page")).toContainText("Consent is not a condition");
  });

  test("Sources keeps every dataset row, its attributions and links", async ({ page }) => {
    await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
    await settle(page, "/sources");
    const manifest = await (await page.request.get(`${OK}/data/fixture/v1/manifest.json`)).json();
    const ids = (manifest.sources as Array<{ source_id: string; title: string }>).map((s) => s.source_id);
    expect(ids.length).toBeGreaterThan(3);
    await expect(page.getByTestId("source-table").locator("> li")).toHaveCount(ids.length);
    for (const s of manifest.sources as Array<{ source_id: string; title: string }>) await expect(page.getByTestId(`source-${s.source_id}`)).toContainText(s.title);
    await expect(page.getByTestId("source-official").getByTestId("official-verification")).toHaveText(/Not verified/);
    const official = page.getByTestId("official-status");
    await expect(official.locator('a[target="_blank"]')).toHaveCount(4);
    await expect(official).toContainText("(800) 553-4133");
    await expect(page.getByTestId("research")).toContainText("Neritic does not run those research models");
    // the live sources, each with its attribution and links
    for (const [id, text, host] of [
      ["wcofs", "WCOFS", "opendap.co-ops.nos.noaa.gov"],
      ["dineof", "Liu, X. & Wang, M. (2023)", "coastwatch.noaa.gov"],
      ["ndbc", "National Data Buoy Center", "www.ndbc.noaa.gov"],
      ["nws", "National Weather Service", "tgftp.nws.noaa.gov"],
      ["coops", "Tides and Currents", "tidesandcurrents.noaa.gov"],
    ] as const) {
      const row = page.getByTestId(`live-source-${id}`);
      await expect(row).toContainText(text);
      await expect(row.locator(`a[href*="${host}"]`).first()).toBeVisible();
    }
    await expect(page.getByTestId("live-source-wcofs")).toContainText("Developmental");
    await expect(page.getByTestId("live-source-dineof")).toContainText("CC0");
  });

  test("About keeps the four kinds, what it cannot tell you, the research, the providers and the review status", async ({ page }) => {
    await settle(page, "/about");
    await expect(page.getByTestId("about-kinds").locator("dt")).toHaveCount(4);
    await expect(page.getByTestId("about-cannot")).toContainText("satellites measure chlorophyll, which is algae, not toxin");
    await expect(page.getByTestId("research-relation")).toContainText("Neritic does not run those research models");
    await expect(page.getByTestId("research")).toContainText("matches, but does not beat, a standard ConvLSTM baseline");
    await expect(page.getByTestId("about-providers").locator("li")).toHaveCount(10);
    // the kinds name today's destinations, never the v4 names
    const kinds = page.getByTestId("about-kinds");
    for (const [name, href] of [["Official notices", "/notices"], ["Explore California Waters", "/explore"], ["Research", "/research"], ["Fisheries", "/fisheries"]] as const)
      await expect(kinds.getByRole("link", { name: new RegExp(`^${name}`) })).toHaveAttribute("href", href);
    await expect(page.getByTestId("about-page")).not.toContainText(/Ocean Map|Bloom Intelligence|four colours/);
    await expect(page.getByTestId("about-page")).toContainText("a ring is an official record not yet checked by a person");
    await expect(page.getByTestId("about-review")).toContainText("have not been checked by a person");
    await expect(page.getByTestId("about-independence")).toContainText("was called CoastWatch until October 2026");
  });

  test("a page without the dataset says so and still points to the agencies", async ({ page }) => {
    await page.goto(`${NODATA}/about`);
    await expect(page.getByTestId("data-unavailable")).toContainText("Official closures and advisories are always on the CDFW and CDPH sites");
    await expect(page.getByTestId("registry-disclosure")).toContainText("Unavailable", { useInnerText: true });
  });

  test("the registry line and the Notices page count the same notices", async ({ page }) => {
    await settle(page, "/sources");
    const line = (await page.getByTestId("registry-disclosure").innerText()).match(/(\d+) official notices? in CA/);
    expect(line).not.toBeNull();
    await page.goto(`${OK}/notices`);
    await expect(page.locator("#nx-notices-title")).toContainText(`${line![1]} current notices on the coast`);
  });

  test("the kept briefing page says what it is when nothing is kept", async ({ page }) => {
    await settle(page, "/coast/kept");
    await expect(page.getByTestId("kept-empty")).toContainText("No briefing is kept on this device");
  });
});

test.describe("modes and motion", () => {
  test("dusk and night: the page takes the mode's paper and ink", async ({ page }) => {
    for (const mode of ["dusk", "night"]) {
      await page.addInitScript((m) => localStorage.setItem("nx-chart-mode-v1", m), mode);
      await settle(page, "/privacy");
      const [bg, fg] = await page.evaluate(() => {
        const el = document.querySelector<HTMLElement>(".n12-margin")!;
        return [getComputedStyle(el).backgroundColor, getComputedStyle(document.querySelector("h1")!).color];
      });
      expect(bg, mode).not.toBe("rgb(255, 255, 255)");
      expect(fg, mode).not.toBe(bg);
    }
  });

  test("first visit keeps the opening with Skip; reduced motion skips it", async ({ page, browser }) => {
    // the default storage dismisses the old intro only: the opening is unseen (as tests/e2e/nx-shell.spec.ts)
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("nx-skip-opening")).toBeVisible();
    await page.getByTestId("nx-skip-opening").dispatchEvent("click");
    await expect(page.getByTestId("nx-still")).toBeHidden({ timeout: 1_500 });
    await expect(page.getByTestId("nx-find-port")).toBeVisible();
    const reduced = await browser.newContext({ storageState: { cookies: [], origins: [] }, reducedMotion: "reduce" });
    const p2 = await reduced.newPage();
    await p2.goto(`${OK}/`);
    expect(await p2.evaluate(() => document.documentElement.getAttribute("data-nx-opening"))).toBeNull();
    await expect(p2.getByTestId("nx-still")).toBeHidden();
    await reduced.close();
  });
});
