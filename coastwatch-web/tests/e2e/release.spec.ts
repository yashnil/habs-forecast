import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** Release readiness: the masthead at every width down to 320 px (WCAG 1.4.10), and the source CSV
 *  link on Bloom Intelligence. */
const NOW = "2026-10-08T20:00:00Z";
const PAGES = ["/", "/bloom", "/fisheries", "/sources", "/about"];
const PHONES = [320, 360, 375, 390];
const DESKTOPS = [1024, 1440];

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).toBeVisible();
  await expect(page.getByTestId("status-line")).toBeVisible();
}

/** The masthead, the status line and the tab bar sit inside the viewport and the page never scrolls sideways. */
async function measure(page: Page) {
  return page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    const inside = (el: Element | null) => {
      if (!el) return null;
      const b = el.getBoundingClientRect();
      return b.width > 0 && b.left >= -0.5 && b.right <= vw + 0.5;
    };
    const pill = document.querySelector('[data-testid="official-pill"]');
    const word = pill?.lastElementChild as HTMLElement | null;
    const header = document.querySelector("header");
    const offscreen = [...(header?.querySelectorAll("a,button") ?? [])]
      .filter((a) => a.getBoundingClientRect().width > 0 && !inside(a))
      .map((a) => a.textContent?.trim());
    const tabs = [...document.querySelectorAll('[data-testid^="tab-"]')].filter((t) => t.getBoundingClientRect().width > 0);
    return {
      vw,
      scrollW: document.documentElement.scrollWidth,
      pillInside: inside(pill),
      word: word?.innerText.trim() ?? "",
      wordVisible: !!word && word.getBoundingClientRect().width > 0,
      statusInside: inside(document.querySelector('[data-testid="status-line"]')),
      statusLinks: [...document.querySelectorAll('[data-testid="status-line"] a')].filter((a) => a.getBoundingClientRect().width > 0).map((a) => a.textContent?.trim()),
      offscreen,
      tabs: tabs.length,
      tabsInside: tabs.every((t) => inside(t)),
    };
  });
}

test.describe("masthead at narrow widths", () => {
  for (const width of [...PHONES, ...DESKTOPS]) {
    test(`${width} px: no sideways scroll; the notices pill keeps its count and verification word; nothing clipped`, async ({ browser }) => {
      const phone = width < 768;
      const ctx = await browser.newContext({ viewport: { width, height: phone ? 740 : 800 }, isMobile: phone, hasTouch: phone });
      const page = await ctx.newPage();
      for (const p of PAGES) {
        await open(page, `${OK}${p}`);
        const m = await measure(page);
        expect(m.scrollW, `${p} scrolls sideways`).toBeLessThanOrEqual(m.vw);
        expect(m.pillInside, `${p} pill clipped`).toBe(true);
        expect(m.wordVisible && m.word.length > 0, `${p} verification word`).toBe(true);
        await expect(page.getByTestId("official-pill")).toContainText(/\d/);
        expect(m.statusInside, `${p} status line`).toBe(true);
        expect(m.statusLinks, `${p} agency links`).toEqual(expect.arrayContaining(["CDFW", "CDPH"]));
        expect(m.offscreen, `${p} header controls off screen`).toEqual([]);
        if (phone) {
          expect(m.tabs, `${p} tabs`).toBe(5);
          expect(m.tabsInside, `${p} tabs inside`).toBe(true);
        } else {
          await expect(page.getByTestId("nav-about")).toBeVisible();
        }
      }
      await ctx.close();
    });
  }

  /** Every masthead and status-line control is a 44 × 44 px target on phones (WCAG 2.5.5), and no two
   *  hit boxes overlap, so a thumb lands on one control only. The drawn pill and menu keep their size:
   *  the box is the control, the drawing is inside it. */
  for (const width of [320, 390]) {
    test(`${width} px: every header control is at least 44 × 44 and no hit boxes overlap`, async ({ browser }) => {
      const ctx = await browser.newContext({ viewport: { width, height: 740 }, isMobile: true, hasTouch: true });
      const page = await ctx.newPage();
      for (const p of ["/", "/bloom", "/coast?place=port-550"]) {
        await open(page, `${OK}${p}`);
        const r = await page.evaluate(() => {
          // the site masthead is the first header; Bloom's station header is the page's own
          const boxes = [...document.querySelector("header")!.querySelectorAll<HTMLElement>("a, button")]
            .filter((e) => getComputedStyle(e).position !== "absolute")
            .map((e) => ({ t: e.dataset.testid || e.textContent?.trim().slice(0, 16) || "", r: e.getBoundingClientRect() }))
            .filter((x) => x.r.width > 1 && x.r.height > 1);
          const small = boxes.filter((x) => x.r.width < 44 || x.r.height < 44).map((x) => `${x.t} ${Math.round(x.r.width)}×${Math.round(x.r.height)}`);
          const overlaps: string[] = [];
          for (let i = 0; i < boxes.length; i++)
            for (let j = i + 1; j < boxes.length; j++) {
              const a = boxes[i].r, b = boxes[j].r;
              if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 0 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 0) overlaps.push(`${boxes[i].t}×${boxes[j].t}`);
            }
          const pill = document.querySelector('[data-testid="official-pill"]')!;
          return {
            n: boxes.length,
            small,
            overlaps,
            masthead: document.querySelector("header")!.firstElementChild!.getBoundingClientRect().height,
            drawnPill: parseFloat(getComputedStyle(pill, "::before").height),
          };
        });
        expect(r.n, `${p} controls`).toBeGreaterThanOrEqual(5);
        expect(r.small, `${p} small controls`).toEqual([]);
        expect(r.overlaps, `${p} overlapping hit boxes`).toEqual([]);
        // nothing was enlarged on screen: the masthead row is still 52 px and the drawn pill 32 px
        expect(r.masthead, `${p} masthead row`).toBeLessThanOrEqual(52);
        expect(r.drawnPill, `${p} drawn pill`).toBe(32);
      }
      await ctx.close();
    });
  }

  test("320 px: the masthead passes axe on the map and Bloom", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    for (const p of ["/", "/bloom"]) {
      await open(page, `${OK}${p}`);
      // contrast is judged on what a reader sees once the entrance fade has finished: measured
      // mid-fade (opacity 0.8) the same colours read 4.1:1 instead of their settled 4.5:1 or better
      await page.waitForFunction(
        () =>
          [...document.querySelectorAll("header *")].every((el) => {
            let o = 1;
            for (let e: Element | null = el; e; e = e.parentElement) o *= Number(getComputedStyle(e).opacity);
            return o === 0 || o > 0.99;
          }),
        null,
        { timeout: 10_000 },
      );
      const r = await new AxeBuilder({ page }).include("header").analyze();
      expect(r.violations.map((v) => v.id), p).toEqual([]);
    }
    await ctx.close();
  });
});

test.describe("source CSV download", () => {
  test("Bloom links to the station's original data on SCCOOS ERDDAP, encoded, and says it is not the chart", async ({ page }) => {
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/bloom`);
    const block = page.getByTestId("source-csv");
    const link = page.getByTestId("source-csv-link");
    await expect(link).toContainText("Download source data (CSV)");
    const href = (await link.getAttribute("href"))!;
    expect(href).toMatch(/^https:\/\/erddap\.sccoos\.org\/erddap\/tabledap\/HABs-SantaCruzWharf\.csv\?time,latitude,longitude,/);
    expect(href).toContain("time%3E%3D2014-01-01T00%3A00%3A00Z");
    expect(href).toContain("orderBy(%22time%22)");
    expect(href).not.toMatch(/["<>\s]|%25/);
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute("rel", /noopener/);
    await expect(block).toContainText("exactly as SCCOOS ERDDAP serves them: not a copy of the chart");
    await expect(block).toContainText("If SCCOOS is unavailable the file will not load");
    await expect(block.getByRole("link", { name: /Dataset page/ })).toHaveAttribute("href", "https://erddap.sccoos.org/erddap/tabledap/HABs-SantaCruzWharf.html");
  });

  test("the link follows the chosen station", async ({ page }) => {
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/bloom?station=HABs-MorroBayFrontBay`);
    await expect(page.getByTestId("source-csv-link")).toHaveAttribute("href", /\/tabledap\/HABs-MorroBayFrontBay\.csv\?/);
  });

  test("a click opens the source in a new tab and leaves Bloom as it was, even when SCCOOS fails", async ({ page, context }) => {
    await context.route("https://erddap.sccoos.org/**", (r) => r.fulfill({ status: 503, body: "Service Unavailable" }));
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/bloom`);
    const [tab] = await Promise.all([context.waitForEvent("page"), page.getByTestId("source-csv-link").click()]);
    await tab.waitForLoadState().catch(() => {});
    expect(tab.url()).toMatch(/erddap\.sccoos\.org/);
    await expect(page.getByTestId("station-detail")).toBeVisible();
    await expect(page).toHaveURL(/\/bloom/);
  });
});

test.describe("research preview", () => {
  test("every page says it is a research preview that does not replace CDFW or CDPH", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }] } });
    const page = await ctx.newPage();
    for (const p of ["/", "/coast?place=port-550", "/bloom", "/fisheries", "/about"]) {
      await page.goto(`${OK}${p}`);
      await expect(page.getByTestId("research-preview-badge"), p).toHaveText("Research preview");
    }
    await page.goto(`${OK}/about`);
    await expect(page.getByTestId("footer-disclaimer")).toContainText("does not replace CDFW or CDPH");
    await expect(page.locator("main")).toContainText("published as a research preview");
    // the site's own title (pages set their own, e.g. "About — CoastWatch")
    await page.goto(`${OK}/`);
    expect(await page.title()).toContain("research preview");
    await ctx.close();
  });
});
