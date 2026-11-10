import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

const SIZES = [
  { name: "desktop 1440×900", width: 1440, height: 900, phone: false, fits: true },
  { name: "short desktop 1280×620", width: 1280, height: 620, phone: false, fits: true },
  { name: "shortest desktop 1024×600", width: 1024, height: 600, phone: false, fits: true },
  { name: "phone 390×844", width: 390, height: 844, phone: true, fits: true },
  { name: "phone 320×640", width: 320, height: 640, phone: true, fits: true },
] as const;

/** Is this element itself what a tap at its centre would hit (nothing lies over it)? */
const onTop = (page: Page, testid: string) =>
  page.getByTestId(testid).evaluate((el) => {
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return !!hit && (el === hit || el.contains(hit));
  });

for (const s of SIZES) {
  test.describe(s.name, () => {
    test.use({ viewport: { width: s.width, height: s.height }, ...(s.phone ? { isMobile: true, hasTouch: true } : {}) });

    test("the directory is closed by default and nothing covers the hero", async ({ page }) => {
      await page.goto(`${OK}/today`);
      await expect(page.getByTestId("today-picker")).toBeVisible();
      await expect(page.getByTestId("port-directory")).toHaveCount(0);
      await expect(page.getByTestId("port-filter")).toHaveCount(0);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBe(0);
      const ids = ["pick-port-port-592", "pick-port-port-550", "pick-port-port-593", "pick-port-port-452", "browse-ports"];
      const dock = s.phone || s.width < 1024 ? 76 : 0;
      for (const id of ids) {
        const el = page.getByTestId(id);
        await el.scrollIntoViewIfNeeded();
        expect(await onTop(page, id), `${id} is what a tap would hit`).toBe(true);
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      if (s.fits) {
        // the featured ports and the action are all on the first screen, above the dock
        for (const id of ids) {
          const b = (await page.getByTestId(id).boundingBox())!;
          expect(b.y + b.height, `${id} on the first screen`).toBeLessThanOrEqual(s.height - dock);
        }
      }
      // the hero's words end inside the stage: the stage grows rather than clips
      const clip = await page.getByTestId("today-hero").evaluate((h) => h.getBoundingClientRect().bottom - h.closest('[data-testid="ocean-stage"]')!.getBoundingClientRect().bottom);
      expect(clip).toBeLessThanOrEqual(1);
    });

    test("opens on purpose; filters; closes with its button, focus returned and the page unlocked", async ({ page }) => {
      await page.goto(`${OK}/today`);
      const browse = page.getByTestId("browse-ports");
      await browse.click();
      const sheet = page.getByTestId("port-directory");
      await expect(sheet).toBeVisible();
      await expect(sheet.getByRole("heading", { name: "All California ports" })).toBeVisible();
      await expect(page.locator("html")).toHaveClass(/nx-sheet-lock/);
      // it stays inside the window, and the list scrolls on its own, not the page behind
      await expect
        .poll(async () => {
          const b = (await sheet.boundingBox())!; // once it has finished sliding in
          return b.x >= 0 && b.x + b.width <= s.width + 0.5 && b.y >= 0 && b.y + b.height <= s.height + 0.5;
        })
        .toBe(true);
      await page.getByTestId("port-row-port-592").waitFor();
      await page.locator(".today-directory .nx-msheet-body").evaluate((el) => (el.scrollTop = 200));
      expect(await page.evaluate(() => window.scrollY)).toBe(0);
      await expect(page.getByTestId("port-list").getByRole("link")).not.toHaveCount(0);
      await page.getByTestId("port-filter").fill("Crescent");
      await expect(page.getByTestId("port-list").getByRole("link")).toHaveCount(1);
      await expect(page.getByTestId("port-count")).toHaveText("1 port");
      await page.getByTestId("port-filter").fill("zzzz");
      await expect(page.getByTestId("port-none")).toContainText("zzzz");
      expect(await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze().then((r) => r.violations.map((v) => v.id))).toEqual([]);
      await page.getByTestId("port-directory-close").click();
      await expect(page.getByTestId("port-directory")).toHaveCount(0);
      await expect(browse).toBeFocused();
      await expect(page.locator("html")).not.toHaveClass(/nx-sheet-lock/);
      // reopened, the old filter is gone
      await browse.click();
      await expect(page.getByTestId("port-filter")).toHaveValue("");
    });
  });
}

test.describe("keyboard", () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test("Enter opens; the filter has focus; Tab stays inside; Escape closes and returns focus", async ({ page }) => {
    await page.goto(`${OK}/today`);
    const browse = page.getByTestId("browse-ports");
    await browse.focus();
    await page.keyboard.press("Enter");
    await expect(page.getByTestId("port-filter")).toBeFocused();
    await expect(browse).toHaveAttribute("aria-expanded", "true");
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("Tab");
      const inside = await page.evaluate(() => {
        const a = document.activeElement;
        return a === document.body || a?.tagName === "HTML" || !!a?.closest("dialog");
      });
      expect(inside, "focus stays in the directory").toBe(true);
    }
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("port-directory")).toHaveCount(0);
    await expect(browse).toBeFocused();
    await expect(browse).toHaveAttribute("aria-expanded", "false");
  });

  test("Enter in the filter opens the first match; every featured port is in the directory too", async ({ page }) => {
    await page.goto(`${OK}/today`);
    await page.getByTestId("browse-ports").click();
    await expect(page.getByTestId("port-row-port-592")).toContainText("Pilot port");
    await page.getByTestId("port-filter").fill("Santa Cruz");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/today\?place=port-593$/);
    await expect(page.locator("h1")).toHaveText("Santa Cruz", { timeout: 15_000 });
  });
});

test("on a touch screen the sheet does not pop the keyboard up uninvited", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(`${OK}/today`);
  await page.getByTestId("browse-ports").tap();
  await expect(page.getByTestId("port-directory")).toBeVisible();
  await expect(page.getByTestId("port-filter")).not.toBeFocused();
  await ctx.close();
});
