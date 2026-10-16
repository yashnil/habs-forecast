import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Research (v4): a station's record on fixed axes over a range the station actually covers, the
 * C-HARM history beside it (never compared numerically), provenance and QC, an export that carries
 * its provenance, and the native C-HARM grid with a transect that keeps gaps as gaps.
 */
const NOW = "2026-10-08T20:00:00Z";

test("a station's record: ranges, the model beside it, provenance, QC", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await expect(page.getByTestId("research-title")).toHaveText("Santa Cruz Wharf");
  await expect(page.getByTestId("rvar-pDA")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("series-ref")).toContainText("C-HARM threshold, 500 ng/L");
  await expect(page.getByTestId("model-panel")).toContainText("never compared with them as numbers");
  await page.getByTestId("range-all").click();
  await expect(page.getByTestId("research-stats")).toContainText("never a zero");
  await expect(page.getByTestId("research-provenance")).toContainText("Method");
  await expect(page.getByTestId("research-qc").locator("li").first()).toBeVisible();
});

test("the export is the window's samples with a provenance header", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("research-export").click()]);
  const text = readFileSync((await dl.path())!, "utf8");
  expect(text).toContain("# Neritic research export: Santa Cruz Wharf");
  expect(text).toContain("empty value = not measured (never zero)");
  expect(text).toContain("Not a seafood test");
  expect(text.split("\n").find((l) => !l.startsWith("#"))).toBe("sample_time_utc,value,qualifier");
});

test("a My Coast place opens on its nearest station; a transect lists cells and keeps gaps", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?place=port-550`);
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await page.getByTestId("tool-transect").click();
  const map = page.getByTestId("research-grid").locator(".maplibregl-canvas");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  const b = (await map.boundingBox())!;
  await page.mouse.click(b.x + b.width * 0.15, b.y + b.height * 0.4);
  await page.mouse.click(b.x + b.width * 0.85, b.y + b.height * 0.6);
  await expect(page.getByTestId("transect")).toContainText("cells crossed");
  await expect(page.getByTestId("transect")).toContainText("no interpolation");
});
