import { mkdirSync } from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { OK } from "../ports";

/** Review stills of Research (docs/coastwatch/v12/implementation/research/). Runs only with NX_MEDIA=1. */
// review media go to the docs only on request (NX_MEDIA=1); an ordinary run keeps them with its own results
const OUT = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/research") : path.resolve(__dirname, "../../../test-results/journey-media/research");
test.skip(!process.env.NX_MEDIA, "media only on request (NX_MEDIA=1)");

const Q = "station=HABs-SantaCruzWharf&var=pDA&from=2023-10-13&to=2026-10-12";
const shots: Array<{ name: string; w: number; h: number; mode?: "dusk" | "night"; q?: string; open?: string; sheet?: boolean }> = [
  { name: "research-1440", w: 1440, h: 900 },
  { name: "research-1440-log", w: 1440, h: 900, open: "Samples" },
  { name: "research-1440-monterey-seriata", w: 1440, h: 900, q: "station=HABs-MontereyWharf&var=pn_seriata&from=2024-06-01&to=2025-09-30", open: "Checks" },
  { name: "research-1440-dusk", w: 1440, h: 900, mode: "dusk" },
  { name: "research-1440-night", w: 1440, h: 900, mode: "night" },
  { name: "research-1920", w: 1920, h: 1080 },
  { name: "research-390", w: 390, h: 844 },
  { name: "research-390-open", w: 390, h: 844, sheet: true },
  { name: "research-390-night", w: 390, h: 844, mode: "night", sheet: true },
  { name: "research-320", w: 320, h: 640, sheet: true },
  { name: "research-844x390", w: 844, h: 390 },
];

for (const s of shots) {
  test(`still: ${s.name}`, async ({ browser }) => {
    mkdirSync(OUT, { recursive: true });
    const ctx = await browser.newContext({ viewport: { width: s.w, height: s.h }, storageState: { cookies: [], origins: [] } });
    await ctx.addInitScript((m) => {
      localStorage.setItem("nx-opening-seen-v1", "1");
      if (m) localStorage.setItem("nx-chart-mode-v1", m);
    }, s.mode ?? "");
    const page = await ctx.newPage();
    await page.goto(`${OK}/research?${s.q ?? Q}`);
    await expect(page.getByTestId("research-strips")).toBeVisible();
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 }).catch(() => undefined);
    if (s.mode) await page.evaluate((m) => document.documentElement.setAttribute("data-chart-mode", m), s.mode);
    if (s.sheet) for (let i = 0; i < 2; i++) await page.getByRole("button", { name: "Show more of this page" }).click();
    if (s.open) await page.getByRole("button", { name: new RegExp(`^${s.open}`) }).click();
    await page.waitForTimeout(2200);
    await page.screenshot({ path: path.join(OUT, `${s.name}.png`) });
    await ctx.close();
  });
}
