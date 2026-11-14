import { expect, type BrowserContext, type Locator, type Page } from "@playwright/test";
import { OK } from "../ports";

/** Shared steps for the My Coast trip workspace specs (journeys/coast.spec.ts, v13/trip.spec.ts). */

export const MOSS = "port-592";

export async function seed(ctx: BrowserContext, o: { mode?: string; fishery?: string } = {}) {
  await ctx.addInitScript(
    ({ mode, fishery }) => {
      if (sessionStorage.getItem("seeded")) return;
      sessionStorage.setItem("seeded", "1");
      localStorage.setItem("nx-opening-seen-v1", "1");
      localStorage.setItem("nx-chart-mode-v1", mode);
      localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery }));
    },
    { mode: o.mode ?? "day", fishery: o.fishery ?? "dungeness_commercial" },
  );
}

/** A typed field by its visible label. */
export const field = (p: Page, label: string): Locator => p.getByRole("textbox", { name: label, exact: true });

export async function type(p: Page, label: string, v: string) {
  const f = field(p, label);
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
}

/** The exact value a ds Figure holds (its accessible text; the visible digits may still be moving). */
export const exact = (p: Page, id: string): Locator => p.getByTestId(id).locator("[data-figure-exact]");

export async function openTrip(p: Page, query = "") {
  await p.goto(`${OK}/coast?place=${MOSS}&view=trip${query}`);
  await expect(p.getByTestId("trip-inputs")).toBeVisible();
  await expect(p.getByTestId("coast-status")).toContainText(/delayed to at least Jan 1, 2027/, { timeout: 15_000 });
}

/** The owner's hands-on test (v13 brief): 400 lb at $9, 30 gal at $8.24, bait and ice $100, fees 0, 1 crew on 20 % after costs. */
export async function fillOwner(p: Page, o: { payers?: boolean } = {}) {
  await type(p, "Catch", "400");
  await type(p, "Price", "9");
  await type(p, "Fuel", "30");
  await type(p, "Fuel price", "8.24");
  await type(p, "Bait and ice", "100");
  await type(p, "Fees", "0");
  await type(p, "Crew", "1");
  await p.getByRole("radio", { name: "After costs" }).click();
  await type(p, "Crew share", "20");
  if (o.payers !== false) await choosePayers(p);
}

export async function choosePayers(p: Page) {
  await p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await p.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
}

/** Wholly inside the viewport, below the app bar and (on a phone) above the destinations bar. */
export async function inView(p: Page, l: Locator): Promise<boolean> {
  const b = await l.boundingBox();
  const v = p.viewportSize()!;
  if (!b) return false;
  const bottom = v.width <= 720 ? v.height - 56 : v.height;
  return b.y >= 0 && b.y + b.height <= bottom + 0.5 && b.x >= -0.5 && b.x + b.width <= v.width + 0.5;
}

export const savedCount = (p: Page) => p.evaluate(() => (JSON.parse(localStorage.getItem("nx-trip-v1:scenarios") ?? "[]") as unknown[]).length);
