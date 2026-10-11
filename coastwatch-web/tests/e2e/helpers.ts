import { expect, type Locator, type Page } from "@playwright/test";

/** Desktop place list (M5): it opens from the place chip at the top of the map. */
export async function openPlaces(page: Page) {
  const card = page.getByTestId("nav-card");
  if (!(await card.isVisible())) await page.getByTestId("place-chip-toggle").click();
  await expect(card).toBeVisible();
  return card;
}

/** Open the port inspector's collapsed sections (30-day history, chlorophyll history). */
export async function openPortHistory(panel: Locator) {
  for (const id of ["port-history", "port-chlorophyll"]) {
    const d = panel.getByTestId(id);
    if ((await d.getAttribute("open")) == null) await d.locator("summary").click();
  }
}

/** The layer dock's second tier (options, method, provenance). */
export async function openDockDetails(page: Page) {
  const b = page.getByTestId("dock-details");
  if ((await b.getAttribute("aria-expanded")) !== "true") await b.click();
}

/** A CoastWatch tile source goes through the tile index (cwtile://<template>#cw-index=<url>);
 *  this returns the published template it requests, and the index it reads (or null). */
export function tileSource(tiles0: string | null | undefined): { template: string | null; index: string | null } {
  if (!tiles0) return { template: null, index: null };
  const raw = tiles0.replace(/^cwtile:\/\//, "");
  const [template, index] = raw.split("#cw-index=");
  return { template, index: index ? decodeURIComponent(index) : null };
}
