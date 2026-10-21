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

/** v7 Explore: the layer list is a popover from the dock's layer button (in the sheet on a phone). */
export async function openLayers(page: Page) {
  const deck = page.getByTestId("layer-deck");
  if (!(await deck.isVisible())) await page.getByTestId("layer-capsule").click();
  await expect(deck).toBeVisible();
  return deck;
}

/** v7 Explore: "About this layer" (provenance, coverage, method and skill, credits), folded in the layer list. */
export async function openAbout(page: Page) {
  await openLayers(page);
  const d = page.getByTestId("layer-about");
  if ((await d.getAttribute("open")) == null) await d.locator("summary").first().click();
  return d;
}

/** v7 Explore: the selection panel's "Evidence and sources" fold (and the claims past the first three). */
export async function openEvidence(page: Page) {
  const d = page.getByTestId("inspect-evidence");
  if ((await d.getAttribute("open")) == null) await d.locator("summary").first().click();
  const more = page.getByTestId("harbor-claims-more");
  if ((await more.count()) && (await more.getAttribute("open")) == null) await more.locator("summary").first().click();
  return d;
}
