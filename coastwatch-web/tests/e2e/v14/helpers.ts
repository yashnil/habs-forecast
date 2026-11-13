import { expect, type Browser, type BrowserContext, type Locator, type Page, type Route } from "@playwright/test";
import { OK } from "../ports";

/**
 * Shared set-up for the v14 guards (tests/e2e/v14). Written against roles, labels and the test ids the v13 pages
 * already carry, so it survives the synthesis restyle; the ids it relies on are listed in guards.spec.ts's header.
 */

export const MOSS = "port-592";
export const FORT_BRAGG = "port-223";
export const BODEGA = "port-473";

/** The owner's five review sizes. */
export const SIZES = [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 1100, height: 800 },
  { width: 390, height: 844 },
  { width: 320, height: 640 },
] as const;
export const PHONES = SIZES.filter((s) => s.width <= 640);
export const isPhone = (s: { width: number }) => s.width <= 640;

export const portRecord = (home: string, last = home) => JSON.stringify({ v: 1, place: home, for: "all", species: "any", lastPort: last, fisheries: ["dungeness_commercial"], vessel: null });

/** A returning reader at a home port: no opening, no tour, the commercial Dungeness fishery chosen. */
export const returning = (home: string = MOSS, last = home) => ({
  "nx-opening-seen-v1": "1",
  "nx-tour-v1": "dismissed",
  "nx-port-v1": portRecord(home, last),
  "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }),
});

/** Put values in the device's storage before the first page of the context loads (and only then). */
export async function seed(ctx: BrowserContext, items: Record<string, string>) {
  await ctx.addInitScript((kv) => {
    if (localStorage.getItem("__e2e_seeded")) return;
    localStorage.setItem("__e2e_seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, items);
}

export async function context(browser: Browser, size: { width: number; height: number }, o: { home?: string; last?: string; reducedMotion?: "reduce" | "no-preference" } = {}) {
  const ctx = await browser.newContext({
    viewport: size,
    ...(isPhone(size) ? { isMobile: true, hasTouch: true } : {}),
    reducedMotion: o.reducedMotion,
    storageState: { cookies: [], origins: [] },
  });
  await seed(ctx, returning(o.home ?? MOSS, o.last));
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return { ctx, page, errors };
}

// ---------------------------------------------------------------- the trip

export const field = (p: Page, label: string): Locator => p.getByRole("textbox", { name: label, exact: true });
export async function type(p: Page, label: string, v: string) {
  const f = field(p, label);
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
}
/** The regression trip: 400 lb at $9, 30 gal at $8.24, bait and ice $100, fees 0, 1 crew at 20 % after costs, costs shared (off the top). */
export async function fillRegressionTrip(p: Page) {
  for (const [l, v] of [
    ["Catch", "400"],
    ["Price", "9"],
    ["Fuel", "30"],
    ["Fuel price", "8.24"],
    ["Bait and ice", "100"],
    ["Fees", "0"],
    ["Crew", "1"],
  ])
    await type(p, l, v);
  await p.getByRole("radio", { name: "After costs" }).click();
  await type(p, "Crew share", "20");
  await p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await p.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
}

// ---------------------------------------------------------------- alerts: two states of the published records

/** The real fixture answer for one port until `publish()`, then the same answer with `edit` applied (v13/helpers.ts). */
export async function secondState(page: Page, port: string, edit: (inputs: Record<string, unknown>) => void) {
  let published = false;
  await page.route(`**/today/data?place=${port}`, async (route: Route) => {
    const res = await route.fetch();
    if (!published) return route.fulfill({ response: res });
    const j = (await res.json()) as { inputs: Record<string, unknown> };
    edit(j.inputs);
    j.inputs.dataGeneratedAt = "2026-10-13T05:30:00Z";
    await route.fulfill({ response: res, json: j });
  });
  return () => {
    published = true;
  };
}
export const dungenessMoves = (inputs: Record<string, unknown>) => {
  const rec = (inputs.fisheries as { records: Array<{ fishery: string; headline: string }> }).records.find((x) => x.fishery === "dungeness_commercial")!;
  rec.headline = "Commercial season delayed to at least Jan 15, 2027.";
};

/** Follow a port on Alerts and make one controlled change arrive (the alerts frame). */
export async function alertsWithOneChange(page: Page, port: string) {
  const publish = await secondState(page, port, dungenessMoves);
  await page.goto(`${OK}/alerts?place=${port}`);
  await page.getByTestId("alerts-follow").click();
  await expect(page.getByTestId("alerts-said")).toContainText("noted as it is now");
  publish();
  await page.getByTestId("alerts-check-now").click();
  await expect(page.getByTestId("alerts-item")).toHaveCount(1);
  await page.evaluate(() => scrollTo(0, 0));
}

// ---------------------------------------------------------------- the five frames as app states

export type Frame = { name: string; root: string; open: (p: Page) => Promise<void> };
export const FRAMES: Frame[] = [
  {
    name: "Today (Fort Bragg)",
    root: "nx-today",
    open: async (p) => {
      await p.goto(`${OK}/today?place=${FORT_BRAGG}`);
      await expect(p.getByTestId("nx-today")).toBeVisible();
    },
  },
  {
    name: "Explore",
    root: "nx-explore",
    open: async (p) => {
      await p.goto(`${OK}/explore?place=${MOSS}`);
      await expect(p.getByTestId("nx-explore")).toBeVisible();
    },
  },
  {
    name: "My Coast trip, empty",
    root: "nx-coast",
    open: async (p) => {
      await p.goto(`${OK}/coast?place=${MOSS}&view=trip`);
      await expect(p.getByTestId("trip-inputs")).toBeVisible();
      await type(p, "Catch", "400");
      await type(p, "Price", "9");
    },
  },
  {
    name: "My Coast trip, worked out",
    root: "nx-coast",
    open: async (p) => {
      await p.goto(`${OK}/coast?place=${MOSS}&view=trip`);
      await expect(p.getByTestId("trip-inputs")).toBeVisible();
      await fillRegressionTrip(p);
      await expect(p.locator('[data-testid="trip-netbar-value"], [data-testid="trip-net"]').first()).toContainText("2,602.24");
    },
  },
  {
    name: "Research (May 27, 2026 sample)",
    root: "nx-research",
    open: async (p) => {
      await p.goto(`${OK}/research?station=HABs-SantaCruzWharf&var=pDA&range=3y&day=2026-05-27`);
      await expect(p.getByTestId("research-plot")).toBeVisible({ timeout: 20_000 });
    },
  },
  {
    name: "Alerts (Fort Bragg, one change)",
    root: "nx-alerts",
    open: (p) => alertsWithOneChange(p, FORT_BRAGG),
  },
];

/** Anything that runs past the sides of the screen: the document itself, and visible text cut off at an edge. */
export async function sideways(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const out: string[] = [];
    const W = document.documentElement.clientWidth;
    const over = document.documentElement.scrollWidth - W;
    if (over > 0) out.push(`the page scrolls sideways by ${over}px`);
    const bodyOver = document.body.scrollWidth - document.body.clientWidth;
    if (bodyOver > 0 && getComputedStyle(document.body).overflowX !== "visible") out.push(`the body hides ${bodyOver}px of sideways overflow`);
    // text that is drawn past an edge of the screen, where no scroller of its own can bring it back
    const clipped = (el: Element) => {
      for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
        const ox = getComputedStyle(a).overflowX;
        if (ox === "auto" || ox === "scroll") return true; // its own scroller can reach it
      }
      return false;
    };
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const seen = new Set<Element>();
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (!el || seen.has(el) || !n.textContent?.trim()) continue;
      seen.add(el);
      // the map draws its own labels and may hold features beyond the frame; screen-reader-only text is not drawn
      if (el.closest(".maplibregl-map, [aria-hidden='true'], [hidden], .sr-only, [data-sr-only]")) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === "hidden" || cs.opacity === "0") continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if ((r.right > W + 1 || r.left < -1) && !clipped(el)) out.push(`“${n.textContent.trim().slice(0, 40)}” runs off the screen (x ${Math.round(r.left)}–${Math.round(r.right)} of ${W})`);
      if (out.length > 12) break;
    }
    return out;
  });
}
