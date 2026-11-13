import AxeBuilder from "@axe-core/playwright";
import { expect, type Browser, type BrowserContext, type Page, type Request, type Route } from "@playwright/test";
import type { CoastGates, CoastSea } from "@/nx/coast/brief";
import { tripFishery } from "@/nx/coast/fisheries";
import { emptyWorkspace, toDraft, type Workspace } from "@/nx/coast/workspace";
import { forecastFromSea } from "@/nx/trips/forecast";
import { takeSnapshot } from "@/nx/trips/snapshot";
import { blankTrip, createTripsStore } from "@/nx/trips/store";
import type { Snapshot, Trip } from "@/nx/trips/types";
import { OK } from "../ports";
import { exact, fillOwner } from "../v13/trip";

/**
 * The v15 acceptance suite's shared set-up (tests/e2e/v15, DIRECTION §17). The fixture servers pin the clock at
 * 2026-10-13T06:00Z (playwright.config.ts); every page here pins the device clock to the same instant, so a planning
 * snapshot, a recheck and the forecast's periods agree on what "now" is.
 *
 * Two ways to have a trip: through the planner (`planTrip`, the owner's journey, Test A's path) or written straight
 * into the device's storage in the trips store's own format (`seedTrip`: the store and the snapshot module build it,
 * from the fixture server's real answers), so a test of reopening, rechecking or completing does not depend on the
 * planner's layout.
 */
export const NOW = "2026-10-13T06:00:00Z";
export const MOSS = "port-592";
export const FISHERY = "dungeness_commercial";
export const GATES = `/coast/data?place=${MOSS}&fishery=${FISHERY}`;
export const SEA = `/coast/data?place=${MOSS}&part=sea`;
export const NS = "nx-trips-v1:";

export const SIZES = { desk: { width: 1440, height: 900 }, phone: { width: 390, height: 844 }, small: { width: 320, height: 640 } } as const;

/** A returning reader at Moss Landing (no opening, no tour), as the v14 guards set up. */
export const returning: Record<string, string> = {
  "nx-opening-seen-v1": "1",
  "nx-tour-v1": "dismissed",
  "nx-port-v1": JSON.stringify({ v: 1, place: MOSS, for: "all", species: "any", lastPort: MOSS, fisheries: [FISHERY], vessel: null }),
  "nx-coast-v1": JSON.stringify({ fishery: FISHERY }),
};

/** Values put in storage before the context's first page, once (a reload keeps what the app wrote, or deleted, since). */
export async function seed(ctx: BrowserContext, items: Record<string, string>) {
  // the mark is a cookie, which "Delete everything" leaves alone, so a deleted device is never seeded again
  await ctx.addInitScript((kv) => {
    if (document.cookie.includes("__e2e_seeded=1")) return;
    document.cookie = "__e2e_seeded=1; path=/; max-age=86400";
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, items);
}

/**
 * `routed`: the test answers /coast/data itself (page.route). The page's service worker would otherwise fetch on its own
 * once it controls the page, past the routes, so it is blocked for those tests.
 */
export async function newContext(browser: Browser, size: { width: number; height: number } = SIZES.desk, items: Record<string, string> = {}, o: { routed?: boolean } = {}) {
  const ctx = await browser.newContext({ viewport: size, ...(size.width <= 640 ? { isMobile: true, hasTouch: true } : {}), storageState: { cookies: [], origins: [] }, ...(o.routed ? { serviceWorkers: "block" as const } : {}) });
  await seed(ctx, { ...returning, ...items });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return { ctx, page, errors };
}

/** The owner's regression worksheet: 400 lb at $9, 30 gal at $8.24, bait and ice $100, 1 crew at 20 % after costs. */
export const ownerWorkspace = (): Workspace => ({
  ...emptyWorkspace(tripFishery(FISHERY)),
  catch: [{ label: "Dungeness crab", ref: "dungeness", amount: "400", price: "9" }],
  fuelGal: "30",
  fuelPrice: "8.24",
  fuelPayer: "shared",
  costs: [{ label: "Bait and ice", usd: "100", payer: "shared" }],
  feesPct: "0",
  crew: "1",
  crewKind: "afterExpenses",
  share: "20",
});

export async function answers(): Promise<{ gates: CoastGates; sea: CoastSea }> {
  const [g, s] = await Promise.all([fetch(`${OK}${GATES}`).then((r) => r.json()), fetch(`${OK}${SEA}`).then((r) => r.json())]);
  return { gates: g as CoastGates, sea: s as CoastSea };
}

export type Seeded = { trip: Trip; planning: Snapshot; storage: Record<string, string> };

/**
 * A planned Moss Landing commercial Dungeness trip with the owner's economics and a planning snapshot taken from the
 * fixture server's answers, as the trips store writes it. `edit` changes the trip before it is stored.
 */
export async function seedTrip(o: { id?: string; depart?: string; limits?: Trip["limits"]; edit?: (t: Trip) => Trip; editSea?: (s: CoastSea) => void; editGates?: (g: CoastGates) => void } = {}): Promise<Seeded> {
  const { gates, sea } = await answers();
  o.editSea?.(sea);
  o.editGates?.(gates);
  const f = tripFishery(FISHERY)!;
  let trip = blankTrip({
    id: o.id ?? "e2e-moss",
    fishery: FISHERY,
    placeKey: MOSS,
    now: NOW,
    name: "Moss Landing, Dungeness crab",
    economics: toDraft(ownerWorkspace(), f),
    state: "planned",
    depart: { date: o.depart ?? "2026-10-14", time: "05:30", timeZone: "America/Los_Angeles" },
    limits: o.limits ?? { windKt: 12, seasFt: 6 },
  });
  if (o.edit) trip = o.edit(trip);
  const planning = takeSnapshot({ trip, gates, sea, forecast: forecastFromSea(sea), now: NOW, mode: "live", savedFrom: null, kind: "planning", id: `${trip.id}-plan` });
  trip = { ...trip, snapshots: { planning: planning.id, rechecks: [], superseded: [] } };
  const map = new Map<string, string>();
  const kv = { getItem: (k: string) => map.get(k) ?? null, setItem: (k: string, v: string) => void map.set(k, v), removeItem: (k: string) => void map.delete(k), key: (i: number) => [...map.keys()][i] ?? null, get length() { return map.size; } };
  const store = createTripsStore(kv, () => new Date(NOW));
  const put = store.put(trip);
  if (!put.ok) throw new Error(`seed: ${put.reason}`);
  const snap = store.putSnapshot(planning);
  if (!snap.ok) throw new Error(`seed snapshot: ${snap.reason}`);
  return { trip, planning, storage: Object.fromEntries(map) };
}

/**
 * The owner's planning journey through My Coast (as tests/e2e/v15/plan.spec.ts, Test A): Moss Landing, commercial
 * Dungeness, a date, the reader's own limits, the regression worksheet, Save trip. Ends on the trip's workspace and
 * returns the new trip's id.
 */
export async function planTrip(page: Page, o: { date?: string; windKt?: string; seasFt?: string } = {}): Promise<string> {
  await page.goto(`${OK}/coast?place=${MOSS}`);
  await expect(page.getByTestId("coast-status").first()).toContainText(/delayed to at least/, { timeout: 15_000 });
  await page.getByTestId("coast-fishery").first().selectOption(FISHERY);
  await page.getByTestId("plan-date").fill(o.date ?? "2026-10-14");
  await page.getByTestId("plan-more").getByRole("button").click();
  await page.getByRole("textbox", { name: "Wind", exact: true }).fill(o.windKt ?? "12");
  await page.getByRole("textbox", { name: "Seas", exact: true }).fill(o.seasFt ?? "6");
  await page.getByTestId("coast-to-trip").click();
  await expect(page.getByTestId("trip-inputs")).toBeVisible();
  await fillOwner(page);
  await expect(exact(page, "trip-net")).toHaveText("$2,602.24");
  await page.getByTestId("trip-plan-save").click();
  await expect(page).toHaveURL(/\/coast\/trips\?id=/);
  await expect(page.getByTestId("trip-workspace")).toBeVisible();
  return new URL(page.url()).searchParams.get("id")!;
}

/** Every trips key this page's storage holds. */
export const tripKeys = (page: Page) => page.evaluate((ns) => Object.keys(localStorage).filter((k) => k.startsWith(ns)), NS);

// ---------------------------------------------------------------- changed answers (Tests B, C, D)

/** Answer the sea endpoint with the real fixture answer, changed by `edit` (or a status code). */
export async function routeSea(page: Page, edit: ((s: CoastSea) => void) | { status: number }) {
  await page.route(`**/coast/data?*part=sea*`, async (route: Route) => {
    if ("status" in edit) return route.fulfill({ status: edit.status, contentType: "application/json", body: JSON.stringify({ ok: false, error: "Service unavailable" }) });
    const res = await route.fetch();
    const j = (await res.json()) as CoastSea;
    edit(j);
    await route.fulfill({ response: res, json: j });
  });
}

/** Answer the gates endpoint (any fishery) with the real fixture answer, changed by `edit`. */
export async function routeGates(page: Page, edit: (g: CoastGates) => void) {
  await page.route(`**/coast/data?*fishery=*`, async (route: Route) => {
    const res = await route.fetch();
    const j = (await res.json()) as CoastGates;
    edit(j);
    await route.fulfill({ response: res, json: j });
  });
}

/** The fixture's NWS periods: change one period's stated wind (its words and its numbers, as NWS would). */
export function strongerWind(s: CoastSea, periodName: string, text: string) {
  const lane = s.sea.lanes.find((l) => l.name === periodName);
  if (!lane) throw new Error(`no period ${periodName}`);
  lane.text = text;
}

// ---------------------------------------------------------------- privacy (Test G)

/** Requests that carry any of `needles` in their address, headers or body. */
export function watchRequests(page: Page | BrowserContext, needles: string[]) {
  const hits: string[] = [];
  const all: string[] = [];
  const on = (r: Request) => {
    const body = r.postData() ?? "";
    const hay = `${r.url()}\n${JSON.stringify(r.headers())}\n${body}`;
    all.push(`${r.method()} ${r.url()}`);
    for (const n of needles) if (hay.includes(n)) hits.push(`${r.method()} ${r.url()} carries "${n}"`);
  };
  page.on("request", on);
  return { hits, all };
}

// ---------------------------------------------------------------- shared assertions

export const VERDICT = /\b(safe|unsafe|legal|allowed|all clear|approved)\b/i;

export async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

export async function noSideways(page: Page) {
  const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(over, "the page scrolls sideways").toBeLessThanOrEqual(0);
}
