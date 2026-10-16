import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/** The detail behind the briefing sits in one disclosure; open it (idempotent) before reading a tab. */
async function openDetails(page: Page) {
  const d = page.getByTestId("coast-details");
  if ((await d.count()) && !(await d.evaluate((el) => (el as HTMLDetailsElement).open))) await d.locator("summary").click();
}


/**
 * Coastal mode (the default for anyone not set up for research): My Coast as a briefing (your
 * waters and species, HAB conditions, official information and ocean conditions, before you go)
 * with nothing to open first and the full science one tap away, and a basemap-first
 * Ocean Map. Ocean conditions come from recorded NOAA responses (tests/conditions-fixtures.json).
 */
const NOW = "2026-10-08T20:00:00Z";
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

/** A visitor with no saved presentation (so Coastal) who has seen the welcome. */
async function coastal(browser: Browser, opts: object = { viewport: { width: 1440, height: 900 } }) {
  const ctx = await browser.newContext({ ...opts, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }, { origin: FAILED, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  return { ctx, page };
}

/** The trip view is rendered on the server; it is interactive once the client-only HAB line is in. */
async function hydrated(page: Page) {
  await expect(page.getByTestId("matter-hab")).toBeVisible();
}

async function noUnsafeClaims(page: Page) {
  const text = await page.locator("main").innerText();
  const bad = text
    .split(/(?<=[.!?])\s+|\n+/)
    .filter((s) => /\b(safe|open|all clear|good fishing)\b/i.test(s) && !/\b(not|no|never|does not|isn't|nothing)\b/i.test(s) && !/Whale.?safe/i.test(s));
  expect(bad, bad.join(" | ")).toEqual([]);
}

test.describe("Coastal mode", () => {
  test("My Coast is one trip briefing: your next trip, what matters, HAB evidence, before you leave, the evidence to explore", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-550&for=recreational&species=anchovy-sardine`);
    await hydrated(page);
    await expect(page.getByTestId("mode-switch-navy").getByTestId("mode-coastal")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("my-coast")).toHaveAttribute("data-view", "trip");
    // your next trip: the link's choices, each a chip that opens its picker
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
    await expect(page.getByTestId("trip-activity")).toContainText("Recreational");
    await expect(page.getByTestId("trip-species")).toContainText("Anchovy or sardine");
    await expect(page.getByTestId("trip-day")).toContainText("Today");
    // what matters: the notices that cover these waters, an alert, with the agency's own source
    const notices = page.getByTestId("matter-notices");
    await expect(notices).toContainText("2 notices cover these waters for anchovy or sardine");
    await expect(notices).toHaveAttribute("data-tone", "alert");
    await expect(notices.locator("[data-mark=official]")).toHaveCount(1);
    await expect(page.getByTestId("matter-notices-source")).toHaveAttribute("href", /^https:\/\/(wildlife\.ca\.gov|www\.cdph\.ca\.gov)\//);
    // one HAB line and one conditions line, each with what the map can show
    await expect(page.getByTestId("matter-hab")).not.toContainText("Reading");
    await expect(page.getByTestId("matter-conditions")).toContainText(/wind|waves|tides/i);
    // the notice copy's provenance is said once, precisely, with the whole list one tap away
    await expect(page.getByTestId("matters-provenance")).toContainText("No person has reviewed that copy");
    await page.getByTestId("matters-all-notices").click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
    await page.keyboard.press("Escape");
    // HAB evidence on one axis, each row with its place and distance; the gap said once
    await expect(page.getByTestId("trip-hab-summary")).toContainText("NOAA's model");
    for (const r of ["cells", "toxin", "satellite", "model"]) await expect(page.getByTestId(`ev-label-${r}`)).toBeVisible();
    await expect(page.getByTestId("evidence-where")).toContainText("km");
    await expect(page.getByTestId("ev-trip")).toBeVisible();
    // before you leave: at most five actions, each opening its source; the rest one tap away
    // (rows can move once the conditions arrive: count after any exit animation)
    const checks = page.getByTestId("checklist").locator('li[data-testid^="check-"]');
    await expect.poll(() => checks.count()).toBeLessThanOrEqual(5);
    expect(await checks.count()).toBeGreaterThanOrEqual(3);
    for (const c of await checks.all()) expect(await c.locator('a[href^="https://"], a[href^="tel:"], a[href^="/bloom?"], a[href^="/fisheries?"]').count()).toBeGreaterThanOrEqual(1);
    await expect(page.getByTestId("check-read-notices")).toContainText("Read the 2 notices that cover these waters");
    await page.getByTestId("check-read-notices-why").click();
    await expect(page.getByTestId("check-read-notices")).toContainText("no person has reviewed the association");
    // a tick is kept for this trip on this device, and means only that the source was opened
    await page.getByTestId("check-read-notices-box").click();
    await expect(page.getByTestId("checklist-progress")).toContainText(/^1 of \d checked$/);
    await page.reload();
    await expect(page.getByTestId("check-read-notices-box")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("checklist")).toContainText("not that anything is safe or open");
    if (await page.getByTestId("checklist-more").count()) {
      await page.getByTestId("checklist-more").click();
      await expect(page.getByTestId("check-history").getByRole("link")).toHaveAttribute("href", "/fisheries?group=northern_anchovy");
    }
    // explore the evidence: the detail, one subject at a time
    await openDetails(page);
    await expect(page.getByTestId("hab-briefing")).toBeVisible();
    // every id on the page is unique, and each evidence section is named by its own heading
    expect(await page.evaluate(() => { const seen = new Set<string>(); const dup: string[] = []; document.querySelectorAll("[id]").forEach((e) => (seen.has(e.id) ? dup.push(e.id) : seen.add(e.id))); return dup; })).toEqual([]);
    await expect(page.getByRole("region", { name: /^HAB evidence near/ })).toHaveAttribute("data-testid", "hab-evidence");
    await expect(page.getByRole("region", { name: /^Bloom and toxin evidence near/ })).toHaveAttribute("data-testid", "hab-briefing");
    await openDetails(page);
    await page.getByTestId("explore-tab-notices").click();
    await expect(page.getByTestId("brief-notice-cdfw-2026-anchovy-take-restriction-monterey-bay").getByRole("link", { name: /CDFW source/ })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
    await openDetails(page);
    await page.getByTestId("explore-tab-conditions").click();
    // water measured in the harbour itself (CO-OPS 6-minute thermistor, recorded 2026-10-13): named as the harbour's, with its flags
    await expect(page.getByTestId("cond-harbor")).toContainText("Water in the harbour · measured");
    await expect(page.getByTestId("cond-harbor")).toContainText(/\d+\.\d °C/);
    await expect(page.getByTestId("cond-harbor")).toContainText("NOAA tide station 9413450, in the harbour");
    await expect(page.getByTestId("cond-marine")).toContainText("W wind 10 to 15 kt");
    await expect(page.getByTestId("cond-marine")).toContainText("NWS PZZ535");
    await expect(page.getByTestId("cond-tides")).toContainText("Monterey");
    await openDetails(page);
    await page.getByTestId("explore-tab-science").click();
    await expect(page.getByTestId("coast-measured")).toBeVisible();
    await noUnsafeClaims(page);
    await ctx.close();
  });

  test("activity, species and day are chosen in place, change the briefing at once and go in the link", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-201`);
    await hydrated(page);
    // no species in the link: a general briefing, and the species chip asks for one
    await expect(page.getByTestId("trip-species")).toContainText("Any species");
    await page.getByTestId("trip-species").click();
    await expect(page.getByTestId("trip-sheet-species")).toBeVisible();
    await page.getByTestId("trip-species-razor-clam").click();
    await expect(page).toHaveURL(/species=razor-clam/);
    await expect(page.getByTestId("trip-sheet-species")).toHaveCount(0);
    await expect(page.getByTestId("trip-species")).toBeFocused();
    await expect(page.getByTestId("matter-notices")).toContainText("razor clams");
    // the checklist follows the species: CDPH's shellfish testing
    if (!(await page.getByTestId("check-cdph-map").count())) await page.getByTestId("checklist-more").click();
    await expect(page.getByTestId("check-cdph-map")).toBeVisible();
    // the activity: shellfish harvest puts the tides first in the conditions
    await page.getByTestId("trip-activity").click();
    await page.getByTestId("trip-for-shellfish").click();
    await expect(page).toHaveURL(/for=shellfish/);
    await openDetails(page);
    await page.getByTestId("explore-tab-conditions").click();
    const first = page.getByTestId("conditions").locator("[data-testid^=cond-][data-state]").first();
    await expect(first).toHaveAttribute("data-testid", "cond-tides");
    // the day: a later day is in the link, the evidence marks it, and a day the model does not reach says so
    await page.getByTestId("trip-day").click();
    const later = page.getByTestId("trip-sheet-day").locator('[role="radio"]').last();
    const date = (await later.getAttribute("data-testid"))!.replace("trip-day-", "");
    await expect(later).toContainText(/Tides only|Model outlook/);
    const tidesOnly = (await later.textContent())!.includes("Tides only");
    await later.click();
    await expect(page).toHaveURL(new RegExp(`date=${date}`));
    await expect(page.getByTestId("ev-trip")).toBeVisible();
    // the model's own line says so; the HAB line keeps to the shore stations
    if (tidesOnly) await expect(page.getByTestId("matter-model")).toContainText("does not reach");
    // Escape closes a picker and gives focus back
    await page.getByTestId("trip-activity").click();
    await expect(page.getByTestId("trip-sheet-activity")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("trip-sheet-activity")).toHaveCount(0);
    await expect(page.getByTestId("trip-activity")).toBeFocused();
    await ctx.close();
  });

  test("Dungeness crab, commercial, North Coast: the season status leads, quoted, and the closure line is on the checklist", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-201&for=commercial&species=dungeness`);
    await hydrated(page);
    const first = page.getByTestId("what-matters").locator("li").first();
    await expect(first).toHaveAttribute("data-testid", "matter-season");
    await expect(first.locator("blockquote")).not.toBeEmpty();
    await expect(page.getByTestId("matter-season-source")).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
    // CDFW's other Dungeness statement is quoted with it, not as a second card
    await expect(page.getByTestId("matter-season-also")).toContainText("marine toxin");
    const checks = page.getByTestId("checklist").locator('li[data-testid^="check-"]');
    await expect(checks.first()).toContainText("Confirm the season status on CDFW's page");
    await expect(page.getByTestId("check-closure-line-go")).toHaveAttribute("href", /^tel:/);
    await noUnsafeClaims(page);
    await ctx.close();
  });

  test("phone: the briefing first, the map one tap away in a sheet that follows what is chosen", async ({ browser }) => {
    const { ctx, page } = await coastal(browser, PHONE);
    await page.goto(`${OK}/coast?place=port-593`);
    await hydrated(page);
    // the trip and what matters fit the first screen without a map in between
    await expect(page.getByTestId("what-matters")).toBeVisible();
    const box = (await page.getByTestId("trip-day").boundingBox())!;
    expect(box.y + box.height).toBeLessThan(844);
    // nothing (not even a chart before it is measured) widens the page: a wider layout makes a phone zoom out
    expect(await page.evaluate(() => window.innerWidth)).toBe(390);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    // the map is not loaded until it is wanted
    await expect(page.getByTestId("coast-map-mobile")).toHaveAttribute("data-open", "false");
    expect(await page.evaluate(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap)).toBe(false);
    const open = page.getByTestId("coast-map-open");
    await expect(open).toBeInViewport();
    // "Show on map" opens the sheet on the evidence
    await page.getByTestId("matter-hab-map").click();
    await expect(page.getByTestId("coast-map-mobile")).toHaveAttribute("data-open", "true");
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
    await expect(page.getByTestId("coast-map-key")).toBeVisible();
    await page.getByTestId("coast-map-close").click();
    await expect(page.getByTestId("coast-map-mobile")).toHaveAttribute("data-open", "false");
    // a picker is a bottom sheet with large targets
    await page.getByTestId("trip-activity").click();
    const sheet = page.getByTestId("trip-sheet-activity");
    await expect(sheet).toBeVisible();
    // focus starts on the current choice, not the close button
    await expect(sheet.locator('[role="radio"][aria-checked="true"]')).toBeFocused();
    const sb = (await sheet.boundingBox())!;
    expect(sb.y + sb.height).toBeGreaterThan(800);
    for (const b of await sheet.locator('[role="radio"]').all()) expect((await b.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    await ctx.close();
  });

  test("the map follows the briefing: the model's real cells and its nearshore gaps, the satellite, a station", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-593`);
    await hydrated(page);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
    const has = (id: string) => page.evaluate((i) => !!(window as unknown as { __cwMap: { getSource: (x: string) => unknown } }).__cwMap.getSource(i), id);
    await expect(page.getByTestId("coast-map-key")).toContainText("Station sampled in the last 30 days");
    expect(await has("forecast")).toBe(false);
    // the model card puts its own grid on the map, with the 15 km ring and the no-value hatch named in the key
    await openDetails(page);
    await page.getByTestId("ev-model-map").click();
    await expect(page.getByTestId("ev-model-map")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("coast-map-key")).toContainText("15 km around your waters");
    await expect(page.getByTestId("coast-map-key")).toContainText("No model value");
    await expect.poll(() => has("forecast")).toBe(true);
    // the satellite card swaps it for the satellite field (one source in view at a time)
    await page.getByTestId("ev-satellite").click();
    await expect(page.getByTestId("ev-satellite")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("ev-model-map")).toHaveAttribute("aria-pressed", "false");
    await expect.poll(() => has("satellite")).toBe(true);
    await expect.poll(() => has("forecast")).toBe(false);
    await expect(page.getByTestId("coast-map-key")).toContainText("algae, not toxin");
    // selecting it again returns the map to the overview
    await page.getByTestId("ev-satellite").click();
    await expect.poll(() => has("satellite")).toBe(false);
    // a station on the map opens its latest measurements, each with its date, and its history
    await page.getByTestId("ev-model-map").click();
    await page.getByTestId("ev-model-map").click();
    await page.waitForTimeout(1500);
    const pt = await page.evaluate(() => {
      const m = (window as unknown as { __cwMap: import("maplibre-gl").Map }).__cwMap;
      const f = m.queryRenderedFeatures(undefined, { layers: ["coast-station"] })[0];
      if (!f) return null;
      const p = m.project((f.geometry as GeoJSON.Point).coordinates as [number, number]);
      const r = m.getCanvas().getBoundingClientRect();
      return { x: r.left + p.x, y: r.top + p.y };
    });
    if (pt) {
      await page.mouse.click(pt.x, pt.y);
      const peek = page.getByTestId("station-peek");
      const card = page.locator('[data-testid="ev-cells"][aria-pressed="true"], [data-testid="ev-toxin"][aria-pressed="true"]');
      // a station the briefing names takes its card into focus; any other opens its readings
      await expect(peek.or(card)).toBeVisible();
      if (await peek.count()) await expect(page.getByTestId("station-peek-history")).toHaveAttribute("href", /^\/bloom\?station=/);
    }
    await ctx.close();
  });

  test("the conditions card shows its own NOAA stations on the map, apart from the HAB evidence", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-550&for=recreational`);
    await hydrated(page);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
    await openDetails(page);
    await page.getByTestId("explore-tab-conditions").click();
    await expect(page.getByTestId("cond-tides")).toContainText("Monterey");
    await page.getByTestId("cond-map").click();
    await expect(page.getByTestId("cond-map")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("coast-map-key")).toContainText("NOAA buoy and tide station (conditions, not HAB)");
    const obs = () =>
      page.evaluate(() => {
        const m = (window as unknown as { __cwMap: import("maplibre-gl").Map }).__cwMap;
        return m.querySourceFeatures("coast-overlay").filter((f) => f.properties?.kind === "obs").map((f) => `${f.properties?.what}`).sort().join(",");
      });
    await expect.poll(obs).toMatch(/tide/);
    // selecting it again returns the map to the overview, without the NOAA stations
    await page.getByTestId("cond-map").click();
    await expect.poll(obs).toBe("");
    await ctx.close();
  });

  test("an honest gap: no station within 60 km at Shelter Cove; said once, as a gap, with what to check", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-215`);
    await hydrated(page);
    const hab = page.getByTestId("matter-hab");
    await expect(hab).toHaveAttribute("data-tone", "gap");
    await expect(hab).toContainText("No recent HAB sample within 60 km");
    await expect(hab).toContainText("not a clean result");
    // the gap is a row of the evidence list, said as a gap, not a reassurance
    await expect(page.getByTestId("gap-pn_cells-recent").or(page.getByTestId("gap-pda-recent")).first()).toContainText(/too far to describe these waters|in the last 30 days/);
    await expect(page.getByTestId("evidence-where")).toContainText("too far to describe these waters");
    if (!(await page.getByTestId("check-toxin-gap").count())) await page.getByTestId("checklist-more").click();
    await expect(page.getByTestId("check-toxin-gap")).toContainText("Look at CDPH's latest toxin test results");
    await openDetails(page);
    await page.getByTestId("explore-tab-conditions").click();
    await expect(page.getByTestId("cond-buoy")).toContainText("No NDBC station reports within 60 km");
    await expect(page.getByTestId("cond-buoy")).toHaveAttribute("data-state", "none");
    await ctx.close();
  });

  test("a NOAA outage reads in plain words, with the technical reason kept for anyone who looks", async ({ browser }) => {
    // a point on the water with no recorded NOAA responses: every live request fails like an outage
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=point-36.700,-122.300`);
    await hydrated(page);
    if (!(await page.getByTestId("check-marine-gap").count())) await page.getByTestId("checklist-more").click();
    await expect(page.getByTestId("check-marine-gap")).toContainText("Read the NWS marine forecast for your zone");
    await page.getByTestId("check-marine-gap-why").click();
    await expect(page.getByTestId("check-marine-gap-reason")).toContainText("did not answer just now");
    await openDetails(page);
    await page.getByTestId("explore-tab-conditions").click();
    const marine = page.getByTestId("cond-marine");
    await expect(marine).toContainText("did not answer just now");
    await expect(marine).toHaveAttribute("title", /HTTP 503/);
    await expect(marine).not.toContainText("HTTP 503");
    await ctx.close();
  });

  test("switching to Research opens the science, is kept on the device, and a link's mode wins without being saved", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-550`);
    await hydrated(page);
    await page.getByTestId("mode-switch-navy").getByTestId("mode-research").click();
    await expect(page.getByTestId("explore")).toHaveCount(0);
    await expect(page.getByTestId("coast-measured")).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("research");
    await page.getByTestId("mode-switch-navy").getByTestId("mode-coastal").click();
    await expect(page.getByTestId("coast-details")).toBeVisible();
    // a shared research link shows Research here, but the visitor's own choice stays Coastal
    await page.goto(`${OK}/coast?place=port-550&mode=research`);
    await expect(page.getByTestId("mode-switch-navy").getByTestId("mode-research")).toHaveAttribute("aria-pressed", "true");
    expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("coastal");
    await ctx.close();
  });

  test("the Ocean Map opens on the basemap in Coastal mode; a link naming a layer still shows it", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    // the depth tint cross-fades (raster-opacity), it is never toggled
    const visible = (id: string) => page.evaluate((i) => (window as unknown as { __cwMap: { getPaintProperty: (x: string, p: string) => number } }).__cwMap.getPaintProperty(i, "raster-opacity") > 0, id);
    await page.goto(`${OK}/?region=monterey_bay`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
    await expect.poll(() => visible("relief-sea-tint")).toBe(true);
    await expect(page.getByTestId("forecast-panel")).toHaveCount(0);
    await page.goto(`${OK}/?region=monterey_bay&layer=forecast`);
    await expect(page.getByTestId("forecast-panel")).toBeVisible();
    await expect.poll(() => visible("relief-sea-tint")).toBe(false);
    await ctx.close();
  });

  test("a failed update is never labelled Current, even for a recent run", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    // the failed fixture's last run was issued on the fixture day; its later update failed
    await page.goto(`${FAILED}/?region=monterey_bay&layer=forecast`);
    const badge = page.getByTestId("forecast-panel").getByTestId("freshness");
    await expect(badge).toHaveAttribute("data-state", "unconfirmed");
    await expect(badge).toContainText("Not confirmed");
    await expect(page.getByTestId("update-failed")).toContainText("a newer run may exist");
    await ctx.close();
  });

  test("reading the device's mode never re-mounts the Bloom chart or the Fisheries history (Research)", async ({ browser }) => {
    // regression: the Research layout used to replace the Coastal tree after the first render, so
    // the chart was detached and drawn again (a flash for researchers; CI caught a detached element)
    const { ctx, page } = await coastal(browser);
    await page.addInitScript(() => localStorage.setItem("cw-mode-v1", "research"));
    for (const [url, id] of [["/bloom?range=3m", "primary-chart"], ["/fisheries", "landings-chart"]] as const) {
      await page.goto(`${OK}${url}`);
      const el = await page.getByTestId(id).elementHandle();
      await page.waitForTimeout(1500);
      expect(await el!.evaluate((n) => n.isConnected), `${id} stayed in the page`).toBe(true);
      await expect(page.getByTestId(id)).toBeVisible();
    }
    await ctx.close();
  });

  for (const [name, opts] of [["desktop", { viewport: { width: 1440, height: 900 } }], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing on the Coastal briefing`, async ({ browser }) => {
      const { ctx, page } = await coastal(browser, opts);
      await page.goto(`${OK}/coast?place=site-clam-beach&for=shellfish&species=razor-clam`);
    await hydrated(page);
      await expect(page.getByTestId("matter-notices")).toBeVisible();
      await expect(page.getByTestId("trip-hab-summary")).toBeVisible();
      await expect(page.getByTestId("matter-hab")).not.toContainText("Reading");
      await expect(page.getByTestId("ev-label-model")).not.toContainText("Reading");
      await expect(page.getByTestId("checklist")).toBeVisible();
      await page.waitForTimeout(600);
      const r = await new AxeBuilder({ page }).include("main").analyze();
      expect(r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
      await ctx.close();
    });
  }
  test("every page links straight to CDFW and CDPH in Coastal mode while the notices are unreviewed", async ({ page }) => {
    for (const path of ["/about", "/bloom", "/fisheries", "/coast"]) {
      await page.goto(`${OK}${path}${path.includes("?") ? "&" : "?"}mode=coastal`);
      const line = page.getByTestId("status-line");
      await expect(line, path).toBeVisible();
      await expect(line.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", /wildlife\.ca\.gov/);
      await expect(line.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", /cdph\.ca\.gov/);
      await expect(line).toContainText(/not verified/i);
    }
  });
});
