import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { FAILED, NODATA, OK } from "./ports";

/**
 * The truth rules (docs/coastwatch/v12/research/E-science-and-perf.md §2) on the v12 pages, ported from the
 * v4–v11 suites that asserted them on layouts that no longer exist (briefing-trust, trust, explore-v11, m3,
 * m6, research, economics, fishing, today, readiness, states, v5-map; see
 * docs/coastwatch/v12/implementation/test-migration.md). Each test names the rule it holds.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const MOSS = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });

async function seed(ctx: BrowserContext, record = MOSS) {
  await ctx.addInitScript((port) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    localStorage.setItem("nx-port-v1", port);
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-tour-v1", "dismissed");
  }, record);
}

const margin = (page: Page) => page.getByTestId("nx-margin");
/** v14: Today's ticket tabs (Status / Record / Fisheries / Notices) */
const tab = (page: Page, name: string) => page.getByTestId("nx-today").getByRole("tab", { name: new RegExp(`^${name}`) }).click();

/** T36: verdict words never appear, in any state. "open" is checked separately where it could be a status. */
const FORBIDDEN = /\b(safe to|is safe|all clear|no-go|good day|good to fish|hotspot|best spot|recommended area|expected catch|price forecast|green light|low risk|toxin forecast|bloom risk)\b/i;

test.describe("Today (Moss Landing, Dungeness commercial)", () => {
  test.beforeEach(async ({ context }) => seed(context));

  test("the official status and its decision date say 'not verified' in words, with source and date (T1, T38)", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-592`);
    const m = margin(page);
    await expect(m.getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027");
    await expect(m.getByRole("button", { name: /^Official · CDFW · checked Oct 12 · not verified\. Open where this comes from$/ }).first()).toBeVisible();
    await expect(m).toContainText("Next whale assessment, subject to change");
    // the marks' words: in the Record tab's legend
    await tab(page, "Record");
    await expect(m).toContainText("Official, not verified");
  });

  test("a test is a named lab result with its age, in the domoic acid gate (T6, T8, T17)", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-592`);
    const m = margin(page);
    // v14: the ticket's Status shows the test against its action level, with its age
    await expect(m.getByTestId("nx-test-cell")).toContainText("Dungeness, highest 9.8 of 30 ppm");
    await expect(m.getByTestId("nx-test-cell")).toContainText("6 samples · Monterey Bay · Sep 28 · 14 days ago");
    // v13: the test is held by the gate it informs (CDPH tests), with the highest result, the count and the age
    await tab(page, "Record");
    const gate = m.getByTestId("gate-domoic");
    await expect(gate).toContainText("CDPH tests");
    await expect(gate).toContainText("Dungeness test, Monterey Bay, Sep 28: highest 9.8 of 30 ppm, 6 samples");
    await expect(gate).toContainText("14 days ago");
    await expect(m.getByRole("button", { name: /^Measured · CDPH · sampled Sep 28/ }).first()).toBeVisible();
    // the figure set on the chart (under Toxin tests) names the action level beside it, never a bare number
    await page.locator('[data-testid="nx-layers"] [data-layer="toxin"]').click();
    await expect(page.locator("[data-nx-sounding=test]")).toContainText("of 30 ppm", { timeout: 15_000 });
  });

  test("the models row says algae, not toxin; C-HARM carries its mandatory words and is a past run (T3, T5, T9, T10)", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-592`);
    const m = margin(page);
    // v14: the models sit quietly under the ticket's Notices tab, after the official notices
    await tab(page, "Notices");
    await expect(m).toContainText("Models · not toxin tests");
    await expect(m).toContainText("Algae higher than usual, not toxin");
    await expect(m).toContainText("P(pDA > 500 ng/L): a probability of conditions in the water, not toxin in seafood.");
    await expect(m).toContainText("C-HARM, past run");
  });

  test("no verdict words anywhere on the page, and the legend states every mark once (T36, README §4)", async ({ page }) => {
    await page.goto(`${OK}/today?place=port-592`);
    const m = margin(page);
    await expect(m.getByRole("heading", { level: 1 })).toBeVisible();
    // v13: CDFW's own clauses and the other test sites are lines of their gates; v14: every tab of the ticket
    let text = await m.innerText();
    for (const name of ["Fisheries", "Notices", "Record"]) {
      await tab(page, name);
      text += await m.innerText();
    }
    expect(text).not.toMatch(FORBIDDEN);
    for (const w of ["Measured", "Model or prediction", "Official, not verified", "No view: no data, not a low value", "(old)"]) await expect(m).toContainText(w);
  });

  test("a failed update keeps the last good records with their own dates (trust S3)", async ({ page }) => {
    await page.goto(`${FAILED}/today?place=port-592`);
    const m = margin(page);
    await expect(m.getByRole("heading", { level: 1 })).toHaveText("Season delayed to at least Jan 1, 2027");
    await tab(page, "Record");
    await expect(m).toContainText("Records published Oct 12, 11:00 AM PT.");
    await tab(page, "Notices");
    await expect(m).toContainText("Past C-HARM run only");
    expect(await m.innerText()).not.toMatch(FORBIDDEN);
  });
});

test("sources differ: an agency release that appears to end a registry notice is a status, never resolved (T40, discrepancy)", async ({ page, context }) => {
  await seed(context, JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-550", fisheries: ["bivalve_shellfish"], vessel: null }));
  await page.goto(`${FAILED}/today?place=port-550`);
  const m = margin(page);
  await expect(m.getByRole("heading", { level: 1 })).toContainText("Sources differ");
  await expect(m).toContainText("Awaiting a person's review. Neither record is chosen here.");
  await expect(m).toContainText("“Annual quarantine of sport-harvested mussels”");
  await expect(m).toContainText("“CDPH announced the 2026 mussel quarantine ended Oct 31; check CDPH before harvesting.”");
  await expect(m).toContainText("Neritic doesn’t decide between them.");
  await expect(m).toContainText("Sources differ · not verified");
});

test.describe("Explore", () => {
  test("chlorophyll: kind, source, the range of observation dates, 'no view' and 'algae, not toxin' beside its ramp (T1, T5, T18, T19)", async ({ page }) => {
    await page.goto(`${OK}/explore?rail=1&place=port-592`);
    const m = margin(page);
    // v15.1: Explore California Waters, the layer named in its switch
    await expect(m.getByRole("heading", { level: 1, name: "Explore California Waters: Chlorophyll" })).toBeVisible();
    await expect(m).toContainText("No view");
    // the newest pixel is Oct 6, past OLCI's 3-day limit on Oct 12: "(old)" beside the dates (T17)
    await expect(m).toContainText("Measured · Sentinel-3 OLCI, 300 m · latest view, pixels Oct 1–6, each its own latest day seen (old). Algae, not toxin.");
    // the first line says what the view is, before any instrument (5-second test)
    // v15.1: one short line in view; the full sentence is in "About this layer"
    await expect(m.getByTestId("nx-explore-lead")).toHaveText("Algae seen from space · not toxin");
    await expect(m.getByTestId("nx-lead-full")).toHaveText("Algae seen from space, Oct 1–6: each spot shows its latest clear view. Not toxin.");
    await expect(m.getByTestId("nx-rail-also")).toBeHidden();
  });

  test("currents: off by default over chlorophyll (T25, T27); switched on, an hour past the radar's limit is still and says (old) (T17, T24)", async ({ page }) => {
    await page.goto(`${OK}/explore?place=port-592`);
    const toggle = page.getByTestId("nx-currents-toggle");
    await expect(toggle).toHaveAttribute("aria-checked", "false");
    await expect(page.getByTestId("nx-currents-key")).toHaveCount(0);
    await toggle.click();
    await expect(toggle).toContainText("on, measured (old)");
    await expect(page).toHaveURL(/cur=1/);
    const key = page.getByTestId("nx-currents-key");
    await expect(key).toContainText("(old). Older than the radar's own freshness limit, so drawn still.");
    await expect(key).toContainText("Relative speed; fainter where the radar measured the cell in fewer hours. No streak is no data, not calm.");
  });

  test("C-HARM: the event and threshold, its mandatory words, classes that are not risk levels, a past run said so (T3, T9, T10, T12)", async ({ page }) => {
    await page.goto(`${OK}/explore?rail=1&place=port-592&layer=charm`);
    const m = margin(page);
    await expect(m.getByTestId("nx-legend")).toContainText("C-HARM");
    await expect(m.getByTestId("nx-legend")).toContainText("P(pDA > 500 ng/L)");
    await expect(m).toContainText("Probability of conditions in the water, not toxin in seafood.");
    await expect(m).toContainText("Classes are display steps, not risk levels");
    await expect(m).toContainText("a lower probability does not mean open or safe");
    await expect(m).toContainText("Past run: every day it covers is before today.");
    await expect(m).toContainText("Currents");
    // v13: the time rail (and its "Also" line) runs along the foot of the chart, outside the margin
    await expect(page.getByTestId("nx-timerail")).toContainText("not over a model");
    expect(await m.innerText()).not.toMatch(FORBIDDEN);
  });

  test("the outlook: its own kind, its typical error in plain words, and 'no signal' cells (T2, T14, v5-map)", async ({ page }) => {
    await page.goto(`${OK}/explore?place=port-592&layer=outlook`);
    const m = margin(page);
    await expect(m).toContainText("Hatched: outlook, a model");
    await expect(m).toContainText("Outlook · Neritic, from NASA MODIS-Aqua 8-day");
    await expect(m).toContainText("Typical error about a factor of 2: read as higher or lower than usual, not as a value.");
    await expect(m).toContainText("Dotted: no signal");
  });

  test("no dataset: a worded unavailable state, never an empty chart (readiness)", async ({ page }) => {
    await page.goto(`${NODATA}/explore`);
    await expect(margin(page)).toContainText("The water can't be shown right now");
    await expect(margin(page)).toContainText("Try again in a few minutes");
  });
});

test.describe("Research", () => {
  test("a record keeps reported 0 in its own lane, chlorophyll says not toxin, the model is a 15 km median with its gap hatched (T6, T7, T11, T19)", async ({ page }) => {
    await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
    const rec = page.getByTestId("research-record");
    await expect(rec).toContainText("Reported 0");
    await expect(rec).toContainText("ng/mL in seawater");
    await page.getByTestId("rail-chl_extracted").click();
    await expect(rec).toContainText("mg/m³, algae, not toxin");
    await page.getByTestId("rail-pDA").click();
    // v14: the model is behind "Show the model"; shown, it is the median of cells within 15 km with its gap said, and
    // its words are visible beside it with nothing else expanded
    await page.getByTestId("research-model-switch").click();
    await expect(rec).toContainText("median of model cells within 15 km");
    await expect(rec).toContainText(/No model runs kept before /);
    await expect(page.getByTestId("research-model-words")).toBeVisible();
    await expect(page.getByTestId("research-model-words")).toHaveText("C-HARM: probability of conditions in the water, not toxin in seafood. No demonstrated skill at the shore; a high reading here is near-constant background, not an alarm.");
  });

  test("a port with no station within 60 km is an honest gap, said in words (briefing-trust Shelter Cove)", async ({ page }) => {
    await page.goto(`${OK}/research?place=port-215`);
    await expect(margin(page)).toContainText("No station is near Shelter Cove: the nearest is 79 km away, so this record says nothing about that water.");
  });

  test("no dataset: nothing is shown rather than an old or partial record (m3, readiness)", async ({ page }) => {
    await page.goto(`${NODATA}/research`);
    await expect(margin(page)).toContainText("The station records can't be read right now");
    await expect(margin(page)).toContainText("Nothing is shown rather than an old or partial record.");
  });
});

test.describe("My Coast", () => {
  test.beforeEach(async ({ context }) => seed(context));

  test("the gates: official records not verified, RAMP assessed, the test's highest result and count, no average (T8, T38, T41)", async ({ page }) => {
    await page.goto(`${OK}/coast?place=port-592`);
    const m = margin(page);
    await expect(m).toContainText("Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
    await expect(m.getByRole("button", { name: /^Official · CDFW RAMP · assessed Oct 30 · not verified\./ })).toBeVisible();
    await expect(m).toContainText("highest 9.8 ppm; none of 6 at or over the 30 ppm action level. 14 days ago.");
    await expect(m).not.toContainText("average 5.5");
  });

  test("the worksheet fills nothing in; a sourced reference enters only on a tap; every output is hypothetical and waits in words (T45, T47, E2)", async ({ page }) => {
    // v13: the worksheet is the trip workspace, My Coast's second state (?view=trip)
    await page.goto(`${OK}/coast?place=port-592&view=trip`);
    const ws = page.getByTestId("nx-coast");
    // v14: the empty trip's stub says what the trip still needs, and that a blank is a gap
    await expect(page.getByTestId("trip-note")).toContainText("A blank is a gap, never 0.");
    await expect(page.getByTestId("trip-note")).toBeVisible();
    const values = await page.getByTestId("trip-inputs").locator("input[type=text], input:not([type])").evaluateAll((els) => els.map((e) => (e as HTMLInputElement).value));
    expect(values.length).toBeGreaterThan(4);
    expect(values.every((v) => v === "")).toBe(true);
    await expect(ws).toContainText("Hypothetical, from your numbers.");
    await expect(page.getByTestId("trip-net")).toContainText("waiting for");
    // no figure reads as $0 while it waits (the waterfall's axis has a $0 tick: a scale, not a figure)
    for (const id of ["trip-hero", "trip-figures"]) expect(await page.getByTestId(id).innerText()).not.toMatch(/\$0\b/);
    await ws.getByRole("button", { name: /Use \$3\.51\/lb, a 2024 statewide average/ }).click();
    await expect(ws.getByRole("textbox", { name: "Price", exact: true })).toHaveValue("3.51");
    await expect(ws).toContainText("CDFW's 2024 statewide average");
    expect(await ws.innerText()).not.toMatch(FORBIDDEN);
  });
});
