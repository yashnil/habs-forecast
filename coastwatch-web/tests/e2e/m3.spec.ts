import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const read = (prefix: string) => JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith(prefix))!), "utf8"));
const obs = read("observations-");
const fish = read("fisheries-");
const OK = "http://localhost:3200";
const FAILED = "http://localhost:3201";
const M2 = "http://localhost:3203";
const NOW = "2026-10-08T20:00:00Z";
const DAY = 86_400_000;

async function open(page: Page, url: string, now = NOW) {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
}

/** Every rendered sentence that says "safe", "open" or "loss" must be negated. */
async function assertNoUnsafeClaims(page: Page) {
  const text = await page.locator("body").innerText();
  const sentences = text.split(/(?<=[.!?])\s+|\n+/);
  const bad = sentences.filter(
    (s) => /\b(safe|open|all clear|losse?s?)\b/i.test(s) && !/\b(not|no|never|does not|isn't)\b/i.test(s) && !/Whale.?safe/i.test(s),
  );
  expect(bad, bad.join(" | ")).toEqual([]);
}

const station = (id: string) => obs.stations.find((s: { station_id: string }) => s.station_id === id);
function measuredIn(id: string, variable: string, from: number) {
  const st = station(id);
  const s = st.series.find((x: { variable: string }) => x.variable === variable);
  return st.sample_times.filter((t: string, i: number) => Date.parse(t) >= from && s.values[i] != null).length;
}

test.describe("Bloom Intelligence", () => {
  test("defaults to Santa Cruz Wharf (Monterey Bay), official notices first, measurements and model separated", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const detail = page.getByTestId("station-detail");
    await expect(detail).toHaveAttribute("data-station", "HABs-SantaCruzWharf");
    // official notices come before any measurement in reading order
    const order = await page.evaluate(() => {
      const a = document.querySelector('[data-testid="bloom-official"]')!;
      const b = document.querySelector('[data-testid="station-detail"]')!;
      return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING;
    });
    expect(order).toBeTruthy();
    await expect(page.getByTestId("bloom-official").getByTestId("official-verification")).toHaveAttribute("data-state", "unverified");
    await expect(page.getByTestId("science-review-pending")).toBeVisible();
    await expect(page.getByTestId("latest-pDA")).toContainText("0.20 ng/mL");
    await expect(page.getByTestId("latest-pn_delicatissima")).toContainText("Not measured at this station");
    const chart = page.getByTestId("chart-pDA");
    await expect(chart).toHaveAttribute("data-measured", String(measuredIn("HABs-SantaCruzWharf", "pDA", Date.parse(NOW) - 365 * DAY)));
    await expect(page.getByTestId("chart-pDA-legend")).toContainText("reported 0 (not quantified, not absent)");
    await expect(page.getByTestId("measured-vs-model")).toContainText("never compared numerically");
    await expect(page.getByTestId("model-section").getByTestId("model-track")).toBeVisible();
    await expect(page.getByTestId("model-section")).toContainText("model, not a measurement");
    await assertNoUnsafeClaims(page);
  });

  test("a station with no recent toxin measurements says so instead of showing zero", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await page.getByTestId("station-row-HABs-MontereyWharf").click();
    await expect(page).toHaveURL(/station=HABs-MontereyWharf/);
    await expect(page.getByTestId("station-detail")).toHaveAttribute("data-station", "HABs-MontereyWharf");
    await expect(page.getByTestId("latest-pDA")).toContainText("2022");
    await expect(page.getByTestId("latest-pDA").getByTestId("freshness")).toHaveAttribute("data-state", "historical");
    await expect(page.getByTestId("chart-pDA-coverage")).toContainText("Not measured in this period. No measurement is not the same as no toxin");
    await expect(page.getByTestId("latest-temp")).toContainText("°C");
  });

  test("the time range widens every chart together", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await page.getByTestId("range-all").click();
    const first = Date.parse(station("HABs-SantaCruzWharf").sample_times[0]);
    await expect(page.getByTestId("chart-pDA")).toHaveAttribute("data-measured", String(measuredIn("HABs-SantaCruzWharf", "pDA", first)));
    await expect(page.getByTestId("chart-pn_seriata")).toHaveAttribute("data-measured", String(measuredIn("HABs-SantaCruzWharf", "pn_seriata", first)));
  });

  test("model history unavailable is explicit and neutral", async ({ page }) => {
    await open(page, `${OK}/bloom?station=HABs-ScrippsPier`);
    await expect(page.getByTestId("model-unavailable")).toContainText("says nothing about bloom conditions");
  });

  test("an old station is flagged as not current", async ({ page }) => {
    await open(page, `${OK}/bloom?station=HABs-TrinidadPier`);
    await expect(page.getByTestId("station-historical")).toContainText("do not describe current conditions");
  });

  test("keyboard: stations are buttons and charts can be read with arrow keys", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const row = page.getByTestId("station-row-HABs-StearnsWharf");
    await row.focus();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/station=HABs-StearnsWharf/);
    const svg = page.getByTestId("chart-pDA-svg");
    await svg.focus();
    await page.keyboard.press("End");
    const tip = page.getByTestId("chart-pDA-tooltip");
    await expect(tip).toBeVisible();
    await page.keyboard.press("ArrowLeft");
    await expect(tip).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(tip).toBeHidden();
  });

  test("unknown station ids fall back to the flagship station", async ({ page }) => {
    await open(page, `${OK}/bloom?station=../../etc/passwd`);
    await expect(page.getByTestId("station-detail")).toHaveAttribute("data-station", "HABs-SantaCruzWharf");
  });
});

test.describe("Fisheries & Economic Exposure", () => {
  test("tiles match the published data; real/nominal and tiers switch; port level unavailable", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await expect(page.getByTestId("exposure-definition")).toContainText("not a prediction of losses");
    await expect(page.getByTestId("port-level-unavailable")).toContainText("permission");
    const y = fish.years.at(-1);
    const t1 = fish.groups.filter((g: { tier: number }) => g.tier === 1);
    const total = (gs: typeof t1, k: string) => gs.reduce((s: number, g: { annual: { year: number }[] }) => s + ((g.annual.find((a) => a.year === y) as Record<string, number>)[k] ?? 0), 0);
    const fmt = (v: number) => `$${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`;
    await expect(page.getByTestId("tile-latest")).toContainText(fmt(total(t1, "dollars_real")));
    await page.getByTestId("tier-12").click();
    await expect(page.getByTestId("tile-latest")).toContainText(fmt(total(fish.groups, "dollars_real")));
    await page.getByTestId("dollars-nominal").click();
    await expect(page.getByTestId("tile-latest")).toContainText(fmt(total(fish.groups, "dollars_nominal")));
    await expect(page.getByTestId("withheld-row")).toContainText("not attributed");
    await expect(page.getByTestId("deflator")).toContainText("missing M10");
    await expect(page.getByTestId("data-through")).toContainText(String(y));
    await assertNoUnsafeClaims(page);
  });
});

test.describe("published M2 data (before any M3 artifact exists)", () => {
  test("bloom and fisheries show explicit unavailable states; the map still works", async ({ page }) => {
    await open(page, `${M2}/bloom`);
    await expect(page.getByTestId("bloom-unavailable")).toContainText("published before measured observations were added");
    await expect(page.getByTestId("bloom-unavailable")).toContainText("does not mean toxin is absent");
    await open(page, `${M2}/fisheries`);
    await expect(page.getByTestId("fisheries-unavailable")).toContainText("published before fisheries data were added");
    await open(page, `${M2}/`);
    await expect(page.getByTestId("official-status")).toBeVisible();
    await expect(page.getByTestId("region-ports")).toBeVisible();
  });
});

test.describe("failed update scenario", () => {
  test("four days later the bloom page still shows the data with their real dates", async ({ page }) => {
    await open(page, `${FAILED}/bloom`, "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("last-sample")).toContainText("Sep 30, 2026");
  });
});

test.describe("accessibility", () => {
  for (const [name, url] of [
    ["map", "/"],
    ["bloom", "/bloom"],
    ["fisheries", "/fisheries"],
    ["sources", "/sources"],
  ] as const) {
    test(`${name}: no serious or critical axe violations`, async ({ page }) => {
      await open(page, `${OK}${url}`);
      await page.waitForLoadState("networkidle");
      const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
      const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
    });
  }
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test("bloom: station picker works and nothing overflows horizontally", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await page.getByTestId("station-select").selectOption("HABs-ScrippsPier");
    await expect(page).toHaveURL(/station=HABs-ScrippsPier/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  });
  test("fisheries: nothing overflows horizontally", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await expect(page.getByTestId("fisheries-tiles")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  });
  test("map: legend is reachable on the map", async ({ page }) => {
    await open(page, `${OK}/`);
    await page.getByTestId("mobile-legend").locator("summary").click();
    await expect(page.getByTestId("mobile-legend").getByTestId("probability-legend")).toBeVisible();
  });
});
