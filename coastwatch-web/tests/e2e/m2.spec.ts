import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const intel = JSON.parse(readFileSync(path.join(FIX, manifest.port_intel_url), "utf8"));
const official = JSON.parse(readFileSync(path.join(FIX, manifest.official_url), "utf8"));
const OK = "http://localhost:3200";
const FAILED = "http://localhost:3201";

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-verification").first()).not.toHaveText(/Checking/);
}

/** Every rendered sentence that says "open" or "safe" must be negated. */
async function assertNoOpenOrSafeClaims(page: Page) {
  const text = await page.locator("body").innerText();
  const sentences = text.split(/(?<=[.!?])\s+|\n+/);
  const bad = sentences.filter((s) => /\b(safe|open|all clear)\b/i.test(s) && !/\b(not|no|never|does not|isn't)\b/i.test(s) && !/Whale.?safe/i.test(s));
  expect(bad, bad.join(" | ")).toEqual([]);
}

test.describe("official notices", () => {
  test("rail lists every active notice, the transcription is flagged as not verified", async ({ page }) => {
    await open(page, OK);
    const panel = page.getByTestId("official-status");
    await expect(panel.getByTestId("official-verification")).toHaveAttribute("data-state", "unverified");
    await expect(panel.getByTestId("official-not-verified-brief")).toContainText("not been checked by a person");
    await panel.getByTestId("official-show-all").click();
    const active = official.registry.records.filter((r: { status: string }) => r.status === "active");
    for (const r of active) await expect(panel.getByTestId(`notice-${r.id}`)).toBeVisible();
    // details show the agency's own words and sources
    await panel.getByTestId("notice-cdfw-rock-crab-commercial-40n").getByRole("button").click();
    await expect(panel.getByTestId("notice-cdfw-rock-crab-commercial-40n").getByTestId("official-text")).toContainText("40° 00.00’ N");
    await expect(panel.getByTestId("notice-cdfw-rock-crab-commercial-40n").getByTestId("notice-flags")).toContainText("does not state when this closure took effect");
    await expect(panel.getByTestId("missing-not-open")).toContainText("does not mean an area is open");
    await assertNoOpenOrSafeClaims(page);
  });

  test("past an expected end date, a notice is flagged as unconfirmed, not lifted", async ({ page }) => {
    await open(page, OK, "2026-11-03T20:00:00Z");
    const q = page.getByTestId("notice-cdph-2026-annual-mussel-quarantine").first();
    await expect(q.getByTestId("notice-critical")).toContainText("has passed, but no lifting notice is recorded");
  });

  test("official notice areas are drawn from verified geometry only", async ({ page }) => {
    await open(page, OK);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { getSource: (s: string) => unknown } }).__cwMap?.getSource("official"));
    const kinds = await page.evaluate(() => {
      const m = (window as unknown as { __cwMap: { getSource: (s: string) => { serialize: () => { data: { features: { properties: { kind: string } }[] } } } } }).__cwMap;
      return m.getSource("official").serialize().data.features.map((f) => f.properties.kind).sort();
    });
    expect(kinds).toEqual(official.geometry.features.map((f: { properties: { kind: string } }) => f.properties.kind).sort());
    expect(kinds).not.toContain("lat_band"); // latitude notices are drawn as limits, never shaded areas
  });

  test("when the data stops updating, official notices say how old the review and the last check are", async ({ page }) => {
    // failed-update dataset: records transcribed Oct 8, official pages last checked Oct 12
    await open(page, FAILED, "2026-10-20T20:00:00Z");
    const v = page.getByTestId("official-status").getByTestId("official-not-verified-brief");
    await expect(v).toContainText("Last review was 12 days ago");
    await expect(v).toContainText("last checked 8 days ago");
    await expect(page.getByTestId("source-failure-banner")).toContainText("Latest update failed");
  });
});

test.describe("ports", () => {
  test("Monterey Bay is the default region with its ports listed", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("region-monterey_bay")).toHaveAttribute("aria-pressed", "true");
    for (const code of [593, 592, 550]) await expect(page.getByTestId(`port-row-${code}`)).toBeVisible();
  });

  test("selecting Monterey shows notices first, then forecast values that match the pipeline", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&var=particulate_domoic&lead=1`);
    await page.getByTestId("port-row-550").click();
    const panel = page.getByTestId("port-panel");
    await expect(panel).toContainText("Monterey County");
    // order: official notices before the forecast
    const officialFirst = await panel.getByTestId("port-official").evaluate(
      (o, f) => !!(o.compareDocumentPosition(f as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
      await panel.getByTestId("port-forecast").elementHandle(),
    );
    expect(officialFirst).toBe(true);
    await expect(panel.getByTestId("official-not-verified-brief")).toBeVisible();
    const p = intel.ports.find((x: { port_code: number }) => x.port_code === 550);
    for (const rel of p.official_relations) await expect(panel.getByTestId(`notice-${rel.record_id}`)).toBeVisible();
    const lead1 = p.charm.leads.find((l: { lead_days: number }) => l.lead_days === 1);
    for (const v of ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic"]) {
      await expect(panel.getByTestId(`port-median-${v}`)).toHaveText(`${Math.round(lead1.variables[v].median * 100)}%`);
    }
    await expect(panel).toContainText("do not describe conditions at the dock");
    await expect(panel.getByTestId("port-history").getByTestId("timeseries")).toBeVisible();
    await expect(panel.getByTestId("chl-port-latest")).toHaveText(p.chlorophyll.latest.median.toFixed(2));
    await expect(page).toHaveURL(/port=550/);
    await assertNoOpenOrSafeClaims(page);
  });

  test("a port without history or satellite values says so instead of inventing them", async ({ page }) => {
    await open(page, `${OK}/?region=southern_california&port=880`);
    const panel = page.getByTestId("port-panel");
    await expect(panel.getByTestId("port-no-cells")).toBeVisible();
    await expect(panel.getByTestId("history-unavailable")).toBeVisible();
    await expect(panel.getByTestId("chl-port-unavailable")).toBeVisible();
  });

  test("region navigation moves the map", async ({ page }) => {
    await open(page, OK);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await page.getByTestId("region-north_coast").click();
    await page.waitForFunction(() => {
      const c = (window as unknown as { __cwMap: { getCenter: () => { lat: number } } }).__cwMap.getCenter();
      return c.lat > 40.4;
    });
    await expect(page.getByTestId("port-row-201")).toBeVisible();
  });
});

test.describe("point inspector", () => {
  test("official notices at a Monterey Bay point come before forecast values", async ({ page }) => {
    await open(page, `${OK}/?inspect=36.80,-121.95&lead=1`);
    const insp = page.getByTestId("inspector");
    await expect(insp.getByTestId("inspect-official")).toBeVisible();
    await expect(insp.getByTestId("inspect-notice-cdph-2026-sn26-019-anchovy-central-coast")).toContainText("between the notice's official latitudes");
    await expect(insp.getByTestId("official-verification")).toHaveAttribute("data-state", "unverified");
  });
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test("slide-up sheet shows the selected port and can be expanded", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&port=592&lead=1`);
    const sheet = page.getByTestId("mobile-sheet");
    await expect(sheet).toHaveAttribute("data-snap", "half");
    await expect(sheet.getByTestId("port-panel")).toContainText("Moss Landing");
    await page.getByTestId("sheet-handle").click();
    await expect(sheet).toHaveAttribute("data-snap", "full");
    await expect(page.getByTestId("detail-panel")).toHaveCount(0); // no desktop side panel on mobile
    await assertNoOpenOrSafeClaims(page);
  });
});
