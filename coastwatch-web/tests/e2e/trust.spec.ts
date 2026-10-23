import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * Scientific trust (v4), restored from the v2 suites that covered the map, notices and freshness
 * (map, m2, m5, p1, p2-currents, p3, coastal): official notices keep every warning and are never
 * presented as verified; old or failed data never reads as current; a value is the published cell or
 * pixel with its own product and time; a gap is said, never drawn as clear water or zero.
 */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const official = JSON.parse(readFileSync(path.join(FIX, manifest.official_url), "utf8"));
const charmPoints = JSON.parse(readFileSync(path.join(FIX, "verification/charm-points.json"), "utf8")).rows as { lat: number; lon: number; layer_id: string; valid_date: string; status: string; grid_value?: number }[];
const satPoints = JSON.parse(readFileSync(path.join(FIX, "verification/satellite-points.json"), "utf8")).rows as { layer_id: string; lat: number; lon: number; value: number | null; observed_date?: string; status: string }[];

type M = { loaded(): boolean; getSource(id: string): unknown; jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number } };
const map = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());

async function open(page: Page, url: string, at = "2026-10-08T20:00:00Z", base = OK) {
  await page.clock.setFixedTime(new Date(at));
  await page.goto(`${base}${url}`);
  await expect(page.getByTestId("notices-button")).toBeVisible();
  await map(page);
}

const statusOf = async (page: Page, id: string) => {
  await page.getByTestId("status-center").click();
  const s = await page.getByTestId("status-panel").getByTestId(`source-${id}`).getAttribute("data-state");
  await page.keyboard.press("Escape");
  return s;
};

test.describe("official notices: every record, every warning, never verified by default", () => {
  test("the panel lists every current notice with its agency wording, says why it is not verified, and keeps the agencies a tap or a call away", async ({ page }) => {
    await open(page, "/?layer=forecast");
    const button = page.getByTestId("notices-button");
    await expect(button).toHaveAttribute("data-state", "unverified");
    await button.click();
    const panel = page.getByTestId("notices-panel");
    // the fixture publishes no agency discrepancy, so every active record is a current notice
    expect(official.conflicts).toEqual([]);
    for (const r of official.registry.records.filter((x: { status: string }) => x.status === "active")) {
      await expect(panel.getByTestId("notices-current").getByTestId(`notice-${r.id}`)).toContainText(r.title);
    }
    await expect(page.getByTestId("notices-unverified")).toContainText("Not verified");
    await page.getByTestId("notices-reasons").locator("summary").click();
    await expect(page.getByTestId("notices-reasons")).toContainText("have not been checked by a person");
    await expect(page.getByTestId("missing-not-open")).toHaveText(/does not mean an area is open or that seafood is safe/);
    await expect(page.getByTestId("notices-hotlines").getByRole("link", { name: "(800) 553-4133" })).toHaveAttribute("href", "tel:+18005534133");
    await expect(page.getByTestId("notices-hotlines").getByRole("link", { name: "(831) 649-2883" })).toHaveAttribute("href", "tel:+18316492883");
    // the agency's own words, one tap deeper
    const first = panel.getByTestId("notices-current").locator("li").first();
    await first.locator("summary").click();
    await expect(first.getByTestId("official-text")).toBeVisible();
    // nothing here says an area is open or safe without negating it
    for (const line of (await panel.innerText()).split("\n").filter((l) => /\b(is|are) (open|safe)\b/i.test(l))) expect(line).toMatch(/\b(not|no)\b/i);
    // Escape closes and returns focus to the button
    await page.keyboard.press("Escape");
    await expect(panel).toHaveCount(0);
    await expect(button).toBeFocused();
  });

  test("past its expected end date, a notice is flagged as unconfirmed, never as lifted", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-11-03T20:00:00Z");
    await page.getByTestId("notices-button").click();
    const r = page.getByTestId("notices-panel").getByTestId("notice-cdph-2026-annual-mussel-quarantine");
    await expect(r.getByTestId("notice-critical")).toContainText("has passed, but no lifting notice is recorded");
    await expect(r.getByTestId("notice-critical")).toContainText("Treat as in effect until the agency confirms");
  });

  test("when the data stop updating, the panel says how old the review and the last check are", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-10-20T20:00:00Z", FAILED);
    await page.getByTestId("notices-button").click();
    await page.getByTestId("notices-reasons").locator("summary").click();
    await expect(page.getByTestId("notices-reasons")).toContainText(/Last review was \d+ days ago/);
    await expect(page.getByTestId("notices-reasons")).toContainText(/Official pages were last checked \d+ days ago/);
    await expect(page.getByTestId("notices-checked")).toContainText("Agency pages last checked");
  });

  test("on a 320 px phone the panel stays on screen and scrolls inside itself", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
    await page.goto(`${OK}/coast?place=port-550`);
    await page.getByTestId("notices-button").click();
    const box = (await page.getByTestId("notices-panel").boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(320);
    expect(box.y + box.height).toBeLessThanOrEqual(640);
    await ctx.close();
  });
});

test.describe("freshness: old or failed data never reads as current", () => {
  test("four days after the run: stale, its dates unchanged, said on the panel and in the status centre", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("layer-forecast")).toContainText("Run of Oct 8 · 4 days ago");
    await expect(page.getByTestId("stale-note").first()).toContainText("No newer C-HARM run has been published since Oct 8 (4 days)");
    await expect(page.getByTestId("stamp-forecast").first()).toContainText("issued Oct 8");
    expect(await statusOf(page, "model")).toBe("stale");
  });

  test("two weeks after the run: out of date, does not describe current conditions, never Ready", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-10-22T20:00:00Z");
    await expect(page.getByTestId("historical-note").first()).toContainText("does not describe current conditions");
    expect(await statusOf(page, "model")).not.toBe("ready");
  });

  test("a failed update keeps the last good run with its real dates, says so, and is never Ready", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-10-08T20:00:00Z", FAILED);
    await expect(page.getByTestId("provenance-status").first()).toContainText(/last update (failed|did not complete)/);
    await expect(page.getByTestId("stamp-forecast").first()).toContainText("issued Oct 8");
    expect(await statusOf(page, "model")).not.toBe("ready");
  });

  test("a failed update on an older run never claims NOAA published nothing newer (S3)", async ({ page }) => {
    await open(page, "/?layer=forecast", "2026-10-12T20:00:00Z", FAILED);
    const note = page.getByTestId("stale-note").first();
    await expect(note).toContainText("Neritic's newest copy is the run of Oct 8 (4 days old)");
    await expect(note).toContainText("could not reach NOAA");
    await expect(note).toContainText("a newer run may exist");
    await expect(note).not.toContainText("No newer C-HARM run has been published");
  });

  test("the forecast says what it is not: no toxin measurement, no closure decision, a low value is not safe; its issue date is inferred", async ({ page }) => {
    await open(page, "/?layer=forecast");
    const c = page.getByTestId("forecast-caveat").first();
    await expect(c).toContainText("not a closure decision");
    await expect(c).toContainText("A low probability does not mean an area is safe");
    await expect(c).toContainText("not risk levels");
    await expect(page.getByTestId("issued-inferred").first()).toContainText("NOAA does not publish it");
    await expect(page.getByTestId("timebar")).toContainText("(inferred)");
  });

  test("with nothing current, the map says so instead of opening on a bare seafloor", async ({ page }) => {
    await open(page, "/", "2026-10-14T20:00:00Z");
    await expect(page.getByTestId("layer-none")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("nothing-current").first()).toContainText("does not mean conditions are normal");
    await page.getByTestId("show-latest-forecast").first().click();
    await expect(page.getByTestId("layer-forecast")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("historical-note").or(page.getByTestId("stale-note")).first()).toBeVisible();
  });

  test("satellite: the view says its sensor and real observation dates, and a stale view is marked", async ({ page }) => {
    await open(page, "/?layer=none");
    await page.getByTestId("layer-satellite").click();
    await expect(page).toHaveURL(/layer=olci300/);
    await expect(page.getByTestId("stamp-satellite").first()).toContainText(/pixels observed \w{3} \d+–\w{3} \d+/);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap: M }).__cwMap.getSource("satellite") && !(window as unknown as { __cwMap: M }).__cwMap.getSource("forecast"));
    await page.clock.setFixedTime(new Date("2026-10-12T20:00:00Z"));
    await page.reload();
    await map(page);
    await expect(page.getByTestId("stale-note").or(page.getByTestId("historical-note")).first()).toBeVisible();
    expect(await statusOf(page, "satellite")).toBe("stale");
  });

  test("a fully clouded day says there was no clear observation and draws nothing", async ({ page }) => {
    await open(page, "/?layer=olci300:2026-10-07");
    await expect(page.getByTestId("sat-clouded").first()).toContainText("No clear observation on Oct 7");
    await expect(page.getByTestId("sat-clouded").first()).toContainText("not clear water");
    expect(await page.evaluate(() => !!(window as unknown as { __cwMap: M }).__cwMap.getSource("satellite"))).toBe(false);
  });

  test("currents: the hour and its age, stale days later, and the regions the radar did not see", async ({ page }) => {
    await open(page, "/?layer=currents");
    await expect(page.getByTestId("stamp-currents").first()).toContainText(/Currents, HF radar · observed .* \(\d+ h ago\)/);
    await expect(page.getByTestId("stamp-currents-gap").first()).toContainText("no data, not calm water");
    await expect(page.getByTestId("stamp-currents-gap").first()).not.toContainText("Monterey Bay");
    await page.getByTestId("time-mean").click();
    await expect(page).toHaveURL(/layer=currents(%3A|:)mean/);
    await expect(page.getByTestId("stamp-currents").first()).toContainText("24 h mean to");
    await expect(page.getByTestId("currents-note").first()).toContainText("tidal");
    await expect(page.getByTestId("currents-note").first()).toContainText("1 m/s ≈ 1.9 knots");
    await page.clock.setFixedTime(new Date("2026-10-11T20:00:00Z"));
    await page.goto(`${OK}/?layer=currents`);
    await map(page);
    await expect(page.getByTestId("stamp-currents").first()).toContainText("Stale:");
  });
});

test.describe("values: the published cell, with its product and time, reachable without a mouse", () => {
  test("a model cell's value equals the pipeline's verification point for every variable", async ({ page }) => {
    const row = (v: string) => charmPoints.find((r) => r.layer_id === `charm_${v}_lead1` && r.status === "pass" && r.grid_value != null && Math.abs(r.lat - 36.8) < 0.01 && Math.abs(r.lon + 121.95) < 0.01);
    for (const v of ["particulate_domoic", "pseudo_nitzschia", "cellular_domoic"]) {
      const r = row(v);
      expect(r, `verification row for ${v}`).toBeTruthy();
      await open(page, `/?layer=forecast&var=${v}&lead=1&sel=cell:${r!.lat},${r!.lon}`);
      await expect(page.getByTestId("cell-value")).toHaveText(`${Math.round(r!.grid_value! * 100)}%`);
    }
  });

  test("a v2 inspect link opens the point with its published values, its notices and why the list is not an all-clear", async ({ page }) => {
    await open(page, "/?var=particulate_domoic&lead=1&inspect=36.80,-121.95");
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "point");
    const f = page.getByTestId("point-forecast");
    await expect(f.getByTestId("point-value")).toHaveText(/^\d+%$/);
    await expect(f.getByTestId("point-when")).toContainText("C-HARM forecast, native 3 km · valid Oct 8");
    await expect(page.getByTestId("point-satellite").getByTestId("point-when")).toContainText(/Sentinel-3 300 m|VIIRS 750 m|No clear/);
    await expect(page.getByTestId("point-notices")).toContainText("not verified");
    await expect(page.getByTestId("point-missing-not-open")).toContainText("does not mean an area is open");
  });

  test("satellite and currents values are read at a clicked point, with sensor, date and an honest gap", async ({ page }) => {
    const r = satPoints.find((x) => x.layer_id === "olci300_chl_latest" && x.status === "pass" && x.value != null)!;
    await open(page, `/?layer=olci300&sel=point:${r.lat},${r.lon}`);
    const s = page.getByTestId("point-satellite");
    await expect(s.getByTestId("point-when")).toContainText(`observed`);
    await expect(s.getByTestId("point-value")).toBeVisible();
    await open(page, "/?layer=currents&sel=point:36.80,-122.10");
    await expect(page.getByTestId("point-currents").getByTestId("point-when")).toContainText(/HF radar, native 2 km · observed/);
    await expect(page.getByTestId("point-currents").getByTestId("point-knots")).toContainText(/≈ \d+\.\d kn/);
    await open(page, "/?layer=currents&sel=point:36.70,-121.65");
    await expect(page.getByTestId("point-currents").getByTestId("point-none")).toContainText("not calm water");
  });

  test("Monterey: official evidence comes first in the harbour panel; a port without data says so and invents nothing", async ({ page }) => {
    await open(page, "/?layer=forecast&sel=harbor:550");
    const kinds = await page.getByTestId("harbor-claims").locator("[data-kind]").evaluateAll((e) => e.map((x) => x.getAttribute("data-kind")));
    expect(kinds[0]).toBe("official");
    // San Diego lies outside the fixture's C-HARM subset: the model's absence is said (folded into the
    // station line, so it is not said twice), and no forecast value is invented
    await open(page, "/?layer=forecast&sel=harbor:880");
    await expect(page.getByTestId("harbor-claims")).toContainText(/NOAA's newest run ends \w+ \d+; nothing for/);
    // no model probability is invented for it (a percentile of a station's own record is not one)
    await expect(page.getByTestId("harbor-claims")).not.toContainText(/\d+% (chance|probability)|NOAA model: \d/);
    await expect(page.getByTestId("harbor-claims").locator('[data-kind="model"]')).toHaveCount(0);
  });
});
