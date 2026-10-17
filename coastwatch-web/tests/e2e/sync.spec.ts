import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Map, panel, chart, evidence and URL stay in step (v4): what is selected on the map is what the panel
 * describes, the chart's newest point is the station's own newest sample with its own date, a model
 * cell's outline is the published 3 km footprint and its value follows the chosen day, and a claim's
 * "Show on map" selects the very node it rests on.
 */
const NOW = "2026-10-08T20:00:00Z";
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const obs = JSON.parse(readFileSync(path.join(FIX, manifest.observations_url), "utf8"));
type Station = { station_id: string; name: string; lat: number; lon: number; sample_times: string[]; series: { variable: string; values: (number | null)[] }[] };
const scw: Station = obs.stations.find((s: Station) => s.station_id === "HABs-SantaCruzWharf");
type M = { loaded(): boolean; jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number }; getSource(id: string): { serialize(): { data: GeoJSON.FeatureCollection } } | undefined };
const W = (page: Page) => page.evaluate.bind(page);

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}${url}`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());
}
const highlight = (page: Page) => W(page)(() => (window as unknown as { __cwMap: M }).__cwMap.getSource("v4-highlight")?.serialize().data.features ?? []);
const newestOf = (s: Station, variable: string) => {
  const v = s.series.find((x) => x.variable === variable)!;
  for (let i = v.values.length - 1; i >= 0; i--) if (v.values[i] != null) return s.sample_times[i].slice(0, 10);
  return null;
};
const label = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });

test("a station chosen on the map: the panel, the ring on the map, the chart's newest point and the URL describe the same station and day", async ({ page }) => {
  await open(page, "/?layer=forecast");
  await page.evaluate(([lon, lat]) => (window as unknown as { __cwMap: M }).__cwMap.jumpTo({ center: [lon, lat], zoom: 11.5 }), [scw.lon, scw.lat]);
  await page.waitForTimeout(600);
  const box = (await page.getByTestId("map").boundingBox())!;
  const pt = await page.evaluate(([lon, lat]) => (window as unknown as { __cwMap: M }).__cwMap.project([lon, lat]), [scw.lon, scw.lat]);
  await page.mouse.click(box.x + pt.x, box.y + pt.y);
  await expect(page.getByTestId("inspect-title")).toHaveText(scw.name);
  await expect(page).toHaveURL(/sel=station(%3A|:)HABs-SantaCruzWharf/);
  const ring = (await highlight(page)).find((f) => f.properties?.kind === "station")!;
  expect((ring.geometry as GeoJSON.Point).coordinates).toEqual([scw.lon, scw.lat]);
  // the newest pDA sample, with its own date, in the latest list and at the chart's newest point
  const newest = newestOf(scw, "pDA")!;
  await expect(page.getByTestId("station-latest")).toContainText(label(newest));
  await expect(page.getByTestId("series-readout").first()).toContainText(label(newest).replace(/^\w+, /, ""));
});

test("a model cell: its outline is the published 3 km footprint containing the point, and its value follows the chosen day", async ({ page }) => {
  await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&sel=cell:36.8,-121.95");
  await expect(page.getByTestId("cell-value")).toHaveText(/^\d+%$/);
  const cell = (await highlight(page)).find((f) => f.properties?.kind === "cell")!;
  const ring = (cell.geometry as GeoJSON.Polygon).coordinates[0];
  const lons = ring.map((c) => c[0]), lats = ring.map((c) => c[1]);
  const [w, e, s, n] = [Math.min(...lons), Math.max(...lons), Math.min(...lats), Math.max(...lats)];
  // the point lies inside, and the footprint is one ~3 km cell (not a smoothed blob)
  expect(-121.95).toBeGreaterThanOrEqual(w);
  expect(-121.95).toBeLessThanOrEqual(e);
  expect(36.8).toBeGreaterThanOrEqual(s);
  expect(36.8).toBeLessThanOrEqual(n);
  const kmNS = (n - s) * 111.32;
  expect(kmNS).toBeGreaterThan(2.5);
  expect(kmNS).toBeLessThan(3.5);
  // the value's day is the timebar's day; another day changes both, the outline stays
  await expect(page.getByTestId("inspect-panel")).toContainText("for Thu, Oct 8");
  await page.getByTestId("time-lead2").click();
  await expect(page.getByTestId("inspect-panel")).toContainText("for Fri, Oct 9");
  await expect(page).toHaveURL(/lead=2/);
  const again = (await highlight(page)).find((f) => f.properties?.kind === "cell")!;
  expect(again.geometry).toEqual(cell.geometry);
});

test("a harbour's measured claim: 'Show on map' selects the station it rests on, whose newest sample has the claim's date", async ({ page }) => {
  await open(page, "/?layer=forecast&sel=harbor:593");
  const measured = page.getByTestId("harbor-claims").locator('[data-kind="measured"]').first();
  await expect(measured).toBeVisible();
  const supports = (await measured.getAttribute("data-supports")) ?? "";
  const sid = supports.split(" ").find((x) => x.startsWith("station:"))?.split(":")[1] ?? supports.split(" ")[0].split(":")[1];
  const headline = await measured.locator("[data-headline]").innerText();
  await measured.getByRole("button", { name: "Show on map" }).click();
  await expect(page).toHaveURL(new RegExp(`sel=station(%3A|:)${sid}`));
  const st: Station = obs.stations.find((s: Station) => s.station_id === sid);
  await expect(page.getByTestId("inspect-title")).toHaveText(st.name);
  // the claim's date is one of the station's own sample dates (a measured claim is never re-dated)
  const day = /\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2})\b/.exec(headline)?.[1];
  expect(day, "the claim states its sample date").toBeTruthy();
  expect(st.sample_times.map((t) => label(t.slice(0, 10)).replace(/^\w+, /, ""))).toContain(day);
  const ring = (await highlight(page)).find((f) => f.properties?.kind === "station")!;
  expect((ring.geometry as GeoJSON.Point).coordinates).toEqual([st.lon, st.lat]);
});

test("Back walks the selections in order and the map follows", async ({ page }) => {
  await open(page, "/?layer=forecast&sel=harbor:550");
  await page.getByTestId("harbor-stations").getByRole("button").first().click();
  await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "station");
  await page.getByTestId("inspect-back").click();
  await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
  await expect(page).toHaveURL(/sel=harbor(%3A|:)550/);
  const kinds = (await highlight(page)).map((f) => f.properties?.kind);
  expect(kinds).toContain("harbor");
  expect(kinds).not.toContain("station");
});
