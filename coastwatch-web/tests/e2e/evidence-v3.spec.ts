import { expect, test } from "@playwright/test";
import { checkGraph } from "../../src/lib/evidenceCheck";
import type { EvidenceGraph } from "../../src/lib/harborEvidence";
import { OK } from "./ports";

/**
 * Harbor intelligence v3, end to end: the server's own graph (GET /api/evidence, live conditions
 * replayed from recordings, CW_FIXED_NOW) keeps every consistency rule for harbours along the
 * coast, each activity and the species with agency facts; a CeNCOOS shore station reaches the
 * harbour's line where NOAA has no thermistor; a station's own record and a species line reach the page
 * (v4's briefing: its claim rows, `claim-*`).
 */
const NOW = "2026-10-13T06:00:00Z";
const PLACES = ["port-201", "port-220", "port-231", "port-473", "port-550", "port-593", "port-606"];
const ASKS = ["for=all&species=any", "for=commercial&species=dungeness", "for=shellfish&species=mussel", "for=recreational&species=salmon"];

for (const place of PLACES) {
  test(`${place}: /api/evidence keeps every consistency rule for each activity and species`, async ({ request }) => {
    for (const ask of ASKS) {
      const g = (await (await request.get(`${OK}/api/evidence?place=${place}&${ask}`)).json()) as EvidenceGraph;
      expect(g.claims.length, ask).toBeGreaterThan(0);
      expect(checkGraph(g), `${place} ${ask}`).toEqual([]);
    }
  });
}

test("Morro Bay: the shore station's water temperature, with its QARTOD pass, where NOAA has no harbour thermistor", async ({ request }) => {
  const c = await (await request.get(`${OK}/api/conditions?lat=35.366&lon=-120.851`)).json();
  expect(c.shore.status).toBe("ok");
  expect(c.shore.station.id).toBe("edu_calpoly_marine_morro");
  const g = (await (await request.get(`${OK}/api/evidence?place=port-606`)).json()) as EvidenceGraph;
  const t = g.nodes.find((n) => n.id === "cencoos:edu_calpoly_marine_morro:water_temperature")!;
  expect(t.text).toBe("19.0 °C");
  expect(t.quality.flags).toEqual(["QARTOD: passed"]);
  // #66 review 2.1: the operator is credited, CeNCOOS as the publisher
  expect(t.source.name).toBe("Morro Bay shore station (Cal Poly San Luis Obispo), via CeNCOOS ERDDAP");
  expect(c.shore.source).toBe("Morro Bay shore station (Cal Poly San Luis Obispo), via CeNCOOS ERDDAP");
  expect(g.claims.find((x) => x.id === "conditions")!.supports).toContain(t.id);
});

test("Santa Cruz: the silent wharf sensor is a gap with the date of its last report", async ({ request }) => {
  const g = (await (await request.get(`${OK}/api/evidence?place=port-593`)).json()) as EvidenceGraph;
  expect(g.gaps.find((x) => x.id === "shore")?.reason).toBe("Santa Cruz Wharf shore station (1.8 km) has not reported since August 19, 2026.");
});

test("My Coast shows the species line and, where the sample is the station's newest, its own record", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-593&for=commercial&species=dungeness`);
  await expect(page.getByTestId("claim-hab")).not.toContainText("Reading");
  const species = page.getByTestId("claim-species");
  await expect(species).toContainText("For Dungeness crab, the agencies' pages name domoic acid");
  await expect(species).toContainText("describe the water, not the crab");
  // #66 review 3.2: an agency page in Neritic's unreviewed copy, never styled as an official notice
  await expect(species).toHaveAttribute("data-kind", "agency_page");
  await expect(species).toContainText("Agency page · Neritic's copy");
  await expect(page.getByTestId("claim-species-provenance")).toHaveText("Neritic's copy of the agencies' pages, read October 16, 2026; not reviewed by a person. The agency pages are the authority.");
  // the agencies' words are one tap away, each its own quote with its own page (#66 review 3.3)
  // v6: a statement is one line; it opens on tap (it may be behind "more statements")
  if (await page.getByTestId("matters-more").isVisible()) await page.getByTestId("matters-more").click();
  await page.getByTestId("claim-open-species").click();
  const agency = page.getByTestId("claim-species-agency");
  await expect(agency).not.toHaveAttribute("open", "");
  await expect(agency.locator("blockquote").first()).toBeHidden();
  await agency.locator("summary").click();
  await expect(agency.locator("summary")).toContainText(/Hide the agencies[’'] words/);
  const words = agency.getByTestId("claim-species-word");
  await expect(words).toHaveCount(2);
  await expect(words.nth(0).locator("blockquote")).toContainText("pre-season testing of Dungeness crab");
  await expect(words.nth(1).locator("blockquote")).toContainText("CDPH recommends consumers not eat the viscera");
  for (const i of [0, 1]) await expect(words.nth(i).locator("a")).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx");
  // Santa Cruz Wharf's September 30 count is its newest: placed among the wharf's quantified samples, zeros said apart (#66 review 1.1)
  await expect(page.getByTestId("claim-hab-history")).toHaveText(/^Compared with earlier years at Santa Cruz Wharf: higher than \d+ of the \d+ quantified (Aug|Sep)–(Oct|Nov) samples there in 2014–2025 \(a small sample\); another \d+ had no cells counted \(not necessarily none present\) and are not ranked\. /);
  await expect(page.getByTestId("claim-hab-history")).not.toContainText("record began");
  await ctx.close();
});

test("the mussel line quotes CDFW's sentence under CDFW, and leaves the quarantine's dates to the official notice", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=shellfish&species=mussel`);
  const species = page.getByTestId("claim-species");
  await expect(species).toContainText("paralytic shellfish toxins (saxitoxins) and domoic acid");
  await expect(species).not.toContainText("October 31");
  if (await page.getByTestId("matters-more").isVisible()) await page.getByTestId("matters-more").click();
  await page.getByTestId("claim-open-species").click();
  const agency = page.getByTestId("claim-species-agency");
  await agency.locator("summary").click();
  const word = agency.getByTestId("claim-species-word");
  await expect(word).toHaveCount(1);
  await expect(word).toHaveAttribute("data-agency", "CDFW");
  await expect(word.locator("a")).toHaveText(/^CDFW ocean fishing health advisories/);
  await expect(word.locator("a")).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
  await ctx.close();
});

test("a species with no named biotoxin test adds no line to the page (it stays in the API)", async ({ browser, request }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-593&for=recreational&species=salmon`);
  await expect(page.getByTestId("claim-hab")).not.toContainText("Reading");
  await expect(page.getByTestId("claim-species")).toHaveCount(0);
  expect(await page.locator('[data-testid="matters"] > li[data-kind]').count()).toBeLessThanOrEqual(6);
  const g = (await (await request.get(`${OK}/api/evidence?place=port-593&for=recreational&species=salmon`)).json()) as EvidenceGraph;
  expect(g.claims.find((c) => c.id === "species")?.kind).toBe("gap");
  await ctx.close();
});
