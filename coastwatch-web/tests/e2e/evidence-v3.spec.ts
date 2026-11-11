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
