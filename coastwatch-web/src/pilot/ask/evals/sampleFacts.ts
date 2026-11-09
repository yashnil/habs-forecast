import { PILOT_PORTS, type PortFact, type PortToday } from "../../contracts";

/**
 * EVAL RECORDS ONLY — sample PortToday records for the four pilot ports, used by the golden and
 * adversarial evals (harness.ts) and the unit tests. No page, route or answer reads this file: the
 * app's facts come from src/pilot/today/portToday.ts (`loadPortTodayFacts`). These records are
 * hand-written from the research notes in docs/coastwatch/pilot/plan.md §1 (as of 2026-11-08) so the
 * evals can pin every number and date; the real fact shape is evaluated by published.json.
 *
 * The C-HARM record is deliberately stale (run of 2026-11-02, window ended 2026-11-05), as it was
 * in the real feed on 2026-11-08, so the stale-data path is exercised.
 */
export const FIXTURE_GENERATED_AT = "2026-11-08T14:00:00Z";
/** The instant unit tests and evals treat as "now". */
export const FIXTURE_NOW = "2026-11-08T15:00:00Z";

type PortCfg = (typeof PILOT_PORTS)[number];

const CDFW_WHALE = "https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries";
const CDFW_ADVISORIES = "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories";
const CDFW_SALMON = "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Salmon";
const CDPH_SHELLFISH = "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx";
const CDPH_DA = "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Domoic-Acid.aspx";

function portFacts(p: PortCfg): PortFact[] {
  const hmb = p.place_id === "port-452";
  const zone = hmb ? { n: "3", label: "Point Arena to Pigeon Point" } : { n: "4", label: "Pigeon Point to Lopez Point" };
  const tideName = { "9413450": "Monterey", "9413745": "Santa Cruz", "9414131": "Pillar Point Harbor" }[p.tide_station] ?? p.tide_station;
  const f = (x: Omit<PortFact, "place_id">): PortFact => ({ place_id: p.place_id, ...x });
  return [
    f({
      id: "status:dungeness_commercial",
      kind: "status",
      severity: "red",
      headline: "Commercial Dungeness crab is delayed until at least 2027-01-01 because of humpback whale entanglements, not domoic acid.",
      detail: "Applies to Zones 1–5, Oregon border to Point Conception. CDFW's next risk assessment is due Dec 15–18.",
      basis: "official",
      verified: false,
      source: { name: "CDFW", url: CDFW_WHALE },
      as_of: "2026-11-05",
    }),
    f({
      id: "status:dungeness_recreational",
      kind: "status",
      severity: "amber",
      headline: "Recreational Dungeness crab opened 2026-11-07 with crab traps banned; hoop nets and snares only.",
      detail: "The trap ban is a whale-entanglement measure.",
      basis: "official",
      verified: false,
      source: { name: "CDFW", url: CDFW_WHALE },
      as_of: "2026-11-05",
    }),
    f({
      id: "status:anchovy_sardine",
      kind: "status",
      severity: "amber",
      headline: "Anchovy and sardine take is restricted in Monterey Bay, Pigeon Point to Point Lobos, since 2026-09-11 because of domoic acid.",
      detail: "CDFW and CDPH advise against eating recreationally caught anchovy and sardine from this area.",
      basis: "official",
      verified: false,
      source: { name: "CDFW", url: CDFW_ADVISORIES },
      as_of: "2026-11-03",
    }),
    f({
      id: "status:salmon_commercial",
      kind: "status",
      severity: "amber",
      headline: "Commercial salmon has reopened with trip limits.",
      detail: "Trip limits and dates are set by CDFW and PFMC in-season actions.",
      basis: "official",
      verified: false,
      source: { name: "CDFW", url: CDFW_SALMON },
      as_of: "2026-11-01",
    }),
    f({
      id: "status:bivalve_shellfish",
      kind: "status",
      severity: "amber",
      headline: "The 2026 annual mussel quarantine order ran to 2026-10-31; Neritic has found no CDPH notice confirming it has ended.",
      detail: "Until CDPH says otherwise, treat sport-harvested mussels as under quarantine.",
      basis: "official",
      verified: false,
      source: { name: "CDPH", url: CDPH_SHELLFISH },
      as_of: "2026-11-06",
    }),
    hmb
      ? f({
          id: "test:pillar-point-dungeness",
          kind: "test",
          severity: "neutral",
          headline: "Dungeness crab from Pillar Point tested below the action level on 2026-09-30.",
          value: "average 3.4 ppm domoic acid in viscera; action level 30 ppm",
          basis: "measured",
          verified: null,
          source: { name: "CDPH crab domoic acid results", url: CDPH_DA },
          as_of: "2026-09-30",
        })
      : f({
          id: "test:bma-g-dungeness",
          kind: "test",
          severity: "neutral",
          headline: "Dungeness crab from Monterey Bay (BMA G) tested below the action level on 2026-09-28.",
          value: "average 5.5 ppm domoic acid in viscera, highest sample 9 ppm; action level 30 ppm",
          basis: "measured",
          verified: null,
          source: { name: "CDPH crab domoic acid results", url: CDPH_DA },
          as_of: "2026-09-28",
        }),
    f({
      id: "test:anchovy-block-509",
      kind: "test",
      severity: "red",
      headline: "Northern anchovy viscera from Block 509 measured 1,200 ppm domoic acid on 2026-08-19.",
      value: "1,200 ppm; action level 20 ppm",
      basis: "measured",
      verified: null,
      source: { name: "CDPH seafood domoic acid results", url: CDPH_DA },
      as_of: "2026-08-19",
    }),
    f({
      id: `sea:nws-${p.nws_zone}-tomorrow`,
      kind: "sea",
      severity: "info",
      headline: `Tomorrow (Mon): NW wind 10 to 15 kt, seas 5 to 7 ft at 10 seconds (NWS zone ${p.nws_zone}).`,
      detail: "Wind building to 20 kt in the afternoon south of Point Pinos.",
      basis: "forecast",
      verified: null,
      source: { name: `NWS marine forecast ${p.nws_zone}`, url: `https://forecast.weather.gov/MapClick.php?zoneid=${p.nws_zone}` },
      as_of: "2026-11-08T11:00:00Z",
      valid_for: { from: "2026-11-09", to: "2026-11-09" },
    }),
    f({
      id: `weather:nws-${p.nws_zone}-today`,
      kind: "weather",
      severity: "info",
      headline: `Today (Sun): NW wind 5 to 10 kt, seas 4 to 6 ft, patchy morning fog (NWS zone ${p.nws_zone}).`,
      basis: "forecast",
      verified: null,
      source: { name: `NWS marine forecast ${p.nws_zone}`, url: `https://forecast.weather.gov/MapClick.php?zoneid=${p.nws_zone}` },
      as_of: "2026-11-08T11:00:00Z",
      valid_for: { from: "2026-11-08", to: "2026-11-08" },
    }),
    f({
      id: `tide:${p.tide_station}-tomorrow`,
      kind: "tide",
      severity: "info",
      headline: `Tide tomorrow at ${tideName}: high 5.4 ft at 7:12 AM, low 0.9 ft at 1:48 PM.`,
      detail: "Astronomical tide predictions; wind and swell change the actual water level.",
      basis: "forecast",
      verified: null,
      source: { name: `NOAA CO-OPS ${p.tide_station}`, url: `https://tidesandcurrents.noaa.gov/stationhome.html?id=${p.tide_station}` },
      as_of: "2026-11-08T00:00:00Z",
      valid_for: { from: "2026-11-09", to: "2026-11-09" },
    }),
    f({
      id: `observation:buoy-${p.buoy}`,
      kind: "observation",
      severity: "info",
      headline: `Buoy ${p.buoy} measured seas of 6.2 ft and NW wind of 12 kt at 5:50 AM.`,
      basis: "measured",
      verified: null,
      source: { name: `NDBC buoy ${p.buoy}`, url: `https://www.ndbc.noaa.gov/station_page.php?station=${p.buoy}` },
      as_of: "2026-11-08T13:50:00Z",
    }),
    f({
      id: "observation:sst",
      kind: "observation",
      severity: "info",
      headline: `Sea surface temperature near ${p.label} was 14.1 °C by satellite on 2026-11-06.`,
      basis: "measured",
      verified: null,
      source: { name: "NOAA CoastWatch SST", url: "https://coastwatch.noaa.gov/" },
      as_of: "2026-11-06",
    }),
    f({
      id: "bloom_forecast:charm",
      kind: "bloom_forecast",
      severity: "amber",
      headline: `C-HARM gave a 0.42 probability of a Pseudo-nitzschia bloom near ${p.label} for 2026-11-02 to 2026-11-05.`,
      detail: "C-HARM predicts water-column conditions, not toxin in seafood.",
      value: "0.42",
      basis: "forecast",
      verified: null,
      source: { name: "NOAA C-HARM", url: "https://coastwatch.noaa.gov/" },
      as_of: "2026-11-02",
      valid_for: { from: "2026-11-02", to: "2026-11-05" },
    }),
    f({
      id: "bloom_outlook:chl8d",
      kind: "bloom_outlook",
      severity: "info",
      headline: `The 8-day chlorophyll outlook near ${p.label} stays close to the seasonal normal through 2026-11-15.`,
      detail: "A conditions outlook for chlorophyll (climatology + AR(1)), not a toxin forecast.",
      basis: "outlook",
      verified: null,
      source: { name: "Neritic chlorophyll outlook", url: null },
      as_of: "2026-11-07",
      valid_for: { from: "2026-11-08", to: "2026-11-15" },
    }),
    f({
      id: `whale:ramp-zone-${zone.n}`,
      kind: "whale",
      severity: "red",
      headline: `${p.label} is in RAMP Zone ${zone.n} (${zone.label}): the commercial Dungeness season is delayed for whale entanglement risk.`,
      detail: "The next CDFW risk assessment is due Dec 15–18; depth limits or gear reductions may follow an opener.",
      basis: "official",
      verified: false,
      source: { name: "CDFW RAMP", url: CDFW_WHALE },
      as_of: "2026-11-05",
    }),
    f({
      id: "data_age:feed",
      kind: "data_age",
      severity: "info",
      headline: "Neritic's data feed last refreshed on 2026-11-08 at 05:40 PT.",
      basis: "derived",
      verified: null,
      source: { name: "Neritic data feed", url: null },
      as_of: "2026-11-08T13:40:00Z",
    }),
  ];
}

export function fixturePortToday(placeId: string): PortToday | null {
  const p = PILOT_PORTS.find((x) => x.place_id === placeId);
  if (!p) return null;
  return { place_id: p.place_id, label: p.label, generated_at: FIXTURE_GENERATED_AT, data_generated_at: FIXTURE_GENERATED_AT, facts: portFacts(p) };
}
