# v3 · verified nearshore observations: CeNCOOS shore stations

Rank 2 of the v2 feasibility matrix (`docs/coastwatch/v2/01-sources.md`), now built:
`src/lib/shoreStations.ts`, the `shore` part of `/api/conditions`, nodes `cencoos:<station>:<variable>`.

## Probed 2026-10-16 (erddap.cencoos.org, HTTP 200 for all five)

| Station (dataset) | Where | Setting | Newest row | Serves (≤ 6 km) |
|---|---|---|---|---|
| Morro Bay (`edu_calpoly_marine_morro`) | Cal Poly T-pier, 35.37085 −120.8589 | in the harbour | 14:10Z, 10-min | Morro Bay (1.0 km) |
| Humboldt Bay (`edu_humboldt_humboldt`) | Humboldt Bay pier, 40.7775 −124.19652 | in the bay | 14:15Z | Eureka (3.6 km) |
| Trinidad (`edu_humboldt_tdp`) | Trinidad Head pier, 41.055 −124.14703 | open coast | 14:15Z | Trinidad (0.6 km) |
| Bodega (`bodega-bay-bml_wts`) | BML seawater intake, 38.31652 −123.0709 | open coast | 13:56Z | Bodega Bay (2.7 km) |
| Santa Cruz Wharf (`edu_ucsc_scwharf1`) | municipal wharf | open coast | **2026-08-19T15:55Z** (silent) | Santa Cruz (1.8 km): a gap |

Variables used: `sea_water_temperature`, `sea_water_practical_salinity`,
`mass_concentration_of_chlorophyll_in_sea_water` (fluorescence, µg/L), each with `_qc_agg`
(QARTOD: 1 PASS, 2 NOT_EVALUATED, 3 SUSPECT, 4 FAIL, 9 MISSING).

Licence (dataset metadata): "These data may be used and redistributed for free but they are not
intended for legal use, since they may contain inaccuracies. For use for publications please
reference … CeNCOOS and NOAA." Carried as a caveat on every node.

## Rules

- One request per harbour: the newest row at or before the clock (`time<=T&orderByMax("time")`,
  0.3–1 s). A fixed clock gives one URL, so the end-to-end replay is exact; a silent station
  answers with its last row, so its silence is said with a date ("has not reported since
  August 19, 2026") instead of a bare 404.
- A row older than 3 h is "not reporting", a gap; never shown as current.
- Only PASS values become readings; any other flag is a gap naming the flag ("flagged “failed” by
  its quality tests, so it is not shown"); NaN is neither.
- Water temperature reaches the conditions line only when NOAA's in-harbour tide-station reading
  is absent; salinity and chlorophyll fluorescence are evidence rows only. Chlorophyll is never on
  the conditions line and always carries "not toxin, and not the same measurement as satellite or
  laboratory chlorophyll".
- Recorded answers at the e2e clock: `coastwatch-web/tests/unit/fixtures/shore/` and five entries
  in `tests/conditions-fixtures.json` (keyed by sha1 of the route's URL; a unit test checks they match).

## Next candidates (not built)

CDIP wave buoys (rank 3; NDBC mirrors already read), CalHABMAP *Alexandrium* counts (the PSP gap
the species line names), WCOFS station forecasts as labelled model guidance.
