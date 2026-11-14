# Ocean model layers: NOAA WCOFS and the gap-filled chlorophyll estimate

Status, 2026-11-14: the server routes and the client contract exist. Explore draws them through Agent A's chart
layers. Nothing is published to `coastwatch-data` or `coastwatch-data-staging`: the app's server reads upstream when
a request arrives and caches the result, the same way `/explore/buoy` works.

Code: [`src/lib/ocean/`](../../../../coastwatch-web/src/lib/ocean/). Routes:
[`src/app/(neritic)/explore/model/`](../../../../coastwatch-web/src/app/(neritic)/explore/model/).
Tests: [`tests/unit/ocean/`](../../../../coastwatch-web/tests/unit/ocean/). Sources: `audit-wcofs.md` and
`audit-gapfill.md` (lead's audits, 2026-11-14).

## 1. Families and how they are kept apart

| Family | What it is | Track | Never call it |
|---|---|---|---|
| `currents.predicted` | WCOFS surface currents (model) | nowcast steps are **Estimates**; forecast steps are **Forecasts** | a measurement. HF radar is the measured current. |
| `sst.model` | WCOFS temperature at 0 m (model) | the same as `currents.predicted` | satellite SST. It is not skin temperature. |
| `chlorophyll.estimate` | NOAA DINEOF gap-filled Level-4 chlorophyll | **Estimates** | observed chlorophyll. It is not toxin and not a bloom forecast. |

Labels (`wcofsLabel` and `chlEstimateLabel` in `contract.ts`):

- `NOAA WCOFS model guidance (developmental) · cycle Nov 14 03z · valid Nov 14, 7:00 PM PST · a model forecast, not a measurement; not a bloom or toxin prediction`.
  Nowcast steps say "a model nowcast" instead of "a model forecast".
- `Gap-filled estimate (NOAA DINEOF) · Nov 2 · reconstructed where satellites saw nothing, not observed · no per-pixel uncertainty`

## 2. WCOFS

- **Endpoint:** CO-OPS THREDDS OPeNDAP, using the `regulargrid` files. The server sends one request per step:
  `https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/WCOFS/MODELS/YYYY/MM/DD/wcofs.t03z.YYYYMMDD.regulargrid.<step>.nc.dods?u_eastward[0][0][349:1:592][509:1:709],v_northward[…],temp[…],time`.
  The brackets are percent-encoded in the real request, because Tomcat rejects them raw.
- **Discovery:** the server reads `catalog.xml` for today's UTC date folder. If that folder has no `regulargrid`
  files, it tries yesterday's, then the day before. The newest cycle and its steps come from the dataset names.
- **Cadence:** one cycle a day at 03z. The files land about 1 h later. Each cycle has 8 nowcast steps (`n003`–`n024`,
  the 24 h up to the cycle time) and 24 forecast steps (`f003`–`f072`), 3 hours apart. Valid time is cycle + lead.
  For nowcast steps the lead is NNN − 24 h. Each packed step is checked against the file's own `time`
  (seconds since 2016-01-01). If they do not match, the step is refused.
- **Grid:** 0.04° on a regular lat/lon grid, no rotation needed. Currents are already eastward and northward.
  - California box: 32.38–42.10°N, 125.02–117.02°W, 244 × 201 cells.
  - Row 0 is the south edge (`lat_step` +0.04).
- **Packing:** uint16 little-endian with gzip, the same encoding as the published grids (`lib/grid` `loadGrid` reads it).
  - u and v: 0.001 m/s steps (offset −32).
  - Temperature: 0.001 °C steps (offset −5).
  - The fill value −99999, NaN and out-of-range values become `nodata` (65535). They never become zero.
- **Mask grid:**
  - 0 = land (no model value);
  - 1 = water;
  - 2 = water whose centre lies within 6 km of a land cell's edge. Draw these faded or hatched, as "coarse near shore".
- **Sizes, measured live 2026-11-14:** 589 kB of float32 per step from upstream in 0.4–0.9 s. Served gzip grids:
  u 30 kB, v 31 kB, temperature 37 kB, mask 1.2 kB. One step's currents are about 60 kB in the browser, and the
  whole cycle is about 3 MB. Steps are fetched only when they are first requested, never as a full cycle.
- **Caching:**
  - Index: 1 h in process; 15 min while today's cycle has not landed; 5 min after an upstream error. During an
    error the last good cycle is served with `fallback: true` and a reason. CDN: `s-maxage=900`.
  - Steps: an in-process LRU (48 steps) and Next's fetch cache (revalidate 1 day). Grid responses are
    `s-maxage=86400, immutable`, because a cycle's step never changes.
  - Timeouts: 6 s for the catalog, 10 s for a step.
  - When nothing is cached, an outage returns `status: "unavailable"` (grid route: 503 with a JSON reason).
- **States:**
  - `current`: the served steps reach now.
  - `expired`: the newest cycle's last step is before now. The response keeps the cycle and its valid interval
    and gives a reason.
  - `unavailable`: nothing could be read.
  - `default_step` is the newest step valid at or before now.

### Caveats (`CAVEATS`, carried in the index)

- NOAA labels WCOFS **developmental**, and its skill drops with lead time. The audit did not read the skill numbers in
  CO-OPS Tech Report 097, so none are quoted here.
- At about 4 km resolution it is unsuitable in bays, harbors and the surf zone. San Francisco Bay is only a few wet
  cells, and Morro, Humboldt and San Diego bays are not resolved at all.
- Temperature is the 0 m level of an interpolated grid. Forecast river inflow uses climatology.
- WCOFS has no biology. C-HARM uses WCOFS as an input, but that attribution belongs on C-HARM.

## 3. Gap-filled chlorophyll estimate (NOAA DINEOF)

- **Source:** NOAA CoastWatch ERDDAP `noaacwNPPN20S3ASCIDINEOF2kmDaily`.
  - Inputs: VIIRS S-NPP, NOAA-20 and Sentinel-3A OLCI.
  - Grid: 1/48° (about 2 km).
  - Units: mg m⁻³. Licence: CC0.
  - Method: Liu & Wang 2023, doi:10.1109/tgrs.2023.3271465.
- **Endpoint:** the newest day comes from `…/noaacwNPPN20S3ASCIDINEOF2kmDaily.csv?time[(last)]`. That day's field
  comes from `….dods?chlor_a[(<date>T12:00:00Z)][(0.0)][(42.1):1:(32.4)][(-125.5):1:(-117.0)]`.
  - The field is 466 × 409 cells and 766 kB, and loads in about 1.6 s.
  - Latitude descends: `lat_step` is negative.
- **Packing:** log10, in 0.0001-decade steps (`transform: "log10"`). The fill value −999 and values outside
  0.001–100 become `nodata`.
- **Mask grid:** 2 marks water within 2 km of a land cell's edge. Fade it there.
- **Caching:** the newest day is checked every 6 h, or 10 min after an error, when the last good day is served with a
  reason. Fields are kept in an in-process LRU and in Next's fetch cache for 1 day.
- **States:**
  - `stale`: the newest day is more than 4 days old. On 2026-11-14 the newest day is **2026-11-02**, 12 days old,
    because NOAA has published nothing newer. Its inputs include S-NPP VIIRS, which ended on 11-02. The index
    carries the real `date`, `age_days` and a reason.
  - `unavailable`: ERDDAP cannot be read.
- **Caveats:**
  - The product has no per-pixel uncertainty and no flag that separates observed cells from filled ones
    (`per_pixel_uncertainty: false`, `filled_flag: false`).
  - Within about 1 km of shore it reads about 3× low: the median log10 bias is −0.48 against same-day OLCI, which is
    not an independent check.
  - Never draw it over observed pixels.
- **Replaceable source:** `ChlEstimateSource` in `chlorophyll.ts`. Copernicus GlobColour gap-free 4 km (`CHL`,
  `CHL_uncertainty`, `flags` INTERPOLATED) can replace DINEOF once the owner creates a CMEMS account and approves the
  attribution text. It is not used now.
- PR #74's staging estimate and 8-day outlook are **not exposed**, because they failed validation (`audit-gapfill.md` §4).

## 4. Client contract (for Explore)

- `contract.ts`: `OceanFamily`, `OceanTrack`, `OceanStatus`, `OceanGrid` (compatible with `ValueGrid`), `WcofsIndex`,
  `WcofsStep`, `ChlEstimateIndex`, `MASK_*`, `wcofsLabel`, `chlEstimateLabel` and `cycleName`.
- `client.ts`: `loadWcofsIndex`, `loadChlEstimateIndex`, `loadWcofsCurrents(step)` (returns the FlowLayer's
  `CurrentField`), `loadOceanCodes(grid)` (uint16 codes for the FieldLayer) and `asValueGrid`.

## 5. Fixture mode and verification

- **Fixture mode:** set `CW_OCEAN_FIXTURE` to a directory with `fixture.json`, a URL-to-recorded-file map. Fixture mode
  is also on, using `tests/ocean-fixtures`, whenever `CW_CONDITIONS_FIXTURE` is set, so the e2e server never reaches
  the network. `CW_OCEAN_FIXTURE=off` turns it off.
  - The clock is `CW_OCEAN_FIXED_NOW`, or the recording time (2026-11-14T10:20Z) when that is unset.
  - Recorded: the WCOFS 2026-11-14 t03z catalog, steps `n024` and `f024`, and DINEOF 2026-11-02. Other steps
    answer as unavailable.
- **Value checks:**
  - WCOFS f024 (valid 2026-11-15 03Z) at 36.70°N 122.10°W gives u +0.049 m/s, v +0.094 m/s and 19.71 °C. This
    matches the audit and NOAA's NCSS point CSV (recorded). The packed grid keeps these values within 0.0005, both
    in unit tests and from the live route.
  - DINEOF 2026-11-02 at the cell nearest 36.70°N 122.10°W (36.698, −122.094) reads 0.330 mg m⁻³ in the recorded
    subset, and the packed grid keeps it. *Not yet checked against an ERDDAP point query: ERDDAP returned 502 for
    every request after the subset was recorded.*
- **Live, 2026-11-14 ~10:40Z (dev server):**
  - The WCOFS index found cycle 2026111403 with 32 of 32 steps, default `f006`.
  - Grids `f024` and `f051` came from upstream on first request (0.4–0.9 s) and from cache afterwards (5 ms).
  - The chlorophyll index answered `unavailable: timed out` while ERDDAP was down. That is the intended outage
    behaviour.
