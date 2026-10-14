# CoastWatch: candidate data sources for the fishermen briefing, measured

Research note for the trip briefing (docs/coastwatch/trip-briefing/README.md). Checks ran 2026-10-14 between 04:55 and 05:45 UTC (the evening of 2026-10-13 in California), with User-Agent `CoastWatch-research/1.0 (+https://github.com/yashnil/habs-forecast)`.

This note builds on [ocean-sources.md](../research/ocean-sources.md), [nearshore-coverage.md](../research/nearshore-coverage.md), doc 27 / appendix A and [fishermen-first/04](../fishermen-first/04-source-feasibility.md). It does not repeat them. What is new here:
- a statewide 14-day measurement of nearshore (0–3 km) coverage and per-port age for OLCI and VIIRS;
- the NOAA-21/20 stall;
- DINEOF flags and coverage;
- NDBC `.ocean` relays of pier sensors;
- `tgftp` NWS text products and their robots status.

The measurement scripts and raw probes were kept outside the repository (one-off requests, not a pipeline); the numbers below are their results.

**Method (coverage).**
- **Reference geography:** ETOPO 2022 15″ water pixels within 3 km of the California shore. There are 39,675 points (12,826 at 0–1 km), split into 6 regions. This is the same `geo.npz` the doc-27 audit built.
- **Sampling:** griddap tiles along the strip.
  - OLCI: 0.25° tiles at stride 2 (about 600 m).
  - VIIRS: 0.5° tiles at native 750 m.
  - Requests were spaced 10 s apart, matching the coastwatch.noaa.gov Crawl-delay.
- **Valid:** the sampled pixel nearest each point is non-NaN.
- **Port age:** taken from any valid pixel within 5 km of the harbor point.
- **"Typical age":** the median, over the last 7 days of the window, of "days since the latest valid pixel near the port".
- **Limits:**
  - "Valid" means the upstream product published a value. It is not a quality judgement.
  - Stride-2 sampling gives a statistical sample, not a census.

---

## 1. Source table

| Source | Operator | Status (last data, checked 10-14 05Z) | Access / robots | Variable + units | Flags | CA nearshore coverage measured | Typical age | Kind | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Sentinel-3A OLCI 300 m daily sectors** `noaacwS3AOLCIchlaSector{CI,DI}Daily` (already used) | EUMETSAT/Copernicus, NOAA CoastWatch | **Active.** CI 2026-10-12T18:46Z, DI 10-12T17:54Z. One scene per day, rolling 90-day window from 07-13. Ready ~1.4 d after overpass | ERDDAP griddap. coastwatch.noaa.gov robots: `Disallow: /erddap/` + `Crawl-delay: 10` (already an open owner question, fishermen-first/04 §5). Licence requires "Contains modified Copernicus Sentinel data" | `chlor_a` mg m⁻³, OC4Me, `NRT S3A_OL_2_WFR` | **No flag variable.** Invalid = NaN (WFR L2 flags applied upstream, not exposed) | 09-29..10-12, 0–3 km: **mean 16.8 % of pixels valid per day** (median 15.7, range 0–48). 0–1 km 13.2 %; 1–3 km 18.5 %. Valid at least once in 14 d: 83 % (0–1 km 73 %). By region (daily mean / ever): North Coast 8.5 / 81; Mendocino-Sonoma 9.9 / 95; SF Bay area 16.8 / 73; Monterey Bay 9.4 / 69; **Central Coast 6.3 / 58**; SoCal 26.3 / 97. Most northern days were 0 %; one clear scene (10-12) supplied nearly all of it | At the latest scene (10-12), the newest valid pixel per point: same day 25 %, 1–2 d 10 %, 3–6 d 43 %, 7–13 d 6 %, none in 14 d 17 %. **Ports** (days with ≥1 valid px within 5 km / newest / typical age): Crescent City 1/14, 10-12, >7 d; Eureka 1/14, 10-12, >7 d; Fort Bragg 1/14, 10-12, >7 d; Bodega Bay 3/14, 10-12, 4 d; San Francisco 4/14, 10-10, 1 d; Santa Cruz 4/14, 10-10, 1 d; **Monterey 2/14, 10-05 (9 d old now)**, 4 d; Morro Bay 2/14, 10-10, 2 d; Santa Barbara 5/14, 10-09, 1 d; San Diego 5/14, 10-09, 1 d | Direct observation (L3, gaps) | **Keep (already adopted).** Best resolution, but on its own it is intermittent near shore |
| **Sentinel-3B OLCI sectors** `noaacwS3BOLCIchlaSector{CI,DI}Daily` (already used as gap fill) | same | **Stalled:** CI last 10-01, DI last 10-03; global 4 km last 10-04. S3A is unaffected | same | same | same | Not measured (no data in window) | n/a | Direct observation | **Keep as optional fill**, but gate on freshness. It currently adds nothing |
| **OLCI multi-day composites (West Coast)** | — | **Do not exist** on either NOAA ERDDAP. Only daily 300 m sectors, S3A/B global 4 km daily, and a Chesapeake OLCI-VIIRS 7-day merge (last 2025-05-27) | — | — | — | — | — | — | **Reject** (not available). CoastWatch already builds its own 7-day "latest clear view" with a per-pixel age (`multisensor.py`) |
| **NOAA-21 VIIRS 750 m daily merge** `noaacwN21VIIRSchlaSectorUSDaily` (PR #46 primary) | NOAA NESDIS STAR / CoastWatch | **Stalled: last 2026-10-05T18:36Z, 8.5 days old.** The kd490/kdpar siblings and the N21 ice products (10-06/10-08) are stalled too, so this looks like a STAR NRT processing outage, not a dataset retirement | ERDDAP, same host/robots as OLCI | `chlor_a` mg m⁻³ (OCI), fill −999 | `l2_flags` (int, per pixel), `qa_score`, `graphics` | 09-22..10-05, 0–3 km: **mean 51.5 % per day** (0–1 km 41.9 %, 1–3 km 56.1 %); ever 99 % | Newest valid at last day: same day 49 %, 1–2 d 31 %. Ports 10–14/14 days with data; typical age **0–1 d while flowing**. **Now 9 d old everywhere** | Direct observation | **Adopt (PR #46) but must ship with the stall gate.** The branch already says "3–8 days behind"; today it is 8.5 |
| **NOAA-20 VIIRS 750 m daily merge** `noaacwN20VIIRSchlaSectorUSDaily` (PR #46 fill) | same | **Stalled: last 10-05T18:38Z** (same outage) | same | same | same | Same window: 47.1 %/day; 0–1 km only **28.7 %** (ever 80 %) vs N21's 41.9 %. N21 ∪ N20 per pixel: **61.3 %/day, 0–1 km 50.0 %**, ever 99 % | Union: same day 58 %, 1–2 d 26 % | Direct observation | **Adopt as fill (PR #46).** It raises 0–3 km daily coverage by ~10 points. It gives no redundancy against this outage |
| **Same-day comparison** (7 common days, 09-29..10-05) | — | — | — | — | — | Daily valid, 0–1 / 1–3 km: OLCI S3A **14.6 / 19.9 %**; N21 **41.7 / 55.8 %**; N21∪N20 **49.8 / 64.4 %**. VIIRS daily merges (several passes, 750 m cells) report a value about 3× as often near shore as OLCI single-scene 300 m. A cell having a value does not mean the value is accurate; PR #46 already notes poorer agreement within 5 km | — | — | Use VIIRS as the frequent layer and OLCI as the sharp one; keep both labelled |
| **S-NPP VIIRS** `erdVHNchla1day/3day/8day` (current main) | NOAA CoastWatch WCN | 1day last 10-10; S-NPP data end **2026-11-02** | pfeg ERDDAP (robots `Disallow: /*?`) | `chla` mg m⁻³ | none exposed | doc 27 pilot only | ~4 d lag | Direct observation | **Retire after 11-02** (PR #46 does this) |
| **MODIS-Aqua West US** `erdMWchla{1,3,8,14}day` | NOAA CoastWatch WCN | Active (1day 10-12), "EXPERIMENTAL" in the title | pfeg ERDDAP | `chlorophyll` mg m⁻³, 0.0125° (~1.4 km) | none | not measured | ~1–2 d | Direct observation (experimental) | **Reject.** Coarser than VIIRS, labelled experimental, and from an end-of-life platform |
| **DINEOF gap-filled chl, 2 km, science quality** `noaacwNPPN20S3ASCIDINEOF2kmDaily` | NOAA STAR (S-NPP + NOAA-20 + S3A) | **Stale: last 2026-09-27** (16.6 d). `history` shows day 270 processed on Oct 8 (~11 d latency). Same for the Kd490/SPM siblings | pfeg redirects (302) to coastwatch.noaa.gov ERDDAP | `chlor_a` "DINEOF Gap-Filled", mg m⁻³ | **Only one variable. No per-pixel observed/filled flag, no reconstruction error field.** `l2_flag_names` lists the input flags only | Single day (09-27), 0–1 km **95.7 %** valid, 1–3 km 99.9 %, ≥3 km 100 %. It fills almost to the shore, and nothing marks which pixels were seen | 17 d now | **Gap-filled analysis** | **Reject for the briefing.** Too stale, too coarse (2 km), and it cannot separate observed from reconstructed pixels |
| **DINEOF NRT 9 km** `noaacwNPPN20VIIRSDINEOFDaily` / `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` | NOAA STAR | Active (10-12 / 10-11) | same | same, 9 km | no fill flag. Summary says "EXPERIMENTAL" | 9 km cells: useless within 3 km | ~1–2 d | Gap-filled analysis, experimental | **Reject.** Too coarse for anything nearshore |
| **CalHABMAP piers** `HABs-*` (already used) | SCCOOS / CalHABMAP labs | Last sample: Monterey Wharf 10-07 (no results posted yet), Cal Poly 10-04, Santa Monica & Scripps 10-05, Santa Cruz 09-30, Stearns 09-28, Morro 09-01, Newport 08-18, **Humboldt 04-07, Trinidad 04-17, Bodega 04-13, Tomales 03-03**. **Latest pDA:** Santa Cruz 09-30; SoCal piers **08-03 to 08-10** (~2 months lab lag); Humboldt/Trinidad April | erddap.sccoos.org: **no robots.txt (404)** | pDA/tDA/dDA ng mL⁻¹, PN cells L⁻¹, chl µg L⁻¹ | none (lab QA) | Points: 4 of the 10 ports have a pier within ~10 km with a sample in the last 14 d (Santa Cruz, Monterey, Santa Barbara, San Diego/Scripps) | weeks to months | Direct observation (lab) | **Keep.** Already used |
| **SCCOOS IFCB** `ca_ifcb_data` | SCCOOS | Files to 2026-10-13 (IFCB-158), 10-09 (IFCB-183); IFCB-151 last 06-23 | ERDDAP **file listing of raw `.adc/.hdr/.roi` binaries**, "Un-reviewed". Note: the full listing is 74 MB (fetched once by mistake; constrain by `name` next time) | raw images, no counts | none | 3 instruments, points | ~hours | Raw observation (needs a classifier) | **Reject now.** No classified PN counts are published. Ask SCCOOS for the classified feed (as doc 27 says) |
| **CeNCOOS / SCCOOS shore stations** (ERDDAP `erddap.cencoos.org`) | CeNCOOS, SCCOOS, Axiom | Live (minutes old): Humboldt Bay pier 04:45Z (13.5 °C, chl 8.6), Trinidad 04:45Z, Bodega BML 02:56Z, Morro Bay 04:40Z (19.1 °C, chl 2.0), Monterey Aquarium intake 10-13 22:59Z, Monterey Wharf 00:20Z (**temp QC=2 for all 202 rows**), Noyo Bay (temp only) 03:35Z, Stearns 04:54Z, Scripps 04:52Z (ECO chl QC=2 only), Santa Monica/Newport 05:00Z. **Stale:** Santa Cruz wharf `edu_ucsc_scwharf1` 08-19 (404 on query), Tiburon 09-24, MLML Monterey 2019 | ERDDAP tabledap. **robots `Disallow: /*?` and `*.csv*`** on erddap.cencoos.org and the Axiom backend | °C, µg L⁻¹ fluorometric chl, PSU | **IOOS QARTOD `*_qc_agg`** (1 pass, 2 not evaluated, 3 suspect, 4 fail) | Points: 8 of 10 ports have a live pier sensor within ~25 km (none at Crescent City, San Francisco open coast, or Santa Cruz now) | minutes–hours | Direct observation | **Adopt, but through NDBC (next row)** until the owner rules on ERDDAP robots. Use ERDDAP only if QARTOD flags are needed |
| **NDBC `.ocean` relays of the same pier sensors** | NOAA NDBC (relaying CeNCOOS/SCCOOS) | Live: HBXC1 Humboldt 04:45Z, TDPC1 Trinidad 04:45Z, BDXC1 Bodega 02:56Z, MBXC1 Morro Bay 04:40Z, CPXC1 Cal Poly 02:42Z, MLSC1 Moss Landing 04:55Z (temp only). **TIBC1 Tiburon stale since 09-24.** Values match ERDDAP exactly | `https://www.ndbc.noaa.gov/data/realtime2/{ID}.ocean`. NDBC robots blocks only named bots; already used by `/api/conditions` | `OTMP` °C, `CLCON` µg L⁻¹ chl (fluorometer), SAL, TURB; `MM` = missing | **No QC flags** (QARTOD is dropped in the relay). BDXC1 chl reads 0.09 µg L⁻¹ (plausible but low; check before trusting) | Points (as above) | minutes | Direct observation | **Adopt now** (§2) |
| **CDPH shellfish toxins** (DA/PSP in mussels; crab/seafood DA) | CDPH | Crab DA PDF LM 2026-10-12 17:23Z (ETag v7); Seafood DA PDF LM 10-12 15:10Z (v4). Monthly biotoxin StoryMap: April 2026. ArcGIS "CDPH Marine Biotoxin Monitoring Results" item is a PDF last modified 2024-01-05. **No feature layer or API with sample results** (ArcGIS Online search, 4 queries) | cdph.ca.gov robots allow these paths; PDFs support conditional GET | DA ppm (µg g⁻¹), PSP µg/100 g (in PDFs/StoryMaps) | n/a | Points by BMA/port | 10–14 d (crab PDF); ~6 months (monthly reports) | Direct observation (regulatory lab) | **Adopt later, human-curated only** (PDF change monitor plus a person's transcription; fishermen-first/04 §2.1). **Never** treat as a feed, and never auto-derive status |
| **WCOFS** (ROMS 4 km, 4D-Var) | NOAA NOS CO-OPS | Active: 2026-10-14 t03z cycle on S3 by 04:07Z. **OPeNDAP `opendap.co-ops.nos.noaa.gov` returned 502/504 on 3 tries (05:00–05:20Z)** | S3 (no robots; NODD open data). Files: stations.nowcast 17.6 MB, 2ds.f024 74 MB, regulargrid.f024 582 MB | temp °C, u/v m s⁻¹, zeta m, salt | none | 4 km grid; NOAA says bays/estuaries unresolved | daily cycle | **Model** (nowcast/forecast) | **Adopt later** (offshore context only; heavy; OPeNDAP unreliable). Not for the briefing |
| **CA ROMS (SCCOOS `roms_ncst/fcst`)** | SCCOOS / UCLA | **Ended 2022-12-01** | erddap.sccoos.org | — | — | — | — | Model | **Reject** (discontinued) |
| **CeNCOOS THREDDS models** | CeNCOOS | — | `thredds.cencoos.org/robots.txt` = **`Disallow: /`** | — | — | — | — | Model | **Reject** (robots) |
| **NWS coastal waters forecast, text** `tgftp…/coastal/pz/pzz{NNN}.txt` | NOAA NWS (EKA, MTR, LOX, SGX, MFR) | Live: PZZ545 issued 833 PM PDT 10-13 (LM 03:34Z), PZZ455 954 PM (04:55Z), PZZ740 128 PM (20:28Z), PZZ356 808 PM. Each carries an `Expires:` line | **tgftp.nws.noaa.gov has no robots.txt (404)**, so nothing is disallowed. Plain HTTPS, `Last-Modified` header. Public domain | Wind kt, seas ft, wave detail ft @ s, by period, out ~5 days; zone headlines | n/a | 25 CA zones (inner 0–10 nm, outer 10–60 nm, bays) | issued 2–4×/day | **Official forecast** | **Adopt now** (§2). This replaces `api.weather.gov` |
| **NWS marine weather messages (MWW)** `tgftp…/data/raw/wh/whus76.k{mtr,eka,lox,sgx}.mww.{…}.txt` | NOAA NWS | Live. MTR MWW 831 PM 10-13 shows SCA **expiring** for PZZ560/535 with VTEC `/O.EXP.KMTR.SC.Y.0123…/` | same host, no robots | Advisory/warning text plus VTEC | VTEC action codes | per zone | event-driven | Official warning | **Adopt with the forecast**, but only as the most recent message. The file holds just the latest MWW per office, so the active-headline line in the zone forecast is the more reliable state |
| forecast.weather.gov / www.weather.gov HTML | NOAA NWS | — | No robots.txt on forecast.weather.gov (404); HTML only | — | — | — | — | Forecast | Link only |
| `api.weather.gov` | NOAA NWS | — | **`Disallow: /`** (re-checked) | — | — | — | — | — | **Keep off** (owner ruling) |

---

## 2. Recommended for the fishermen briefing now

Three sources fit the brief: each needs no more than a day of engineering, and none touches the strictly validated pipeline manifest. All three are live, server-side fetches in the web route that already serves `/api/conditions` (NDBC, CO-OPS). That route is not a pipeline artifact, so **no manifest field is added**. Each value is shown with its source, its observation or issue time, and its distance from the port.

1. **NWS coastal waters forecast and marine messages from `tgftp`, replacing the disabled `api.weather.gov` call.**
   - **Fetch:**
     - `GET https://tgftp.nws.noaa.gov/data/forecasts/marine/coastal/pz/pzz{NNN}.txt` for the port's zones:
       - Crescent City: 450 (356 just north)
       - Eureka: 450
       - Fort Bragg: 455
       - Bodega: 540
       - SF: 545/530
       - Santa Cruz/Monterey: 535/560
       - Morro Bay: 670
       - Santa Barbara: 650
       - San Diego: 740
     - This mapping is from zone names. **Superseded:** the shipped table (`coastwatch-web/src/lib/nwsText.ts`) assigns each port and harvest site the inner (0–10 NM) zone that contains it, checked against each zone file's own description (Morro Bay PZZ645, not the outer PZZ670; San Diego PZZ740).
     - Optionally also `GET https://tgftp.nws.noaa.gov/data/raw/wh/whus76.k{off}.mww.{off}.txt` with `off` ∈ {eka, mtr, lox, sgx}.
   - **Handling:**
     - Use conditional GET (`If-Modified-Since`).
     - Respect the `Expires:` line.
     - Show the text verbatim with its issue time.
     - Do not parse numbers out of the text.
   - **Robots:** none on the host. Confirm with the owner that this satisfies the 2026-10-13 ruling. It should, because the ruling concerned a host whose robots.txt is `Disallow: /`.
   - **Outcome:** this restores the weather line the fishermen-first contract dropped.

2. **Live pier water temperature and chlorophyll fluorescence via NDBC `.ocean` relays.**
   - **Fetch:** `GET https://www.ndbc.noaa.gov/data/realtime2/{ID}.ocean`, using the existing NDBC parser pattern ("latest non-MM per field").
   - **Station per port:**

     | Port | Station |
     |---|---|
     | Eureka | HBXC1 |
     | Crescent City | TDPC1 (Trinidad, ~75 km; label the distance) |
     | Bodega Bay | BDXC1 |
     | Monterey | MLSC1 (temp only) |
     | Morro Bay | MBXC1 |
     | Avila/Port San Luis | CPXC1 |

   - **Rules:**
     - Hide a station after 6 h without data (TIBC1 is already stale).
     - Label chl as "chlorophyll fluorescence at the pier (plant biomass, not toxin)".
   - **Where QARTOD flags matter:** for Santa Barbara, San Diego and Monterey Wharf, which are not relayed by NDBC, the same values with flags exist on `erddap.cencoos.org` (`stearns-wharf-automated-shore-st-1`, `scripps-pier-automated-shore-sta-1`, `monterey-wharf-real-time-samplin`). That host's robots.txt disallows `/*?`, so it waits for the owner's ERDDAP ruling.

3. **VIIRS NOAA-21/NOAA-20 (PR #46), shipped with a visible stall state.** This is a pipeline change already in review, not a new source; nothing new is required for the briefing.
   - **Coverage:** measured here, it is the only chlorophyll layer that reaches a typical port almost daily: 11–14 of 14 days with a valid pixel within 5 km of every port, against 1–5 of 14 for OLCI.
   - **Stall:** both satellites have been frozen at 10-05 since then, so the briefing must say "VIIRS last observed Oct 5 (9 days ago)" rather than present it as current. Use the existing `outcome` / `last_success_at` fields; no schema change.
   - **Fetch (unchanged from PR #46):** `https://coastwatch.noaa.gov/erddap/griddap/noaacwN21VIIRSchlaSectorUSDaily.nc?chlor_a[(t0):1:(t1)][(0.0)][(lat0):1:(lat1)][(lon0):1:(lon1)]`, then the same with N20 for the fill.

Not recommended now: DINEOF (stale, unflagged), WCOFS (heavy, OPeNDAP down, 4 km), IFCB (raw only), CDPH (PDF; human curation only), MODIS-Aqua (experimental).

---

## 3. Is multi-source chlorophyll compositing legitimate, and how is provenance kept?

**Per-pixel selection: yes, and it is what CoastWatch already does.** In `multisensor.py` and PR #46, each pixel shows exactly one sensor's own value. Under that rule:
- choose by a documented rule (newest valid; primary sensor unless the other is clearly newer);
- keep native grids;
- never average, blend or bias-correct;
- publish the per-pixel **sensor, platform and observation date** (the age grid and platform grid);
- state the measured inter-sensor offsets: N21 about 4 % below S-NPP and N20 about 10 % below (PR #46); OLCI OC4Me vs VIIRS OCI are not harmonised for this coast.

This is a "latest clear view", and it is honest because a reader can always tell what was observed, by which sensor, and when.

**Averaging or merging sensors is not legitimate without a validated harmonisation.**
- NOAA itself publishes an OLCI-VIIRS "weighted merge" only for Chesapeake Bay, it is stale (last 2025-05-27), and no such merge exists for the West Coast.
- The measured nearshore valid rates differ about 3× between the sensors (OLCI 14.6 % vs N21 41.7 % at 0–1 km on the same days). That shows their masking and adjacency handling differ, which is exactly where a merge would inject bias.

**Gap-filled (DINEOF) analysis is legitimate science, but CoastWatch can't present it as a separate layer today.**
- DINEOF (EOF-based reconstruction, Beckers & Rixen 2003; Alvera-Azcárate et al.) is accepted, and C-HARM itself uses it internally.
- To display it as an analysis distinct from observations, a product must:
  1. carry a per-pixel observed/filled flag and ideally an error field;
  2. be recent;
  3. be visually separate, e.g. "estimated between observations", hatched, never in the observed colour ramp.
- NOAA's DINEOF products fail (1): one `chlor_a` variable, no flag, no error.
- The 2 km version also fails (2): 16.6 days old, ~11-day production latency.
- It fills 95.7 % of 0–1 km pixels, so most nearshore values shown would be reconstructions that look like measurements.
- **Verdict: do not display it.** If it is revisited, the only acceptable form is a separate, opt-in "model-estimated chlorophyll" layer. It would need a fill flag, derived by CoastWatch from its own published observation grids (same day and sensor) only if the grids are provably the inputs; otherwise the layer is not shown.

**Never fill missing toxin probabilities.** C-HARM's nearshore NaNs (12.6 % valid within 1 km) must stay "no model value here". No chlorophyll source (observed, composited or gap-filled) may stand in for domoic acid or Pseudo-nitzschia risk, and no interpolation or extrapolation of C-HARM to the shore is acceptable (doc 27 §2: Frolov 2013, Anderson 2016).

---

## 4. Findings that need action outside this note

1. **NOAA-21 and NOAA-20 chlorophyll have been stalled since 2026-10-05**, with S-NPP ending 11-02. If STAR does not recover before then, CoastWatch will have no fresh VIIRS at all. Recommend a dataset stall alarm (no new `time` for more than 3 days while the server answers) and a note on PR #46.
2. **S-3B OLCI has been stalled since 10-01/10-03.** S-3A is currently the only fresh 300 m source.
3. **Robots:**
   - coastwatch.noaa.gov (`/erddap/` disallowed, Crawl-delay 10), pfeg and cencoos (`/*?` disallowed) remain the open owner question.
   - erddap.sccoos.org, tgftp.nws.noaa.gov, NDBC (named bots only), the NOAA OFS S3 bucket, and cdph.ca.gov (for these paths) do not restrict these fetches.
4. **The WCOFS OPeNDAP endpoint failed** (502/504) for the whole check. S3 is the only reliable path.
