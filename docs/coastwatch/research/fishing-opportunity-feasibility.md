# Can CoastWatch responsibly recommend fishing areas or dates? A feasibility study

*Prepared 2026-10-12. This is research only: no data access was requested, nobody was contacted, and nothing was written to any repository. Every claim has a URL. Anything not verified in this session is marked **[UNVERIFIED]**.*

---

## 0. Bottom line

| Question | Verdict |
|---|---|
| (a) Show users fishing recommendations ("fish here / on this date") now | **NO-GO** |
| (b) An offline research prototype built only on public data | **GO, narrowly.** It must be a *fishery-independent habitat/density* model for trawl-survey groundfish (sablefish first). It is not a catch-opportunity model. |
| (c) A "seasonal context" feature showing public historical landings by port area and month (descriptive only) | **Conditional GO.** It must keep confidentiality suppression, label the data as landings and not catch location or opportunity, keep it separate from HAB/forecast content, and confirm reuse terms with CDFW first. |

**Why (a) is a no-go.** No public dataset pairs California **catch with fishing effort at a fine spatial and temporal scale** (trip, set or block by day or week). The data that does exist is all confidential:
- CDFW landing receipts
- logbooks (CPFV, trawl, squid, lobster, Dungeness crab bi-weekly reports and electronic-monitoring tracks)
- PacFIN confidential tables
- WCGOP observer data

Releasing it requires a confidentiality agreement with a postsecondary research institution or an agency (Fish & Game Code §8022; MSA §402(b)). CoastWatch also has no way to account for closures, quotas, gear restrictions, bycatch, or whale-entanglement risk. A user-facing tool that steers effort would be scientifically unsupported and could do ecological harm.

---

## 1. Verified API test: NWFSC West Coast Groundfish Bottom Trawl Survey (FRAM Data Warehouse)

**The URL pattern in the brief (`/data/api/v1/...`) is out of date.** It returned **HTTP 404** (`Cannot GET /api/v1/source/trawl.catch_fact/selection.json`). The official `nwfscSurvey` R package notes that the warehouse moved in April 2025 to `https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/`. The old `/data/api/v1/` paths are commented as "pre-april 2025 site" ([get_url.R](https://raw.githubusercontent.com/pfmc-assessments/nwfscSurvey/main/R/get_url.R)).

**Working requests, run 2026-10-12, HTTP 200, JSON array of objects:**

```
# Haul (effort / station) table: one row per tow, includes zero-catch tows
https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/trawl.operation_haul_fact/selection.json?filters=project=Groundfish%20Slope%20and%20Shelf%20Combination%20Survey,year>=2024,year<=2024&variables=trawl_id,datetime_utc_iso,sampling_start_hhmmss,latitude_dd,longitude_dd,depth_hi_prec_m,temperature_at_gear_c_der,temperature_at_surface_c_der,salinity_at_gear_psu_der,area_swept_ha_der,performance

# Catch table: positive tows only for the species filter
https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/trawl.catch_fact/selection.json?filters=project=Groundfish%20Slope%20and%20Shelf%20Combination%20Survey,field_identified_taxonomy_dim$common_name=sablefish,date_dim$year>=2019,date_dim$year<=2019&variables=common_name,scientific_name,project,year,vessel,tow,total_catch_numbers,total_catch_wt_kg,trawl_id,cpue_kg_per_ha_der,depth_m
```

**Fields actually returned:**
- **`trawl.operation_haul_fact`:** `project, year, vessel, pass, tow, datetime_utc_iso, sampling_start_hhmmss, depth_hi_prec_m, longitude_dd, latitude_dd, area_swept_ha_der, trawl_id, performance, station_invalid, temperature_at_gear_c_der, temperature_at_surface_c_der, salinity_at_gear_psu_der`.
  - `datetime_utc_iso` holds the date only (time part `00:00:00Z`). Time of day is in `sampling_start_hhmmss`.
  - Other variables the package requests include gear start/end lat/lon, net width/height, door width and turbidity ([pull_haul.R](https://raw.githubusercontent.com/pfmc-assessments/nwfscSurvey/main/R/pull_haul.R)).
- **`trawl.catch_fact`:** `common_name, scientific_name, project, year, vessel, tow, total_catch_numbers, total_catch_wt_kg, trawl_id, cpue_kg_per_ha_der, depth_m`.
  - Lat/lon/date were **not** returned when requested from the catch table. Join on `trawl_id` to the haul table, as `nwfscSurvey::pull_catch` does ([pull_catch.R](https://raw.githubusercontent.com/pfmc-assessments/nwfscSurvey/main/R/pull_catch.R)).
  - The catch table holds **positive tows only**. Zeros come from the haul table, which is what the package's join does.

**Observed volumes:**

| Measure | Value |
|---|---|
| Hauls, 2003–2025 | 15,133 |
| Hauls per year, typical | ~640–770 |
| 2013 | 497 hauls (reduced) |
| 2019 | 369 hauls (only 2 vessels: *Excalibur* pass 2, *Last Straw* pass 1) |
| 2020 | **no survey** |
| 2025 | 719 hauls (already online) |
| California (lat < 42°N), "Satisfactory" hauls, 2003–2025 | 7,846 |
| Positive tows, coast-wide, 2003–2024 | Dover sole 11,626; shortspine thornyhead 7,081; lingcod 4,597; **California halibut 106** (too few) |
| Sablefish positive tows, 2003–2025 | 9,695 coast-wide; **4,733 in California** |
| Ranges seen in 2024 | lat 32.39–48.44°N; depth 14.7–1268 m |
| Bottom temperature present, 2024 | 627 of 682 hauls |

The year-by-year haul counts come from the API calls themselves (scratch copies are in `research2/api/`).

**Survey design:**
- Runs **May–October**, two passes, chartered commercial vessels, stratified random design ([NOAA survey page](https://www.fisheries.noaa.gov/west-coast/science-data/us-west-coast-groundfish-bottom-trawl-survey)).
- Covers **trawlable** shelf and slope habitat at **55–1,280 m** from the Canada border to the Mexico border, since 2003 ([Keller et al. 2017, NOAA TM-NWFSC-136](https://repository.library.noaa.gov/view/noaa/14179); [PFMC survey description](https://pam.pcouncil.org/wp-content/uploads/document-blocks/2023/07/49ed4cf8-9678-4803-9a7a-799364e606ce-Description_of_West_Coast_NWFSC_fishery-independent_surveys.pdf)).
- **Licence / terms:** a formal licence on the endpoint was **[UNVERIFIED]**. The data is a U.S. federal data product, and stock assessments and the open-source `nwfscSurvey` package routinely pull it with no credential.

**Caveat: survey abundance ≠ fishing opportunity.**
- The survey measures density per hectare swept by a standardised Aberdeen trawl in trawlable habitat during the survey season.
- It does not measure what hook-and-line, pot or longline gear would catch, or what a recreational angler would catch.
- It says nothing about legality: Rockfish Conservation Areas, EFH conservation areas, MPAs, quotas and seasons all apply.
- Rocky, untrawlable nearshore habitat, where many rockfish live, is **not sampled**.

---

## 2. Data inventory

Key to "usable as labels":
- **YES:** actual catch with effort, spatially and temporally resolved, legally usable.
- **PARTIAL:** usable for a narrower target than "fishing opportunity".
- **NO:** not usable.

| # | Dataset | Owner | URL | Catch? | Effort? | Location precision | Time precision | Public / confidential | Licence / terms | Access path | Latency | Usable as labels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Marine Fisheries Data Explorer (MFDE)**: commercial landings from landing receipts, 1980→most recent complete year | CDFW | [MFDE](https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE), [User Guide](https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE/User-Guide) | Landed lb and $ | Participation counts only (vessels / businesses), not trips × hours | Port, port area; **Block Summary** (526 10'×10' blocks, 21 larger blocks up to 30'×30', 10 four-digit areas to the EEZ) | Annual or monthly depending on view. Block Summary and Top Species are **annual**; "Monthly Landings in Pounds" (ex-CCL Tables 8–14) | Public aggregates. Cells marked "Confidential" where too few contributors. No numeric threshold stated for landings; CPFV reports use **<3 CPFVs** | Provided "as-is"; "not intended for management"; site Conditions of Use apply | Power BI-style dashboards with a "Data Export" option (format not stated) | Electronic receipts since 2019-07-01, "near real time", but views go to the last complete calendar year; data may be revised | **NO** for opportunity (no effort, port ≠ catch location, annual blocks). **YES for descriptive seasonal context** (port area × month) |
| 2 | **Landing receipts (raw, trip-level)** | CDFW | [FGC §8022 (FindLaw)](https://codes.findlaw.com/ca/fish-and-game-code/fgc-sect-8022/), [Onecle](https://law.onecle.com/california/fish/8022.html) | Yes (per trip, species, block of catch) | Trip only | Block | Day | **Confidential** (§8022): "shall not be public records"; summaries only | n/a | §8022(b): release to a postsecondary institution under a legally binding confidentiality agreement, or to federal fishery agencies | Near real time to CDFW | **NO** (not obtainable by CoastWatch; possible via an academic partner's NDA) |
| 3 | **Commercial logbooks**: groundfish trawl, market squid (vessel and light-boat), spiny lobster, nearshore etc. | CDFW (and PacFIN for trawl) | [MFSU](https://wildlife.ca.gov/Conservation/Marine/MFSU) | Yes | Yes (tows, traps, hours) | Set / tow positions or blocks | Day | **Confidential** (§8022) | n/a | MFSU (MFSU@wildlife.ca.gov) plus an NDA. Program-by-program details **[UNVERIFIED]** | Months | **NO** publicly; YES under an agreement |
| 4 | **Dungeness crab bi-weekly reports and electronic monitoring (EM)** under 14 CCR §132.8 (RAMP) | CDFW | [RAMP 2.0 update Oct 2025](https://opc.ca.gov/wp-content/uploads/2025/11/CDFWRAMP-2.0Updates_Oct2025DCTF-508.pdf), [§132.8 RAMP 2.0 text](https://opc.ca.gov/wp-content/uploads/2009/04/132.8-Reg-Text-RAMP-2.0.pdf), [DCTF EM 2021](https://opc.ca.gov/wp-content/uploads/2009/04/DCTF-EM-presentation-October-2021.pdf) | Bi-weekly reports: zone, depth range, traps deployed. Catch comes via landing receipts | Traps deployed; EM GPS pings every ≥1 min | Zone (reports); GPS (EM) | Bi-weekly; per-minute (EM) | **Confidential** | n/a | CDFW agreement | n/a | **NO** publicly. A trap-level crab logbook as such is **[UNVERIFIED]** for California |
| 5 | **CPFV logbooks** (mandatory, trip-level: block, anglers, hours, species kept / released) | CDFW | [NOAA InPort 2325](https://www.fisheries.noaa.gov/inport/item/2325), [MFDE CPFV reports](https://wildlife.ca.gov/Conservation/Marine/Data-Management-Research/MFDE/User-Guide) | Yes | Yes (angler-hours) | Block | Day | Trip records **confidential**. Public: MFDE "CPFV Statewide" (41 species, statewide, annual) and "CPFV Portside" (30 species, N/S CA and port clusters, annual); suppressed if <3 CPFVs | NMFS/SWFSC staff need a non-disclosure statement; CDFW must approve products | NDA (e.g., the 2000–2012 logs went to Point 97/Ecotrust under an NDA: [CA open data](https://lab.data.ca.gov/dataset/spatial-and-economic-human-uses-california-south-coast-mpa-baseline-study-1992-to-2012/0d12d13e-f03a-4bce-8710-c4b08ec3b337)) | Annual for public summaries | **NO** publicly (no public block × month CPUE). Historic 1936–1978 block summaries exist in hard copy (InPort) **[UNVERIFIED digital availability]** |
| 6 | **CRFS / RecFIN catch estimates** | CDFW / PSMFC | [CRFS survey design](https://wildlife.ca.gov/Conservation/Marine/CRFS/Survey-Design), [CRFS Methods 2022](https://recfin.org/wp-content/uploads/2024/04/CRFS_Methods_2022.pdf), [RecFIN catch-estimate table (InPort 55977)](https://www.fisheries.noaa.gov/inport/item/55977) | Estimated catch (retained / released) | Estimated angler trips | **6 districts** (county-based) × mode (shore, pier, CPFV, private / rental boat) | **Month** | Public estimates. Intercept microdata partly restricted **[UNVERIFIED which fields]** | PSMFC/RecFIN terms **[UNVERIFIED]** | recfin.org reports (the report code "CTE001" could not be confirmed) | Months | **PARTIAL**: district-month CPUE (catch / trips) is computable but far too coarse to direct anyone where to fish, and it is an estimate, not an observation |
| 7 | **PacFIN** commercial landings, fish tickets, logbooks | PSMFC | [Confidential data access policy](https://pacfin.psmfc.org/pacfin_pub/confagree.php), [PacFIN FAQ](https://pacfin.psmfc.org/?p=589), [PacFIN notices](https://pacfin.psmfc.org/?p=699) | Yes | Logbook effort (trawl) | Port group (public); block / tow (confidential) | Month (public APEX); trip (confidential) | Public APEX reports withhold species detail when **<3 vessels or dealers**. Confidential access limited to agency staff with an FMP-related need | n/a | APEX CSV/Excel (public); written request for confidential access | Weeks | **NO** for spatial labels; descriptive only |
| 8 | **WCGOP observer data** (haul-level catch and discards on observed trips) | NOAA NWFSC | [Observer logistics / database notice](https://www.webapps.nwfsc.noaa.gov/apex/ifq/f?p=505%3A28), [FR 2023-27834](https://public-inspection.federalregister.gov/2023-27834.pdf) | Yes | Yes | Haul lat/lon | Haul | **Confidential** (MSA §402(b), NAO 216-100) | n/a | NOAA data request / NDA **[UNVERIFIED process]**; public products are aggregated mortality reports | Months to years | **NO** publicly |
| 9 | **NWFSC West Coast Groundfish Bottom Trawl Survey** (fishery-independent) | NOAA NWFSC | [Survey page](https://www.fisheries.noaa.gov/west-coast/science-data/us-west-coast-groundfish-bottom-trawl-survey), API base `https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/` (verified §1) | Yes (kg, numbers, kg/ha per species per tow, including zeros via haul table) | Yes (area swept, ha) | Tow lat/lon (~1 km tow) | Date (time of day separately) | **Public** | Federal data. Explicit licence **[UNVERIFIED]** | Anonymous JSON API; `nwfscSurvey` R package | Previous year available (2025 already online) | **YES as labels for fishery-independent density / habitat.** **NO as fishing-opportunity labels** |
| 10 | **CalCOFI larval fish counts** | SIO / NOAA SWFSC | [ERDDAP erdCalCOFIlrvcnt](https://coastwatch.pfeg.noaa.gov/erddap/info/erdCalCOFIlrvcnt/index.html) | Larvae counts (`larvae_count`, `larvae_10m2`, `larvae_100m3`) | Standardised tow (`volume_sampled`, `standard_haul_factor`) | Station lat/lon (CalCOFI grid) | Cruise (quarterly) | Public | "may be used and redistributed for free" (ERDDAP licence) | ERDDAP | Coverage verified **1951-01-09 → 2023-01-25** (multi-year lag) | **PARTIAL**: recruitment / spawning indices, not fishable abundance |
| 11 | **SWFSC Rockfish Recruitment & Ecosystem Assessment Survey (RREAS)** catch | NOAA SWFSC | [ERDDAP FED_Rockfish_Catch](https://coastwatch.pfeg.noaa.gov/erddap/info/FED_Rockfish_Catch/index.html) | Counts of pelagic juvenile rockfish, krill, squid, anchovy etc. per haul | Standard midwater haul | Station lat/lon, 32.7–41.5°N | Haul time; **1990-05-13 → 2025-06-19** | Public | Free-use ERDDAP licence | ERDDAP | ~1 year | **PARTIAL**: forage / recruitment indices (useful for salmon and squid context), not fishery labels |
| 12 | **Global Fishing Watch** apparent fishing effort (AIS) | GFW | [API licence & rate limits](https://globalfishingwatch.org/our-apis/documentation/docs/license-rate-limits), [commercial-use FAQ](https://globalfishingwatch.org/faqs/can-i-use-global-fishing-watch-apis-for-commercial-purposes/), [GEE daily 2012–2017 (CC BY-SA)](https://developers.google.com/earth-engine/datasets/catalog/GFW_GFF_V1_fishing_hours) | **No catch** | Fishing hours (model-inferred from AIS) | 0.01° (historic daily product); 0.1° in public bulk releases | Daily / monthly | Public | API data **CC BY-NC 4.0, non-commercial only**, attribution "Powered by Global Fishing Watch". The 2012–2017 GEE product is CC BY-SA 4.0 | API key, bulk download | Days | **NO**. Effort without catch, and heavily biased in CA: U.S. AIS carriage applies to fishing vessels **≥65 ft** ([46 USC 70114 summary](https://panbo.com/more-ais-in-the-usa-the-new-uscg-requirements/)). Only ~12% of U.S. commercial fishing vessels are ≥65 ft ([Oceana 2023](https://usa.oceana.org/wp-content/uploads/sites/4/2023/07/Fact-Sheet-on-AIS-Vessel-Tracking-2023.pdf)), so most CA crab, salmon troll, squid light and nearshore boats are invisible |
| 13 | **CA MPA boundaries (ds582)** | CDFW | [BIOS metadata](https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS0582.html), [data.gov](https://catalog.data.gov/dataset/california-marine-protected-areas-ds582) | n/a (exclusion mask) | n/a | Polygons | Snapshot as of 2019-01-01 (newer edition **[UNVERIFIED]**) | Public | CC BY 4.0, "not for navigational use or defining legal boundaries" | Download / FeatureServer | Irregular | Mask, not labels |
| 14 | **Domoic acid fishery closures** (crab, razor clam) | CDFW / CDPH / OEHHA | [CDFW razor clam closure news](https://wildlife.ca.gov/News/Archive/razor-clam-fishery-closes-in-humboldt-county-due-to-public-health-hazard1) | n/a | n/a | Area / county | Dates | Public | — | Web notices | Days | Mask, not labels. CoastWatch already treats official notices strictly; never present as lifted |

---

## 3. What drives each key California species, and how strong the evidence is

| Species | Drivers with published evidence | Weak, missing, or misleading | Notes for CoastWatch |
|---|---|---|---|
| **Albacore** | SST, chlorophyll, and distance to coastal SST fronts explain troll CPUE (boosted regression trees, 1988–2011, leave-one-year-out CV). Higher catches near warm, low-chlorophyll oceanic water and near fronts ([NOAA repository summary](https://repository.library.noaa.gov/view/noaa/37600/noaa_37600_DS1.pdf); attributed to Nieto et al. 2017 **[UNVERIFIED author list]**). Classic front association in [Laurs et al. 1984](https://swfsc-publications.fisheries.noaa.gov/publications/CR/1984/8452.PDF) | The model could not predict very high CPUEs. Habitat models found suitable habitat fairly stable across years, and the SST correlation with centre of gravity was weak ([Muhling et al. 2019, CalCOFI Rep 60](https://repository.library.noaa.gov/view/noaa/62817) **[UNVERIFIED exact URL mapping]**). Laurs warns fleets aggregate and share information, so troll CPUE is not a random sample | The best mechanistic case for "fronts → fish", but the labels (logbooks) are confidential |
| **Market squid** | Paralarval density falls sharply with warm SST / El Niño. GAM with SST, zooplankton volume and log chl explains >40% of deviance ([Van Noord & Dorval 2017](https://repository.library.noaa.gov/view/noaa/54936/noaa_54936_DS1.pdf) **[UNVERIFIED that this PDF is that paper]**). Landings drop by an order of magnitude in El Niño and recover after La Niña ([OPC rapid assessment citing Zeidberg et al. 2006](https://opc.ca.gov/webmaster/ftp/project_pages/Rapid%20Assessments/Market%20Squid.pdf)). CalCOFI paralarvae relate to ENSO and the PDO ([Koslow & Allen 2011](https://calcofi.com/publications/calcofireports/v52/Vol_52_205-213.Koslow.pdf)) | Spawning-ground location within a season is driven by fleet / light-boat behaviour. No public spatial catch | Strong **interannual** signal (ENSO) and good for an annual descriptive index, not spatial guidance |
| **Chinook salmon** | Early-season (Mar–May) upwelling intensity and krill / juvenile-rockfish forage on the shelf drive juvenile growth and survival, so adult returns 2–4 years later ([Wells et al. 2016 MEPS](https://filelib.wildlife.ca.gov/Public/TownetFallMidwaterTrawl/LFS_Synthesis/References/Ocean/2016_Wells_MarineEcosystemPerspectivesChinookSalmonRecruitment_MEPS.pdf); [Wells et al. 2008](https://www.waterboards.ca.gov/waterrights/water_issues/programs/bay_delta/docs/cmnt091412/sldmwa/wells_et_al_2008.pdf)) | Abundance is dominated by freshwater / hatchery conditions and **season closures** (the California salmon fishery has had recent closures **[UNVERIFIED current status]**). Within-season ocean "hotspot" evidence is thin in public data | Recommending salmon fishing locations is especially inappropriate: stocks are depleted and seasons are often closed |
| **Rockfish / groundfish** | Depth is dominant. Temperature and oxygen shape distributions in trawl-survey sdmTMB models (quadratic temperature and log depth, breakpoint on oxygen) ([English et al. 2022/2023, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10225861/)) | Nearshore rocky rockfish are not in the trawl survey. Rockfish Conservation Areas and depth limits govern legal fishing | **Best public labels** (WCGBTS) |
| **Sablefish** | Depth / slope habitat. Survey density well sampled (4,733 positive CA tows) | Survey depth cap 1,280 m truncates adult habitat **[UNVERIFIED magnitude]** | Prototype candidate |
| **California halibut** | Nearshore soft bottom, temperature-linked distribution **[UNVERIFIED citation]** | Only **106** positive WCGBTS tows coast-wide 2003–2024, so the survey is useless for it | No public labels |
| **Dungeness crab** | Recruitment linked to larval transport / spring transition **[UNVERIFIED citation in this session]**. Season openings are set by meat-fill, domoic acid and whale-entanglement (RAMP) rules ([Santora et al. 2020](https://repository.library.noaa.gov/view/noaa/27585)) | Catch is mostly a function of season opening and fleet "derby" behaviour, not environment within the season | **Highest ecological risk**: steering pot effort can raise entanglement risk. 2014–16 heatwave plus a domoic-acid delay compressed whales and gear into the same space (Santora et al. 2020) |
| **Red sea urchin** | Food (kelp) availability. North-coast fishery collapsed after kelp loss: >90% bull-kelp loss, the 2014–17 heatwave and a purple-urchin boom ([UC Davis summary of Rogers-Bennett & Catton 2019](https://www.ucdavis.edu/news/californias-crashing-kelp-forest)) | No public spatial catch / effort; diver-based fishery | No |
| **Spiny lobster** | Larval supply rises in El Niño through the Davidson Current. Phyllosoma abundance is a possible spawning-biomass index ([Koslow et al. 2012 CalCOFI Rep 53, via LTER bibliography](https://ccelter.ucsd.edu/wp-content/uploads/bibliodocs/0278ccelter.pdf) **[UNVERIFIED mapping]**). Management uses logbook CPUE ([CDFW draft FMP](https://fisheries.legislature.ca.gov/sites/fisheries.legislature.ca.gov/files/u8/Lobster_Fishery_Management_Plan_010616_Draft.pdf)) | No published CA logbook-CPUE vs temperature model found. Short-term temperature–CPUE links are inconsistent in other lobster fisheries | Labels confidential; MPAs are central |
| **Razor clams** | Closures driven by **domoic acid**, not abundance ([CDFW](https://wildlife.ca.gov/News/Archive/razor-clam-fishery-closes-in-humboldt-county-due-to-public-health-hazard1)) | — | The only relevant output is the official closure status, which CoastWatch already handles. Never imply "safe to harvest" |

### Environmental covariates and where each is used

| Covariate | Species with evidence | Source CoastWatch could use |
|---|---|---|
| Depth / bathymetry, slope, distance to shelf break | Groundfish, sablefish, rockfish (dominant) | GEBCO / NOAA CRM **[standard, not re-verified]** |
| Bottom temperature, dissolved oxygen | Groundfish (English et al. 2023) | In-situ on WCGBTS hauls (`temperature_at_gear_c_der`); ROMS reanalysis for prediction **[UNVERIFIED product URL]** |
| SST, SST fronts | Albacore (strong); squid paralarvae (cool SST); upwelling proxy | Already in CoastWatch pipelines (MUR / OISST) |
| Chlorophyll (surface), chl fronts | Albacore (low chl / oceanic side), squid (low chl) | Already in CoastWatch. **Chlorophyll is not a toxin proxy** and is only weakly linked to most target species |
| Upwelling indices (CUTI / BEUTI) | Salmon (Mar–May), squid, krill | Public indices **[UNVERIFIED URL in this session]** |
| ENSO / PDO | Squid (strong), lobster larvae, crab | NOAA indices |
| Forage (krill, juvenile rockfish) | Salmon | RREAS (ERDDAP, verified) |

---

## 4. Modelling options compared with simple baselines

Any model must beat these baselines **on out-of-sample data** before anyone believes it:
1. **Climatology:** mean density by depth bin × latitude band × month or pass, from training years only.
2. **Persistence:** last survey year's density at the nearest stations.
3. For landings questions, **seasonal naïve:** same month last year, or a multi-year monthly mean.

| Option | Strengths | Weaknesses / risks | Suitability |
|---|---|---|---|
| **Spatiotemporal GLMM: sdmTMB** (delta-gamma or Tweedie, SPDE mesh, AR1 year fields) ([English et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10225861/)) or **VAST** | Standard in West Coast stock assessment. Handles zeros, area-swept offset, spatial autocorrelation and uncertainty | Spatial random fields can "memorise" and inflate in-sample skill. Forecasting a new year relies on AR1 or covariates | **Primary choice** for the prototype |
| **GAM / habitat suitability** (mgcv, environment only) | Interpretable response curves; easy to transfer to new years | Ignores residual spatial correlation, so CIs are too optimistic | Second model / ablation |
| **Gradient boosting** (LightGBM / BRT, as in albacore studies) | Flexible interactions | Needs strict blocked CV. Poor native uncertainty; needs a conformal or quantile wrapper | Comparator only |
| **CPUE standardisation** (GLM / GLMM on logbooks) | The right tool for fishery-dependent data | Needs confidential logbooks. Fleet behaviour and hyperstability bias | Only after a data agreement |
| **Calibrated probabilistic occurrence** (P(present) with isotonic or Platt calibration) | Communicates uncertainty honestly | Presence ≠ catchable quantity | Use as a component |
| **Dynamic ocean management** (EcoCast-style multi-species target vs bycatch surfaces) ([NOAA EcoCast](https://www.fisheries.noaa.gov/news/new-tool-helps-fisheries-avoid-protected-species-near-real-time); [CoastWatch EcoCast](https://coastwatch.noaa.gov/cwn/news/2019-02-15/ecocast-tool-help-fishers-minimize-bycatch-us-west-coast.html)) | The responsible prior art: weighs target catch against protected-species risk, built with observer, tagging and fishery data under agreements | Needs confidential observer data plus tagging data; it was a NOAA-led co-production with the fleet | The template for any future real product |

### Required validation design (non-negotiable)

1. **Strict temporal holdout.**
   - Train 2003–2016, tune on 2017–2019, test once on **2021–2025**. 2020 has no survey; 2019 has half coverage and must be flagged.
   - Covariates must be ones available *before* the predicted period, for any forecast claim.
2. **Geographic holdout.** Leave-one-latitude-band-out (e.g., ~1° bands, or Conception / Monterey / Mendocino regions) plus spatially blocked CV with blocks larger than the residual autocorrelation range.
3. **Vessel / pass holdout.** Leave one charter vessel out, so the model is not learning vessel effects. Fit vessel-year as a random effect in final models.
4. **No leakage.**
   - Climatology and baseline means are computed from training years only.
   - Covariates are extracted at the tow date and location, not from a future composite.
   - No standardisation statistics are computed on test data.
   - Zero tows come from the haul table (the catch table is positives only).
5. **Environmental alignment.**
   - Satellite SST / chl matched to tow date (±3–8 day composite, documented). Bottom temperature in-situ for fitting.
   - For prediction, use the same-source gridded product, or fit with that product to avoid training/serving skew.
   - Record clouds and gaps; never interpolate silently.
6. **Effort correction.** Offset `log(area_swept_ha_der)`. Drop `performance != "Satisfactory"` and `station_invalid == 1`.
7. **Metrics** (each against climatology and persistence; report skill scores):
   - Encounter: AUC, Brier score and **reliability diagrams / calibration slope**.
   - Positive density: RMSE on log scale.
   - Combined: Tweedie deviance or CRPS.
   - Interval coverage (50 / 80 / 95%) and PIT histogram.
   - Design-based index comparison (model index vs `nwfscSurvey` design-based index).
8. **Uncertainty.** Report predictive intervals on maps. Suppress cells with high CV and anything outside the 55–1,280 m domain, beyond the sampled latitude range or in untrawlable habitat.
9. **Species-specific evaluation.** One model and one evaluation per species. No pooled "fish score".
10. **Confidentiality compliance.** Public WCGBTS only. No fusion with confidential data unless done under an agreement by the agreement-holder, and only aggregated outputs released (≥3-contributor rule; CDFW approval of products, per InPort 2325 terms).
11. **Closures and MPAs.** Mask MPAs (ds582), Rockfish Conservation Areas and EFH conservation areas **[UNVERIFIED layer URLs]** in any map. State plainly that masks are illustrative, not legal boundaries (ds582 says so itself).
12. **Ecological risk review.**
    - Never rank areas for depleted or constrained stocks.
    - Never produce pot / trap guidance (whale entanglement: Santora et al. 2020; RAMP).
    - Never produce salmon guidance.
    - Assess whether outputs could concentrate effort on spawning aggregations.

---

## 5. Verdict: is a defensible offline prototype possible today with only public data?

**Yes, for one narrow class:** fishery-independent **trawl-survey density / habitat models for well-sampled groundfish off California**. There is no defensible public-data prototype for **catch opportunity** for any species.

### The single most promising prototype: Sablefish habitat-density model (California, WCGBTS)

**Species:** sablefish (*Anoplopoma fimbria*). It is commercially important in California, has 4,733 positive CA tows 2003–2025, and a strong depth / temperature structure. Run the same pipeline on Dover sole (11,626 positives coast-wide) as a replication species and on shortspine thornyhead as a third.

**Labels:**
- **Haul / effort table:** `https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/trawl.operation_haul_fact/selection.json?filters=project=Groundfish%20Slope%20and%20Shelf%20Combination%20Survey,year>=2003,year<=2025&variables=trawl_id,year,pass,vessel,datetime_utc_iso,latitude_dd,longitude_dd,depth_hi_prec_m,area_swept_ha_der,temperature_at_gear_c_der,temperature_at_surface_c_der,performance,station_invalid`
- **Catch table:** `https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/trawl.catch_fact/selection.json?filters=project=Groundfish%20Slope%20and%20Shelf%20Combination%20Survey,field_identified_taxonomy_dim$common_name=sablefish,date_dim$year>=2003,date_dim$year<=2025&variables=trawl_id,common_name,total_catch_wt_kg,total_catch_numbers,cpue_kg_per_ha_der`
- **Join:** left-join catch onto hauls by `trawl_id` and fill zeros.
- **Filter:** `latitude_dd < 42`, `performance == "Satisfactory"`, `station_invalid == 0`, and drop the 4 null-lat hauls.
- **Response:** `total_catch_wt_kg` with `log(area_swept_ha_der)` offset (or `cpue_kg_per_ha_der`).

**Covariates:**
- log depth (in-situ) and bathymetric slope
- bottom temperature (in-situ for fitting; for mapping, ROMS reanalysis **[UNVERIFIED product]**, or drop it and use depth plus a spatial field)
- satellite SST on the tow date (CoastWatch's existing pipeline)
- year and pass as factors / AR1, vessel random effect

Chlorophyll is included only as an ablation, to test the hypothesis that it adds nothing.

**Unit:**
- Fitting is at the tow level.
- Prediction runs on a ~2–3 km grid restricted to the survey domain (CA, 55–1,280 m, trawlable strata) at **annual / pass (May–Oct) resolution**. No daily or weekly resolution.

**Baselines:** depth-bin × latitude-band climatology (training years only), and last-year persistence.

**Models:**
1. Delta-gamma sdmTMB with spatial plus AR1 spatiotemporal fields (primary).
2. Environment-only GAM.
3. LightGBM with conformal intervals (comparator).

**Validation:** the design in §4. Train 2003–2016; validate 2017–2019; test 2021–2025 once. Add leave-one-region-out and leave-one-vessel-out.

**Metrics:** AUC, Brier, calibration slope, log-RMSE on positives, Tweedie deviance / CRPS skill vs both baselines, 80% / 95% interval coverage, and agreement of the model-based annual index with the design-based index.

**The prototype could claim:**
- "Where and at what depths / temperatures the NOAA bottom-trawl survey has found higher sablefish density off California, 2003–2025, and how well that pattern held in held-out years, with uncertainty."
- Possibly a retrospective habitat-shift analysis.

**It could not claim:**
- Catch rates for any fishing gear, or for recreational anglers
- Where to fish legally (RCAs, EFH areas, MPAs, quotas)
- Anything about nearshore or rocky habitat, waters shallower than 55 m or deeper than 1,280 m, or months outside May–October
- Day-to-day or week-to-week prediction
- Anything about other species
- Any HAB or toxin link

Survey density ≠ fishing opportunity.

**Secondary research idea (annual, descriptive):** a market-squid interannual study. Relate CalCOFI / RREAS paralarvae and juvenile-squid indices and ENSO to CDFW public **monthly / annual squid landings** (MFDE). This tests whether public indices anticipate poor squid years. The output is a research note, not guidance; landings reflect market and regulation (the seasonal catch cap) **[UNVERIFIED current cap]**.

---

## 6. Data-acquisition plan for genuine catch-opportunity guidance (for the record; do not start without user approval)

| Step | Partner | Mechanism | Rough timeline |
|---|---|---|---|
| 1. Academic host | A UC / CSU lab with fisheries-data experience (e.g., one holding CDFW or PacFIN agreements) | Required: §8022(b) permits release to a **postsecondary institution under a legally binding confidentiality agreement** | 1–3 months to establish |
| 2. CDFW data request | CDFW MFSU (MFSU@wildlife.ca.gov) | Request block-level landing receipts plus the relevant logbooks for one species (e.g., CPFV logs for rockfish / lingcod, which have angler-hour effort), via an NDA. All products need CDFW approval ([InPort 2325 terms](https://www.fisheries.noaa.gov/inport/item/2325)) | 3–12 months **[UNVERIFIED; no published SLA]** |
| 3. Federal data | PacFIN / NWFSC (trawl logbooks, WCGOP) | Eligibility is restricted to agency staff and contractors with FMP-related needs ([PCFDC policy](https://pacfin.psmfc.org/pacfin_pub/confagree.php)). Realistically only through a NOAA collaborator | 6–18 months |
| 4. Fleet co-production | Port associations, Dungeness Crab Task Force, CPFV operators | Voluntary data sharing and co-design, EcoCast-style | 6–24 months |
| 5. Ecological safeguards | CDFW, NOAA protected-resources staff, PFMC advisory bodies | Review of bycatch / entanglement / stock-status constraints before any public output | Ongoing |

**Legal and ethical constraints:**
- Confidential data cannot reach the public except as ≥3-contributor aggregates approved by the owner.
- SB 609 (2025–26) would change §8022 disclosure of identifiers **[UNVERIFIED status]** ([LegiScan](https://legiscan.com/CA/text/SB609/2025)).
- GFW data is **non-commercial** (CC BY-NC 4.0).
- MPA layers are not legal boundaries.
- Guidance must never steer effort into protected-species risk (whales, sea turtles) or onto depleted stocks.
- Per project rules: never present observations as forecasts, never treat chlorophyll as toxin, and never imply that a harvest closure has been lifted.

---

## 7. GO / NO-GO

- **(a) User-facing fishing recommendations now: NO-GO.**
  - No public catch-plus-effort labels at a useful resolution.
  - The data that does exist is confidential.
  - Survey density ≠ opportunity.
  - Legal masks are incomplete.
  - The ecological risks (entanglement, depleted stocks) are high.
- **(b) Offline research prototype: GO, scoped as above.**
  - Sablefish (plus Dover sole and shortspine thornyhead) WCGBTS habitat-density model, California, annual May–Oct, sdmTMB vs climatology and persistence.
  - Strict temporal, regional and vessel holdouts.
  - Internal research only. Not shown in the app.
- **(c) Seasonal-context feature (public historical landings by port area × month): conditional GO.** Conditions:
  1. Use MFDE "Monthly Landings in Pounds" / port tables as published, keeping "Confidential" cells as-is, with no imputation or back-calculation.
  2. Label clearly: "commercial landings at port of landing; not where fish were caught; not catch rates; not a forecast; not a recommendation."
  3. Show it beside, never fused with, HAB observations and official closures.
  4. Confirm CDFW reuse / redistribution terms. The User Guide says "as-is" and "not intended for management", and redistribution rights are **[UNVERIFIED]**. Ask CDFW only with the user's approval.
  5. Exclude species under closure, or flag them prominently. No salmon "season context" while the fishery is closed **[UNVERIFIED current status]**.
  6. Integration note: per project memory, the deployed web and demo reject unknown manifest fields. Any such data would need its own versioned artifact, not new fields in the existing pipeline manifest.

---

### Things I could not verify in this session

- An explicit licence statement for the FRAM / TRIPS API
- The numeric MFDE landings confidentiality threshold (only CPFV "<3" is stated)
- RecFIN report codes (e.g., "CTE001") and microdata restrictions
- Whether California has a trap-level Dungeness crab logbook (only bi-weekly reports plus EM are verified)
- Squid / lobster logbook regulation citations
- ROMS bottom-temperature product URLs, CUTI / BEUTI URLs, and RCA / EFH layer URLs
- Current salmon season status
- SB 609 status
- Author attributions marked inline
