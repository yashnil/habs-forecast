# Nearshore data coverage for California HAB information: C-HARM's coastal gap and what can (and cannot) fill it

Research note for CoastWatch. Prepared 2026-10-12 (America/Los_Angeles); server checks ran around 2026-10-13 04:00–05:15 UTC.
This is research only. Nobody was contacted and nothing was written to any git repository.

**Status labels used throughout**
- **documented**: stated in a cited source, or measured directly from data in this session (the HTTP result and timestamp are logged).
- **estimated**: inferred from physics, resolution or pilot data, not from an official statement.
- **unknown / UNVERIFIED**: not confirmed. Treat these as open questions.

---

## 0. Key findings (read first)

1. **The live C-HARM v3.1 IDs are `wvcharmV3_0day`, `wvcharmV3_1day`, `wvcharmV3_2day` and `wvcharmV3_3day`**, each also served as `_LonPM180`. The grid is 0.03° (about 3 km). The inputs are **S‑NPP VIIRS** chlorophyll and Rrs486/551, gap-filled with DINEOF over 180 days, plus **WCOFS ROMS** salinity, temperature and currents (documented in the ERDDAP `summary` and `history`). `charmForecast*V2` ended on 2022‑11‑30, and `charmForecast*day` (v1) covers 2018–2022.
2. **Why the pDA mask differs from the Pseudo-nitzschia mask** (documented by measurement on the 2026‑10‑11 nowcast slice):
   - `pseudo_nitzschia` is valid wherever `chla_filled` is valid (77,690 vs 77,692 cells).
   - `particulate_domoic` and `cellular_domoic` are valid only where **both** `chla_filled` **and** the WCOFS `salinity`/`water_temparture` are valid (77,172 cells, exactly the intersection).
   - The 518 cells that have PN but no pDA all lack WCOFS salinity. The pDA model therefore needs model salinity, and its footprint is the satellite mask clipped by the WCOFS land/sea mask. This matches Anderson et al. 2016, where the pDA model uses ROMS salinity.
   - Because both variables share the `chla_filled` footprint, the dominant nearshore loss comes from the **ocean-color/DINEOF land mask** (estimated). WCOFS removes only a few hundred more cells.
3. **The pipeline is three weeks from a source cutoff.** NASA Earthdata reports that delivery of new **Suomi NPP data, VIIRS included, ceases 2 Nov 2026 13:00 UTC** ("subject to change"; posted 2026‑08‑07, updated 2026‑10‑05) — https://www.earthdata.nasa.gov/data/alerts-outages/suomi-npp-observations-cease-november-2-2026 . C-HARM v3.1 and `erdVHNchla*` both name S‑NPP VIIRS. Whether CoastWatch will switch C-HARM to NOAA‑20/21 is **unknown**. CoastWatch should expect a possible C-HARM outage or a version change and must not show the resulting gaps as "no risk".
4. **The C-HARM authors document the nearshore limitation themselves.** Anderson et al. (2016) write that "satellite coverage was exceptionally poor in the first 4‑10 km distance from the coast (mean gap of 8 to 10 days)" along parts of California (citing Frolov et al. 2013). They also write that 3 km nearest-pixel matchups "are only rough approximations of HAB activity in the nearshore zone". Both quotes are in §1.
5. **No validated, gridded nearshore toxin product exists for California** (documented by absence across NOAA NCCOS, SCCOOS and CeNCOOS). The nearshore toxin truth is **point data**:
   - CalHABMAP weekly pier samples (pDA, Pseudo-nitzschia counts);
   - CDPH shellfish-tissue biotoxin results (mostly mussels, volunteer-collected);
   - the California IFCB network (cell images, near-real-time, *not* toxin).
6. **Nearshore physics is far better covered than nearshore toxin.** MUR SST is gap-free to the coast by construction (it is interpolated). OLCI 300 m chlorophyll and the HF radar 1 km/2 km products reach closer to shore than C-HARM does. SCCOOS/CeNCOOS automated shore stations report every few minutes. All of these are **environmental context**, not toxin.

---

## 1. Why the C-HARM nearshore mask exists

### 1a. Documented by the C-HARM developers

| Claim | Quote / evidence | Source | Status |
|---|---|---|---|
| Satellite coverage collapses near the coast | "for many places along the California coast (north of San Diego, off Point Sur, and between Cape Mendocino and Point Arena), satellite coverage was exceptionally poor in the first 4‑10 km distance from the coast (mean gap of 8 to 10 days)" | Anderson et al. 2016, *Harmful Algae* 59:1–18, accepted MS p.~5, https://repository.library.noaa.gov/view/noaa/33076 (citing Frolov et al. 2013) | documented |
| Cloud cover drives gaps; DINEOF is used to fill them | "limited by their relatively short duration and cloud cover (Frolov et al., 2013) … Temporal and spatial averaging is routinely used to reduce the gaps due to cloud cover" | same | documented |
| Pixels near land drop out from navigation and view-angle errors | "This can occur when data close to land are included, since small errors in the satellite navigation or in viewing angle result in the pixels being classified as land (NaN values)" | same, §3.4 | documented |
| Resolution limits nearshore skill | "the spatial resolution of the final HAB model predictions is constrained by the lowest resolution input product, which is the 3‑km CA ROMS model, such that these moderate-resolution modeled matchups (nearest pixels) are only rough approximations of HAB activity in the nearshore zone" | same, Discussion | documented |
| Pier samples do not match offshore water | "overall problems with model performance when assessed at the pixel level relative to very nearshore, pier-sampling matchups is corroborated by in situ observations from the Santa Barbara Channel where Pseudo-nitzschia and pDA at Stearns Wharf rarely align with offshore levels" | same | documented |
| Skill was poor at pier pixels for PN in some tests | The paper reports AUC values from 0.77 down to 0.33 and 0.04 for some model/station combinations; PICES 2016 slides give the PN model at pier stations "AUC ~0.50" | same; https://dev.meetings.pices.int/publications/presentations/PICES-2016/HAB-Anderson.pdf | documented |
| Nearshore and offshore biomass decorrelate within 2–4 km | Abstract of Frolov, Kudela & Bellingham 2013, "Monitoring of harmful algal blooms in the era of diminishing resources: a case study of the U.S. West Coast", *Harmful Algae* 21–22:1–12 (abstract via https://modarch.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=08442) | full text not read | documented (abstract only) |
| Original CA ROMS grid | "covering the entire California coastal ocean from north of Crescent City, CA to Ensenada, Mexico and extending approximately 1000 km offshore at a resolution of 3.3 km" | Anderson et al. 2016 §2.1 | documented |
| v3 swapped in WCOFS and VIIRS; WCOFS salinity hurts pDA | SCCOOS bulletins: "serious issues with WCOFS salinity accuracy that have notable consequences for the Particulate Domoic Acid predictions". The VIIRS change has small impact "relative to the replacement of CA ROMS Classic with WCOFS" | https://sccoos.org/california-hab-bulletin/february-2023/ , https://sccoos.org/c-harm-v3/ (HTTP 403 to curl on 2026‑10‑12; content seen via search index) | documented (secondary) |
| Anderson et al. 2019 | Frontiers in Marine Science 6:250, "Scaling Up From Regional Case Studies to a Global HAB Observing System". It recommends "sustained, preferably automated, near real-time information from nearshore and offshore sites" and includes a C-HARM schematic | https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2019.00250/pdf | documented (abstract); full text on nearshore masking not read: UNVERIFIED |

**Not found:** I found no explicit statement from NOAA or SCCOOS on *how* the v3.1 coastal mask is defined (e.g. a distance buffer or an L2 flag set). The exact mask definition is **UNVERIFIED**. The variable-intersection analysis in §0.2 is the best evidence available.

### 1b. General ocean-color physics behind coastal masking (estimated mechanisms, with citations)

- **Adjacency effect:** bright land scatters photons into water pixels. In the NIR/SWIR this corrupts the aerosol estimate, so blue water-leaving reflectance is underestimated and chlorophyll from blue/green band ratios is usually *over*estimated. One MODIS study measured land-adjacency effects against offshore pixels more than 12 pixels from land. Sources: Bélanger et al. (SPIE, sea-ice adjacency analogue), https://proceedings.spiedigitallibrary.org/conference-proceedings-of-spie/5885/588509/ ; search-index summaries of MODIS land-adjacency studies. *Specific numbers for the California coast: UNVERIFIED.*
- **Stray light:** the NASA OBPG Level-2 flag `STRAYLIGHT` marks pixels within a set number of pixels of bright targets (cloud or land), and the standard L3 chlorophyll binning excludes it. The flag page is https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/ (HTTP 200, but the page now renders the Earthdata OB.DAAC landing page, so I could not quote the bit table). **The exact pixel buffer is UNVERIFIED.** Other relevant flags in the standard L2 set (from memory, needs confirmation): `LAND`, `CLDICE`, `HIGLINT`, `HILT`, `ATMFAIL`, `HISATZEN`, `COASTZ`, `TURBIDW`, `CHLFAIL`, `NAVFAIL`, `PRODFAIL`.
- **Optically complex water:** CDOM, suspended sediment (river plumes, surf-zone resuspension) and **bottom reflectance** in shallow clear water break the open-ocean OCx algorithms. Kd/SPM fields may still be valid where chlorophyll is not (Frontiers in Remote Sensing 2024 review, https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2024.1359709/epub). *Estimated.*
- **Grid geometry:** a 750 m swath binned to a 3 km grid. Any 3 km cell that overlaps land or the stray-light zone loses most of its contributing samples. With a 3 km cell, every cell centred within about 1.5 km of shore overlaps land (estimated). This alone makes C-HARM's 12.6% validity within 1 km plausible.
- **DINEOF cannot create data where there is never data.** If a pixel is masked in nearly every scene (persistent land/stray-light flag), DINEOF treats it as land and leaves it NaN. *Estimated, consistent with Anderson 2016 §3.4.*

### 1c. Model-side limitation

- **WCOFS grid:** "grid resolution of about 4 km", 348×1016 points, 40 terrain-following levels, daily cycle with a 24 h nowcast and 72 h forecast, and 4D‑Var assimilation of SST, HF-radar currents and altimetry. https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html (HTTP 200). **Documented.**
- **WCOFS rivers:** the info page lists "USGS real-time river discharge observations for Columbia River" plus climatological discharge for 14 Washington rivers and the Fraser. **No California rivers are listed** (documented by absence on that page). That is consistent with the salinity problems SCCOOS flagged and implies poor plume salinity off SF Bay, the Russian, Eel, Klamath and Santa Clara rivers (estimated).
- An AGU 2020 abstract describes the forecast domain as running to "the 10‑meter isobath" (pre-implementation, https://agu.confex.com/agu/osm20/preliminaryview.cgi/Paper643703.html). It also describes a 2 km research grid with 4 km DA. Whether the operational grid resolves the 10 m isobath is **UNVERIFIED**.
- Wetting/drying is **not mentioned** for WCOFS (documented absence). Bays such as Tomales, Morro, Humboldt and Mission Bay, and most harbors, are sub-grid at 4 km (estimated).

---

## 2. Which variables and products cover the nearshore better

All HTTP codes and latest timestamps were observed by me on 2026‑10‑12/13 (UTC).

| Product | Dataset ID / endpoint | Native res. | Cadence | Nearshore behaviour | HTTP / latest time seen |
|---|---|---|---|---|---|
| **C-HARM v3.1** (PN, pDA, cDA, chla_filled, salinity, temp) | `wvcharmV3_0day` … `_3day` (coastwatch.pfeg.noaa.gov/erddap) | 0.03° (~3 km) | daily nowcast + 1/2/3‑day | Your audit: 12.6% valid <1 km, 31.5% <3 km, 75% <10 km | 200; 0day `time_coverage_end` 2026‑10‑11T12:00Z; 3day 2026‑10‑14T12:00Z |
| **MUR SST L4** | `jplMURSST41` | 0.01° (~1 km grid) | daily | Gap-free over water by construction (optimal interpolation of MODIS/VIIRS/AMSR2 etc.). Pilot: 100% valid in every distance bin. **"Valid" does not mean "observed"**: under persistent fog the value is interpolated (estimated). Land mask flag 2 = land | 200; latest 2026‑10‑11T09:00Z |
| **VIIRS chl 750 m (ERD)** | `erdVHNchla1day/3day/8day` | 0.0075° | daily composite | Pilot (Monterey Bay, 13 published days 20 Sep–6 Oct 2026): mean daily valid fraction **0.22 in 0–1 km**, 0.46 in 1–3 km, 0.43 in 3–10 km, 0.36 beyond 10 km; 78% of 0–1 km cells valid at least once. S‑NPP ends 2 Nov 2026 | 200; latest 2026‑10‑06T12:00Z (lag about 6 days) |
| **VIIRS chl 750 m sector (STAR)** | `noaacwNPPVIIRSchlaSectorUYDaily` (Monterey), `…VYDaily` (San Diego), plus `…SCI…` science-quality versions on coastwatch.noaa.gov/erddap | 750 m | daily | as above | 200; UY latest 2026‑10‑10T11:42Z |
| **OLCI 300 m chl (S3A/S3B)** | `noaacwS3AOLCIchlaSectorCIDaily` (29.8–45.2°N, ‑140 to ‑120°) and `noaacwS3AOLCIchlaSectorDIDaily` (‑120 to ‑100°, covers the Southern California Bight); same for S3B | 0.0025° (~300 m) | daily, **rolling 90-day window only** | Best ocean-color nearshore reach. Each 300 m pixel is 10× closer than a 3 km cell, but it is still subject to adjacency and stray-light flags (estimated). OC4ME algorithm. Must be archived locally for time series | 200; S3A‑CI 2026‑10‑11T19:12Z; S3B‑CI 2026‑10‑01T19:32Z (**S3B about 10 days stale**); S3A‑DI 2026‑10‑11T17:31Z (start 2026‑07‑12) |
| **Merged DINEOF chl 2 km** | `noaacwNPPN20S3ASCIDINEOF2kmDaily` | 2 km | daily, science quality | Gap-filled. Kd490 and SPM siblings exist (`…kdSCIDINEOF…`, `…spmSCIDINEOF…`) | 200; latest 2026‑09‑27 |
| **Sentinel‑2 MSI** | no operational chl product on these ERDDAPs; requires ACOLITE/C2RCC processing of Copernicus L1C | 10/20/60 m | ~5 days (S2A+B) | Resolves surf zone, harbors and estuaries. No validated California chl/HAB product (estimated). Research only | not checked |
| **HF radar surface currents** | `ucsdHfrW1`, `ucsdHfrW2`, `ucsdHfrW6` (PFEG ERDDAP); HFRNet https://hfradar.ioos.us/ | 1/2/6 km | hourly | The HFRNet grid design masks "Grid cells falling within 0.5 km of land" (https://hfradar.ioos.us/hfrnet/documents/HFRNet_RTVgrids_20060127.pdf). Totals need two-site overlap, so coverage near headlands and in coastline-parallel geometry is patchy, and the first range cell sits ~1–3 km from the antenna (**estimated, UNVERIFIED**). The 1 km product exists only around specific short-range sites (SF Bay mouth, Santa Barbara Channel, San Diego; estimated) | W1 200, latest 2026‑10‑13T02:00Z; W2 200, 2026‑10‑13T03:00Z; W6 **503** at query time |
| **WCOFS** | NOS OFS https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html; THREDDS/NODD | ~4 km | daily, 72 h forecast | Coastline at grid scale, no wetting/drying, no California rivers (see §1c) | page 200 |
| **UCSC CCS ROMS 4D‑Var** | https://oceanmodeling.ucsc.edu/ | ~0.1° (≈10 km) (**UNVERIFIED** from memory) | daily | Coarser than WCOFS. Research-grade context only | 200 |
| **SF Bay OFS (FVCOM)** | tidesandcurrents.noaa.gov/ofs/sfbofs (**not fetched**) | unstructured, ~100 m in bay (UNVERIFIED) | 4×/day (UNVERIFIED) | Resolves SF Bay itself (estimated) | not checked |
| **CalHABMAP weekly pier samples** | SCCOOS ERDDAP `HABs-*` (17 datasets): pDA, tDA, dDA, PN *seriata*/*delicatissima* groups, chl, nutrients, T, S | point at pier | weekly (nominal) | **This is the only nearshore *toxin* measurement in the water column** | all 200; latest: Monterey Wharf 2026‑10‑07; Cal Poly 2026‑10‑04; Scripps 2026‑10‑05; Santa Monica 2026‑10‑05; Stearns 2026‑09‑28; Santa Cruz 2026‑09‑30; Morro Bay 2026‑09‑01; Newport 2026‑08‑18; **Trinidad 2026‑04‑17, Humboldt 2026‑04‑07, Humboldt S. Bay 2026‑03‑30, Bodega Lab 2026‑04‑13, Bodega buoy 2026‑01‑28, Tomales (3 sites) 2026‑03‑03** |
| **Automated shore stations** (SCCOOS/CeNCOOS) | CeNCOOS ERDDAP: `scripps-pier-automated-shore-sta-1`, `newport-pier-automated-shore-sta`, `santa-monica-pier-automated-shor`, `stearns-wharf-automated-shore-st-1`, `san-luis-bay-cal-poly-pier-shore`, `monterey-wharf-real-time-samplin`, `edu_humboldt_humboldt`, `edu_ucsc_scwharf1`, `tiburon-water-tibc1` | point | minutes | T, S, chl fluorescence, turbidity, O₂, fDOM, pH (varies by station) | all 200. Real-time (end ≥ 2026‑10‑13T02Z): Scripps, Newport, Santa Monica, Stearns, Cal Poly, Monterey Wharf, Humboldt. Stale: Santa Cruz 2026‑08‑19, Tiburon 2026‑09‑24, `mlml_monterey` 2019, Bodega Head 2015 |
| **IFCB network** | https://ifcb.caloos.org/ (HTTP 200; WHOI dashboard v5.0, Jan 2026) lists 20 datasets incl. Trinidad Pier, Hog Island Oyster (Tomales), Bodega Marine Lab, SF Pier 17, Santa Cruz Wharf, MBARI Power Buoy, Stearns Wharf, Newport Pier, Scripps Pier (×3 instruments), Del Mar Mooring, and cruises. SCCOOS ERDDAP `ca_ifcb_data` is a **file index** ("Un-reviewed data from IFCB sensors") | point | ~20 min samples, near-real-time (estimated) | Gives cell images and ML classifications (Pseudo-nitzschia counts). **Not toxin** | `ca_ifcb_data` .das 200; orderByMax query 400, so latest time **unknown** |
| **CDPH Marine Biotoxin Monitoring** | https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx ; monthly/annual StoryMap reports, e.g. https://storymaps.arcgis.com/stories/aa81d7b243b543df8796035146d32a83 | point (shellfish beds, mostly mussels; razor clams in the north) | irregular, often weekly to monthly by site (estimated) | **Tissue toxin**: the regulatory truth for shellfish. "relies on volunteers and partnerships to collect the shellfish (mostly mussels) and plankton samples"; "Phytoplankton samples help inform where shellfish sampling is needed; no advisories are issued related to plankton abundance". DA alert level 20 ppm in tissue (federal action level; the 17 ppm figure in one report summary is UNVERIFIED) | page 200 (TLS chain error with WebFetch; fetched via curl) |

**Pilot method (documented here so it can be rerun):**
- Box: 36.5–37.0 N, ‑122.2 to ‑121.75 E.
- Land reference: MUR `mask` bit 2 at 0.01°, nearest-neighbour onto the VIIRS grid.
- Distance: brute-force great-circle approximation.
- Caveat: the 0–1 km bin is limited by the 1 km land mask. Treat the numbers as **estimated**.
- VIIRS outages: the requested window 1 Sep–6 Oct contained **only 13 time steps** (none 1–19 Sep; 21 Sep, 27–28 Sep and 5 Oct also missing). That is a **source/processing outage signature** (missing time axis), distinct from cloud gaps (NaN within an existing time step).

---

## 3. Alternative validated HAB products for the California nearshore

| Product | What it is | Status / validation | Nearshore toxin? |
|---|---|---|---|
| C-HARM v3.1 | PN >10,000 cells/L, pDA >500 ng/L, cDA >10 pg/cell probabilities (https://habs.sccoos.org/hab-forecast) | Operational at CoastWatch since about Jan 2023. Skill was assessed only at pier pixels for 2014–15 (Anderson 2016). v3 has a known WCOFS salinity issue. **No published v3.1 skill paper found (UNVERIFIED)** | No (masked or 3 km) |
| California HAB Bulletin (SCCOOS) | **Monthly** synthesis of C-HARM, HABMAP, IFCB and stranding data (https://sccoos.org/california-hab-bulletin/; HTTP 403 to curl) | Expert-interpreted and experimental. NOAA NCCOS lists it as an "External Partner HAB Forecast" (https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/ , HTTP 200) | Narrative, not gridded |
| CalHABMAP alerts | Weekly listserv alerts from pier data (Kudela et al. 2021, *Oceanography*, https://tos.org/oceanography/article/the-california-harmful-algal-bloom-monitoring-and-alert-program-a-success-story-for-coordinated-ocean-observing) | Observational, about 9–17 sites | Yes, at piers only |
| CA IFCB network | 12+ IFCBs as "early warning". HABDAC (NOAA-funded data assembly center) is **proposed/under development** (https://coastalscience.noaa.gov/?p=52592) | Classifier output is un-reviewed. **No validated automated toxin alert (estimated)** | No (cells) |
| CDPH quarantine/advisories | Annual mussel quarantine (1 May–31 Oct) plus site-specific advisories (https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Annual-Mussel-Quarantine.aspx , HTTP 200) | Regulatory | Yes (tissue) |
| PNW HAB Bulletin (NANOOS) | Pseudo-nitzschia/pDA bulletins for WA/OR beaches, using LiveOcean (https://www.nanoos.org/products/habs/forecasts/bulletins.php , HTTP 200) | Seasonal; status in 2026 not confirmed | Not California |
| NCCOS forecasts elsewhere | Florida/Texas (*Karenia brevis*), Gulf of Maine (*Alexandrium*), Lake Erie (cyanobacteria) | Operational/experimental, varies | Not California |

**Conclusion:** no validated, spatially continuous toxin product exists inside about 3 km of the California coast. The only legitimate nearshore toxin information is point measurements: HABMAP pDA, CDPH tissue results, and SPATT where it is reported.

---

## 4. How coastal ocean models handle estuaries, beaches and harbors

- **Resolution:** WCOFS is about 4 km (documented), and the original CA ROMS was 3.3 km (documented). Features narrower than about 2 grid cells (8 km) are not dynamically resolved (estimated, standard ROMS rule of thumb). These include harbors, Morro Bay, Tomales Bay, Humboldt Bay, lagoon mouths and the surf zone.
- **Coastline:** land/sea masking at grid scale moves the effective "coast" offshore by up to one cell. The minimum depth is clipped (often to 5–10 m in shelf ROMS). Both points are estimated; the 10 m isobath comes from the AGU abstract.
- **Wetting/drying:** not reported for WCOFS. Intertidal flats in estuaries are absent (estimated). The SF Bay FVCOM OFS is the exception for SF Bay (UNVERIFIED details).
- **Rivers:** WCOFS forces the Columbia, WA rivers and the Fraser only (documented). The California river plumes that set nearshore salinity, and that matter for the C-HARM pDA salinity term, are missing or climatological (estimated).
- **Waves and surf zone:** wave-driven nearshore circulation and rip currents are not modelled.
- **Implications for CoastWatch:**
  1. A model cell centred 3 km offshore describes shelf water, not the beach.
  2. Salinity-driven pDA probability near river mouths is the least trustworthy part of C-HARM.
  3. Estuary and harbor users must be told the model does not represent their water body.

---

## 5. Satellite missingness: causes and how to tell them apart

| Cause | Signature in data / metadata | How to detect |
|---|---|---|
| **Source outage / processing outage** | The time step is *absent* from the ERDDAP time axis, or `time_coverage_end` lags | Compare the expected daily axis with the actual one, e.g. `erdVHNchla1day` had no steps for 1–19 Sep 2026. Check `time[last]` against today. Read NASA/NOAA alert pages (S‑NPP end 2 Nov 2026) |
| **Server outage** | HTTP 5xx | `ucsdHfrW6` and an ERDDAP search returned **503** during this session |
| **Clouds / marine-layer fog** | Time step exists, large contiguous NaN regions, seasonal (May–Sep) and coast-hugging | Correlate NaN fraction with month. Cross-check with VIIRS/GOES cloud mask. L2 `CLDICE` flag |
| **Sun glint** | Swath-geometry stripes. L2 `HIGLINT` | Requires L2 flags; L3 composites lose this info |
| **Stray light / adjacency** | A persistent NaN band of fixed width along coastlines and cloud edges, present even on clear days | NaN fraction vs distance-to-coast on *clear* days. L2 `STRAYLIGHT` |
| **Algorithm failure / turbid water** | Isolated NaNs near river plumes. L2 `CHLFAIL`, `ATMFAIL`, `TURBIDW` | Compare with Kd/SPM products that still return values |
| **Land/grid mask** | The *same* cells are NaN in every time step | Count, per cell, the fraction of time steps that are ever valid. Cells at 0% are mask, not weather. In C-HARM, compare `chla_filled` vs `salinity` masks (§0.2) |
| **Gap-filling artefacts** | Values present but not observed (DINEOF, MUR OI) | MUR provides `analysis_error`. DINEOF products state gap-filling in `history`. Never present gap-filled values as observations |

The L3 ERDDAP products used here do **not** carry per-pixel flag provenance. Attributing an individual NaN to clouds vs stray light requires L2 data from OB.DAAC (S‑NPP/NOAA‑20/21 VIIRS, OLCI) or NOAA CoastWatch L2 swaths (estimated).

---

## 6. Recommended honest presentation of differing temporal coverage

1. **Label every value with its kind and age:** "Measured (lab, pier sample) · 6 days ago"; "Satellite (observed) · yesterday"; "Model estimate (C-HARM nowcast) · today"; "Gap-filled satellite/analysis". Never merge these into one undated colour.
2. **Show cadence explicitly.**
   - HABMAP is **weekly**. Show the sample date and a staleness state (e.g. greyed after 14 days). Several North/North-Central series currently end in Jan–Apr 2026. That may be an ingest lag rather than stopped sampling (unknown), but it must still read as "no recent sample", not "safe".
   - CDPH is irregular; link to the official notice.
   - C-HARM is a daily model run. Show the run date and the forecast lead.
3. **Represent "no data" as its own state, with a reason when known:** outside model coverage (mask), no satellite view (clouds), source outage, or not sampled. Never fill with the lowest risk colour.
4. **Never move a value across space.** A pier pDA result describes that pier on that date. C-HARM at the nearest valid cell describes water about 3+ km offshore. If the nearest valid cell is shown for a beach, state the distance ("nearest model estimate is 4.2 km offshore").
5. **Separate toxin evidence from environmental context visually** (different panels or sections). Context such as SST, chlorophyll, currents and upwelling never uses the toxin/risk palette.
6. **Defer to regulatory sources.** CDPH advisories and quarantines are authoritative for harvest decisions. Model and observation layers are informational.

---

## Deliverable (a): coverage matrix

Legend:
- **D** = documented; **E** = estimated; **U** = unknown.
- Fraction refers to the chance of a usable value on a given day for gridded products, or presence of a station for point products.
- Regions:
  - **NC** = North Coast (OR border–Cape Mendocino)
  - **NCC** = North-Central (Cape Mendocino–Pigeon Point)
  - **C** = Central (Pigeon Point–Point Conception)
  - **S** = Southern (Point Conception–Mexico)

| Product (cadence) | Region | <1 km | 1–3 km | 3–10 km | >10 km | Notes / citation |
|---|---|---|---|---|---|---|
| **C-HARM v3.1 pDA** (daily, 3‑day fcst) | all CA | 12.6% of coastal water valid (**D**, your audit) | 31.5% (**D**, cumulative <3 km) | 75% (**D**, <10 km) | ~100% within domain (**E**) | Regional split **U**: needs the §(d) script. Expect worst in bays and along steep, cloudy NC/NCC coasts (Anderson 2016 names Cape Mendocino–Point Arena and Point Sur) (**E**) |
| C-HARM v3.1 PN (daily) | all | slightly > pDA (**D**: +518 cells statewide, where WCOFS lacks salinity) | same | same | same | §0.2 |
| MUR SST (daily L4) | all | 100% values (**D** pilot, C region) but interpolated under fog (**E**) | 100% (**D** pilot) | 100% | 100% | `jplMURSST41`. Show `analysis_error` |
| VIIRS chl 750 m L3 (daily) | C (Monterey pilot) | 22% of days (**D** pilot, Sep–Oct 2026) | 46% (**D**) | 43% (**D**) | 36% (**D**) | other regions **U**; NC/NCC lower in summer fog (**E**). S‑NPP ends 2 Nov 2026 (**D**) |
| OLCI 300 m chl (daily, 90‑day window) | all (sectors CI + DI) | better than VIIRS but still flagged near land (**E**) | **E** high on clear days | **E** | **E** | not measured; see §(d) |
| Merged DINEOF chl 2 km | all | low (cells overlap land) (**E**) | **E** moderate | **E** high | **E** ~100% | gap-filled, not observed |
| Sentinel‑2 MSI (≈5‑day) | all | **E** resolves surf zone on clear days | **E** | **E** | not designed for open ocean | no operational chl product (**E**) |
| HF radar 6 km (hourly) | all | 0% (**E**; 6 km cells) | low (**E**) | moderate–high where 2-site overlap (**E**) | high to ~150 km (**E**) | W6 returned 503 (**D**) |
| HF radar 1–2 km (hourly) | NCC (SF Bay mouth), C (Monterey Bay), S (SB Channel, San Diego) (**E**) | 0% within 0.5 km by grid design (**D**, HFRNet doc); otherwise sparse (**E**) | moderate (**E**) | high where covered (**E**) | — | W1/W2 latest 2026‑10‑13 (**D**) |
| WCOFS (daily, 72 h) | all | mostly absent (land-masked 4 km cells) (**E**) | partial (**E**) | yes (**E**) | yes | no CA rivers (**D**); no wet/dry (**D** absence) |
| CalHABMAP pier pDA/PN (weekly) | NC | Trinidad, Humboldt Bay ×2 (**D**; latest Mar–Apr 2026) | — | — | — | SCCOOS ERDDAP |
| | NCC | Bodega ×2, Tomales Bay ×3 (**D**; latest Jan–Apr 2026). **No SF/San Mateo pier in HABMAP** (**D** absence) | — | — | — | |
| | C | Santa Cruz, Monterey, Cal Poly, Morro Bay ×2 (**D**; latest Sep–Oct 2026) | — | — | — | |
| | S | Stearns, Santa Monica, Newport, Scripps (**D**; Newport latest Aug 2026) | — | — | — | |
| Automated shore stations (minutes) | NC | Humboldt Bay (**D**, real-time) | — | — | — | CeNCOOS ERDDAP |
| | NCC | Tiburon (stale 2026‑09‑24), Bodega Head (ended 2015) (**D**) | | | | |
| | C | Monterey Wharf, Cal Poly (real-time); Santa Cruz (stale 2026‑08‑19) (**D**) | | | | |
| | S | Stearns, Santa Monica, Newport, Scripps (real-time) (**D**) | | | | |
| IFCB (≈20‑min, NRT) | NC | Trinidad (**D**, listed) | | | | ifcb.caloos.org; current status **U** |
| | NCC | Hog Island/Tomales, Bodega, SF Pier 17 (**D**, listed) | | | | |
| | C | Santa Cruz Wharf (**D**); MBARI Power Buoy (offshore, distance **U**) | | | | |
| | S | Stearns, Newport, Scripps (**D**); Del Mar mooring at ~1–3 km (**E**) | | | | |
| CDPH shellfish tissue (irregular, often weekly) | all four | mussel/razor-clam sites, coastal and bays (**D** program; site list and cadence **U** here) | — | — | — | Authoritative for shellfish |

---

## Deliverable (b): where legitimate environmental data can fill nearshore gaps (context, NOT toxin)

- **Water temperature near the beach:** MUR SST (labelled "satellite analysis, may be interpolated"), shore-station temperature, and SST anomaly (`jplMURSST41anom1day`) as upwelling context.
- **Upwelling and currents:** HF radar 2/6 km totals (labelled with the coverage mask), WCOFS currents offshore, and the upwelling indices CUTI/BEUTI (https://mjacox.com/upwelling-indices/ , not fetched, UNVERIFIED).
- **Phytoplankton biomass (not toxin):** OLCI 300 m chl, VIIRS 750 m chl, and shore-station chlorophyll fluorescence. Label these "chlorophyll — a measure of all algae, not of toxins".
- **Water clarity and turbidity:** Kd490/SPM DINEOF products and shore-station turbidity (river plumes after storms).
- **Community composition:** IFCB Pseudo-nitzschia counts as "cells observed", with the caveat that "cell presence does not indicate toxin".
- **Salinity / freshwater influence:** shore-station salinity, which is better than WCOFS near river mouths.

---

## Deliverable (c): what must NOT be done

1. **Do not interpolate, extrapolate or kriging-fill toxin** (C-HARM pDA, HABMAP pDA, CDPH tissue DA) into masked nearshore cells or between stations. Nearshore and offshore decorrelate within 2–4 km (Frolov 2013 abstract), and piers "rarely align with offshore levels" (Anderson 2016).
2. **Do not use chlorophyll, SST, fluorescence or an IFCB Pseudo-nitzschia count as a toxin proxy** or colour them on a risk scale. Domoic acid production varies by species and condition, and high chlorophyll is often a non-toxic bloom. CDPH states that a visible bloom is "not an indication of biotoxins in shellfish".
3. **Do not present gap-filled values (DINEOF, MUR OI) as observations**, and do not present observations as forecasts. Follow the project rule: never present observations as forecasts or chlorophyll as toxin.
4. **Do not render masked, missing or outage cells as "low risk"** or drop them silently.
5. **Do not snap a beach to the nearest C-HARM cell without showing the distance**, and do not do so at all for enclosed bays and harbors.
6. **Do not imply regulatory status.** Do not mark anything as safe to harvest or eat; defer to CDPH. Do not mark official records verified, lifted or expired (project rule).
7. **Do not re-run C-HARM's logistic models on higher-resolution inputs and publish the result** as a C-HARM product. It would be unvalidated nearshore.
8. **Do not quote C-HARM skill numbers from v1 (MODIS/CA ROMS) as if they apply to v3.1.**

---

## Deliverable (d): research plan to measure actual nearshore coverage (2–3 products)

**Candidates**
1. `wvcharmV3_0day` (pDA, PN, chla_filled, salinity): reproduce the audit by region.
2. `noaacwS3AOLCIchlaSectorCIDaily` + `…DIDaily` (OLCI 300 m). Start archiving now, because only 90 days are retained.
3. `erdVHNchla1day` (VIIRS 750 m), with a follow-on for NOAA‑20/21 equivalents after the S‑NPP end. Optional fourth product: `jplMURSST41` (expect 100%; report `analysis_error` vs distance instead).

**Reference geometry**
- Coastline: GSHHG full-resolution or NOAA NOS medium-resolution shoreline, rasterised to a common 100 m "truth" grid. Use one shoreline for all products.
- Distance-to-coast: `scipy.ndimage.distance_transform_edt` on a local equal-area projection (EPSG:3310 California Albers).
- Regions: latitude cut-points at Cape Mendocino 40.44°N, Pigeon Point 37.18°N and Point Conception 34.45°N, and the OR border at 42.0°N.
- Exclude enclosed bays with a separate polygon flag, so open coast and bays are reported separately.

**Script sketch (Python, `xarray` + `erddapy`)**
```python
from erddapy import ERDDAP
e = ERDDAP(server="https://coastwatch.pfeg.noaa.gov/erddap", protocol="griddap")
e.dataset_id = "wvcharmV3_0day"; e.griddap_initialize()
e.constraints.update({"time>=": "2024-10-01", "time<=": "2026-10-11"})
e.variables = ["particulate_domoic", "pseudo_nitzschia", "chla_filled", "salinity"]
ds = e.to_xarray()             # chunk by month for large requests
# per cell: water fraction (from shoreline raster), dist_km, region
# metrics below, grouped by region x distance bin x month
```

For OLCI and VIIRS use `server="https://coastwatch.noaa.gov/erddap"` (sector IDs above) and `erdVHNchla1day` respectively. Loop over months and save to Zarr.

**Metrics** (all reported per region × {<1, 1–3, 3–10, >10 km} × month)
1. **Mask coverage:** the fraction of coastal water area with any valid value over the full period. This separates the permanent mask from weather.
2. **Daily availability:** the mean fraction of valid cells per available day. Separately, the fraction of *expected* days missing from the time axis (outages).
3. **Revisit gap:** the distribution of consecutive missing-day run lengths per cell (median and 90th percentile). This is comparable to the "8–10 day" figure in Anderson 2016.
4. **Seasonality:** monthly availability, to quantify the summer marine-layer effect, especially in NC/NCC.
5. **Distance to nearest valid cell** from each beach / HABMAP / CDPH site (km). This is what users actually experience.
6. **Mask intersection audit** (C-HARM): valid(pDA) vs valid(chla_filled) ∧ valid(salinity). Check that it holds daily, not only on 2026‑10‑11.
7. **Resolution-normalised comparison:** report both area-weighted (water km²) and cell-count fractions, because cell area differs by 100× between 300 m and 3 km.
8. **Optional matchup:** for HABMAP piers, the fraction of sample dates with an OLCI/VIIRS observation within 1 km and ±1 day. This measures data *availability*, not skill.

**Outputs:** one CSV per product, plus a single tidy table and plots of availability vs distance by region. Provenance should record the dataset ID, query URL, `time_coverage_end` at run time, and the shoreline version.

---

## Appendix: server checks log (this session)

| URL / dataset | HTTP | Latest timestamp seen |
|---|---|---|
| coastwatch.pfeg.noaa.gov/erddap/search?searchFor=charm | 200 | lists v1 (2018–2022), v2 (2022‑present; actual end 2022‑11‑30), v3.1 `wvcharmV3_*` |
| …/griddap/wvcharmV3_0day.das | 200 | time_coverage_end 2026‑10‑11T12:00Z; history "2026‑10‑12: DINEOF gap filling … S‑NPP NOAA VIIRS … 180 days" |
| …/griddap/wvcharmV3_1day.das | 200 | 2026‑10‑12T12:00Z |
| …/griddap/wvcharmV3_3day.das | 200 | 2026‑10‑14T12:00Z |
| …/griddap/charmForecast0dayV2.das | 200 | 2022‑11‑30T12:00Z |
| …/griddap/wvcharmV3_0day.nc (full-domain last slice) | 200 | 2026‑10‑11T12:00Z; 137,241 grid cells; valid: PN 77,690, pDA 77,172, cDA 77,172, chla 77,692, sal/temp 77,254 |
| …/griddap/jplMURSST41 time[last] | 200 | 2026‑10‑11T09:00Z |
| …/griddap/erdVHNchla1day.das | 200 | 2026‑10‑06T12:00Z |
| …/griddap/noaacwNPPN20S3ASCIDINEOF2kmDaily.das | 200 | 2026‑09‑27T12:00Z |
| …/griddap/ucsdHfrW1 time[last] | 200 | 2026‑10‑13T02:00Z |
| …/griddap/ucsdHfrW2 time[last] | 200 | 2026‑10‑13T03:00Z |
| …/griddap/ucsdHfrW6 time[last] | **503** | — |
| …/search?searchFor=HF radar | **503** | — |
| coastwatch.noaa.gov/erddap/griddap/noaacwS3AOLCIchlaSectorCIDaily.das | 200 | 2026‑10‑11T19:12Z |
| …/noaacwS3BOLCIchlaSectorCIDaily.das | 200 | 2026‑10‑01T19:32Z |
| …/noaacwS3AOLCIchlaSectorDIDaily.das | 200 | 2026‑10‑11T17:31Z (start 2026‑07‑12) |
| …/noaacwNPPVIIRSchlaSectorUYDaily.das | 200 | 2026‑10‑10T11:42Z |
| erddap.sccoos.org/erddap search "HABs" | 200 | 17 CalHABMAP datasets (dates in matrix) |
| erddap.sccoos.org/erddap/tabledap/ca_ifcb_data.das | 200 | file index; orderByMax query → 400 |
| erddap.cencoos.org/erddap/search (several terms) | 404 (no match) | full listing via /info/index.csv → 200 (1,078 datasets) |
| CeNCOOS shore stations (11 checked) | 200 | see §2 |
| hfradar.ioos.us | 200 | — |
| hfrnet-tds.ucsd.edu/thredds | 000 (connection failed) | — |
| ifcb.caloos.org | 200 | dashboard v5.0 (Jan 2026) |
| sccoos.org/ifcb, sccoos.org/california-hab-bulletin | 403 (bot block) | — |
| habs.sccoos.org, calhabmap.org, data.caloos.org | 200 | — |
| coastalscience.noaa.gov/science-areas/habs/hab-forecasts/ | 200 | — |
| tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html | 200 | — |
| CDPH biotoxin & quarantine pages | 200 | the CDPH page carried a banner saying the federal government is shut down, so expect possible federal data delays (**E**) |
| earthdata.nasa.gov S‑NPP cessation notice | 200 | posted 2026‑08‑07, updated 2026‑10‑05; cessation 2026‑11‑02 13:00Z |

**Unverified items to confirm before quoting:**
- the exact C-HARM v3.1 mask rule;
- the OBPG STRAYLIGHT buffer width;
- the HF radar first-range-cell distance and per-region 1 km coverage;
- UCSC ROMS resolution;
- SFBOFS details;
- CDPH sampling cadence and DA alert value;
- IFCB station operating status;
- whether the stale HABMAP North/North-Central series are an ingest lag or a sampling stop;
- whether CoastWatch will move C-HARM to NOAA‑20/21 VIIRS after 2 Nov 2026.
