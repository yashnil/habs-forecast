# Nearshore HAB information for California: sources and literature

**Prepared:** 2026-10-12, read-only research for CoastWatch. Nothing in the repository was changed.

**Method.** Web sources were opened where possible. "VERIFIED (query)" means I ran the ERDDAP or API request myself on 2026-10-12, and the script is in a local working folder (not committed). "Repo evidence" means the existing notes in `docs/coastwatch/evidence/*.md`. Where I could only see a secondary summary of a paper, or could not open the full text, I say so.

**Verdict vocabulary.**
- **TOXIN:** measures domoic acid (DA), or predicts it, at or near the shore.
- **BLOOM-PROXY:** *Pseudo-nitzschia* (PN) counts, or bulk biomass near shore. It says nothing about toxin.
- **CONTEXT:** physical or regional drivers only.

---

## 0. Bottom line

1. **There is no public, validated, gridded product that gives domoic-acid information inside ~3 km of the California shore.** C-HARM v3.1 is the only gridded DA product, at 3 km. Its toxin layers are blank 2–7 km from several CalHABMAP piers. I measured this with the latest run (§1.1). Its only peer-reviewed skill assessment is of v1 (Anderson et al. 2016). At pier pixels that assessment found *Pseudo-nitzschia* skill at or below chance, and pDA skill that varied with the site (AUC 0.77 at Santa Cruz, 0.04 at Stearns Wharf).
2. **Measured nearshore toxin data exist only at points:**
   - CalHABMAP weekly pier pDA (public ERDDAP; latency from about one week to more than 2 months);
   - CalHABMAP SPATT (public, irregular);
   - CDPH shellfish toxin results (published only as monthly narrative StoryMaps, months late, plus county advisories);
   - CDPH seafood and crab PDFs.
3. **Nearshore bloom-proxy data:**
   - CalHABMAP and CDPH PN counts, weekly;
   - CA IFCB images, hourly at about 9 pier or mooring sites (public dashboard; the classified PN counts are not obviously exposed as a public feed);
   - OLCI 300 m and Sentinel-2 MSI chlorophyll, which is biomass and not PN.
4. **Ocean colour cannot detect PN or DA directly.** Smith et al. (2018) state there is no "novel optical signal for domoic acid or *Pseudo-nitzschia* abundances" in current multispectral sensors.
5. **No published work validates interpolating or extrapolating offshore C-HARM values to the shore.** The published evidence points the other way:
   - Frolov et al. (2013): nearshore biomass is "decoupled" from biomass ~2–4 km offshore.
   - Anderson et al. (2016): nearest-pixel matchups are "only rough approximations of HAB activity in the nearshore zone".
   - The SCCOOS bulletin: pier data "do not always correspond with C-HARM predictions for the open coast".
6. **Station-level DA prediction is plausible at about a 1-week horizon,** using autoregressive toxin data or molecular or chemical precursors. But the label base is thin. Since 2014 each main pier has had **6–46 weekly samples ≥ 0.5 ng/mL pDA** (C-HARM's 500 ng/L threshold), clustered in a few years (§2.4).

---

## 1. Source comparison

### 1.1 C-HARM v3.1 (NOAA CoastWatch West Coast)

| Attribute | Finding |
|---|---|
| Variable | **Modelled** probabilities: PN > 10,000 cells/L, pDA > 500 ng/L, cDA > 10 pg/cell. These are logistic GLMs on gap-filled VIIRS chl, Rrs486 and Rrs551, plus WCOFS SST and salinity ([ERDDAP info](https://coastwatch.pfeg.noaa.gov/erddap/info/wvcharmV3_0day/index.html); GLM basis in [Anderson et al. 2016](https://repository.library.noaa.gov/view/noaa/33076)) |
| Resolution / coverage | 0.03° (~3 km), "California and Southern Oregon coast" ([ERDDAP info](https://coastwatch.pfeg.noaa.gov/erddap/info/wvcharmV3_0day/index.html)) |
| Time | Daily nowcast plus 1–3 day forecasts. Archive from **2022-11-01**, but only 1,154 time steps over ~1,440 days, so about 20% of days are missing (VERIFIED (query), `wvcharmV3_0day` info). v1 covers 2018-06-19 to 2022-03-17 (`charmForecast0day`). v2 covers 2021-01 to 2022-12 and used CA-ROMS ([v2 info](https://coastwatch.pfeg.noaa.gov/erddap/info/charmForecast3dayV2/index.html)) |
| Nearshore missingness | Toxin layers are masked 1–2 cells further offshore than PN (repo evidence: `docs/coastwatch/evidence/sources-charm.md` §1.5). Distance from each pier to the nearest **pDA-valued** cell centre, run of 2026-10-10 (VERIFIED (query), `charm_pier.py`): Scripps 2.8 km, Newport 2.9, Santa Monica 4.2, Stearns 2.4, Cal Poly 3.8, **Santa Cruz 5.4**, **Monterey Wharf 6.8**, Bodega 1.7, Trinidad 2.0, **Humboldt South Bay 6.4**. The nearest PN-valued cell is 0.7–4.2 km away |
| Documentation of the mask | **I found no producer document that describes or explains the toxin mask.** The "3–6 km" phrasing in our docs comes from our own measurements. The bulletins (Dec 2022 to Apr 2023) describe a WCOFS **salinity** bias that inflates pDA ("biased by the overestimated salinity fields produced by WCOFS"). They note that cDA "does not rely on the salinity prediction" ([Jan 2023 bulletin](https://sccoos.org/california-hab-bulletin/january-2023/)). That fits a salinity-driven mask, but this is my inference and is UNVERIFIED |
| Inputs and their nearshore limits | WCOFS is ~4 km ROMS with 4D-Var, one cycle a day, 72 h forecasts ([WCOFS info](https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html)). In the v1 assessment, ROMS missed salinity drops below 30 psu at Santa Cruz, "likely due to the lack of terrestrial freshwater input". pDA was more sensitive to substituting salinity (r² 0.37) than to substituting chlorophyll (r² 0.16) ([Anderson et al. 2016](https://repository.library.noaa.gov/view/noaa/33076/noaa_33076_DS1.pdf)) |
| Skill | v1 only, 2014–2015, at pier pixels (3×3 grid point nearest each station):<br>• PN model across all stations: ROC "almost overlies the 1:1 line", optimised accuracy 43%. At Santa Cruz and Stearns the PN ROC is below chance ("AUC = 0.33").<br>• pDA at Santa Cruz: **AUC 0.77**. pDA at Stearns: **AUC 0.04**.<br>• Leads at the Santa Cruz pixel: 5 days to mussel DA, 9–12 days to SPATT (r ≈ 0.40), 5 days to strandings (r = 0.48) ([Anderson et al. 2016, PDF](https://www.marinemammalcenter.org/storage/app/media/Misc/PDF/Research%20Publications/2016/anderson-et-al-2016.pdf)).<br>**No published skill for v2, v3 or v3.1** (also noted in repo evidence). Bulletins report "a high rate of false positives" from VIIRS pixelation ([Jan 2023](https://sccoos.org/california-hab-bulletin/january-2023/)) |
| Naming | ERDDAP says `product_version 3.1`. The [Mar–Apr 2025 bulletin](https://sccoos.org/california-hab-bulletin/march-april-2025/) calls it "C-HARM v4" while linking `wvcharmV3_3day`. Treat the label as inconsistent upstream |
| Access / licence | ERDDAP griddap; WMS only for `_LonPM180`. Free use, "not intended for legal use" ([ERDDAP info](https://coastwatch.pfeg.noaa.gov/erddap/info/wvcharmV3_0day/index.html)) |
| Producer caveat | "These nearshore data do not always correspond with C-HARM predictions for the open coast"; "C-HARM output may be more closely correlated with marine mammals that strand" ([CA HAB Bulletin page](https://sccoos.org/california-hab-bulletin/), quoted in repo evidence) |
| **Verdict** | **Modelled TOXIN probability offshore. Not nearshore TOXIN information.** Inside the toxin mask it is at most offshore CONTEXT |

**Finer or regional HAB forecasts.**
- **I found no finer-resolution or nearshore C-HARM product.**
- The **Pacific Northwest HAB Bulletin** (NANOOS) covers **Washington and Oregon beaches**. Its [forecast page](https://www.nanoos.org/products/habs/forecasts/home.php) does not mention California. It combines offshore and beach PN and pDA sampling with a LiveOcean (ROMS) particle-transport forecast. LiveOcean covers the northern California Current, Oregon, Washington and the Salish Sea ([COMT LiveOcean](https://comt.ioos.us/projects/liveocean)).
- A retrospective test of that transport model as a decision tool is [Stone et al. 2022, Harmful Algae 119:102334](https://strathprints.strath.ac.uk/83364). An earlier hindcast "captures most observed Pn HAB beach events from 2004 to 2007, though numerous false positives occur" (Giddings et al. 2014, as summarised via [strathprints](https://strathprints.strath.ac.uk/54079/); full text not opened).
- **None of these can be used in California as they stand.** The domain, the Juan de Fuca eddy source mechanism and the beach-sampling network are all PNW-specific.
- A Santa Barbara Channel model (Anderson et al. 2009/2011) is the ancestor of C-HARM, not a separate operational product.
- UCLA's mechanistic ROMS-BEC PN/DA module ([Moreno et al. 2022, Harmful Algae 118:102296](https://atmos.ucla.edu/dbianchi/publications/)) is a research model, not a forecast.

### 1.2 In-situ HAB observations

| Source | Measured variable | Space / coverage | Time / latency | History | Nearshore | Access / licence | Limits | Verdict |
|---|---|---|---|---|---|---|---|---|
| **CalHABMAP pier stations** (SCCOOS ERDDAP, 17 `HABs-*` datasets) | **Measured** pDA, tDA, dDA (ng/mL); PN *seriata*/*delicatissima* group counts (cells/L); other HAB taxa; nutrients, chl, T, S (dataset info, repo evidence) | Points at about 9 university or municipal piers, San Diego to Humboldt ([IOOS NHABON summary](https://cdn.ioos.noaa.gov/media/2021/10/NHABON-Funding-Awards-FY21_v2.pdf)); 17 datasets | Weekly ("collects weekly phytoplankton…"). **Latency varies by station:** as of 2026-10-08 the last pDA was 2026-09-30 at Santa Cruz, but 2026-08-03/10 at the SCB piers (repo evidence) | 2005–2008 onward for the core piers. **pDA samples since 2014-01-01** (VERIFIED (query)): Scripps 657, Newport 604, Santa Monica 649, Stearns 627, Cal Poly 644, Santa Cruz 621, Monterey Wharf 165 (pDA ends 2022-08-31), Trinidad 108, Humboldt 89, Humboldt S. Bay 104. Morro Bay, Tomales and Bodega have no pDA | **At the shore** (the defining nearshore data) | ERDDAP tabledap, no key. "Free to use, not for legal use" | Single point; surface grab; weekly, so it misses short events (4-day decorrelation off N. California; [Frolov et al. 2013](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=08442)) | **Nearshore TOXIN** (point) |
| **CalHABMAP SPATT** (`SPATT-ScrippsPier`, `-NewportBeachPier`, `-TrinidadPier`) | **Measured** time-integrated dissolved toxin on resins (HP20, SP700, SP207) | 3 piers on SCCOOS ERDDAP | Deployment-integrated (days to weeks); latest 2025-03 to 2026-02 | 2020 onward; 178 / 126 / 82 rows (VERIFIED (query)) | At shore | ERDDAP | Integrative and semi-quantitative; v1 C-HARM correlated best with SPATT ([Anderson 2016](https://repository.library.noaa.gov/view/noaa/33076)) | **Nearshore TOXIN** (integrated, lagged) |
| **CA IFCB network** | **Measured** images of particles (~15 mL hourly per the [CalCOFI poster](https://calcofi.org/downloads/CalCOFI_Conference/2022_posters/P1.2Ellman_Monitoring-Phytoplankton-in-the-California-Current.pdf)); ML classification ("~80% accurate" random forest, 24 classes; [Kudela 2022 slides](https://new.globalhab.info/images/symposium2022/session5/Kudela_CA_Network_v3.pdf)) | Hog Island, Bodega, SF Pier 17, Santa Cruz Wharf, MBARI power buoy, Stearns, Newport, Del Mar mooring, Scripps, plus cruises ([SCCOOS IFCB](https://sccoos.org/ifcb/)) | Near real time. Latest bins (VERIFIED (query), [ifcb.caloos.org](https://ifcb.caloos.org/datasets)): Stearns 2026-10-12, Santa Cruz 2026-10-10, **Newport 2026-09-08**, Scripps IFCB158 2024-10 (other Scripps units not checked) | Santa Cruz back to 2016 (time-series endpoint) | At shore | Public IFCB dashboard (ifcbdb, HTML/JSON endpoints). **No licence stated** on the SCCOOS page. I found no documented public feed of classified PN counts. Bulletins quote IFCB PN in "images/L" with cells/L conversions ([Jan–Feb 2026 bulletin](https://sccoos.org/california-hab-bulletin/january-february-2026/)) | Genus-level PN only (no species, no toxin); classifier error; instrument downtime | **Nearshore BLOOM-PROXY** (high frequency) |
| **California HAB Bulletin** | Narrative synthesis | Statewide | "Monthly to bi-monthly"; in practice bimonthly; no Jul–Aug 2026 issue as of 10-08 (repo evidence) | Archive since about 2018 | Mixed | Web or email, not machine-readable | Experimental | CONTEXT |
| **CDPH Marine Biotoxin Monitoring** (shellfish toxins) | **Measured** PSP and DA in shellfish (mussels, razor clams…) | Coastal sampling sites; published by county | **Monthly StoryMap reports**; latest listed April 2026, so about 5–6 months behind (repo evidence; [example report](https://storymaps.arcgis.com/stories/aa81d7b243b543df8796035146d32a83)) | Monthly reports from 2023; annual 2020–2024 (repo evidence) | At shore (intertidal shellfish) | **Raw results are not published as data.** Reports give toxin *ranges*; advisories via ArcGIS web-map JSON (repo evidence) | Tissue toxin lags water-column DA (5-day model lead in [Anderson 2016](https://repository.library.noaa.gov/view/noaa/33076)); mussels depurate fast | **Nearshore TOXIN** (but late and aggregated) |
| **CDPH Phytoplankton Monitoring** (volunteers) | **Measured** PN relative abundance index / % composition / density | Volunteer pier and shore sites; 82 records in a rolling ~6-week window (VERIFIED (query): `Pseudo_nitzschia/FeatureServer/6` count = 82) | About weekly; `dataLastEditDate` 2026-10-01 | **Rolling only.** You must archive it yourself. The program ran about 1,700 samples a year in 2015 ([CDPH to OPC 2016](https://opc.ca.gov/webmaster/ftp/pdf/agenda_items/20160203/NA/Item6-3-CDPH-Biotoxin-Monitoring.pdf)) | At shore | ArcGIS FeatureServer (machine-readable) | Net tows give relative abundance; CDPH issues no advisories from plankton (repo evidence) | **Nearshore BLOOM-PROXY** |
| **CDPH/FDB crab and seafood DA** | **Measured** DA in crab viscera and meat, other seafood | Port or zone | Episodic PDFs | Seasons | Fishery zones | PDF tables (repo evidence) | Not water | TOXIN for fisheries, not for water |
| **MBARI ESP** (Monterey Bay) | **Measured** PN (molecular) and pDA on mooring | Monterey Bay moorings | Results to shore "within an hour" during campaigns ([MBARI](https://www.mbari.org/?p=110304)); 2015 pDA > 30 µg/L ([NCCOS](https://coastalscience.noaa.gov/?p=15208)) | Campaign-based | Bay or shelf | **No current public feed found** | Campaign only | TOXIN (research, not operational) |
| PNW ESP (La Push, WA), for comparison | ~100 DA measurements over 5 months in 2023, fed to the PNW Bulletin ([NCCOS 2023](https://dev.coastalscience.noaa.gov/news/real-time-obs-improve-domoic-acid-forecasts)) | WA offshore | Daily | 2023 | — | NANOOS | Not California | n/a for CA |

### 1.3 Satellite

| Source | Variable | Resolution / coverage | Time / latency | History | Nearshore missingness | Access / licence | Limits | Verdict |
|---|---|---|---|---|---|---|---|---|
| **OLCI S-3A/3B chl** (`noaacwS3[AB]OLCIchlaSector{CI,DI}Daily`) | **Measured** (retrieved) chl-a, mg m⁻³ | 0.0025° (~300 m); sectors CI and DI cover California | One overpass per sector per day; ~2 days behind; **rolling 90-day window** (`pipeline/.../satellite.py` header; time_coverage 2026-07-11 to 2026-10-10, VERIFIED (query)) | **90 days only** on this server | **Shore-normal transects**, Jul 15 to Oct 9, 2026 (VERIFIED (query), `transect2.py`). Valid-pixel fraction by distance from shore:<br>• **Santa Cruz**, S-3A, 78 days: **10% at 0–1 km**, 16% at 1–2 km, 20% at 2–3 km, 17–19% at 3–15 km. S-3B is similar.<br>• **Stearns Wharf**, S-3A: **9% at 0–1 km**, 23% at 1–2 km, 29–30% at 2–5 km, 24–28% at 5–15 km.<br>• **NOAA-20 VIIRS daily merge off Santa Cruz**, same period: 6% at 0–1 km, 38–50% beyond 1 km. VIIRS gives more valid days offshore but nearly nothing in the first km.<br>So most clear-sky days are lost to cloud or fog everywhere, and the first ~1 km loses a further half or more to land and adjacency masking. Off central California, cloud-gap statistics are worst in "the first 4–10 km" (mean gap 8–10 days; Frolov et al. via [Anderson 2016](https://repository.library.noaa.gov/view/noaa/33076/noaa_33076_DS1.pdf)) | ERDDAP. Must state "Contains modified Copernicus Sentinel data" (VERIFIED (query), dataset licence) | Chl ≠ PN. Adjacency effects detectable to ~36 km offshore for MODIS-type sensors (20 km for MSI) ([Bulgarelli & Zibordi 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5890359/)). OC algorithms underperform above ~1 mg m⁻³ (Kahru et al. 2014, cited in [Anderson 2016](https://repository.library.noaa.gov/view/noaa/33076/noaa_33076_DS1.pdf)) | **Nearshore BLOOM-PROXY (biomass)**, intermittent |
| **VIIRS chl** (`erdVHNchla1day`, NOAA-20/21 sector US) | Retrieved chl-a | 0.0075° (~750 m) | Daily; ~3–4 days latency | NRT sector datasets only from 2026-02 (repo evidence); science-quality 4 km from 2012 | Coarser pixels lose more coastal water. C-HARM bulletins cite VIIRS "pixelation" | ERDDAP, NOAA licence | As above | BLOOM-PROXY at ≥ 1 km; CONTEXT nearshore |
| **MODIS-Aqua** | chl, FLH | 1 km / 4 km | NASA says Terra/Aqua "will begin shutting down operations in late 2026/early 2027" ([Earthdata](https://www.earthdata.nasa.gov/data/alerts-outages/transition-from-modis-viirs)) | 2002 to present | Coarse | — | **End of life.** Note the repo's MODIS-trained model inputs (repo evidence) | CONTEXT / legacy |
| **Sentinel-2 MSI** | TOA radiance → Rrs → chl or indices (e.g. NDCI), after atmospheric correction (ACOLITE, C2RCC…) | 10/20/60 m; 290 km swath | 5-day revisit with two satellites at the equator ([ESA brochure](https://www.fe-lexikon.info/material/texte/sentinel_2_brochure.pdf)); S-2C (launched 2024) is not verified here | 2015 onward | **Resolves the surf zone, piers and bays.** Built mainly for land, with lower SNR than ocean-colour sensors. Atmospheric-correction uncertainty can exceed 100% in the red and 1,000% in the NIR in turbid water ([Stirling study](https://www.stir.ac.uk/research/hub/file/1254294)). Sun glint and adjacency handling are processor-dependent. Coastal fog and the marine layer limit clear views (no CA-specific statistic found) | Free and open Copernicus data (policy page not opened, UNVERIFIED). **No operational chl product for California.** Copernicus Marine HR-OC (100 m, Sentinel-2) covers **only a 20 km European coastal strip** ([CMEMS help](https://help.marine.copernicus.eu/en/articles/5194057-introduction-to-ocean-colour-sentinel-2-high-resolution-products)) | Small high-biomass blooms mapped at 10 m with ACOLITE+NDCI (a *Lingulodinium* bloom, Spain; [PMC7250863](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250863/)). Weak for the low-biomass PN blooms typical of CA. You would have to run your own processing | **Potential nearshore BLOOM-PROXY** (research effort, sparse in time) |
| **PACE OCI** | Hyperspectral Rrs, chl | ~1 km | ~1-day L3 latency (repo evidence) | 2024 onward | Coarse near shore | Earthdata login | A preprint (not peer-reviewed) claims a distinct PN spectral fingerprint ([bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.02.24.707776v1)) | CONTEXT (research) |
| **MUR SST** (`jplMURSST41`) | L4 *analysis* (gap-filled, blended) | 0.01° grid, described as "1-km" | Daily; ~1 day | 2002 to present (VERIFIED (query)) | Gap-filled; inputs include ~25 km microwave, so effective resolution in cloudy periods is coarser than the grid ([URSI paper](https://2021.ursi.org/proceedings/procAP19/papers2019/sstvalidationSummaryPaperOct15.pdf)); weaker coastal gradients than OSTIA in one study ([Scholar Commons](https://scholarcommons.sc.edu/biol_facpub/386)) | ERDDAP | Interpolated, not observed | CONTEXT (upwelling, warm anomalies) |
| **GOES-West SST** (`goes_west`) | Retrieved SST, hourly | 0.02° (~2 km) | Hourly NRT | — | Cloud-limited | ERDDAP | — | CONTEXT |

**PN and DA are not directly detectable from ocean colour.**
- Smith et al. (2018), Harmful Algae 79:87–104 ([NOAA repository](https://repository.library.noaa.gov/view/noaa/57789)), state there is no "novel optical signal for domoic acid or *Pseudo-nitzschia* abundances from the currently available multi-spectral ocean color satellite sensors". They add that chlorophyll-anomaly methods "have yet to be successfully applied to predict *Pseudo-nitzschia* or domoic acid events with high accuracy".
- **Qualifications:**
  - Empirical reflectance-ratio models give *probabilities* of toxigenic conditions. In the Santa Barbara Channel they "correctly estimate 98% of toxic bloom situations, with a 7–29% rate of false positive identification" ([Anderson et al. 2009 abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=01250)). The abstract does not say this was independent out-of-sample validation, and the dataset covered 1.5 years.
  - A MERIS model in Galicia explained 73% of pDA variance ([MDPI Water 2019](https://mdpi.com/2073-4441/11/9/1954/htm)).
  - Hyperspectral discrimination is preprint-stage only.

### 1.4 Physical models and in-situ physical observations

| Source | Variable | Resolution | Time | Nearshore | Access | Verdict |
|---|---|---|---|---|---|---|
| **WCOFS** (NOAA CO-OPS) | **Modelled** currents, T, S, water level | ~4 km (348×1016 grid, 40 levels) | Daily cycle; 72 h forecasts; 4D-Var assimilation ([WCOFS info](https://tidesandcurrents.noaa.gov/ofs/wcofs/wcofs_info.html)) | Shelf scale. Known salinity bias in C-HARM use ([Jan 2023 bulletin](https://sccoos.org/california-hab-bulletin/january-2023/)). Repo has a WCOFS-vs-HF-radar check (repo `p1/currents-contract.md`) | NODD S3, open | CONTEXT |
| **LiveOcean** (UW) | ROMS physics and biogeochemistry; PN particle tracking | ~1.5 km is suggested, not confirmed ([OOIFB slide](https://ooifb.org/wp-content/uploads/2022/06/17_June2022_NEPacOOI_Workshop_OOIModeling.pdf)) | Daily, 3-day forecasts ([COMT](https://comt.ioos.us/projects/liveocean)) | WA/OR only | NANOOS | Not CA |
| **HF radar** (`ucsdHfrW{6,2,1,500}`) | **Measured** surface currents | 6 km / 2 km / 1 km / 500 m by site ([CORDC data stream plan](https://cencoos.org/wp-content/uploads/2024/06/Appendix-E2.1-High-Frequency-Radar-Data-Stream-Plan.pdf)); 2 km arrays reach out "30–50 km" | Hourly; 4–15 h latency (repo `15-p2-currents-multisensor.md`) | Coverage starts "a few kilometers offshore" ([NOAA dataset](https://data.amerigeoss.org/dataset/high-frequency-radar-radial-ranges)); gaps hour to hour; CeNCOOS funding covers only part of the original network ([CORDC plan](https://cencoos.org/wp-content/uploads/2024/06/Appendix-E2.1-High-Frequency-Radar-Data-Stream-Plan.pdf)). No site-specific first-range-cell figure found (UNVERIFIED) | ERDDAP | CONTEXT (transport toward shore) |
| **SCCOOS automated shore stations** (Scripps, Newport, Stearns, Santa Monica) | **Measured** T, S, chl fluorescence (WetSTAR), O₂/pH at some sites | Pier points, 2–5 m depth | Every 4–6 min; "limited data quality checks" in live mode ([SCCOOS autoss](https://sccoos.org/autoss); [sensor plan](https://sccoos.org/wp-content/uploads/2021/01/Automated-Shore-Station-Sensor-Plan.pdf)) | At shore | ERDDAP/THREDDS. Current dataset IDs **not verified**: a search of the SCCOOS ERDDAP for "shore station" did not return them | Nearshore CONTEXT (fluorescence is a BLOOM-PROXY for biomass) |
| **CeNCOOS shore stations** (Santa Cruz Wharf YSI, Monterey Wharf) | **Measured** T, S, chl, turbidity, O₂, pH | Points | 15 min (Monterey); the Monterey ERDDAP dataset ends **2026-07-14**, so possibly stale ([ERDDAP](https://erddap.cencoos.org/erddap/info/monterey-wharf-real-time-samplin/index.html)) | At shore | ERDDAP | Same as SCCOOS |
| **CUTI/BEUTI upwelling indices** | Modelled and assimilated transport and nitrate flux, 1° latitude | Daily, 1988 to present; ends 2026-09-27, so ~2 weeks of latency (VERIFIED (query), [erdCUTIdaily](https://oceanview.pfeg.noaa.gov/erddap/info/erdCUTIdaily/index.html)) | Coastal band | ERDDAP | CONTEXT (a documented PN driver) |
| NDBC / CO-OPS buoys and tide stations | T, winds, water level | Points | Real time | Some nearshore | NDBC/CO-OPS APIs (not separately verified here) | CONTEXT |

### 1.5 Regulatory notices (CDFW, CDPH, OEHHA)

| Source | Machine-readable? | Notes |
|---|---|---|
| CDPH bivalve advisories and quarantine | **Partly.** The ArcGIS web-map JSON is toggled by `definitionExpression`/`visibility`; county polygons are in FeatureServer | Advisories are at county granularity. News releases are HTML. No RSS found (repo evidence) |
| CDPH phytoplankton map | **Yes** (FeatureServer, rolling) | See §1.2 |
| CDPH monthly biotoxin reports | No (StoryMaps) | About 5–6 months behind |
| CDFW domoic-acid closures, Director's Declarations, RAMP | **No** (HTML sections and PDFs at nrm.dfg). RSS covers only general news. Zone geometry is in BIOS FeatureServer, but status is **not** an attribute (repo evidence) | Needs human-verified curation |
| OEHHA recommendations | No (PDF memos; bot protection) | Upstream provenance only |

These give **TOXIN-derived decisions**, not data. Per CLAUDE.md they must never be auto-marked lifted or expired.

---

## 2. Literature: nearshore PN/DA prediction (California / US West Coast)

### 2.1 Key studies

| Study | Target, site | Data / method | Reported skill | Validation | Notes for us |
|---|---|---|---|---|---|
| Anderson et al. 2009, *Harmful Algae* 8:478 ([abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=01250)) | Toxic PN bloom, Santa Barbara Channel | 1.5-yr time series; regression on MODIS/SeaWiFS reflectance, AVHRR SST | "98% of toxic bloom situations, 7–29% false positive" | Not stated in abstract to be independent | Better at presence or absence than at concentration ([Smith 2018](https://repository.library.noaa.gov/view/noaa/57789)) |
| Lane, Raimondi & Kudela 2009, *MEPS* 383:37 ([abstract](https://int-res.com/abstracts/meps/v383/p37-51)) | Toxigenic PN bloom, Monterey Bay (incl. pier data) | 8.3 years of multi-project sampling; logistic regression; annual and seasonal models. Chl and silicic acid in all models; T, upwelling, river discharge and nitrate seasonally | **Not verified** (full text blocked). A 2010 PICES slide by the lead author suggests ROC values above 0.8 for 3-predictor models ([slides](https://dev.pices.int/Publications/Presentations/PICES-2010/2010-S9/S9-1005-Lane.pdf)) | — | Relies on in-situ nutrients, which are not available in real time |
| Anderson et al. 2011, *GRL* 38:L04603 ([abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=03744)) | Toxic PN, SB Channel; GLM on MODIS + ROMS | Nowcast mode | Monthly maps "matched" ship and state data; flagged offshore events missed by nearshore monitoring | Qualitative | The "full" model with nutrients beat the remote-sensing-only model, but only the latter is operational ([Smith 2018](https://repository.library.noaa.gov/view/noaa/57789)) |
| **Anderson et al. 2016**, *Harmful Algae* 59:1 ([PDF](https://repository.library.noaa.gov/view/noaa/33076/noaa_33076_DS1.pdf)) | C-HARM v1 at pier pixels, 2014–2015 | Nowcasts vs weekly HABMAP, SPATT, mussels, strandings | See §1.1 (PN ~chance or worse; pDA AUC 0.77 at Santa Cruz, 0.04 at Stearns) | Out-of-sample in time (operational output) | **The only CA skill test at the shore; it explicitly warns about the nearshore spatial mismatch** |
| Frolov, Kudela & Bellingham 2013, *Harmful Algae* 21–22:1 ([abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=08442)) | Observing-system design | MODIS FLH, in situ | Decorrelation ~4 d (N. CA) to ~17 d (SCB); nearshore biomass "decoupled" from ~2–4 km offshore; piers "likely informative of bloom magnitudes in the immediate proximity" | — | **Directly argues against extrapolating offshore fields to piers, and vice versa** |
| McCabe et al. 2016, *GRL* 43:10,366 ([NOAA repo](https://repository.library.noaa.gov/view/noaa/20927)) | 2015 coastwide event | Warm anomaly + upwelling + storms | Mechanistic attribution, not a forecast | — | Explains anomalous years that dominate labels |
| McKibben et al. 2017, *PNAS* ([climate.gov summary](https://climate.gov/news-features/event-tracker/scientists-link-toxic-algal-blooms-along-us-west-coast-warm-waters)) | Oregon razor clam DA | ~20-yr record vs ONI/PDO | Risk model by climate state (paper not opened) | — | Seasonal or interannual, not weekly |
| Smith et al. 2018, *Harmful Algae* 79:87 ([NOAA repo](https://repository.library.noaa.gov/view/noaa/57789)) | SCB review | ~4,500 pDA plankton samples (Fig. 3, n = 4528) | Review | — | No optical PN/DA signal; nutrients (Si) matter |
| Moreno et al. 2022, *Harmful Algae* 118:102296 | Mechanistic PN/DA module for ROMS-BEC | Lab-calibrated | Si limitation ~2× DA production vs N limitation (search summary) | Lab, not field forecast | A path to a physics-based nearshore DA model, but research-stage |
| Sandoval-Belmar et al. 2023, *Harmful Algae* 126:102435 ([PDF](https://ftp.sccwrp.org/pub/download/DOCUMENTS/JournalArticles/1323_HABCrossRegionalExam.pdf)) | ~20-yr in-situ pDA synthesis; Monterey, SB Channel, San Pedro | Correlational | DA outbreaks correlate with upwelling, chl and Si limitation. **Drivers differ by region**: Monterey events under weaker upwelling and nutrient-poor conditions; SB and San Pedro under cold, N-rich, strong upwelling | — | **A single statewide model will be mis-specified; it needs region-specific terms** |
| Brunson et al. 2024, *PNAS* ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.02.565333.full.pdf); [JCVI](https://www.jcvi.org/publications/molecular-forecasting-domoic-acid-during-pervasive-toxic-diatom-bloom)) | DA, Monterey Wharf II, 2015 | 52 near-weekly net tows; metatranscriptomics | Co-expression of DA-biosynthesis genes predicts DA **one week ahead** | Single bloom year | 1-week molecular lead; not routine |
| Trapp, Kudela et al. (PNAS, ~2025; paper **not opened**) ([press summary](https://www.globalseafood.org/advocate/new-technique-could-give-fisheries-advance-warning-before-harmful-algal-blooms/)) | Copepodamides via SPATT, Santa Cruz Wharf, 28 months | Chemical cue of grazers | Reportedly PN blooms 6 weeks ahead and mussel DA 7 weeks ahead, but **only 22% of high-toxin events** caught at 7 weeks vs 14% at 1 week by toxin tracking | Station time series | Long lead, **low recall**; numbers UNVERIFIED against the paper |
| Trapp et al. 2025, *Toxins* 17:511 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12567695/)) | Two Santa Cruz DA events (2023, 2024) | SPATT metabolomics | Similar events, **different drivers** | — | Non-stationarity within a single site |
| Scripps Pier random-forest study ([NOAA repo](https://repository.library.noaa.gov/view/noaa/64372/noaa_64372_DS1.pdf)) | HAB **taxa abundance** (not DA), Scripps Pier, Aug 2017 to Nov 2020 | RF ensembles; daily IFCB-type data (cf. SCCOOS `spc_hab_daily`) | Paper not opened in full. Nutricline depth mattered more than ambient nutrients (search summary) | — | The closest ML work to station level in CA; **no ML DA-at-pier forecast for California was found** |
| Stone et al. 2022, *Harmful Algae* 119:102334 ([strathprints](https://strathprints.strath.ac.uk/83364)) | PNW beach PN events via LiveOcean transport | Retrospective particle tracking | Sensitive to "beaching" definitions and thresholds | Retrospective | Transport-based beach forecasting; WA/OR only |
| Portugal RF (arXiv 2026, preprint) ([arXiv](https://arxiv.org/pdf/2607.07834)) | PN HAB occurrence | Satellite predictors | Mean ROC-AUC ≈ 0.74 | Preprint | Non-CA; satellite-only ceiling about 0.7–0.75 AUC |

### 2.2 Is station-level DA prediction feasible?

- **Nowcast to 1 week:** yes, with **in-situ inputs**. Evidence:
  - DA autocorrelation (the "most accurate at one week" baseline in the copepodamide study);
  - the 1-week transcript lead (Brunson 2024);
  - the 5–12-day C-HARM leads at Santa Cruz (Anderson 2016).
- **2–7 weeks:** the only evidence is the copepodamide study, and its recall is low (22%).
- **Satellite and model inputs alone, at the pier:** skill is weak and site-dependent (C-HARM v1: pDA AUC 0.77 at one pier, 0.04 at another; PN ≈ chance). The best non-CA satellite-only ML reports AUC ≈ 0.74.
- **No study reports a validated, operational, multi-station California DA forecast with out-of-sample skill by lead time.**

### 2.3 Has anyone validated interpolating offshore C-HARM to the shore?

**No. I found no such study.** The published evidence argues against it:
- Frolov et al. 2013: nearshore biomass is decoupled from biomass ~2–4 km offshore.
- Anderson et al. 2016:
  - nearest-pixel matchups are "only rough approximations";
  - Stearns Wharf pDA "rarely align[s] with offshore levels in the middle of the channel";
  - strandings occur "in the absence of nearshore pDA", which suggests an offshore or subsurface source.
- The SCCOOS bulletin: "nearshore data do not always correspond with C-HARM predictions for the open coast".
- The C-HARM toxin fields are themselves driven by WCOFS salinity, which lacks river freshwater near shore (Anderson 2016, for ROMS).
- Our current UI choice of "nearest offshore cell, with distance" is a display convenience, not a validated estimate.

### 2.4 CalHABMAP label volume (VERIFIED (query), SCCOOS ERDDAP, 2014-01-01 to 2026-10-12)

The table counts samples with pDA ≥ 0.5 ng/mL (= 500 ng/L, the C-HARM threshold) by year.

| Station | pDA samples | ≥ 0.5 ng/mL | By year | PN *seriata* samples / ≥ 10⁴ cells/L |
|---|---|---|---|---|
| Scripps | 657 | 9 | 2017:2, 2024:2, 2025:5 | 637 / 58 |
| Newport | 604 | 6 | 2021:1, 2023:1, 2025:4 | 595 / 110 |
| Santa Monica | 649 | 12 | 2014:1, 2017:5, 2025:5, 2026:1 | 659 / 69 |
| Stearns | 627 | 13 | 2014:1, 2017:6, 2021:1, 2024:1, 2025:4 | 633 / 96 |
| Cal Poly | 644 | 13 | 2015:2, 2016:4, 2023:2, 2025:4, 2026:1 | 653 / 76 |
| Santa Cruz | 621 | 46 | 2014–17: 29; 2018–22: 7; 2025–26: 10 | 568 / 65 |
| Monterey Wharf | 165 (ends 2022-08) | 39 | 2014–2017 only | 473 / 144 |
| Trinidad | 108 | 10 | 2020–2025 | n/a |
| Humboldt / S. Bay | 89 / 104 | 2 / 5 | — | n/a |

Totals: **4,268 pDA samples** and **155 threshold exceedances** across these 10 sites. Exceedances cluster in a few years per site (2015–2017, 2025). Caveats:
- Counts are rows. Duplicates at the same time were not removed, although distinct-time counts matched within 1%.
- The seriata group is not identical to toxigenic species.

### 2.5 Lessons for any CoastWatch station model

1. **Label scarcity and clustering.**
   - 6–13 positives per SCB pier over 12 years, concentrated in 1–3 bloom years.
   - Use pooled or hierarchical models with **leave-one-year-out (or leave-one-bloom-season-out) CV**.
   - Random k-fold leaks through week-to-week autocorrelation (4–17-day decorrelation, Frolov 2013).
   - Report PR-AUC and Brier score, not only accuracy. Under the class imbalance here, ROC AUC flatters.
2. **Irregular sampling and latency.**
   - Sampling is weekly with gaps (Monterey pDA stopped in 2022). Results arrive 1 week to more than 2 months late.
   - Any model using lagged pDA must use only values *available at issue time* (as-of joins), or the skill is fictitious.
   - C-HARM misses about 20% of days; OLCI keeps only 90 days on the NRT server.
3. **Regional non-stationarity.** Drivers flip sign between Monterey and the SCB (Sandoval-Belmar 2023), and between events at a single site (Trapp 2025).
4. **Proxy ≠ toxin.** PN counts exceed 10⁴ cells/L far more often than pDA exceeds 0.5 ng/mL. For example, Newport had 110 PN exceedances and only 6 pDA exceedances. Bloom proxies should not be presented as toxin risk (CLAUDE.md science rule).
5. **Spatial representativeness.** A pier sample represents its immediate vicinity only (Frolov 2013). Neither C-HARM cells nor satellite pixels should be labelled with pier values without a matchup-distance caveat.

---

## 3. Gaps I could not close

- No producer documentation of the C-HARM toxin mask or its cause (salinity-based masking is an inference).
- Lane et al. 2009 skill numbers, and the McKibben 2017 and copepodamide PNAS papers, were not opened; their figures come from slides or press.
- Whether the CA IFCB dashboard or HABDAC publish classified PN counts as an API.
- Current SCCOOS automated shore-station ERDDAP IDs and freshness.
- The Copernicus licence page; Sentinel-2C status; site-specific HF-radar first-range-cell distances.
- The transects are single shore-normal lines over one 3-month season, so they are indicative only. The `erdVHNchla1day` run returned only 12 time steps (server errors) and is not used.
