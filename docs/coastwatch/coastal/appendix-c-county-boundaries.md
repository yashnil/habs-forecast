# California county boundaries for CoastWatch: sources, download, and marine-jurisdiction caveats

Research date: 2026-10-12. Scratch only; no repository was edited.

## TL;DR

- **Recommended layer:** CDT **"California County Boundaries and Identifiers"** (the version without coastal buffers, cut at the shoreline). It has 58 features, one per county, all with valid geometry. The licence is **Public Domain** (the ArcGIS Hub lists it as CC0-1.0). The data was last edited on **2026-09-21**. The item metadata reports "Last Processed 4/30/2026, CDTFA updates as of 4/20/2026".
- Downloaded file: `ca_counties_shoreline_cdt.geojson` (local, not committed; the built layer is `coastwatch-web/public/basemap/ca-counties.json` in PR #34) (28,420,373 bytes, EPSG:4326, about 772k vertices). **It needs simplification or tiling before it ships to the web.**
- The "with Coastal Buffers" variant (81 features: 58 land + 15 ocean + 8 bay polygons) was also downloaded. **Do not use it on the map.** Its ocean polygons reach roughly **3 nautical miles** offshore, which looks like a statutory or regulatory water boundary. It is in fact a tax-jurisdiction artifact.
- County polygons are legitimate as **land reference / orientation context** and as a **lookup key for CDPH notices that are worded by county**. They are **not** fishery-closure, crab-zone, state-waters or advisory extents.

---

## 1. Authoritative datasets

### 1a. CDT "California County Boundaries and Identifiers" (recommended)

| Item | Value |
|---|---|
| Landing page | https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers (also `https://gis.data.ca.gov/datasets/60b7e0f3d33b4064a4b43bf14589bfe3_1`) |
| ArcGIS item | https://www.arcgis.com/home/item.html?id=60b7e0f3d33b4064a4b43bf14589bfe3 (owner `CDT_GIS`) |
| FeatureServer layer | `https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_County_Boundaries_and_Identifiers_Blue_Version_view/FeatureServer/1` (**layer id 1, not 0**; a layer-0 request returns HTTP 400) |
| Publisher | California Department of Technology (CDT), Office of Digital Services. Contact: gis@state.ca.gov. Boundary source: California Department of Tax and Fee Administration (CDTFA). Credits field: "California Department of Tax and Fee Administration, California Department of Technology" |
| Version / dates | Item created 2024-09-16; item modified 2026-04-30. Layer `editingInfo.lastEditDate` = 1790012459470, which is **2026-09-21 17:40:59 UTC**. Metadata says: "Last Processed: 4/30/2026 by Nick Santos using code path at https://github.com/Office-of-Digital-Services/gis-city-county at commit ce2e1f41… It incorporates updates from CDTFA as of 4/20/2026." The data is refreshed about every two weeks, and CDT says: "If you require unchanging point-in-time data, export a copy for your own use rather than using the service directly." The schema changed in February 2025 and is described as stable since then. Status page: https://gis.data.ca.gov/pages/city-and-county-boundary-data-status |
| Native CRS | EPSG:3310 (California Albers / Teale). The service reprojects on request (`outSR=4326`) |
| Geometry | 58 polygons, one per county (51 Polygon and 7 MultiPolygon). `OFFSHORE` is null for every feature |
| Coastline | **Cut at the shoreline.** CDT removes the CDTFA coastal buffers. In Northern California the county cut "runs near the Golden Gate Bridge", so San Francisco Bay water remains inside the bay counties. Where a county line runs parallel to the coastline, the CDTFA line is kept "even if it extends into the ocean a small amount" (for example Fort Bragg, Point Arena, SF, Pacifica, Half Moon Bay and Capitola) |
| Fields | `OBJECTID`, `CDTFA_COUNTY` ("Alameda County"), `CDTFA_COPRI` ("01000"), `CENSUS_PLACE_NAME`, `CENSUS_GEOID` (5-digit FIPS, "06001"), `CENSUS_PLACE_TYPE`, `GNIS_PLACE_NAME`, `GNIS_ID` (integer), `CDT_COUNTY_ABBR` (3-letter, e.g. "ALA"), `PRIMARY_DOMAIN` (null), `CENSUS_POPULATION` (null), `CDT_NAME_SHORT` ("Alameda"), `OFFSHORE`, `AREA_SQMI` (computed in EPSG:3310), `GlobalID` (**not persistent; do not key on it**. Use `CENSUS_GEOID` or `GNIS_ID`) |
| Formats / APIs | The query endpoint supports JSON, geoJSON and PBF (maxRecordCount 2000). Service exports: csv, shapefile, sqlite, geoPackage, filegdb, featureCollection, geojson, kml, excel. Hub download pattern: `https://gis.data.ca.gov/api/download/v1/items/60b7e0f3d33b4064a4b43bf14589bfe3/geojson?layers=1` (not tested; I used the FeatureServer query) |
| Accuracy caveat (quote) | "The boundaries in this map are representations of taxing jurisdictions for the purpose of determining sales and use tax rates and should not be used to determine precise city or county boundary line locations." County lines were "originally derived from a 1:24,000 accuracy dataset" |

**Licence / terms of use (quoted from item `licenseInfo`):**
> Public Domain. Every reasonable effort has been made to assure the accuracy of the maps and data provided; nevertheless, some information may not be accurate. The California Department of Technology assumes no responsibility arising from use of this information. THE MAPS AND ASSOCIATED DATA ARE PROVIDED WITHOUT WARRANTY OF ANY KIND, either expressed or implied, including but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Before making decisions using the information provided on this map, contact the California Department of Technology to confirm the validity of the data provided.

The ArcGIS Hub API record (`https://hub.arcgis.com/api/v3/datasets/60b7e0f3d33b4064a4b43bf14589bfe3_1`) reports `license: CC0-1.0` and `recordCount: 58`.

### 1b. CDT "California County Boundaries and Identifiers with Coastal Buffers"

| Item | Value |
|---|---|
| Landing page | https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers-with-coastal-buffers (`…/28c9f9dd8c3d4eb5a534cb30ddb3ce39_1`) |
| FeatureServer layer | `https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_County_Boundaries_and_Identifiers_with_Coastal_Buffers/FeatureServer/1` |
| Version | Item created 2024-10-24, modified 2026-04-30. Data last edited 2026-09-21, the same pipeline run as 1a |
| Licence | "Public Domain Government Work", with the same disclaimer as above. Hub lists CC0-1.0 |
| Schema / CRS | Identical to 1a. Native CRS is EPSG:3310 |
| Features | **81**: 58 land polygons, 15 `OFFSHORE='ocean'` and 8 `OFFSHORE='bay'`. The coastal counties have 2 or 3 rows each, sharing the same identifiers. CDT says to dissolve on every field except `OFFSHORE` and `AREA_SQMI` to get a single polygon per county |

**How the buffer differs (measured here).** The ocean buffers reproduce the CDTFA tax-boundary water extents. Measured from the CDT shoreline version, the median distance of the outer-edge vertices is about **3.0–3.2 nautical miles** (about 3.5 statute miles) for nearly every ocean county. Santa Cruz is the exception at about 2.6 nm. Maximum distances grow where the line closes across bays or wraps around islands and points: Del Norte 9.2 nm, Monterey 9.0, Santa Barbara 8.0, Santa Cruz 7.7, Humboldt 7.2. Ocean buffer areas range from about 149 sq mi (Orange) to 1,429 sq mi (Santa Barbara, which includes the Channel Islands waters). The bay buffers cover SF Bay, San Pablo Bay and Suisun Bay (Alameda, Contra Costa, Marin, SF, San Mateo, Santa Clara, Solano and Sonoma).

What the buffers are for: CDT describes them only as the parts of the CDTFA jurisdictions that lie in water, kept so that tax-rate-area / jurisdiction coverage is complete. The 3 nm figure lines up with the state seaward boundary (Gov. Code §170: three nautical miles from the baseline). Some county statutes describe the line as running "three English miles" into the Pacific (for example Gov. Code §23137, San Diego). These polygons are **not** an authoritative state-waters, fishery or advisory boundary. They are CDTFA tax-jurisdiction water extents, and CDT publishes no claim that they match Gov. Code §170 or the BOEM line. **Validity:** 4 of the 81 geometries are invalid ("Nested shells" in Los Angeles, San Francisco, Santa Barbara and Ventura, around the islands) and would need `make_valid` before use.

### 1c. Alternative: US Census Bureau (public domain)

| Product | URL | Vintage | Coast |
|---|---|---|---|
| Cartographic Boundary, 1:500k | https://www2.census.gov/geo/tiger/GENZ2025/shp/cb_2025_us_county_500k.zip (11.8 MB, all US) | 2025 (published 2026-04-23). 2024 and 2023 are also available | **Clipped to the shoreline** (generalized) |
| Cartographic Boundary, 1:5M | https://www2.census.gov/geo/tiger/GENZ2025/shp/cb_2025_us_county_5m.zip (3.0 MB) | 2025 | Clipped, very generalized |
| TIGER/Line counties | https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip (84 MB) | 2025 (2025-09-23) | **Includes offshore water** (`AWATER`) out to roughly the state limit, so it is unsuitable for a land layer |

Filter to California with `STATEFP='06'`. The key is `GEOID`, which matches CDT `CENSUS_GEOID`. Census geographic products are US federal works with no copyright (public domain, 17 U.S.C. §105). The suggested credit is "Source: U.S. Census Bureau, 2025 Cartographic Boundary Files". Census boundaries are statistical boundaries. The CDT/CDTFA layer is the state-maintained source and updates more often, so **CDT is preferred**. The Census CB 1:500k file is a fine, lighter fallback. It was not downloaded, because the CDT downloads succeeded.

Other copies seen (not authoritative for this purpose): the CalHHS republication of the same CDT layer (`bfd61a35…`), the older CDT "California County Boundaries" item (`eb861a3d…`, 2023), CAL FIRE (`8713ced9…`), Cal OES (`59d92c1b…`) and CDFW "Basic Plus" (`8783d3d6…`).

## 2. Downloads and verification

Directory: a local working folder (not committed)

| File | Exact URL | Downloaded (UTC) | Bytes | SHA-256 |
|---|---|---|---|---|
| `ca_counties_shoreline_cdt.geojson` | `https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_County_Boundaries_and_Identifiers_Blue_Version_view/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=geojson&returnGeometry=true` | 2026-10-12 14:21:25Z to 14:23:05Z (both files) | 28,420,373 | `1ebbbf55cf4e9593cd3d125982f39f0aa4e1d9160fa5d71d5e899c4932239fcf` |
| `ca_counties_coastal_buffers_cdt.geojson` | `https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_County_Boundaries_and_Identifiers_with_Coastal_Buffers/FeatureServer/1/query?where=1%3D1&outFields=*&outSR=4326&f=geojson&returnGeometry=true` | same window | 45,728,282 | `9558bec11f100fb35ba97eba54ff119542ad8949546ea4bee1e98805d9118ff5` |

Metadata snapshots in the same folder: `item_60b7e0f3….json`, `item_28c9f9dd….json`, `layer_*.json` and `download_time.txt`. Verification scripts are in a local working folder (not committed) (`verify_counties.py`, `buffer_extent.py`, `county_lines.py`, `cdph_cmp.py`).

Verification results for the shoreline file:
- 58 features. `exceededTransferLimit` is not set.
- 58 unique `CDT_NAME_SHORT` values, which **exactly match** the official list of 58 counties (Alameda … Yuba), with none missing and none extra. `CENSUS_GEOID` (06001 … 06115), `CDT_COUNTY_ABBR` and `GNIS_ID` are each unique (58 values).
- All 58 geometries are valid in shapely 2.1.2 (51 Polygon, 7 MultiPolygon).
- CRS member `EPSG:4326`. The coordinate order is lon/lat.
- Bbox: **[-124.40981, 32.53432, -114.13078, 42.00950]**. The buffer file's bbox is [-124.48202, 32.52884, -114.13078, 42.00950].
- About 771,926 vertices, so **simplify before shipping**. For example, mapshaper `-simplify` with topology kept so shared county edges stay coincident, or tippecanoe to PMTiles/MVT. Expect around 0.5–2 MB after simplification.

Where coastal county lines meet the shore (from the shoreline file): Del Norte/Humboldt 41°27.87′N; **Humboldt/Mendocino 40°00.09′N**; **Mendocino/Sonoma 38°46.12′N**; Sonoma/Marin 38°17.74′N; SF/San Mateo 37°42.49′N; **San Mateo/Santa Cruz 37°11.20′N**; Santa Cruz/Monterey 36°50.83′N; Monterey/SLO 35°47.73′N; SLO/Santa Barbara 34°58.46′N; Santa Barbara/Ventura 34°22.65′N; Ventura/LA 34°02.73′N; LA/Orange 33°44.49′N; Orange/San Diego 33°23.22′N.

## 3. Marine regulatory boundaries: how notices are actually defined

**CDPH (public-health advisories and the annual mussel quarantine).**
- Sources: https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx, the Recreational Bivalve Shellfish Advisory map https://experience.arcgis.com/experience/394836318cfe4f7494e1c09097a43559/, the Domoic Acid page https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx, and the Marine Biotoxin Monitoring Program https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx.
- Bivalve advisories are worded **by county**, for example "do not eat sport-harvested mussels, clams, scallops or oysters from Sonoma County". They refer to the county's beaches, bays and estuaries (shoreline harvest). They do **not** define any offshore distance. The annual mussel quarantine (normally May 1–Oct 31) covers "the entire coast of California … including all bays and estuaries".
- CDPH's own map does not use CDT counties. It uses its own generalized polygon layer, `https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services/California_Coastal_Counties/FeatureServer/3` ("Areas"), with 20 units. That layer **splits out "Northern Channel Islands" and "Southern Channel Islands" as separate advisory areas** from the mainland counties. Its polygons spill only slightly past the CDT shoreline (1–18% of area for mainland counties), which reflects generalization and bays, not a 3 nm band.
- Crab and other viscera advisories are worded with **county lines plus latitude or named points**. Examples: "from the California/Oregon border to the Mendocino/Sonoma county line", and "near Shelter Cove (40°01′ N) to Point Arena (38°57.5′ N)".

**CDFW fishery closures and delays.**
- Source: https://wildlife.ca.gov/fishing/ocean/health-advisories. Closures are made by CDFW (Director's declaration) on OEHHA's recommendation under Fish & Game Code §5523, with CDPH as the testing and health partner.
- They are defined by **latitude lines tied to named points or county lines**. Example from the page: rock crab closure "from the Mendocino/Humboldt county line (40° 00.00′ N. lat.) to Cape Mendocino, Humboldt County (40° 30.00′ N. lat.)". For the 2025–26 season, Dungeness crab in Biotoxin Management Area B was described as "from the southern boundary of the Reading Rock State Marine Protected Areas (41° 17.6′ N) to Cape Mendocino (40° 10.0′ N)" (OEHHA memo https://oehha.ca.gov/sites/default/files/media/2026-01/DGCrabCommMemo_B_FINAL010826%20Signed.pdf).
- Commercial delays usually apply to **state waters** (0–3 nm), and federal action may follow for 3–200 nm.
- **The "40°00′N" in notices is a regulatory latitude stand-in for the Humboldt/Mendocino county line.** The CDT county line actually meets the coast at 40°00.09′N, about 160 m north. The RAMP Zone 2/3 line is 38°46.125′N, while the CDT Mendocino/Sonoma coast point is 38°46.12′N. Regulatory lines should be drawn as **straight latitude lines**, not as county edges.

**CDFW Dungeness crab RAMP fishing zones** (14 CCR §132.8(a)(7)).
- Zone 1: OR border to Cape Mendocino 40°10′N.
- Zone 2: 40°10′N to the Sonoma/Mendocino line 38°46.125′N.
- Zone 3: 38°46.125′N to Pigeon Point 37°11′N.
- Zone 4: 37°11′N to Lopez Point 36°N.
- Zone 5: 36°N to Point Conception 34°27′N.
- Zone 6: Point Conception to the Mexico border.
- Zone 7 is the leatherback foraging area (Point Arena 38°57.5′N to Point Pinos 36°38.314′N).
- Zones run from the shore to 200 nm.
- GIS: CDFW BIOS ds3120. Metadata: https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3120.html. Service: https://services2.arcgis.com/Uq9r85Potqm3MfRV/ArcGIS/rest/services/biosds3120_fpu/FeatureServer/0. Data.gov: https://catalog.data.gov/dataset/risk-assessment-and-mitigation-program-ramp-fishing-zones-r7-cdfw-ds3120. The metadata says the layer is current as of September 2023.

**CDFW groundfish recreational management areas** (federal 50 CFR 660 Subpart G).
- Northern: OR to 40°10′N.
- Mendocino: 40°10′ to 38°57.5′N (Point Arena).
- San Francisco: to 37°11′N (Pigeon Point).
- Central: to 34°27′N, split at 36°N.
- Southern: to Mexico.
- GIS: BIOS ds3143, https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds3143_fpu/FeatureServer/0 (metadata https://ftp.dfg.ca.gov/Public/Metadata/CDFW_BIOS/ds3143_fgdc.xml).

**State waters / 3 nm line.**
- Legal basis: Gov. Code §170 (three nautical miles seaward of the baseline, closing across bays and around outer islands) and the federal Submerged Lands Act.
- Datasets:
  - BOEM "Digital Offshore Cadastre – Pacific83 – Submerged Lands Act (SLA) Boundary Line" (https://www.boem.gov/environment/pcsla). BOEM warns that the "GIS files are NOT an OFFICIAL record". Background: https://www.boem.gov/about-boem/fixing-california-sla-boundary.
  - NOAA Office for Coastal Management "U.S. State Submerged Lands", MarineCadastre (https://catalog.data.gov/dataset/u-s-state-submerged-lands, ZIP https://marinecadastre.gov/downloads/data/mc/USStateSubmergedLands.zip).
  - NOAA OCM "Federal and State Waters" (InPort 54383).
  - NOAA OCM "Coastal Zone Management Act Boundary" (InPort 53132, https://www.fisheries.noaa.gov/inport/item/53132). It is labelled unofficial.
  - All are US federal works or CC0. None of them was downloaded here.

**Marine Protected Areas.** CDFW BIOS ds582 (https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS0582.html; service https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds582_fpu/FeatureServer/0). Its licence is CC BY 4.0, so attribution is required. It is a best-effort representation and "not for navigation".

### Conclusion: what county polygons can legitimately do on the map

1. **Land-side reference and orientation.** Use subtle outlines and labels so users can find "Sonoma County" and similar places. Use the shoreline-cut version, so nothing is drawn over water.
2. **Interaction key for CDPH county-worded notices.** When CDPH says "Sonoma County", highlighting that county's **coastline** is a reasonable visual cue. A better cue is a shoreline segment, for example the CDT boundary intersected with the coast. Label it "advisory issued for Sonoma County (shore-harvested shellfish)". Caveat: CDPH also uses "Northern/Southern Channel Islands" units that county polygons cannot represent, so those need separate island geometries, for example from CDPH's own layer.
3. **Joining by `CENSUS_GEOID`** to other county data, such as economic exposure.

**Do not:**
- use county polygons, or the coastal-buffer variant, to show where a fishery is closed;
- extend counties offshore;
- treat a county edge as a closure line;
- imply a state-waters extent.

Draw CDFW closures and RAMP zones as latitude-bounded bands (from the regulatory coordinates or BIOS ds3120/ds3143), clipped to state waters (BOEM/NOAA 3 nm line) only where the notice says "state waters". Show the 3 nm line from BOEM/NOAA if needed.

## 4. Citation and attribution text

- **On-map attribution (short):** `County boundaries: CA Dept. of Technology / CDTFA (public domain)`
- **Sources/About page (full):** "California County Boundaries and Identifiers. California Department of Technology, Office of Digital Services; boundaries from the California Department of Tax and Fee Administration (CDTFA). Data last edited 2026-09-21 (CDTFA updates through 2026-04-20 per item metadata); downloaded 2026-10-12 from https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers. Public domain; provided without warranty. Boundaries represent sales-and-use-tax jurisdictions and are not legal survey lines."
- **Disclaimer to show near the layer toggle or legend:** "County lines are shown on land for reference only. They do not define fishery closures, crab zones, state waters, or the extent of health advisories. See the CDFW and CDPH notices for the exact areas."
- No attribution is legally required (public domain). The CDT disclaimer ("without warranty … contact CDT to confirm validity") should be kept in the docs or provenance.
- If Census is used instead: "Source: U.S. Census Bureau, 2025 Cartographic Boundary Files (cb_2025_us_county_500k)."
- If CDFW BIOS layers (ds3120, ds3143 or ds582) are added later, they need CC BY 4.0 attribution, for example "California Department of Fish and Wildlife, BIOS dsXXXX".
