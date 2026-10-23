# 02 · Offline field mode: what may be stored, and how

Research date 2026-10-16. Scope: one harbour pack (about 40 × 40 km around Monterey, z8–14) stored in the
browser (Cache Storage / IndexedDB) by the Neritic v4 PWA. **[V]** = verified against the cited page today.
**[U]** = unverified or my interpretation. This is not legal advice. Nothing here has been checked by counsel.

## 1. Sources: may we cache them offline?

### Basemap and cartography

| Source | Offline cache? | Conditions / attribution | Evidence |
|---|---|---|---|
| **OpenFreeMap public tiles** (`tiles.openfreemap.org/planet`) | **Yes, with limits [V]** | Each download can be at most 3,000 tiles or 50 MB, use at most 4 parallel requests, and send a unique User-Agent. For more, email info@openfreemap.org. Other automated collection is prohibited. | [openfreemap.org/tos](https://openfreemap.org/tos/): "Your app may let users download areas for offline use if each download is: at most 3,000 tiles or 50 MB · made with at most 4 parallel requests · sent with a unique User-Agent identifying your app". Prohibited: "Collect data from the service in automated ways, except for offline downloads". The FAQ at [openfreemap.org](https://openfreemap.org/) says the same. |
| **OpenFreeMap planet downloads** (Btrfs / MBTiles / **PMTiles**, weekly) | **Yes, self-host an extract [V]** | The project code is MIT. Data is OSM (ODbL) and the schema is OpenMapTiles. Keep the attribution. Runs are deleted after about 4 weeks (monthly ones after 6 months), so pin and copy the run you use. | [GitHub README](https://github.com/hyperknot/openfreemap): "You can download them in Btrfs, MBTiles and PMTiles formats… `https://btrfs.openfreemap.com/areas/planet/{version}/tiles.btrfs (and .mbtiles, .pmtiles)`". Checked today: `areas/planet/20261011_080001_pt/tiles.pmtiles` is 86,991,227,316 bytes, `accept-ranges: bytes`. |
| **OpenStreetMap data** (inside every tile) | Yes [V] | ODbL. Credit "© OpenStreetMap contributors". **[U]** Publishing a regional extract file is distributing a derivative database or Produced Work. Keep it under ODbL and say so in the pack credits. | [osm.org/copyright](https://www.openstreetmap.org/copyright); Protomaps: "Open Database License Produced Work… OpenStreetMap attribution required" ([docs](https://docs.protomaps.com/basemaps/downloads)). |
| **OpenMapTiles schema / design** | Yes [V] | Design is CC-BY 4.0. "visibly credit 'OpenMapTiles.org'… the credit should appear in the corner of the map." | [OpenMapTiles LICENSE.md](https://github.com/openmaptiles/openmaptiles/blob/master/LICENSE.md) |
| **Glyph PBFs (Noto Sans)** | **Yes, bundle [V]** | SIL OFL 1.1. Fonts can be bundled with software but "may not be sold by itself". Keep the licence text. We already self-host 8 ranges × 2 stacks (`coastwatch-web/public/fonts`, about 1.2 MB). | [OFL.txt](https://github.com/notofonts/latin-greek-cyrillic/blob/main/OFL.txt): "can be bundled, embedded, redistributed and/or sold with any software". The OFM LICENSE says "Noto Sans is released under the SIL Open Font License, Version 1.1". |
| **Protomaps basemap builds** | Yes [V] | Licensed as an ODbL Produced Work with OSM attribution. "hotlinking to these downloads are discouraged", so copy to your own storage. | [docs.protomaps.com/basemaps/downloads](https://docs.protomaps.com/basemaps/downloads) |
| **ETOPO 2022** (relief rasters) | **Yes [V]** | No copyright. Cite NCEI (DOI 10.25921/fd45-gt74). Not for legal use. | ERDDAP `ETOPO_2022_v1_15s` licence: "The data may be used and redistributed for free but is not intended for legal use". [NCEI product page](https://www.ncei.noaa.gov/products/etopo-global-relief-model) gives the citation. |
| **NOAA Coastal Relief Model** (Monterey inset) | **Yes [V]** | No US copyright. Cite NCEI. "**cannot be used for navigation**", which matters for a fishing app. | [NCEI CRM](https://www.ncei.noaa.gov/products/coastal-relief-model): "Products derived primarily from NCEI data are not subject to copyright protection in the United States." "The U.S. CRM cannot be used for navigation." **[U]** The page summary placed Vol. 7 in the "Central Pacific (Hawaii)". Re-check which volume covers Monterey before quoting the source in the credits. |

### Data snapshots

| Source | Offline cache? | Conditions | Evidence |
|---|---|---|---|
| **NOAA C-HARM** (CoastWatch ERDDAP `wvcharmV3_*`) | Yes [V] | Free use and redistribution, no warranty. Our own rules still apply: it is a model nowcast/forecast of *probability*, not toxin. | [ERDDAP info](https://coastwatch.pfeg.noaa.gov/erddap/info/wvcharmV3_1day/index.html): "may be used and redistributed for free but is not intended for legal use" |
| **NOAA CO-OPS** (tides, water level) | Yes [V] | Public domain. "shall not be modified in content and then presented as official government material." NOS "requests that attribution be given". Label preliminary data as preliminary. | [tidesandcurrents.noaa.gov/disclaimers.html](https://tidesandcurrents.noaa.gov/disclaimers.html) |
| **NWS marine forecasts** | Yes [V] | Public domain. Do not alter the text and present it as official. Do not use the NWS logo to imply endorsement. Material incorporated in our product must be identified (17 U.S.C. § 403). | [weather.gov/disclaimer](https://www.weather.gov/disclaimer) |
| **NDBC** buoys / HF radar | Yes **[U]** | Presumed public domain like other NOAA/NWS data. I did not fetch NDBC's own disclaimer page. | (to verify: ndbc.noaa.gov/disclaimer) |
| **CalHABMAP / SCCOOS ERDDAP** | Yes [V] for the dataset checked | Same ERDDAP default licence: "may be used and redistributed for free". There is no acknowledgement attribute; credit CalHABMAP/SCCOOS as a courtesy. **[U]** I did not find an SCCOOS-wide data policy, so check each dataset's `license` attribute in the pipeline. | [erddap.sccoos.org … HABs-MontereyWharf](https://erddap.sccoos.org/erddap/info/HABs-MontereyWharf/index.html) |
| **CDPH / CDFW web pages** | Yes, for text and facts [V] | "information presented on this web site, unless otherwise indicated, is considered in the public domain." Photos may be third-party copyright. CDFW: "official seals and marks may not be used without permission". Cache our *extracted* notice records, not their page chrome or images. | CDPH [conditions of use](https://www.cdph.ca.gov/Pages/conditions-of-use.aspx) (fetched with curl; TLS chain fails in WebFetch); CDFW [Conditions-of-Use](https://wildlife.ca.gov/Conditions-of-Use) |

**Summary.** Nothing blocks a one-harbour pack legally. The only quantitative restriction is OpenFreeMap's
**public-instance** offline rule. Every federal and state source allows redistribution, subject to: no
alteration presented as official, attribution, and "not for navigation / not for legal use".

## 2. Measured size, Monterey box

Bbox `-122.124,36.440,-121.676,36.800` (±20 km around 36.62 N, -121.90 W). I downloaded every z0–14
tile once from OpenFreeMap, run `20261011_080001_pt`. That is 628 tiles, sent 4 at a time with an
identifying User-Agent, so the download stayed within the ToS. **[V] measured:**

| Zoom | Tiles | gzip KiB | raw KiB |
|---|---|---|---|
| 0–7 | 10 | 1,334 | 2,406 |
| 8–10 | 8 | 466 | 672 |
| 11 | 12 | 203 | 286 |
| 12 | 36 | 346 | 481 |
| 13 | 121 | 1,021 | 1,385 |
| 14 | 441 | 3,506 | 5,112 |
| **Total** | **628** | **6.7 MiB** | **10.1 MiB** |

OpenFreeMap stops at z14; MapLibre overzooms beyond it. Other parts of the pack:
- Glyphs. Regular + Italic, 8 ranges each, are already in `public/fonts` (about 1.2 MB). Latin-only 0-255 + 8192-8447 is about 0.22 MB for the two stacks.
- Sprite. 28 KB JSON + 119 KB @2x PNG, only if we use OFM icons.
- Relief and isobaths. 0.4 MB (`relief-*.webp` + `isobaths.geojson`).
- Data snapshot. **[U]** <1 MB.

**One-harbour pack ≈ 9–13 MB.** A 40 km box at z15 would add about 1,700 tiles and about 13 MB **[U, extrapolated ×4]**, which is not worth it. The whole California coast at z0–12 is **[U]** tens of MB.

## 3. Recommended architecture

**Option A, recommended: self-hosted regional PMTiles per harbour.**
1. **Build, offline in the pipeline, not in the browser.** Run
   `pmtiles extract https://btrfs.openfreemap.com/areas/planet/<run>/tiles.pmtiles monterey.pmtiles --bbox=-122.124,36.44,-121.676,36.8 --maxzoom=14`.
   The CLI fetches only the byte ranges it needs ("The source archive may be local or remote… must be clustered"; [CLI docs](https://docs.protomaps.com/pmtiles/cli)). Expect about 7 MB, since tiles stay gzipped, plus a small directory. Pin the run id and record it in the pack manifest. Rebuild monthly at most.
   **[U]** I have not run it, because the `pmtiles` CLI is not installed here.
2. **Host** the file on our own static origin (Vercel `public/` or R2), next to glyphs, relief and a
   `pack.json`. The pack.json holds the version, bbox, bytes, sha256, data `as_of` and the attribution strings. Keep it as a *separate* file; never add fields to the existing pipeline manifest (manifest-schema rule).
3. **Download** when the user taps "Save Monterey for offline". `fetch` each pack file, check the sha256 and
   `cache.put()` it into a versioned Cache Storage bucket (`neritic-pack-monterey-v<n>`). Check
   `navigator.storage.estimate()` first, call `navigator.storage.persist()` after the download, and show the result honestly.
4. **Read** in MapLibre with the `pmtiles` library's `Protocol`. Register it once with
   `maplibregl.addProtocol("pmtiles", protocol.tile)` ([docs](https://docs.protomaps.com/pmtiles/maplibre)). Give it a `PMTiles` instance built on a **custom `Source`** whose `getBytes(offset, length)` returns `blob.slice()` of the cached Response's blob (`protocol.add(p)`). The JS library exports `Source`, `FileSource` and `FetchSource` in `js/src/index.ts`, and `Protocol.add(p: PMTiles)` in `adapters.ts` **[V, source code]**. The style points `omt` at `pmtiles://monterey` when the pack is active, and at OFM online otherwise.
   This needs **no service-worker tile interception**, which is useful because Cache API `match()` does not serve HTTP Range requests (byte-range slicing would have to be hand-written in the SW) **[U, from spec knowledge]**.
5. **Service worker.** A hand-written `public/sw.js` precaches the app shell (the field-mode route, JS/CSS chunks, the style, glyphs and relief) and serves it network-first with a cache fallback. Next's PWA guide uses a hand-written SW and says that for full offline caching "one option is Serwist" (Turbopack + webpack examples). It sets `Cache-Control: no-cache, no-store, must-revalidate` on `/sw.js` ([nextjs.org PWA guide](https://nextjs.org/docs/app/guides/progressive-web-apps), updated 2026-07-30) **[V]**. App Router chunk names are hashed per build. Either generate the precache list at build time (Serwist does this) or cache chunks at runtime the first time they load. **[U]** A hand-written SW can't know the hashed chunk list without a build step.
6. **Data.** Store the last-good JSON snapshots (C-HARM pixel series, CO-OPS tide predictions for N days, NWS
   zone text, CalHABMAP latest, official notices) in IndexedDB with their source `as_of`. Offline UI must
   say "Saved <time> — not live" and must never relabel observations as forecasts. Tide *predictions* stay valid offline; observations go stale.
7. **Attribution offline.** MapLibre's attribution control shows the source `attribution` strings. Keep the OSM, OpenMapTiles,
   OpenFreeMap and NOAA NCEI credits on the pmtiles source too, plus "Not for navigation" in the pack sheet.

**Option B, fallback: prefetch the 628 OFM tiles directly from the browser.** This is permitted (628 < 3,000, 6.7 MB < 50 MB, 4 parallel). Problems:
- (a) The ToS requires a "unique User-Agent identifying your app". Browsers in practice do not let page or SW script override `User-Agent` on `fetch`, so we cannot comply literally **[U, ask OFM]**.
- (b) Tile URLs carry the weekly run id (`/planet/20261011_080001_pt/…`), so a cached pack misses once the TileJSON advances. We would have to pin the TileJSON offline too, and old runs are deleted.
- (c) Every user hits OFM's free infrastructure.

Option A sends one request per user to *our* host, and no user traffic goes to OFM.

**Option C, Protomaps build** as the source: same ODbL terms, but a different schema, so `basemap.ts` would need restyling. Not needed.

## 4. Quotas and eviction

| Fact | Source |
|---|---|
| Chromium: one origin can use up to 60% of total disk, and the browser at most 80%. | [MDN quotas](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria) [V] |
| WebKit (Safari 17+): browser-app origin quota is up to 60% of disk and the overall quota up to 80%. Other apps that embed web content get 15% / 20%. A Home Screen web app "has the same origin quota and overall quota as when it is opened in a browser app." | [WebKit storage policy](https://webkit.org/blog/14403/updates-to-storage-policy/) [V] |
| WebKit eviction is LRU by origin, under quota or storage pressure, or after ITP finds no use "for some time". Origins with persistent storage can be excluded. | same [V] |
| Safari 7-day rule: with tracking prevention on, an origin with no user interaction "in the last seven days of browser use" loses script-written data. web.dev says this "does not apply to installed PWAs". | MDN [V]; [web.dev storage](https://web.dev/articles/storage-for-the-web) [V] |
| `persist()`: WebKit "grants a request based on heuristics like whether the website is opened as a Home Screen Web App". Chromium grants it silently on engagement, install/bookmark or notification permission, otherwise "silently denied". Firefox prompts the user. | WebKit blog [V]; [web.dev persistent-storage](https://web.dev/articles/persistent-storage) [V] |
| `estimate()` returns usage and quota. The quota "is an upper limit, not a guarantee", so handle `QuotaExceededError`. Browsers pad cross-origin data in usage reports. | WebKit blog, MDN [V] |

What this means for us: about 13 MB is far below every quota. **The real risk is eviction, not size.** Tell iOS users to install to the Home Screen; that gives the 7-day exemption and the persist() heuristic. Re-check that the pack exists (`caches.has`) on every launch and show "Pack missing, re-download" rather than a blank sea. **[U]** Safari 18/26 behaviour has not been checked on a device.

## 5. Open risks

1. **[U]** The OFM User-Agent clause cannot be met from a browser, which is why Option A (no user traffic to OFM) is recommended. If we ship B, email info@openfreemap.org first.
2. **[U]** ODbL obligations for a *published* extract file. Keep it unmodified, under ODbL, with a licence note in `pack.json` and the credits.
3. **"Not for navigation."** The CRM, ETOPO and our basemap are not charts. Field mode must say so; fishermen will use it at sea.
4. **Staleness.** Offline data must show its `as_of` time and must never imply a live or forecast status. C-HARM stays a probability, never toxin. Official notices: show their saved status only, and never infer lifted or expired.
5. **OFM run retention** (about 4 weeks): keep our extract as the source of truth.
6. **[U]** Not measured or checked: `pmtiles extract` output size, the NDBC disclaimer, the SCCOOS-wide policy, CRM volume numbering.
7. **[U]** iOS: an installed PWA has storage separate from Safari, so download the pack inside the installed app.
8. Do not cache opaque (no-cors) responses. Serve pack files from our own origin.
