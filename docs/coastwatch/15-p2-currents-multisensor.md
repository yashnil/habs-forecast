# 15 — P2: observed currents, multi-sensor satellite view, pipeline health

Status:
- **Branch:** `feat/coastwatch-p2-currents-multisensor`, draft PR #13. **Not merged.**
- **Production:** current datasets are not published to production; publishing them needs your approval.
- **Website:** not deployed.

P1 (PR #11) is merged and publishing in production (section 1).

## 1. P1 integration and production publication

**Pre-merge checks** (all passed; details in the PR #11 body and [14-p1-ocean-map.md](14-p1-ocean-map.md)):

- **Inventory.** Before publishing, `check-published` now validates every tile, not only samples: each tile directory holds exactly `n_tiles` files and each one decodes. Staging had 228 files plus 1,614 of 1,614 tiles. The earlier "2,882 satellite files vs 217 checked" was sample-only tile checks plus the previous run's artifacts, which prune keeps for caches.
- **Pages size.** Typical 35–60 MB. The worst case, a cloud-free week with the previous run kept, is about 300 MB, roughly 30 % of the 1 GB limit.
- **Stale or failed data never reads as recent.** Freshness follows observation dates. A product the last run didn't refresh says so. Carried-over layers keep their dates.
- **No regressions.** All 12 C-HARM grids are byte-identical between `main` and the branch on the same live data, and official-notice records are identical to production.
- **Repeated 403s.** At most 4 attempts per request; a per-run circuit breaker per ERDDAP dataset; refused sources keep their files byte for byte.

**Merged** as `2deb2ba` (merge commit).

**First production refreshes:**

| Run | Result |
|---|---|
| 38000436793 (dispatched) | Published, but NOAA refused the runner: C-HARM HTTP 403, carried over with its own issue date; Sentinel-3 HTTP 502 and VIIRS 403. Production had no previous satellite layers, so **none were published**. Honest, but empty. |
| 38000446094 (scheduled) | **Every source updated.** C-HARM 156/156 point comparisons. Satellite 108/108. 228 files plus 1,614 tiles validated before publishing; 217 OK at the public URL with CORS. |
| Pages 38002073718 | Deployed and checked. `https://yashnil.github.io/habs-forecast/v1/manifest.json` serves run 38000446094: Sentinel-3 observed 2026-10-08, VIIRS 2026-10-04, CORS `*`. |

## 2. Multi-sensor satellite view

**Rule, per pixel:**
- Sentinel-3 300 m if it has a valid observation in its 7-day window, unless VIIRS observed that place more than 2 days more recently;
- otherwise VIIRS 750 m;
- otherwise nothing.

**What it is, and isn't:**
- **One sensor per pixel.** Each pixel is one sensor's own published value and date. Nothing is averaged, blended, resampled or bias-corrected.
- **Native grids kept.** Tiles are drawn with each sensor's own grid (`render_stack`), so VIIRS cells stay 750 m.
- **Members still published.** The individual Sentinel-3 and VIIRS layers remain in the dataset and are what the inspector reads. It names the sensor shown, its value, date and resolution, and what the other sensor has at the same place.
- **Gaps stay gaps.** Places neither sensor saw remain unobserved.
- **Overlays.** "Show which sensor" and "Show each pixel's observation date" are categorical overlays.

**Verification.** `verify-satellite` re-derives the display pixel by pixel from the member grids and checks the sensor, colour and age tiles: 384 pixels per run, 0 failures, locally statewide and in staging.

**Live statewide comparison** (2026-10-09; evidence in [p2/evidence/multisensor-live-statewide.json](p2/evidence/multisensor-live-statewide.json)):

| Region | Sentinel-3 alone | VIIRS alone | Either | VIIRS only |
|---|---|---|---|---|
| Whole domain | 22.7 % | 74.9 % | 76.0 % | 53.2 % |
| North Coast | 29.0 % | 33.4 % | 47.8 % | 18.8 % |
| Mendocino–Sonoma | 15.7 % | 53.2 % | 56.6 % | 41.0 % |
| SF & Farallones | 0.8 % | 94.3 % | 94.4 % | 93.6 % |
| Monterey Bay | 30.6 % | 93.5 % | 95.4 % | 64.8 % |
| Central Coast | 10.4 % | 89.2 % | 89.7 % | 79.3 % |
| Southern California | 98.6 % | 99.4 % | 99.8 % | 1.2 % |

**More complete is not more accurate.** Same-day agreement, where both sensors saw a VIIRS cell, comparing the geometric mean of the Sentinel-3 pixels inside it:

| Region | Pairs | Sentinel-3 relative to VIIRS (median) | RMSD (log10) | r (log10) |
|---|---|---|---|---|
| Whole domain | 14,696 | **53 % lower** (log ratio −0.33) | 0.37 | 0.81 |
| North Coast | 86 | 39 % lower (−0.21) | 0.72 | 0.91 |
| Southern California | 58 | **2.4× higher** (+0.38) | 0.47 | 0.82 |

- **The sensors are not interchangeable.** They correlate well but differ by a factor of 2–5 at the same place and day, and the sign of the difference changes by region.
- **VIIRS fill is old.** The VIIRS pixels shown are a median 5 days old (publication latency), against Sentinel-3's 2 days.

**Decision.** The multi-sensor view is offered as a **labelled option, not the default**. The panel states the coverage gained, the VIIRS age, and the agreement in words: "Sentinel-3 read about 53 % lower than VIIRS … a colour step at a sensor edge may be the sensors, not the water." Sentinel-3 300 m stays the default.

**Size.** About 8.6 MB and 1,614 files (display, sensor and age tiles).

## 3. Observed ocean currents (HF radar)

**Source.** HFRNet total vectors, 2 km, hourly, via NOAA CoastWatch ERDDAP `ucsdHfrW2`.

**Availability (2026-10-09):**
- **History:** a rolling ~92 days, hourly.
- **Latency:** newest hour about 6 h old (4–15 h seen).
- **Volume:** 9,200–11,200 valid cells statewide every hour, and hours don't fill in later.
- **Coverage this hour:**

| Region | Coverage |
|---|---|
| Monterey Bay | 82 % |
| Central Coast | 81 % |
| SF & Farallones | 47 % |
| Southern California | 33 % |
| Mendocino–Sonoma | 19 % |
| North Coast | **0 %** (no radar totals, in the recording and live) |

- **Upstream precision:** values are reported to 1 cm/s.
- **Upstream QC:** HFRNet already requires at least 2 radars; no cell has HDOP above 1.24.
- **Backup host:** HFRNet's own THREDDS server timed out from here, so there is no second source today.

**Ingestion (`hf_radar`):**
- **Exact requests.** Grid indices are requested exactly; latitude, longitude and time axes are checked against the dataset's own.
- **QC.** Cells are re-checked: fill values, ≥ 2 radars, HDOP ≤ 1.25, speed ≤ 2 m/s. Failures are dropped and counted, never repaired.
- **Hours.** The last 24 hours are one layer each, with u and v grids of about 15 KB each.
- **24-hour mean.** Published where a cell has at least 18 valid hours.
- **No filling.** No interpolation in space or time. An hour with no valid cell is missing, not empty.
- **Reuse.** Hours older than 6 h are reused.
- **Outages.** The previous hours are kept with their own times.
- **Cost.** About 7 s statewide locally (55 s on the runner), about 0.8 MB published per day of hours.

**Accuracy checks (`verify-currents`, in the data workflow):**
- **Live point queries:** value and cell centre for sample cells. Locally **63/63**; staging run 38008465866 **32/32**.
- **Empty cells:** cells published without a value are either empty upstream or fail a documented check.
- **Mean:** the 24 h mean is re-derived from the published hours.
- **Unreachable upstream:** an ERDDAP that disappears mid-check, as happened in run 38009420295, is reported as UNVERIFIABLE and blocks publication rather than counting as a mismatch.

**Map:**
- **Arrows (default).** Built in the browser from the grids: one per 2 km cell when zoomed in, thinned at fixed grid positions when zoomed out (to about 32 km statewide). Length and opacity show speed; there is no colour ramp, so currents never read as chlorophyll or risk. The legend is in m/s, with knots.
- **Animated flow (optional).** Particles move through the one observed field shown, take their cell's value, vanish where there is no observation, live 1–2 s and move at a fixed screen speed per m/s. It's off under reduced motion.
- **Why arrows are the default.** Particles imply continuity and trajectories that hourly snapshots don't support. They are also unreadable in a still image and cost CPU on phones. Arrows show exactly one observed value per cell.
- **Time controls.** Hourly selection (previous/next and a slider over the hours that exist) or the 24-hour mean, each with its observation time in PT and UTC, age in hours and freshness. Every view says "observed, not a forecast".
- **Inspector.** Speed (m/s and knots), direction toward, observation hour, or "no data, not calm water".

## 4. Forecast currents (WCOFS)

The full write-up is [p2/wcofs-evaluation.md](p2/wcofs-evaluation.md).

- **Method.** Two runs, 44 forecast steps, four regions. Hourly and daily-mean comparisons against persistence and zero baselines.
- **Against persistence.** WCOFS does not beat persistence in any region except, marginally, the Gulf of the Farallones.
- **Monterey Bay.** The daily-mean pattern is rotated 134–167°: the bay circulation isn't resolved at 4 km. That explains P1's weak agreement; it is not a convention bug, since the shelf comparison is rotated only 6–8°.
- **Status.** Research only.
- **Validation gate defined.** At least 60 days covering both seasons; skill against latency-aware persistence at 24–48 h; mean rotation within ±30°; results by bays, shelf and offshore. The hindcast would be about 50 GB and 2 hours on free infrastructure.

## 5. Reliability and monitoring

**`cwp health`** runs every data refresh. It writes `v1/health.json` (history) and raises alerts. Thresholds and rationale are in [`health.py`](../../pipeline/coastwatch_pipeline/health.py):

| Alert | Rule | Not an alert |
|---|---|---|
| source_failing / source_degraded | A NOAA source fully or partly fails 3 runs in a row (about 18 h) | One refused run (routine on ERDDAP) |
| satellite_stale | Newest Sentinel-3 pixel older than 4 days, VIIRS older than 8 | Low coverage from cloud or fog |
| charm_missing | Newest C-HARM issue older than 2 days | Missing leads within a run |
| hfr_stale | Newest HF-radar hour older than 18 h | Normal 4–15 h latency |
| hfr_coverage_drop | Newest hour under half the 24 h median of valid cells | Hour-to-hour variation |
| hfr_region_lost | A region that had radar coverage before has none for 24 h | Regions never covered (North Coast today) |
| publish_stale | Starting dataset older than 14 h (missed runs) | — |

**Notifications (production only):**
- one "CoastWatch pipeline health" issue: opened when an alert appears, commented only when alerts change, closed when clear;
- a "CoastWatch data refresh failed" issue when the refresh fails. Verification failures block publication on purpose, and the last valid data stay live with their own dates.

**Hardening:**
- the build job times out after 40 minutes;
- each download has a 240 s total limit (a trickling upstream stalled one staging run for 50 minutes);
- progress and slow fetches are logged;
- an "unknown datasetID" 404 is reported as an outage, not as "no new data".

## 6. Tests and CI

| Suite | Result |
|---|---|
| Pipeline `pytest` | **184 passed**: multi-sensor (rule table, pixel-level parity in three regions, 750 m cells kept, members kept, unobserved kept, agreement recovers a known ratio), currents (real recorded responses for Monterey Bay, the North Coast and the Southern California Bight: upstream values, window, mean, QC drops, misaligned or mistimed responses rejected, direction convention, no-coverage region, reuse, outage carry-over, vanished dataset unverifiable), health (thresholds, no alert for one failure or for clouds, persistence and clearing), download time cap |
| Web unit (`vitest`) | **96 passed**, including rule parity with the pipeline, arrow thinning levels, cell sampling without interpolation, and sensor colours equal to the pipeline's |
| Playwright e2e | **83 passed**, including `p2-multisensor` and `p2-currents`: labels, tiles, overlays, hourly selection, mean, particles off under reduced motion, inspector values equal to the published grids, stale state, no-currents dataset; **axe clean** on both new panels at 1440 and 390 |
| `tsc`, `eslint` | clean |

## 7. Screenshots

These are of the running app against the **staging dataset**, real NOAA data from run 38010854678 on 2026-10-10 at 00:5x UTC. They are in [p2/screenshots/](p2/screenshots/).

| View | 1440 × 900 | 1280 × 720 | 390 × 844 |
|---|---|---|---|
| Sentinel-3 300 m (default satellite view, for comparison) | [02](p2/screenshots/desktop-1440/02-map-satellite.png) | [02](p2/screenshots/laptop-1280/02-map-satellite.png) | [02](p2/screenshots/mobile-390/02-map-satellite.png) |
| Multi-sensor, Monterey Bay | [11](p2/screenshots/desktop-1440/11-map-multisensor.png) | [11](p2/screenshots/laptop-1280/11-map-multisensor.png) | [11](p2/screenshots/mobile-390/11-map-multisensor.png) |
| Multi-sensor, which sensor | [12](p2/screenshots/desktop-1440/12-map-multisensor-sensor.png) | | [12](p2/screenshots/mobile-390/12-map-multisensor-sensor.png) |
| Multi-sensor, statewide | [13](p2/screenshots/desktop-1440/13-map-statewide-multisensor.png) | [13](p2/screenshots/laptop-1280/13-map-statewide-multisensor.png) | |
| Currents, newest hour (arrows) | [14](p2/screenshots/desktop-1440/14-map-currents.png) | [14](p2/screenshots/laptop-1280/14-map-currents.png) | [14](p2/screenshots/mobile-390/14-map-currents.png) |
| Currents, 24-hour mean | [15](p2/screenshots/desktop-1440/15-map-currents-mean.png) | | [15](p2/screenshots/mobile-390/15-map-currents-mean.png) |
| Currents, animated flow (optional) | [16](p2/screenshots/desktop-1440/16-map-currents-flow.png) | [16](p2/screenshots/laptop-1280/16-map-currents-flow.png) | |
| Currents, statewide | [17](p2/screenshots/desktop-1440/17-map-currents-statewide.png) | [17](p2/screenshots/laptop-1280/17-map-currents-statewide.png) | |
| Currents with a port selected (inspector) | [18](p2/screenshots/desktop-1440/18-map-currents-point.png) | [18](p2/screenshots/laptop-1280/18-map-currents-point.png) | [18](p2/screenshots/mobile-390/18-map-currents-point.png) |
| Empty state: North Coast, no radar coverage | [19](p2/screenshots/desktop-1440/19-map-currents-north-coast-gap.png) | | [19](p2/screenshots/mobile-390/19-map-currents-north-coast-gap.png) |

**Review notes:**
- **Coastline:** above all data in every view.
- **Legends:** each matches what is drawn. Arrows use glyphs in m/s with knots; flow uses streaks; the sensor overlay uses swatches.
- **Provenance and freshness:** shown in every panel. For currents the hour age is in hours; the badge omits a misleading calendar-day count.
- **Empty states:** North Coast reads "0 % observed this hour" with no arrows; an inspector point without radar says "no data, not calm water".
- **Mobile:** the collapsed dock keeps the hour, time, legend and the "not a forecast" line.
- **Arrows:** statewide arrows thin to about 32 km and are drawn smaller; Monterey Bay shows one arrow per 4 km at default zoom.

**Staging runs on this branch:**

| Run | Result |
|---|---|
| 38004433095 | Stuck 50 min in the pipeline step with no output (cancelled). This led to the 240 s download cap, progress logging and the 40-minute job timeout. |
| 38008465866 | Success. Pipeline about 4 min; C-HARM 156/156; satellite 396 checks (384 multi-sensor pixels); currents 32/32; 296 files plus 3,038 tiles; health clear. Sentinel-3 404 "unknown datasetID" was reported as "no scenes", which led to the outage fix. |
| 38009420295 | Blocked publication (correct): `ucsdHfrW2` vanished mid-verification and the 404s counted as mismatches. This led to the UNVERIFIABLE fix. |
| **38010854678** | **Final. Success, all 9 sources updated.** C-HARM 156/156; satellite 408 checks (384 multi-sensor pixels); currents 6 new hours, 18 reused, 32/32 live checks; 271 files plus 2,842 tiles validated before publishing, 258 OK at the public URL with CORS; health "All checks clear". |

## 8. NOAA reliability: what happened on 2026-10-09

- **Sentinel-3 sectors** on `coastwatch.noaa.gov` were "Currently unknown datasetID" twice. The second outage was still ongoing at 00:40 UTC on 10-10. The central server also returned **502 Proxy Error** during the first production run.
- **HF radar `ucsdHfrW2`** was "Currently unknown datasetID" at 00:38 UTC on 10-10.
- **Runner refusals.** `coastwatch.pfeg.noaa.gov` refused GitHub's runners (**HTTP 403**) for C-HARM and VIIRS in several whole runs, while answering a desktop.
- **No backup.** HFRNet's THREDDS server did not answer at all.
- **What the pipeline does.** It keeps the last valid data with their dates, says what failed, and alerts only on sustained problems. It cannot make NOAA answer.

## 9. Recommendation for the next phase

1. **Approve publishing HF-radar observed currents to production.** Verified in staging, small (about 1 MB/day), honest about gaps. If approved, merge PR #13, or a currents-only subset if you prefer to keep multi-sensor off. I would also watch the first week of health issues.
2. **Keep the multi-sensor view as an option.** Before considering it a default, run a 30–60-day Sentinel-3 vs VIIRS match-up by region and season. A documented regional offset might justify a seam warning on the map; it does not justify blending.
3. **Ask NOAA CoastWatch about runner access.** A contact or an allow-listed path for the 403s, and whether a mirror of `ucsdHfrW2` and the OLCI sectors exists. This is the biggest reliability risk, and it isn't fixable in code.
4. **WCOFS stays research** until the 60-day gate in section 4 passes. No transport or bloom outlook.
5. **Next user-facing step:** currents over chlorophyll, as a combined view. Today groups are exclusive: one legend, one product. It should be designed carefully, so that arrows over chlorophyll don't imply "the bloom goes there".
