# 01 — Codebase audit

Audit date: 2026-10-08. Scope: `coastwatch-web/`, `dashboard/`, `GOALS.md`, research README and result files, model artifacts, data-processing scripts. Line references are to the files as of commit `9c42f6b`.

---

## 1. Summary

| Area | State | Verdict |
|---|---|---|
| Research pipeline (`convLSTM/`, `pinn/`, `tft/`, `new_ds/`) | Complete for a paper; reproducible only with the data freeze, which is **not on this machine** | Keep as-is; do not import into the web app |
| Trained models | Checkpoints exist (`runs/pinn_best.pt`, `Models/convLSTM_best.pt`, `~/HAB_Models/*`) but normalization stats are not persisted and checkpoint provenance is ambiguous | **Not deployable** without retraining work |
| `coastwatch-web/` | Small, clean Next.js 15 + Mapbox shell (~1,300 lines). Live layer = NASA GIBS VIIRS + PACE chlorophyll tiles. Everything else is static text or links | **Extend** — good foundation, little to unwind |
| `dashboard/` (Streamlit) | Duplicate front end; its Python export utilities are the only producer of `snapshot.json` | **Retire the Streamlit UI**; salvage `ocean_mask.py` ideas into the new ingestion package |
| `GOALS.md` | Thoughtful compliance-first vision; over-scoped for a first release | Use as north star; MVP takes a narrow slice |

---

## 2. Reusable assets

### 2.1 `coastwatch-web/`

| Asset | File | Reuse |
|---|---|---|
| App shell, Tailwind 4, Geist fonts | `src/app/layout.tsx`, `globals.css` | Keep; replace ad-hoc slate palette with design tokens |
| Map component (react-map-gl, raster + GeoJSON sources, geolocate, harbor click) | `src/components/CoastMap.tsx` | Keep structure; generalize to a layer registry |
| GIBS capabilities parser + server route with fallback | `src/lib/gibs.ts`, `src/app/api/gibs-chl-meta/route.ts` | Keep pattern (server-side "latest valid date" discovery with explicit fallback flag) |
| CA map extent constants | `src/lib/caMapExtent.ts` | Keep |
| Haversine + nearest port | `src/lib/geo.ts` | Keep `distanceKm`, `nearestHarbor`; delete deprecated `gibsChlDate` |
| Compliance banner (official-rules-win copy + links) | `src/components/ComplianceBanner.tsx` | Keep concept; must become data-driven (actual status, not just links) |
| Provenance footer | `src/components/ProvenanceStrip.tsx` | Keep concept; generalize to per-layer provenance |
| Zone drawer (mobile bottom sheet / desktop side panel) | `src/components/ZoneDrawer.tsx` | Keep layout; becomes the Port drawer |
| NWS MapClick links | `src/lib/nws.ts` | Replace with api.weather.gov marine zone mapping |

Dependencies (`package.json`): `next 15.5.14`, `react 19.1`, `mapbox-gl ^3.9.4`, `react-map-gl ^8.0.1`, Tailwind 4, ESLint 9. No test framework, no state library, no data-fetching library. Lean, which is good.

### 2.2 Python

| Asset | File | Reuse |
|---|---|---|
| Land/ocean masking (Natural Earth + fallback polyline) | `dashboard/ocean_mask.py` | Reuse logic in ingestion package |
| RGBA rendering + manifest writer | `dashboard/snapshot_utils.py` | Replace (see §4.6 projection bug); keep "plain-language viewer text" idea |
| 4 km MODIS grid definition | `new_ds/modis_4km_grid.npz`, `new_ds/00_make_target_grid.py` | Needed for any future model inference |
| Data-freeze recipe | `config/data_freeze_v1.yaml`, `convLSTM/prepare_data.py` | Needed to rebuild model inputs operationally |
| Predicted fields 2003-02-18 → 2021-06-26 | `Diagnostics_*/predicted_fields.nc` (845 × 240 × 240; `log_chl_pred`, `log_chl_pers`, `log_chl_true`, `valid_mask`) | Use for a **historical** model-showcase / hindcast page, never as current conditions |
| Skill maps, monthly and bin skill | `Diagnostics_*/skill_maps.npz`, `metrics_*.csv` | Use in a model card |

---

## 3. Fabricated, synthetic, or unsourced data — complete inventory

| # | Location | What it is | Where it surfaces | Action |
|---|---|---|---|---|
| F1 | `dashboard/data/snapshot.json` = `coastwatch-web/public/data/snapshot.json` | `"data_source": "synthetic_demo"`, `"variable": "log_chl_demo"`, `"time": "demo-static"`. Regional tiers computed on a synthetic field | Web: "App bundle generated" timestamp, "Bundled notes" card ("Sample map — not today's ocean"), provenance footer | Delete from the public build. The "Bundled notes" card currently tells users the map is fake while the map shows real GIBS tiles — contradictory |
| F2 | `dashboard/data/overlay.png` = `coastwatch-web/public/data/overlay.png` | Rendered synthetic field (`make_demo_snapshot.synthetic_logchl`: Gaussian "gyre" + sine) | **Not rendered** by the web app (no reference in `src/`), but shipped in `public/` | Delete |
| F3 | `dashboard/scripts/make_demo_snapshot.py`, `generate_synthetic_demo_overlay.py` | Synthetic generators | CI workflow `.github/workflows/dashboard-demo.yml` runs one weekly | Move to `tests/fixtures` generator clearly named; never written to a path the public app reads |
| F4 | `public/data/fisheries_context.json` (copy of `dashboard/fisheries_context.json`) | Hand-written "typical targets" and "operational notes" per region, no citations, no date. E.g. lists salmon as a target in every northern region although the California ocean salmon fishery has been closed in recent seasons (verify current status before publishing anything) | Insight panel, zone drawer | Replace with sourced, dated content (landings-derived species lists + curated regulatory status with `last_verified`) |
| F5 | `public/data/ca_harbors.geojson` | 17 hand-placed points, no source. Several are wrong: Santa Barbara at 34.25°N (harbor ≈ 34.40°N, point is in the channel); "San Pedro / LA" at −118.50 (≈ 0.2° west of the port); "Point Reyes / Inverness" at 38.40°N (Inverness ≈ 38.10°N); "Pillar Point" and "Half Moon Bay" are the same harbor; Pillar Point placed at 37.63°N (≈ Pacifica). Missing major landing ports (e.g., Trinidad, Shelter Cove, Port Hueneme/Oxnard, Dana Point, Oceanside, Newport) | Map markers, nearest-port snapping | Replace with ports derived from CDFW port / port-area definitions; keep a reviewed coordinate per port |
| F6 | `src/lib/nws.ts` `REGION_CENTERS` | Arbitrary points used to build NWS MapClick links | Brief, drawer | Replace with official NWS marine zone (PZZ) polygons/IDs |
| F7 | `dashboard/regional.py` `CA_REGIONS` | Six latitude bands drawn by hand; regional medians include all ocean pixels in the band out to 126°W (not nearshore) | Snapshot `regional_algae` (not currently rendered) | Retire; use port-centered nearshore buffers or official zones |
| F8 | `README.md` §1 & §5, `PAPER_READY_SUMMARY.md`, `PAPER_STATUS_REPORT.md` | "8.3% validation improvement (0.699 vs 0.762)" compares val RMSE from the imputation-sensitivity run (`paper_results/imputation`) with val RMSE from the diagnostics run (`Diagnostics_ConvLSTM`). Same-pipeline numbers (`Diagnostics_*/metrics_global.csv`): ConvLSTM val 0.762 / test 0.801; PINN-optimized val 0.765 / test 0.801. "<3% degradation in the 2019 marine heatwave" has no supporting result file | Would surface on any model showcase page | Public site must quote only same-pipeline numbers (Table 4.1). Flag for the paper too |

---

## 4. Misleading or unsupported logic

| # | Location | Problem | Fix |
|---|---|---|---|
| M1 | `dashboard/regional.py:41-70` | Tiers ("Lower / Typical / Higher") are within-map tertiles: some region is always "Higher" even on a uniformly calm day, and the label changes if the map extent changes | Never use within-map percentiles as risk. Use (a) official C-HARM probabilities with their published thresholds, or (b) anomalies vs a **fixed** per-pixel, per-season climatology with stated baseline years |
| M2 | `src/lib/recommendations.ts` `fishTargetsForTier`, `economicRiskBullets` | Dead code today, but encodes "lower chlorophyll → favor usual nearshore finfish and crab programs" — chlorophyll as a go signal | Delete. Recommendations must never derive from chlorophyll alone |
| M3 | `CoastMap.tsx:150-187` | VIIRS NOAA-20 tiles drawn at full opacity over PACE tiles at 42% — two products with different algorithms and color scales blended into one picture, with no single legend | Show one chlorophyll product at a time with its own legend; offer the other as an alternate |
| M4 | `CoastShell.tsx:162-168`, `api/gibs-chl-meta/route.ts` | The app requests tiles for GIBS's `<Default>` (newest) date. Verified 2026-10-08: for that date GIBS returns HTTP 500s or corrupt striped RGBA tiles for both `VIIRS_NOAA20_Chlorophyll_a` and `OCI_PACE_Chlorophyll_a`; the previous day is clean. **The live map is likely showing broken tiles today.** The fallback also silently uses "today − 2 days" | Use `Default − 1 day` or probe one ocean tile (palette-mode PNG) before adopting a date; carry a `freshness` status (`verified` / `fallback`) and display it |
| M4b | `src/lib/gibs.ts:14,22` | Legend URLs `legends/VIIRS_NOAA20_Chlorophyll_a_H.svg` and `legends/OCI_PACE_Chlorophyll_a_H.svg` return **404**; capabilities list `VIIRS_Chlorophyll_H.svg` and `MODIS_Chlorophyll_H.svg`. The app's only quantitative legend is a dead link | Read legend URLs from capabilities, or render our own legend from known palette |
| M4c | `README.md` §8, `coastwatch-web/README.md` | Still describe "MODIS Aqua L3S 8-day" GIBS tiles; the code uses VIIRS NOAA-20 + PACE | Correct docs |
| M5 | `ComplianceBanner.tsx` | Collapsed by default; "official rules always win" is shown as links, not as the actual current status of closures | Compliance status must be content, at the top of the hierarchy, not a collapsed footnote |
| M6 | `fisheries_context.json` `operational_note`s | Uncited claims ("Historic HAB hot spot… higher chlorophyll here is common during upwelling") rendered next to the map as guidance | Replace with sourced copy or remove |
| M6b | `ComplianceBanner.tsx:43`, `fisheries_context.json` (north_bay) | Link `https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/MarineBiotech.aspx` returns **404** (verified 2026-10-08). The primary official link in the compliance banner is dead | Replace with `…/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx`; add an automated link checker (`06` §5) |
| M7 | `snapshot_utils.build_manifest` | `what_map_shows` text says colors rank *relative* abundance — accurate for the synthetic overlay but now displayed beside an absolute-scale GIBS layer | Remove; each layer gets its own description |

---

## 5. Forecast artifacts and inference requirements

**What exists**

- Checkpoints: `runs/pinn_best.pt` (540 KB), `Models/convLSTM_best.pt` (208 KB), and in `~/HAB_Models/`: `best_pinn.pt`, `best_pinn_v1p3.pt`, `convLSTM_PINN_best.pt`, `convLSTM_best.pt`, `vanilla_best.pt`, TFT `.ckpt` files.
- Hindcast fields for all models, 2003-02-18 → 2021-06-26, 8-day steps, MODIS 4 km grid.
- MC-dropout / ensemble / quantile code (`pinn/pinn_model_uncertainty.py`, `pinn/inference_uncertainty.py`), **never trained** — no `pinn_uncertainty_best.pt` or ensemble checkpoints exist.

**Why the models cannot run operationally today**

1. **Missing inputs.** The data freeze `HAB_convLSTM_core_v1_clean.nc` (expected at `~/Desktop/HABs_Research/Data/Derived/`) is not present. `config.yaml` points at `~/Desktop/HABs_Research/Processed`, also absent.
2. **Normalization stats are not persisted.** `pinn/pinn_model.py:69-76` computes per-variable mean/std from the freeze's training split at runtime; only `state_dict` is saved (`:325`). `data/mean_std.npz` holds 18 values and `Models/feature_names.npy` holds 67 XGBoost tabular features — neither matches the 28-channel ConvLSTM/PINN input. Without the freeze, the stats cannot be reconstructed exactly.
3. **Ambiguous checkpoint provenance.** `pinn_model.py` writes to `OUT_DIR/'convLSTM_best.pt'` — the same filename as the ConvLSTM baseline — and `Diagnostics_PINN_Optimized/diagnostics_summary.json` records `ckpt_path: ~/HAB_Models/convLSTM_best.pt`. Which weights produced which diagnostics cannot be verified from files alone.
4. **Input sources are not operational.** Training used MODIS-Aqua L3 (`log_chl`, `Kd_490`, `nflh`), ERA5 reanalysis, and CMEMS GLORYS reanalysis. MODIS-Aqua is being decommissioned (NASA projections range from late 2026 to Sept 2027), GLORYS is a delayed-mode reanalysis, and ERA5 lags ~5 days (ERA5T). An operational version needs VIIRS/PACE ocean color, CMEMS analysis-forecast physics, and NRT atmospheric forcing — i.e., **domain shift → retraining and re-validation**.
5. **Skill is modest and specific.** Test RMSE 0.80 ln(mg m⁻³) vs 0.89 persistence (≈ 10% skill), 6% in the lowest-chlorophyll quartile, peaking ~13% in Apr–Jun. Bloom-event detection metrics (hit rate, FAR) exist only in `spatial_bias.py` scripts and are not in the published diagnostics. The target is chlorophyll, not toxin.

**Operational inference requirements (for a later phase)**

- 6 × 8-day composites (48 days) of 28 channels on the 240 × 240 4 km grid.
- Persisted `{variable: (mean, std)}` and channel order stored inside the checkpoint or a sidecar `model_card.json`.
- CPU inference < 10 s per forecast (per README) — cheap enough for a scheduled batch job; no GPU or server required.

---

## 6. Technical debt and simplification opportunities

| Item | Recommendation |
|---|---|
| Two front ends (Streamlit + Next.js) | Retire Streamlit UI and its CI smoke test; keep only Python utilities that move into `pipeline/` |
| `sync-data.mjs` copies files from `dashboard/` into `public/` | Replace with a pipeline that publishes versioned artifacts + manifest; web reads manifest |
| `snapshot.json` mixes data, UI copy (markdown strings), and refresh instructions | Split: data manifests (machine) vs UI copy (in the app, versioned with code) |
| Region keys tie ports to 6 latitude bands | Ports become first-class entities with their own IDs |
| Mapbox GL v3 (proprietary license, token, metered map loads) | Evaluate MapLibre GL via `react-map-gl/maplibre` (same API) + an open basemap; see `04-architecture.md` |
| Equirectangular PNG overlay placed by four corners in a Web Mercator map (`export_map_snapshot.py` → image source) | Mercator stretch is non-linear in latitude; for a 32–42°N image the mid-domain misplacement reaches ≈ 18 km (computed) — larger than the nearshore band itself. Reproject to EPSG:3857 before rendering |
| Build health | `tsc --noEmit` and `eslint src` pass clean (2026-10-08). Next 16.x and mapbox-gl 3.32 are available; pin and decide on the Next 16 upgrade in M0 rather than mid-build |
| No tests, no type-checked data contracts | Add JSON Schema/Zod contracts shared by pipeline and app; Vitest + Playwright |
| `tsconfig.tsbuildinfo`, `.next/` present in tree | Ensure ignored |
| Large local files (`Models/X_test.npy` 638 MB) | Already git-ignored; keep out of any deploy context |
| Research Markdown claims (F8) | Correct before quoting anywhere public |
