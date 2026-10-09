# CoastWatch forecast and data upgrade: investigation

Status: **research, 2026-10-09.** No production code, pipeline or deployment was changed. Every endpoint was
tested against live data on 2026-10-09. Failures and access limits are recorded as they happened.

## Executive summary

1. **No validated 7-day harmful-bloom forecast exists for California.** The only operational HAB forecast is **C-HARM v3.1**: nowcast plus 3 days, at about 3 km.
   - Its 1–3-day forecasts already *are* a transport model: gap-filled VIIRS fields are moved with NOAA WCOFS currents, then the bloom and toxin models are re-run.
   - Its horizon is capped by WCOFS's 72 hours.
   - It leaves much of the first 1–3 km from shore without values (in the SoCal Bight, only 17 % of water within 1 km of shore has a value).
   - No public skill numbers exist for v3.x.
   - A SCCOOS bulletin calls it "v4", but the served data say `product_version 3.1`.
2. **The big, honest upgrade is observational detail, not a longer forecast.**
   - **Sentinel-3 OLCI chlorophyll at 300 m** is served openly by NOAA CoastWatch about 2 days after overpass. On clear days it shows the nearshore bloom-scale structure that C-HARM's 3 km cells cannot.
   - In the central and northern fog season it is often missing: Monterey Bay had one usable 300 m day in 14. It therefore needs a **latest clear view with per-pixel age**, never silent gap-filling.
3. **Ocean physics is current and open.**
   - **WCOFS** gives hourly 72-hour surface-current forecasts at about 4 km on a public AWS bucket. California's surface costs about 57 MB of range reads per step.
   - **HF radar** gives observed hourly currents at 2 km with about 6 h lag.
   - Together they support a beautiful, honest currents layer and an experimental "where water may move" drift layer. They do not support a bloom forecast.
4. **PACE** (hyperspectral, online about 2.5 h after overpass) and **Copernicus Marine** (10-day global physics and biogeochemistry at 9–25 km) are real and current. Both need free accounts you would have to create: PACE downloads returned HTTP 401.
5. **A 7-day statistical model is research, not a feature.** The published pier record has 4,267 particulate-DA samples since 2014 but only **155 above 500 ng/L**, concentrated in a few events. Any station-level "next sample" model must beat both persistence and C-HARM in a blocked backtest before it is shown.

### Strongest data combination available today

| Role | Source | Native | Freshness |
|---|---|---|---|
| Official status | CDFW/CDPH registry (M2) | — | human-curated |
| HAB forecast, days 0–3 | **C-HARM v3.1** | 3 km | daily, 2-day-old nowcast |
| Bloom-scale observation | **OLCI 300 m (S3A, S3B fallback) latest clear view with age**, VIIRS 750 m fallback, PACE once an account exists | 250 m / 750 m / 1.2 km | 2 days (OLCI), hours (PACE) |
| Transport context | **WCOFS 72 h surface currents** + **HF-radar observed currents** | 4 km / 2 km | daily run / hourly |
| Ground truth | **CalHABMAP** pier samples (M3) | 17 piers | weekly |
| History | NOAA FOSS landings (M3) | statewide | annual |

### Should the design-reset plan change?

**Yes, moderately** ([10-updated-p1-roadmap.md](10-updated-p1-roadmap.md)).
- **What holds:** the design system and revision 2 map structure carry over, and the map lab confirms them with 300 m data.
- **What changes in P1:**
  - The forecast dock becomes a **layer system** (Model / Satellite / Ocean physics) with native-resolution chips.
  - Its time control depends on the layer: forecast leads for C-HARM (never past +3 d), overpass days with coverage bars for satellite layers.
  - Further work on smoothing C-HARM's appearance is dropped: the detail users want comes from observations.
- **What ships next:** quantitative 300 m chlorophyll (U1) and currents (U2).
- **What stays research:** anything labelled a 4–7-day bloom outlook, until it passes [6-validation](06-validation-backtesting.md).

## Map lab (real data)

| C-HARM, native 3 km cells | OLCI 300 m, same bay, clear day | OLCI + WCOFS currents + 72 h surface drift |
|---|---|---|
| ![](maps/01-monterey-charm-native.png) | ![](maps/03-monterey-olci300-0924.png) | ![](maps/08-monterey-olci-currents-drift.png) |

More comparisons are in [04-map-experiments.md](04-map-experiments.md). The lab itself is at [`lab/index.html`](lab/index.html).

## Deliverables

| # | Document |
|---|---|
| 1 | [Verified source inventory and endpoints](01-source-inventory.md) |
| 2 | [Operational freshness and coverage report](02-freshness-coverage.md) |
| 3 | [Resolution and forecast-horizon comparison](03-resolution-horizon.md) |
| 4 | [High-resolution map experiments](04-map-experiments.md) |
| 5 | [Seven-day modelling feasibility (approaches A, B, C)](05-seven-day-feasibility.md) |
| 6 | [Validation and backtesting requirements](06-validation-backtesting.md) |
| 7 | [Recommended technical architecture](07-architecture.md) |
| 8 | [Realistic implementation phases](08-implementation-phases.md) |
| 9 | [Costs, API dependencies and blockers](09-costs-dependencies-blockers.md) |
| 10 | [Recommended updated P1 roadmap](10-updated-p1-roadmap.md) |
| 11 | [Practical fisheries implications](11-fisheries-decision-support.md) |

## Reproducing

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install xarray netCDF4 h5netcdf h5py scipy numpy pillow fsspec aiohttp requests
.venv/bin/python scripts/fetch_samples.py /tmp/fu/samples 14                # ERDDAP subsets, 3 regions
.venv/bin/python scripts/fetch_wcofs.py /tmp/fu/wcofs 20261009              # WCOFS surface, byte ranges
for L in 0 1 2 3; do curl -g -o /tmp/fu/state/charm_lead$L.nc \
  "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_${L}day.nc?particulate_domoic[(last)][(31.3):(43.0)][(232.5):(243.0)]"; done
.venv/bin/python scripts/nearshore_coverage.py /tmp/fu/samples
.venv/bin/python scripts/render_lab.py /tmp/fu lab
node scripts/shoot_lab.mjs                                                   # Playwright from coastwatch-web
```

Raw evidence (probe outputs, listings, logs) is in [`evidence/`](evidence/). Downloaded data is not committed. Re-running
fetches the then-current data, so dates will differ.
