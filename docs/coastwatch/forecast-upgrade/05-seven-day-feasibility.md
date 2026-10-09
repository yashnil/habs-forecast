# 5. Seven-day bloom outlook: feasibility

## 5.0 First, separate four things that the word "bloom forecast" blurs

| Process | What predicts it | Time scale | What CoastWatch can observe |
|---|---|---|---|
| **Water transport**: where surface water goes | Currents (WCOFS, HF radar, Copernicus) | hours–days | Physics models + HF radar |
| **Bloom growth or decay**: biomass and *Pseudo-nitzschia* abundance | Upwelling, nutrients, light, temperature, grazing | days–weeks | Satellite chlorophyll (biomass, not species), weekly pier counts |
| **Toxin production**: domoic acid per cell and in water | Species and strain, silicate/iron stress, growth phase; not visible from space | days–weeks | Weekly pier pDA and cDA (CalHABMAP) |
| **Seafood toxicity and closures** | Accumulation in shellfish, crab and fish, depuration; agency testing and decisions | weeks–months | CDPH/CDFW notices only |

A surface-current simulation addresses only the first row. It must never be presented as a forecast of the
other three.

## A. An existing validated operational HAB forecast

- **What exists:** C-HARM v3.1: nowcast plus 1–3 days, 3 km, probabilities of pDA > 500 ng/L, cDA > 10 pg/cell and *Pseudo-nitzschia* > 10⁴ cells/L.
- **Inputs:** gap-filled VIIRS chlorophyll and reflectance (486, 551 nm), WCOFS salinity, temperature and currents.
- **Method:** statistical (generalized linear) habitat models. Forecasts come from moving the satellite fields with WCOFS currents (C-HARM metadata).
- **Seven days:** **not available.** The horizon is capped by WCOFS's 72 hours. No other operational California HAB outlook was found (§1).
- **Skill:**
  - **Earlier version:** the published assessment (Anderson et al. 2016, 2014–15 pier data) found the domoic-acid models informative and the *Pseudo-nitzschia* model prone to false positives.
  - **v3 and v3.1:** no public skill numbers found.
  - **Known issues:** SCCOOS bulletins from 2022–23 flagged WCOFS salinity errors that affect the pDA model.
  - **Coverage:** much of the nearshore 0–3 km strip has no values (§2.3).
- **Verdict:** **use it, unchanged, for days 0–3.** It is the only HAB-specific operational forecast. CoastWatch should show its horizon honestly (three days, not seven) and add its nearshore limits to the caveats.

## B. Experimental transport outlook (currents + observed fields)

| | |
|---|---|
| Inputs | WCOFS hourly surface currents (72 h, 4 km); HF-radar observed currents for the starting state and for verification; latest clear-view chlorophyll (OLCI 300 m / VIIRS 750 m) or C-HARM probability as the field being transported. For days 4–7 only Copernicus 1/12° physics (≈ 9 km) exists. |
| Target | **Where surface water now in a given place may be in 1–3 days.** Not bloom extent, not toxin. |
| Resolution | Effective resolution is the currents' (4 km; 9 km beyond 72 h). Uncertainty grows with lead time. |
| Horizon | 72 h with WCOFS. Days 4–7 only with Copernicus physics, which barely resolves bays (Monterey Bay ≈ 4–5 cells). |
| Uncertainty | Drifter separation grows quickly in eddies and fronts. It needs an ensemble (seed spread, WCOFS vs Copernicus vs HF-radar persistence) and a spread display, never a single line. |
| Validation data | HF-radar hourly currents (2012–present) for current errors; drifter archives (NOAA Global Drifter Program); next-overpass satellite features (does a filament move where predicted?). |
| Scientific limits | Ignores growth, decay, sinking, vertical migration and toxin production. Surface-only. Wind-driven nearshore flow and bays are poorly resolved at 4 km. Moving a 300 m image with 4 km currents fabricates precision. |
| Complexity | **Moderate.** WCOFS ingest (0.7 GB/day of range reads, §9), a particle tracker (≈ 200 lines, prototyped in [`render_lab.py`](scripts/render_lab.py)), verification against HF radar. 2–3 weeks to a validated experimental layer. |
| Verdict | **Worth building as an explicitly physical, 72-hour "surface drift" layer** with an ensemble spread and the label "where water may move, not where a bloom will be". It is what C-HARM does internally, made visible. **Not a 7-day bloom outlook.** |

## C. New statistical / ML model trained on observations and ocean forecasts

| | |
|---|---|
| Target | Station-level: probability that the **next weekly** CalHABMAP sample exceeds pDA 0.5 ng/mL (= 500 ng/L) or *Pseudo-nitzschia* 10⁴ cells/L, at each pier, about 7 days ahead. A map-level target has no ground truth. |
| Inputs | Recent pier measurements (persistence is the strongest predictor); satellite chlorophyll and reflectance near each pier (VIIRS 2012–, OLCI 2016–, PACE 2024–); SST; upwelling indices (CUTI/BEUTI, daily, public); WCOFS or a reanalysis for currents and salinity; C-HARM probabilities (archive only since Nov 2022); season. |
| Training data (actual, from the published observations) | **4,267 pDA samples at 10 piers, 2014–2026, with only 155 at or above 0.5 ng/mL (3.6 %)**, clustered in a few events (2015, 2022–24) and dominated by Santa Cruz Wharf (46) and Monterey Wharf (39). 9,744 *Pseudo-nitzschia* counts, 1,510 at or above 10⁴ cells/L. See [`evidence/calhabmap-training-data.txt`](evidence/calhabmap-training-data.txt). |
| Resolution | 10–17 piers, weekly. No spatial map. |
| Horizon | 1 sampling interval (about 7 days). |
| Expected uncertainty | High. With roughly 150 positive events, event-blocked cross-validation leaves only tens of independent events per fold. Skill against persistence is likely modest. The repository's earlier research model reached only about 10 % skill over persistence for chlorophyll (see `01-codebase-audit.md`). |
| Validation needed | Leave-one-year-out and leave-one-event-out, compared with (1) persistence, (2) climatology and (3) C-HARM at the nearest valid cell. Brier skill score, reliability diagrams, hit and false-alarm rates at decision thresholds (§6). |
| Scientific limits | Toxin production depends on drivers satellites don't see. Few events and non-stationarity (2015 marine heatwave) make extrapolation risky. WCOFS inputs exist only since 2021, so models using them have about 4 seasons. |
| Complexity | **High.** 4–8 weeks of research, plus an independent HAB-scientist review before any public display. |
| Verdict | **Research track only.** Worth starting as a backtest (§6) because a calibrated "next sample at this pier" probability would be genuinely useful. Ship only if it beats persistence and C-HARM out of sample, with reliability shown. |

## 5.4 Recommendation

| Horizon | What CoastWatch can honestly show | Status |
|---|---|---|
| Now (observations) | Latest clear-view satellite chlorophyll at 300–750 m with per-pixel age; HF-radar currents; pier measurements | **Ship-ready data** (pipeline work) |
| Days 0–3 | C-HARM probabilities (official agency forecast); WCOFS currents; experimental surface-drift spread | C-HARM ready now; drift needs validation |
| Days 4–7 | **No validated HAB forecast exists.** Optionally Copernicus physics (≈ 9 km) as ocean context only. | Do not present as a bloom outlook |
| Next pier sample (≈ 7 days) | Calibrated station probabilities, if the backtest succeeds | Research |
