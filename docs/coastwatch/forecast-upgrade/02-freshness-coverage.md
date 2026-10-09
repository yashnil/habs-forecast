# 2. Operational freshness and coverage report

Test window: the 14 most recent time steps of each product as of 2026-10-09, subset to three regions:

| Region | Bounds |
|---|---|
| Monterey Bay | 36.4–37.2 °N, 122.6–121.75 °W |
| North Coast (Humboldt/Trinidad) | 40.4–41.4 °N, 124.7–123.95 °W |
| Southern California Bight | 32.6–34.1 °N, 119.0–117.1 °W |

Scripts: [`scripts/fetch_samples.py`](scripts/fetch_samples.py), [`scripts/nearshore_coverage.py`](scripts/nearshore_coverage.py).
Raw output: [`evidence/coverage-summary.txt`](evidence/coverage-summary.txt), [`evidence/daily-valid-fraction.txt`](evidence/daily-valid-fraction.txt), [`evidence/nearshore-coverage.txt`](evidence/nearshore-coverage.txt).

## 2.1 How current is "current"?

| Layer | Newest data on 2026-10-09 | Age | Cadence observed |
|---|---|---|---|
| PACE OCI L2 (NRT) | 10-08 21:02 UTC overpass | **≈ 0.5 day** (online 2.5 h after overpass) | daily overpass |
| HF radar currents | 10-09 09:00 UTC | **≈ 6 h** | hourly |
| WCOFS currents/T/S | run 10-09 t03z, valid to 10-12 03 UTC | forecast | daily run |
| C-HARM v3.1 | nowcast 10-07; forecast valid 10-08…10-10 | nowcast 2 days | daily, **with gaps** (no 10-05 nowcast) |
| OLCI 300 m (S3A) | 10-07 overpass | 2 days | daily |
| VIIRS NOAA-20 4 km | 10-06 | 3 days | daily |
| VIIRS 750 m West Coast | 10-04 | 5 days | irregular (missing days in the axis) |
| DINEOF 2 km gap-filled | 09-27 | 12 days | daily, science-quality latency |

So the freshest *quantitative* view of the water is 2 days old (OLCI or C-HARM). PACE is fresher, but only behind an Earthdata Login.

## 2.2 Cloud and fog gaps (share of a region's water with a value, per day)

Water = cells with a value on at least one day in the window.

| Layer | Monterey: median day | Monterey: days ≥ 50 % | North Coast: median | North: days ≥ 50 % | SoCal Bight: median | SoCal: days ≥ 50 % |
|---|---|---|---|---|---|---|
| OLCI 300 m | **1 %** | 1 of 14 | **0 %** | 1 of 14 | 26 % | 6 of 14 |
| VIIRS 750 m | 6 % | 3 of 13 | 15 % | 4 of 13 | 70 % | 10 of 14 |
| VIIRS NOAA-20 4 km | 32 % | 5 of 14 | 36 % | 6 of 14 | 95 % | 12 of 14 |
| C-HARM, DINEOF (gap-filled) | 100 % | 14 of 14 | 100 % | 14 of 14 | 100 % | 14 of 14 |

Day by day in Monterey Bay, the 300 m OLCI was 97 % clear on 09-24 and then never above 16 % for two weeks; the newest
overpass (10-07) is 0 % clear. Coarser sensors look clearer because one valid pixel covers more area and
their composites merge more passes, not because they see through fog.

**Implication:** high-resolution observations are excellent when the sky allows and absent much of the time,
especially in the central and northern coast's fog season. Gap-filled products always "have" a value because
it is statistically estimated. Any honest map needs:
- a per-pixel **date of the last clear view**;
- a visible no-data state.

It must never silently fall back to a gap-filled field.

## 2.3 Nearshore coverage (where piers, ports and harvest happen)

Share of water with a value, by distance from shore. C-HARM nowcast versus one clear OLCI day:

| Region | 0–1 km | 1–3 km | 3–5 km | 5–10 km | > 10 km |
|---|---|---|---|---|---|
| Monterey: C-HARM | 52 % | 73 % | 96 % | 100 % | 100 % |
| Monterey: OLCI 300 m, 09-24 | 68 % | 92 % | 99 % | 100 % | 100 % |
| SoCal Bight: C-HARM | **17 %** | **48 %** | 88 % | 99 % | 100 % |
| SoCal Bight: OLCI 300 m, 10-06 | 75 % | 100 % | 100 % | 100 % | 100 % |

- **C-HARM** leaves much of the first 3 km without a value. It masks nearshore cells, and its 3 km cells straddle the shore.
- **On a clear day, OLCI 300 m** reaches most of that strip. Its nearshore pixels are also the ones most affected by bottom reflectance, river plumes and adjacency to land, so they need quality flags before quantitative use.
- **The North Coast numbers are not reliable.** Two weeks of near-total cloud left too little clear water to build the water mask, so I've left them out.

Method: distances come from a distance transform on the OLCI water mask at about 250 m pixels. They are approximate by construction.

## 2.4 What this means for "genuinely current information"

1. The C-HARM nowcast is about 2 days old, and C-HARM's 3-day forecast is the only operational HAB-specific outlook.
2. The most current **physical** information is observed hourly HF-radar currents (about 6 h lag) and the WCOFS 72-hour forecast.
3. The most current **biological** observation with values is OLCI 300 m (2 days, cloud permitting), or PACE (hours, once an Earthdata account exists).
4. Nothing currently available observes toxin from space. Chlorophyll and reflectance are biomass and optics, not domoic acid.
