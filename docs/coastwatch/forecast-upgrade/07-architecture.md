# 7. Recommended technical architecture

The M1–M3 architecture stays: a scheduled Python pipeline publishes validated, versioned static artifacts.
The Next.js app reads the manifest, and freshness is computed in the browser. The upgrade adds sources and
layer types inside that pattern. There's no database, no tile server, and no runtime calls to data providers.

```
             ┌────────────── every 6 h (GitHub Actions) ─────────────────┐
 ERDDAP ──►  │ sources/olci300.py   (NOAA CW sectors CI+DI, S3A then S3B) │
 ERDDAP ──►  │ sources/viirs750.py  (erdVHNchla1day)                      │
 ERDDAP ──►  │ sources/charm.py     (unchanged)                           │──► process/ ──► publish/ ──► Pages (or R2)
 ERDDAP ──►  │ sources/hfradar.py   (ucsdHfrW2, last 24 h)                │      │
 AWS S3 ──►  │ sources/wcofs.py     (regulargrid, surface only, 6-hourly) │      ├─ latest-clear-view composite + age grid
 (later) ──► │ sources/pace.py      (Earthdata token, L2 BGC NRT)         │      ├─ Mercator PNG (display) + u16 grid (values)
             └────────────────────────────────────────────────────────────┘      ├─ currents → u/v texture PNG + arrows GeoJSON
                                                                                 └─ drift ensemble (experimental, gated)
```

## 7.1 Layer contract (extends the existing manifest)

Every layer entry gets the following fields:
- `product_class`: existing values (`official_forecast`, `observation`, `experimental_model`, …).
- `native_resolution_m` and `grid` (the u16 value grid, as today).
- `valid_time` / `observed_date` / `issued_date`.
- `coverage`: the share of the domain and of each region with a value.
- For composites, `age_grid`: days since each pixel's observation.
- `render`: `nearest` or `display_interpolation` (only allowed for `official_forecast`, never for readouts).
- `palette.id`.
- `caveats`.

The schema change is additive. The app's Ajv validation and the M2-compat tests extend the same way M3 did.

## 7.2 Satellite chlorophyll at 300 m without a tile server

- **Statewide California at 300 m** is about 4,000 × 3,600 cells, 14.4 M per day. As a u16 grid that is 29 MB raw, a few MB gzipped (most of the area is land, cloud or no data).
- **For display,** pre-render a **PMTiles** pyramid (zoom 5–11) of the latest clear-view composite and its age layer. It's a single static file served with HTTP range requests, which works on GitHub Pages or Cloudflare R2. MapLibre reads it with the `pmtiles` protocol.
- **Values for the readout** come from the u16 grid in regional chunks (one per region, about 0.5–2 MB), fetched on demand. This is the same pattern as `lib/grid.ts`.
- **Keep a rolling 14 days** for the time slider. Older days live only in the archive bucket if backtesting needs them.

## 7.3 Currents

- **WCOFS:** read only the surface level of `regulargrid` files every 6 hours from +3 h to +72 h, about 12 reads × 57 MB of range requests. Publish:
  - a **u/v texture PNG** per step (two channels, quantized) for a WebGL particle layer;
  - **thinned arrows GeoJSON** as a static fallback.
- **HF radar:** a 24-hour mean of the 2 km product with a coverage mask, published as observed currents (`observation`).

## 7.4 Experimental products (gated)

Drift ensembles and any station-level ML model are produced in a separate pipeline job. Each publishes to a
separate manifest group with `product_class: experimental_model` and is hidden by a feature flag until its
§6 gate passes.

## 7.5 Front-end

The design-reset map (revision 2) already has the structure this needs: a navigation card, a dock with
quantity, day and legend, an inspector, and a value readout. Changes:
- **Layer groups in the dock:** *Model* (C-HARM), *Satellite* (latest clear view, single day) and *Ocean physics* (currents, drift). Each layer carries its native-resolution chip.
- **Time slider:** forecast leads for C-HARM; overpass days with coverage bars for satellite layers.
- **Readout:** shows the value, the layer's native cell, and for composites, **the observation date**.
- **Inspector for a port:** adds "latest clear view near this port: X mg/m³, observed N days ago" to the satellite section.
