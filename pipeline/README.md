# CoastWatch pipeline

Fetches public data, validates it, renders map-ready artifacts, and writes a versioned manifest the web app reads. No database or server: output is static files.

```bash
cd pipeline
uv sync                                  # Python 3.11+; numpy, scipy, pillow, pydantic
uv run cwp run                           # live run -> ../coastwatch-web/public/data/v1
uv run cwp run --out /tmp/cw --only charm
uv run cwp verify-charm                  # compare published C-HARM with ERDDAP at 13 points
uv run cwp fixture --out /tmp/cw-fixture # deterministic dataset from recorded fixtures, no network
uv run cwp schema                        # re-export ../schemas/v1/*.schema.json after model changes
uv run pytest                            # offline tests (fixtures)
uv run pytest -m live                    # live end-to-end check against ERDDAP
```

## Sources (M1)

| Source | Module | Product class | Output |
|---|---|---|---|
| C-HARM v3.1 (`wvcharmV3_{0..3}day`, NOAA CoastWatch West Coast ERDDAP) | `sources/charm.py` | `official_forecast` | Per lead and variable: Web-Mercator PNG + uint16 value grid on the source grid |
| NASA GIBS chlorophyll (VIIRS NOAA-20, PACE OCI) | `sources/gibs.py` | `observation` | Tile URL template for the newest date whose probe tiles pass validation, verified legend URL |
| CDFW landing ports [ds3081] + `data/curated/ports.json` | `sources/ports.py` | `reference` | `ports.geojson` |

## Output layout

```
v1/
  manifest.json                         # written last; schema: schemas/v1/manifest.schema.json
  ports.geojson                         # schema: schemas/v1/ports.schema.json
  charm/<issued-date>/lead<k>/<variable>.png
  charm/<issued-date>/lead<k>/<variable>.u16.gz
  verification/charm-points.json        # from `cwp verify-charm`
```

## Rules the code enforces

- **Validation before publish.** C-HARM: NetCDF signature, required variables, product version 3.1, one time step equal to the probed valid time at 12:00Z, regular 0.03° grid inside the published domain, probabilities in [0, 1], at least 10% valid cells. Fill values (−99999) become "no value", never zero.
- **Issue time is derived and labelled.** C-HARM publishes valid days only; issue date = nowcast valid day + 1 (`issued_date_derived: true`).
- **No mixing runs, no back-fill.** A lead whose newest valid day belongs to an older run is reported missing, not shown.
- **Failures stay visible.** Each source runs in isolation. On failure the previous artifacts stay published with their real dates, and `sources[].outcome = "failed"` with the error. The browser decides current / stale / historical from those dates, so a dead pipeline cannot make old data look current.
- **Correct georeferencing.** Rasters are resampled (nearest neighbour) onto pixels uniform in Web-Mercator y, so placing them by their corners is exact. Value grids stay on the source grid.
- **GIBS dates are tested, not trusted.** The newest advertised date is used only if three probe tiles over central California are HTTP 200 palette PNGs with data in at least 2% of pixels; otherwise the pipeline steps back up to 6 days.

## Scheduling and publishing

`.github/workflows/coastwatch-data.yml` runs `cwp run` every 6 hours and pushes `v1/` to the `coastwatch-data` branch (free, versioned by git). The web app reads it via `NEXT_PUBLIC_CW_DATA_BASE_URL`. Scheduled workflows only run from the default branch, so nothing publishes until this is merged.
