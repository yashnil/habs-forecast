# California nearshore HAB context dashboard

## Next.js + Mapbox (recommended for stakeholders)

For a **WellWatch-style** interactive map (geolocation, zoom, NASA GIBS chlorophyll tiles, ports, side panel), use **`coastwatch-web/`** in this repo:

```bash
cd coastwatch-web
npm install
cp .env.example .env.local   # add NEXT_PUBLIC_MAPBOX_TOKEN
npm run sync-data
npm run dev
```

See [`coastwatch-web/README.md`](../coastwatch-web/README.md).

---

## Streamlit (lightweight)

Streamlit app that maps a **single snapshot** of near-coastal log-chlorophyll (or model output) over California, with links to official marine biotoxin and fisheries resources.

This is **decision support**, not a regulatory or public-health product. Always follow CDPH, CDFW, and NOAA guidance for harvesting and consumption.

## Quick start

From the repository root:

```bash
python -m venv .venv-dashboard
source .venv-dashboard/bin/activate   # Windows: .venv-dashboard\Scripts\activate
pip install -r dashboard/requirements.txt
python dashboard/scripts/make_demo_snapshot.py
streamlit run dashboard/app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Refreshing data (periodic updates)

1. **From project NetCDF** (observations or exported predictions):

```bash
python dashboard/scripts/export_map_snapshot.py \
  --nc /path/to/your_cube.nc \
  --var log_chl \
  --time -1 \
  --outdir dashboard/data \
  --source observations
```

For PINN/ConvLSTM exports with `log_chl_pred`:

```bash
python dashboard/scripts/export_map_snapshot.py \
  --nc /path/to/pinn_export.nc \
  --var log_chl_pred \
  --time -1 \
  --outdir dashboard/data \
  --source pinn_forecast
```

2. **Schedule** the same command with cron, GitHub Actions, or a cloud scheduler (daily–weekly is typical for composite-based fields).

Outputs:

- `dashboard/data/overlay.png` — RGBA georeferenced image for the map (land pixels are transparent so coloring stays on the ocean)
- `dashboard/data/snapshot.json` — bounds, colormap limits, and **plain-language** metadata for the Streamlit UI

Optional flags on export:

- `--composite-days 8` — stored in the manifest so the app can explain typical composite length (metadata only).
- `--mask-fallback` — use a fast California polyline instead of **Natural Earth land polygons** (GeoPandas downloads `ne_110m_land`). Use offline or in CI.

Regenerate the demo with the same mask you use in production:

```bash
python dashboard/scripts/make_demo_snapshot.py --outdir dashboard/data
# offline / CI:
python dashboard/scripts/make_demo_snapshot.py --outdir dashboard/data --mask-fallback
```

## Optional path override

To point the app at alternate files without editing code, add a JSON file at the repo root named `.dashboard_paths.json`:

```json
{
  "manifest": "/absolute/path/to/snapshot.json",
  "overlay": "/absolute/path/to/overlay.png"
}
```

## Synthetic demo

`make_demo_snapshot.py` writes a **non-physical** patterned field so the UI runs without your NetCDF or checkpoints. Replace it with `export_map_snapshot.py` for real use.
