# California Coastwatch (web)

WellWatch-style **Next.js + Mapbox** front end for the `habs-forecast` dashboard data: interactive zoom/pan, **your location**, California **fishing ports**, a **NASA GIBS** 8-day chlorophyll layer (continually updated by NASA’s tile service), and the **research grid** (`overlay.png` + `snapshot.json`) from the Python pipeline.

## Quick start

From repo root:

```bash
cd coastwatch-web
npm install
cp .env.example .env.local
# Edit .env.local — add NEXT_PUBLIC_MAPBOX_TOKEN
npm run sync-data
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

`npm run sync-data` copies `dashboard/data/snapshot.json`, `overlay.png`, and `dashboard/fisheries_context.json` into `public/data/`. Re-run after each Python export.

## Stack (aligned with [WellWatch](https://github.com/yashnil/WellWatch))

- **Next.js 15** (App Router)
- **TypeScript** + **Tailwind CSS 4**
- **Mapbox GL** + **react-map-gl** — navigation, geolocate, raster + image sources
- **Public data:** [NASA GIBS](https://wiki.earthdata.nasa.gov/display/GIBS/) WMTS tiles for MODIS Aqua L3S 8-day chlorophyll

## What users get

- **Interactive map** — zoom, pan, locate me, optional layers
- **Clear legend** — research grid chlorophyll scale
- **Ports** — GeoJSON harbors keyed to regional summaries
- **ML / grid insight** — regional tier (Lower / Typical / Higher) from `snapshot.json`
- **Fish & economics** — rule-based recommendations from tier + curated `fisheries_context.json`
- **Refresh story** — `generated_at` and recommended cadence from the manifest

## Production

- Set `NEXT_PUBLIC_MAPBOX_TOKEN` in the host environment.
- Run `npm run sync-data` (or CI) before `npm run build` so `public/data` is populated.
- GIBS tile dates are chosen client-side (~6 days back); adjust in `src/lib/geo.ts` if tiles are blank.

## License

Same as parent repository (MIT unless otherwise noted).
