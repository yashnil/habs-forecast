# CoastWatch — web app

Next.js 15 (App Router) + TypeScript + Tailwind 4 + **MapLibre GL** (via `react-map-gl/maplibre`) with a navy basemap on OpenFreeMap vector tiles. No map token or paid service.

The app renders static artifacts produced by [`../pipeline`](../pipeline/README.md). It never calls upstream agencies at request time.

## Run it

```bash
npm install
npm run data:live     # runs the pipeline (needs uv) -> public/data/v1
npm run dev           # http://localhost:3000
```

Without network access, use the recorded fixture data:

```bash
npm run data:fixture                               # -> public/data/fixture/v1
CW_DATA_BASE_URL=/data/fixture/v1 npm run dev      # a "Test data" banner is shown
```

## What it shows (Milestone 1)

| Section | Product class | Source |
|---|---|---|
| Official closures and health advisories | (pointer only) | Not ingested yet: the card says so and links to CDFW / CDPH pages and hotlines |
| Bloom and toxin forecast | Official forecast | C-HARM v3.1, NOAA CoastWatch West Coast ERDDAP: P(*Pseudo-nitzschia* > 10,000 cells/L), P(particulate DA > 500 ng/L), P(cellular DA > 10 pg/cell); nowcast and +1 to +3 days |
| Satellite chlorophyll | Observation | NASA GIBS VIIRS NOAA-20 and PACE OCI tiles (validated date, verified legend) |
| Landing ports | Reference | CDFW ds3081 |

Bloom Intelligence, Fisheries & Economic Exposure and My Coast appear in navigation as **Upcoming** and are not linked.

## How it stays honest

- The server validates `manifest.json` and `ports.geojson` against `src/generated/schemas` (copied from `../schemas/v1`) before rendering. Missing or invalid data shows an explicit "Live data unavailable" state.
- Freshness (current / stale / historical / unavailable) is computed **in the browser** from the dates in the artifacts and each source's published policy. A stalled pipeline therefore cannot make old data look current.
- Failed updates are shown in a banner and on `/sources`; the last good data keeps its real dates.
- Only one raster is on the map at a time, each with its own legend.
- The point inspector reads the published value grids. Where a nearshore cell has no forecast value, it shows the nearest forecast cell, labelled with its distance.
- Safety-relevant copy lives in `src/content/copy.ts` and is checked by `tests/unit/safety.test.ts`.

## Code map

```
src/app/page.tsx                 Live Ocean Map (server: loads + validates data)
src/app/sources/page.tsx         Data & sources (status, errors, notes, freshness rules)
src/components/LiveOceanMap.tsx  client container: URL state (?var, lead, layer, inspect)
src/components/map/              MapCanvas (MapLibre), Inspector
src/components/panels/           OfficialStatusCard, ForecastPanel, ObservationPanel
src/lib/                         data (server), freshness, grid decoding, layers, time, basemap
src/content/copy.ts              user-facing copy with safety meaning, official links
src/generated/                   types + schemas generated from ../schemas/v1 (npm run gen:types)
scripts/                         gen-types, copy-fixture-data, copy-maplibre-worker
tests/unit, tests/e2e            vitest, playwright (fixture data in tests/fixture-data*)
```

## Checks

```bash
npm run lint && npm run typecheck && npm test   # unit tests
npm run test:e2e                                # production build + Playwright on fixture data
```

Configuration: see `.env.example`.
