# Station lists (bundled, regenerate when stations change)

- `ndbc-ca.json`: NOAA National Data Buoy Center stations in California waters reporting meteorological data
  (`met="y"`), from https://www.ndbc.noaa.gov/activestations.xml, fetched 2026-10-13. Public domain (NOAA).
- `coops-tide-ca.json`: NOAA CO-OPS tide-prediction stations in California (reference "R" and subordinate "S"),
  from https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=tidepredictions, fetched 2026-10-13.
  Public domain (NOAA).

- `coops-watertemp-ca.json`: NOAA CO-OPS stations in California with a water-temperature sensor, from
  https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=watertemp, fetched 2026-10-15 (16 stations;
  re-checked against the same URL on 2026-10-15, identical). Public domain (NOAA).
  Each entry also carries `sensorNote`: the water-temperature sensor's (E1) elevation relative to MLLW and its
  status from https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{id}/sensors.json, read
  2026-10-15 (feet in the metadata, converted to metres). That metadata states no horizontal placement and no
  "depth below the surface": the thermistors sit 0.3–5.6 m below mean lower low water, so the depth below the
  surface changes with the tide. Four entries are kept although they do not report: Los Angeles 9410660 has no
  E1 sensor in its sensor list, and San Francisco 9414290, Redwood City 9414523 and Martinez 9415102 are marked
  inactive ("Suspect Data - under review") since 2025-12 / 2026-03. Keeping them makes the route say "sensor not
  reporting" for a harbour that has a station, rather than pretending there is none. Audit:
  `docs/coastwatch/v2/03-coops-audit.md`.

Only id, name, position and type (plus `sensorNote` for the thermistors) are kept. Live values are fetched by
`src/app/api/conditions/route.ts`.
