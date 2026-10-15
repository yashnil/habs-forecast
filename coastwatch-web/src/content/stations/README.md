# Station lists (bundled, regenerate when stations change)

- `ndbc-ca.json`: NOAA National Data Buoy Center stations in California waters reporting meteorological data
  (`met="y"`), from https://www.ndbc.noaa.gov/activestations.xml, fetched 2026-10-13. Public domain (NOAA).
- `coops-tide-ca.json`: NOAA CO-OPS tide-prediction stations in California (reference "R" and subordinate "S"),
  from https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=tidepredictions, fetched 2026-10-13.
  Public domain (NOAA).

- `coops-watertemp-ca.json`: NOAA CO-OPS stations in California with a water-temperature sensor, from
  https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=watertemp, fetched 2026-10-15. Public domain (NOAA).
  Several (Santa Barbara, Los Angeles, San Francisco, Redwood City) answered "No data" that day: a sensor can be down.

Only id, name, position and type are kept. Live values are fetched by `src/app/api/conditions/route.ts`.
