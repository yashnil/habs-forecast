# Station lists (bundled, regenerate when stations change)

- `ndbc-ca.json`: NOAA National Data Buoy Center stations in California waters reporting meteorological data
  (`met="y"`), from https://www.ndbc.noaa.gov/activestations.xml, fetched 2026-10-13. Public domain (NOAA).
- `coops-tide-ca.json`: NOAA CO-OPS tide-prediction stations in California (reference "R" and subordinate "S"),
  from https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=tidepredictions, fetched 2026-10-13.
  Public domain (NOAA).

Only id, name, position and type are kept. Live values are fetched by `src/app/api/conditions/route.ts`.
