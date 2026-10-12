# Analysis scripts (coastal coverage audit, 2026-10-12)

Run with `uv run --script <file>` (dependencies are declared in each file). They read only public data and never write to the data branch:

| Script | Reads | Writes |
|---|---|---|
| `audit.py` | ETOPO 2022 15″ for California (`ca15s.nc`, NOAA NCEI); the published C-HARM nowcast grids (`charm/<issue>/lead0-*/{variable}.u16.gz` from the production dataset); 41 C-HARM v3.1 nowcasts from NOAA ERDDAP `wvcharmV3_0day` (the 1st and 15th of each month, Nov 2024 to Sep 2026, where a nowcast exists, plus 2026-10-10) | `data/charm_audit.json`, coverage maps |
| `sat_audit.py` | the published OLCI 300 m daily grids (6 days, 2026-10-05 to 10) | `data/olci_audit.json` |
| `ports_audit.py` | production `port-intel`, `observations` and the official notices registry | `data/ports_audit.json` |
| `coverage_geo.py` | the published nowcast grid | `data/charm_coverage_2026-10-10.geojson` (prototype geometry for the missing-data UX comparison) |
| `mycoast_data.py`, `mycoast_mock.py` | the files above | `data/mycoast.json`, `mockups/*.html` |

Input paths are relative to the working folder the audit was run in; the downloads themselves are not committed (ETOPO is 26 MB; the ERDDAP files are 60 MB).
