# Station baselines

Data: `observations-6d325da89a.json` (generated 2026-10-08T18:00:00Z); split 2022-01-01; next sample 3–14 days later.

## pDA >= 0.5 ng/mL

| Station | train pairs | test pairs | test events | Brier base | BSS season | BSS persist | AUC season | AUC persist |
|---|---|---|---|---|---|---|---|---|
| Santa Cruz Wharf | 372 | 236 | 11 | 0.047 | -0.06 | -0.06 | 0.75 | 0.57 |
| Cal Poly Pier (Avila Beach) | 402 | 237 | 7 | 0.029 | -0.01 | 0.05 | 0.46 | 0.63 |
| Stearns Wharf (Santa Barbara) | 390 | 233 | 5 | 0.021 | -0.01 | 0.15 | 0.64 | 0.69 |
| Santa Monica Pier | 410 | 236 | 6 | 0.025 | 0.03 | 0.34 | 0.83 | 0.83 |
| Newport Beach Pier | 388 | 202 | 5 | 0.025 | 0.01 | -0.06 | 0.70 | 0.49 |
| Scripps Pier (La Jolla) | 416 | 240 | 7 | 0.029 | 0.00 | 0.11 | 0.64 | 0.71 |
| **pooled** | | 1384 | 41 | 0.029 | -0.01 | 0.07 | 0.67 | 0.67 |

## Pseudo-nitzschia >= 10,000 cells/L

| Station | train pairs | test pairs | test events | Brier base | BSS season | BSS persist | AUC season | AUC persist |
|---|---|---|---|---|---|---|---|---|
| Bodega Marine Lab | 60 | 227 | 2 | 0.009 | -0.31 | 0.16 | 0.12 | 0.75 |
| Santa Cruz Wharf | 321 | 235 | 27 | 0.102 | 0.14 | 0.25 | 0.82 | 0.75 |
| Monterey Wharf | 294 | 166 | 73 | 0.246 | 0.26 | 0.16 | 0.77 | 0.70 |
| Cal Poly Pier (Avila Beach) | 402 | 245 | 96 | 0.289 | 0.11 | 0.20 | 0.73 | 0.67 |
| Stearns Wharf (Santa Barbara) | 390 | 239 | 56 | 0.213 | 0.01 | 0.13 | 0.63 | 0.64 |
| Santa Monica Pier | 410 | 246 | 45 | 0.151 | 0.13 | 0.14 | 0.77 | 0.71 |
| Newport Beach Pier | 385 | 195 | 72 | 0.234 | 0.09 | 0.10 | 0.68 | 0.66 |
| Scripps Pier (La Jolla) | 393 | 240 | 97 | 0.241 | 0.07 | 0.11 | 0.65 | 0.67 |
| **pooled** | | 1793 | 468 | 0.184 | 0.11 | 0.15 | 0.76 | 0.77 |

