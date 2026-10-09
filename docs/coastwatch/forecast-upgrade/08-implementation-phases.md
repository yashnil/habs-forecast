# 8. Realistic implementation phases

Two tracks.
- **The product track** ships observations and existing operational models: no new science, only the usual pipeline review and staging runs.
- **The research track** produces evidence. Nothing from it is public until its §6 gate passes and a HAB scientist has reviewed it.

## Product track

| Phase | Scope | Depends on | Size | Ships |
|---|---|---|---|---|
| **U0 — Design-reset P1 (adjusted)** | Map rebuild from revision 2: navigation card, dock, inspector, readout, opaque raster with hatching, C-HARM display palette (pipeline palette PR + staging). The dock becomes a **layer system** with Model / Satellite / Ocean-physics groups, a native-resolution chip, and a time slider with coverage bars for satellite layers. | P0 (PR #9) | 2–3 weeks | Yes |
| **U1 — High-resolution satellite chlorophyll** | Adds quantitative layers next to the existing GIBS daily imagery (VIIRS NOAA-20, PACE), which has no values: pipeline sources `olci300` (NOAA CW sectors CI + DI; S3A, falling back to S3B) and `viirs750`; latest clear-view composite with an age grid; per-day layers for 14 days; PMTiles for display, u16 chunks for readouts; port inspector "latest clear view near this port". Matchup statistics against pier chlorophyll in methods. | U0, hosting decision (§9) | 2–3 weeks | Yes |
| **U2 — Ocean currents** | `wcofs` source (surface currents, 6-hourly to 72 h) and `hfradar` (24 h mean, 2 km). Static arrows + WebGL particle layer (from a u/v texture), "Ocean physics · Model / Observation" chips, 4 km and 2 km resolution chips. | U0 | 1–2 weeks | Yes, as physics, not HAB |
| **U3 — PACE** | Earthdata token as a CI secret; `pace` source for L2 BGC NRT chlorophyll; same composite pipeline (about 1.2 km). Hyperspectral phytoplankton products evaluated in research first. | Your Earthdata account | 1 week | Yes (chlorophyll) |
| **U4 — Fisheries decision support** | Per-species "what applies today", the port watch list, port freshness card (§11). | U0–U1 | 1–2 weeks | Yes |

## Research track

| Phase | Question | Output | Size |
|---|---|---|---|
| **R1 — Score C-HARM** | How does C-HARM v3.1 (Nov 2022–now) score against CalHABMAP pDA and *Pseudo-nitzschia*, by lead and station? | A methods page with Brier skill score and reliability, published as information about the agency model | 1–2 weeks |
| **R2 — Surface drift** | Does a WCOFS particle ensemble beat HF-radar persistence at 24–72 h in Monterey Bay, the SF Bight and the SoCal Bight? | Hindcast report; if it passes, an experimental "where water may move" layer (72 h, spread shown) | 2–3 weeks |
| **R3 — Next-sample station model** | Can a calibrated model beat persistence *and* C-HARM for "next weekly sample ≥ 0.5 ng/mL" at each pier? | Backtest report; possibly a station-level experimental probability | 4–8 weeks |
| **R4 — PACE phytoplankton types** | Do PACE hyperspectral products separate diatom-dominated water in California nearshore matchups? | Feasibility memo | open-ended |

## Sequencing

```
now ─ U0 (P1 adjusted) ─┬─ U1 satellite 300 m ── U3 PACE
                        ├─ U2 currents
                        └─ U4 fisheries support
R1 (score C-HARM) runs in parallel from now; R2 after U2; R3 after R1.
```

The 7-day horizon enters the product only if R2 or R3 passes its gate, and then only as what was validated:
"water movement, 72 h" or "next pier sample".
