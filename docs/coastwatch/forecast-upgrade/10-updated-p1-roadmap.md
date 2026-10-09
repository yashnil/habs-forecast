# 10. Recommended updated P1 roadmap

## Should the design-reset plan change?

**Partly.** The design system and the map structure from revision 2 hold up well against real high-resolution data.
- The navigation card, the dock, the port-only inspector, opaque rasters over hatched no-data, the readout, and native-resolution honesty are exactly what the new layers need.
- Three things change:
  1. **Less effort on making C-HARM look smooth.** The 3 km C-HARM field is coarse by nature, and display interpolation adds no information. Keep the approved banded palette and nearest-cell rendering. Drop further C-HARM polish: the visual detail users want comes from **observations**, not from re-rendering a 3 km model.
  2. **The dock becomes a layer system,** not a three-quantity C-HARM switch. It holds a Model group (C-HARM pDA / cDA / *Pseudo-nitzschia*), a Satellite group (latest clear view, single days) and an Ocean-physics group (currents, later drift). Each layer has its native-resolution chip and its own legend.
  3. **The time control depends on the layer:** forecast leads (now, +1, +2, +3, and **never** +4…+7 for C-HARM), or overpass days with **coverage bars** for satellite layers.

## P1, revised

| Step | Change from the original P1 | Notes |
|---|---|---|
| P1.1 Palette pipeline PR (`cw-probability-classes-v1`) + staging | unchanged | Still needed: C-HARM remains the only HAB forecast |
| P1.2 NavCard, ForecastDock, PortInspector, ValueReadout | **ForecastDock → LayerDock** with groups, resolution chips, per-layer legend, and a time control that depends on the layer | The manifest already lists layers by `product_class` |
| P1.3 Basemap: lighter land, coastline above data, opaque rasters, hatched no-data | unchanged | Verified with real 300 m data in the map lab |
| P1.4 Mobile bottom sheet | Layer picker in the sheet (native select per group) | |
| **P1.5 (new)** | Extend the **existing GIBS source** (`pipeline/…/sources/gibs.py`, today VIIRS NOAA-20 and PACE) with the Sentinel-3A/3B OLCI layers, and give imagery layers a "picture, not values" readout state, until U1 adds quantitative 300 m grids | A configuration change to an existing source, inside the P1 pipeline PR |
| P1.6 Tests | Add: a layer's native-resolution chip always present; forecast lead control never exceeds the layer's published leads; satellite time slider shows coverage; no readout from imagery-only layers | |

## What can ship immediately vs. what needs research

| Ships with P1 / U1–U4 (observations and operational products) | Needs validation first (research) |
|---|---|
| C-HARM 0–3 days with banded palette, nearshore caveat and its true horizon | Any 4–7-day bloom, toxin or closure outlook |
| Same-day satellite imagery (GIBS: VIIRS and PACE already live, OLCI added) → quantitative 300 m OLCI / 750 m VIIRS latest clear view with per-pixel age | Advected (moved-forward) satellite chlorophyll |
| WCOFS forecast currents (72 h) and HF-radar observed currents, labelled physics | Drift ensembles as a decision aid (R2) |
| PACE chlorophyll once an Earthdata token exists | PACE phytoplankton-type products (R4) |
| Per-species official status, port watch list, freshness card | Station-level next-sample probability (R3) |
| Published C-HARM skill scores as information (R1) | — |
