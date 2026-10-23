# Neritic v6: one product direction (lead synthesis of audits A–E)

Audit evidence: `scratchpad/v6-audit/{A,B,C,D,E}/`. Harbour ids: Princeton / Half Moon Bay = **port-452**; port-592 = Moss Landing.

## The product
- **Explore** = understand the coast (the map is the canvas).
- **My Coast** = decide and plan.
- **Research** = investigate the evidence.

There is no new workspace and no separate economics tab. Trip economics lives inside My Coast. Quality comes from subtraction and from the links between these three.

## P0 (must fix)

| # | Problem | Owner |
|---|---|---|
| 1 | "3 here" in the top bar vs "None tied to these waters" in the glance and fold. The counts must agree, and wording must separate "here" from "naming your species" | M |
| 2 | A commercial crabber sees the sport reopen date as the season | M |
| 3 | The Worth tile shows another species' price (bluefin for albacore, sablefish for nearshore) and reads as current; `landings.ts:43` pounds total is wrong (177,684,524) | M |
| 4 | Offline saved copy: tiles say "Tonight" and "Today" on a later day. Match NWS periods by their own start time; date every tile "as saved" | M |
| 5 | 8-day outlook drawn over SF and San Pablo Bays, with its tightest error bars there | DATA (mask) + X (hide reads in bays) |
| 6 | Shoreline-strip estimates described as "filled from nearby clear pixels" though they are extrapolated ×4.6 | DATA (class grid) + X (own class, wording, legend) |

## P1
- **Trip economics** (ECON). The highest-value new capability; specification below.
- **My Coast IA** (M):
  - Compact header.
  - What matters now: at most 3 one-line items, details on tap.
  - At a glance as rows:
    - official for my fishery;
    - sea vs my limits, with inline "Set limits";
    - trip cost;
    - break-even;
    - missing / uncertain;
    - inspect next.
  - "Since last check" only when something changed.
  - Sections in order: Trip economics, Sea, Seasons, Before you go (one line per check), Evidence, Notices, Unknowns.
  - Remove repeats: season ×11, NWS ×5, caveats ×8.
  - Map as a strip under the glance on phones; sticky map plus economics summary card at 1440.
- **Science wording** (X):
  - "NOAA publishes no toxin probability here", not "does not model".
  - SST is an "Analysis" (MUR foundation temperature) everywhere; "may mark a front where satellites saw clear sky".
  - Turbid-water caveat on observed bay chlorophyll.
  - Outlook skill table: "this outlook *is* climatology + AR(1)".
- **Cross-links** (X), all keeping context (`place`, `for`, `species`, `date`, `from`):
  - Every inspector ends with **Brief this**, keeping the trip, and **Research this**, carrying station/cell, day and variable.
  - Research shows "Arrived from <place> · nearest station N km" with a back link, and reads `from`.
  - Search covers species and harvest sites.
- **Offline, mobile, shell** (O):
  - Service worker: 5 s network deadline for navigations and `/api`; unmapped paths get the offline page.
  - Manifest `start_url` is `/coast`.
  - Launch screen on the **first visit ever** only; returning users have no hold.
  - Daylight mode (tokens; ⋯ menu; `nx-daylight-v1`).
  - Touch targets ≥ 44 px.
  - Compact top bar when the viewport is short.
  - Consistent age rounding.

## Deferred / rejected
- Rejected (no data, or they cross the science and economics boundary):
  - catch, price or "where to fish" forecasts;
  - trip score or go/no-go;
  - port ranking;
  - fuel-dock scraping;
  - route planner;
  - pre-filled price or catch;
  - cloud sync;
  - economics in Research;
  - PINN.
- Deferred:
  - `/fisheries` merge into My Coast and Research (P2);
  - landscape-specific Explore redesign beyond the compact top bar;
  - EIA diesel as a pipeline feed (shown as a dated static reference instead).

## Trip economics specification (ECON)
- **Pure library:** `src/lib/tripEconomics.ts`, with no React, no `window`, floats inside and rounding only at display.
  - Inputs:
    - `roundTripNm`, `cruiseKt`, `steamGph`;
    - `fishingHours`, `fishingGph` (or direct `fuelGal`), `fuelUsdPerGal`;
    - costs as lines, each `{label, usd, shared: boolean}` (shared = deducted before crew shares);
    - `feesPctOfGross`, `feesUsdPerLb`;
    - catch as lines `{species, lb, usdPerLb}`, with the primary line first;
    - crew model `{kind: "afterExpenses" | "ofGross" | "dayRate", pct?, crew?, usdPerDay?, days?}`.
  - Model A, share after expenses:
    - N = R(1−τ) − uQ − D;
    - crew = s·N, but crew = 0 if N ≤ 0;
    - owner net = (1−s)·N − B.
  - Model B, share of gross: owner net = R(1−s−τ) − uQ − D − B.
  - Model C, day rate: crew = crew × $/day × days, a boat-paid cost.
  - Break-even catch for the primary species (other lines held fixed): Q*. Denominator ≤ 0 means "no break-even at this price".
  - Break-even price p*, at the primary line's catch.
  - Sensitivity: price ±20 % × catch ±30 % grid; fuel ±20 %; the break-even fuel price.
  - Units: lb (default) or kg (display only); squid in $/ton ⇄ $/lb (÷2000).
  - Partial inputs return what they can, plus a `missing` list. Reject NaN and negatives; clamp absurd values.
- **Presets:** `src/content/fisheryPresets.ts`. Every default is editable and carries a source:
  - NWFSC-121 (2008) gph by gear;
  - crew shares 28.1 / 30.7 % (fixed gear) and 22.5 / 25.7 % (trawl);
  - buyback fee 2.25 % for LE trawl;
  - crab pot tiers.
  - Unknown values (albacore, urchin, squid fuel) are blank and user-entered.
- **References beside inputs, never pre-filled:**
  - CDFW By The Numbers 2024 statewide $/lb. Label: "Past average, not today's dock price; ask your buyer". Species-exact only, otherwise "no species-level reference".
  - EIA California diesel, weekly, as a dated static value with a staleness chip after 14 days. Label: "on-road retail incl. taxes; your fuel dock will differ".
- **Store:** `src/lib/tripEconomicsStore.ts`.
  - localStorage namespace `nx-trip-v1:` (`scenarios`, `vessels`, `meta`); records `{schema:1, id, createdAt, updatedAt, …}`; caps of 5 vessels and 20 scenarios.
  - Newer schema = read-only; corrupt = kept and reported, never dropped.
  - Export / Import (JSON); "Delete all trip data"; "Kept on this device only".
  - Never sent to the server; never in URLs unless the user presses "Copy scenario link" (numbers only, no names).
- **UI:** `src/v4/coast/economics/`.
  - `TripEconomics` (section; worksheet lazy via `next/dynamic`).
  - `useTripSummary` (for the glance rows).
  - `Fields` (NumberField, Select).
  - Inputs in four groups (Trip, Vessel and fuel, Costs and crew, Catch and price); live results in tabular figures.
  - Phone: results bar fixed above the dock.
  - Scenarios A/B compare: only the differing inputs plus the result rows.
  - Copy, always: "Scenario from your inputs, not a forecast." Banned words: safe, go/no-go, good day, hotspot, expected catch (from Neritic), price forecast.
  - Each saved scenario keeps snapshots of the references it used: `{value, source, period, fetchedAt}`.

## Contracts between tracks
- **ECON → M:**
  - `import { TripEconomics, useTripSummary } from "@/v4/coast/economics"`.
  - `<TripEconomics place={{ key, name, portCode }} species={SpeciesId} audience={Audience} tripDate={string|null} conditions={{ line: string; over: boolean | null; asSaved: string | null } | null} notices={{ here: number; namingSpecies: number }} />`
  - `useTripSummary(placeKey, species)` returns `{ name, costUsd, breakEvenLb, breakEvenUsdPerLb, netUsd } | null`.
  - Anchor id `trip-economics`.
- **DATA → X** (`derived/index.json`, never the manifest):
  - The `chl_estimate_latest` entry gains an optional `class_grid`, with the same grid shape and `transform:"none"`, values:
    - 1: filled from nearby clear water;
    - 2: extrapolated shoreward, never observed (0–0.6 km strip);
    - 3: bay, borrowed background;
  - plus a `class_legend`.
  - The `chl_outlook_8d` grid has its bay cells masked, and its caveat says so.
- **M ↔ X:** Explore's harbour inspector gets "Brief this" (X builds the href with `coastHref`, keeping `for/species/date` when present in `from`). Research reads `from`, `day` and `variable` (X).
- **O:** owns `theme.css`, `globals.css`, `v4/shell/**`, `sw.js`, `v4/offline/**`, `app/manifest.ts`, `Splash`/`MotionRoot`. Other tracks use existing tokens and classes only and request any token change in their report.
