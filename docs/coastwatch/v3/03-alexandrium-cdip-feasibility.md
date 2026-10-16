# v3 · feasibility: CalHABMAP *Alexandrium* counts, then CDIP wave buoys

Owner's brief (2026-10-16): check *Alexandrium* first, because HAB coverage near shore is the larger
scientific gap; quantify which harbours gain genuinely new recent observations; do not force the
integration if the feed is inaccessible, stale, geographically redundant or poorly documented, and
recommend CDIP instead. Never imply that *Alexandrium* abundance measures PSP toxins or decides
shellfish safety. Probes made 2026-10-16, one request at a time; nothing was integrated.

## Verdict

**Neither is built now.** *Alexandrium* is accessible and fresh but **geographically and temporally
redundant**: 0 of 22 harbours gain a new observation place or time. CDIP is **redundant too**:
26 of 27 live California buoys are already read through their NDBC mirrors, and the remaining one is
more than 30 km from every harbour. So the CDIP fallback does not add new harbour observations.
Recommendation: add *Alexandrium* later as a **pipeline variable**, after the v2 release, web-first
(see Cost). Do not build CDIP.

## *Alexandrium* spp. (CalHABMAP via SCCOOS ERDDAP)

| Test | Result |
|---|---|
| Accessible | `Alexandrium_spp` (cells/L, "Phytoplankton Species") in all 17 `HABs-*` datasets Neritic already reads; HTTP 200 |
| Fresh | As fresh as each pier's sampling: Cal Poly Oct 11, Santa Monica Oct 12, Stearns Sep 28, Santa Cruz Sep 30; northern stations' last samples are Mar–Apr 2026 |
| Redundant | **Counted on exactly the same visits as Pseudo-nitzschia at every station**, set-equal by timestamp over the last year (e.g. Santa Cruz 48 = 48, Cal Poly 52 = 52) |
| Documented | Units and long name only; no method, counting limit or taxonomic note in the dataset metadata (the dataset summary still says "eight piers"). Pseudo-nitzschia counts have the same gap (the pipeline says "not published as a limit") |

**Harbours that would gain something** (recent = ≤ 30 days at 2026-10-16, within 60 km):
- New observation place or time: **0 of 22**.
- A new variable from visits already read: **9 of 22**: Santa Cruz (2.2 km), Moss Landing, Monterey
  (via Santa Cruz Wharf); Avila / Port San Luis (1.4 km), Morro Bay (23.9 km) (via Cal Poly Pier);
  Santa Barbara (0.4 km), Ventura, Port Hueneme (via Stearns Wharf); San Pedro (via Santa Monica Pier).
- A non-zero recent count: **1 station, 2 harbours**. Cal Poly Pier counted 7,463 (Sep 27), 2,132
  (Oct 4) and 5,330 cells/L (Oct 11), serving Avila / Port San Luis and Morro Bay. Every other recent
  count is 0.

**Scientific value.** It is the only seawater signal for the PSP gap the species line names, so its
value is a new variable, not new coverage. If integrated, it is said as "*Alexandrium* cells at
<pier>", and the line must keep saying that cell counts do not measure saxitoxins. Only CDPH's tests
of shellfish say anything about toxin in the animal. No line may treat a zero as clean, or a count
as a closure or a safety statement.

**Cost and risk.** The released web and the demo validate `observations.json` strictly, and the
variable id is a closed enum (`pDA, tDA, dDA, pn_seriata, pn_delicatissima, chl_extracted, temp`).
Adding `alexandrium` in the pipeline would make the live site reject the whole observations file
([[manifest-schema-compat]]). The safe path, after v2 is released: (1) a web release whose schema
accepts the new id; (2) only then, the pipeline adds the column to the request it already makes.
A web-route live read would avoid the schema but duplicate CalHABMAP reads on every page view; not
worth it for a variable already in the pipeline's request.

## CDIP wave buoys (fallback)

- THREDDS realtime catalog: 86 stations, **27 in California**, all reporting at 15:00–16:00Z on 2026-10-16.
- **26 of 27 have a WMO id already in Neritic's NDBC station list** (`src/content/stations/ndbc-ca.json`),
  so their waves and SST already reach `/api/conditions` through NDBC realtime2 (e.g. 156 = 46236,
  076 = 46215, 201 = 46254, 142 = 46237).
- Not listed: 203 (46251, Santa Cruz Basin, 33.75 N 119.59 W), within 30 km of no harbour.
- Harbours gaining a new wave observation: **0 of 22**. CDIP's own feed would add only latency
  (about 6 min vs NDBC's hourly mirror), peak wave direction and spectra, at buoys already shown.

## What would add genuinely new nearshore HAB coverage

Nothing probed today adds a new place. Five of the 22 harbours have no CalHABMAP shore station within
60 km: Crescent City (77 km to Trinidad Pier), Shelter Cove (79 km to Humboldt Bay, South Bay),
Point Arena (85 km) and Fort Bragg (139 km) (both to the Bodega Marine Lab buoy), and Princeton /
Half Moon Bay (73 km to Santa Cruz Wharf). They need a new sampling programme, not a new feed. Worth asking the owner: whether to approach the IFCB
dashboard contacts about the licensing of class scores (v2 sources §3: not usable as counts today).
