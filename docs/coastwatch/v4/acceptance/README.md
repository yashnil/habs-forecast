# Neritic v4: acceptance pass (2026-10-16)

The owner asked for a design, functionality and integration acceptance pass of v4 before any merge
or release. This folder holds the audits behind it; the PR (#68) and the report link to them.

| File | What it is |
|---|---|
| [test-reconciliation-A-map.md](test-reconciliation-A-map.md) | The 75 deleted v2 map tests (map, m2, m5, p2-currents, p2-multisensor, counties), one row each |
| [test-reconciliation-B-briefing.md](test-reconciliation-B-briefing.md) | The 58 deleted My Coast tests (coast, coastal, p1, p3, m8) |
| [test-reconciliation-C-shell-kept.md](test-reconciliation-C-shell-kept.md) | The 47 deleted shell tests (shell, release, tour, welcome) and every assertion weakened in kept specs |
| [research-audit.md](research-audit.md) | Research against the brief's nine scientific functions, before the Research tools were built |
| [task-walkthrough.md](task-walkthrough.md) | A scripted v2/v4 walkthrough of eight tasks (keystroke-level, not a user study) |
| [feature-parity.md](feature-parity.md) | The final v2 → v4 parity matrix: each capability, where it is now, its provenance, its test |
| [final-walkthrough.md](final-walkthrough.md) | The final scripted walkthrough: 5 harbours × 3 viewports × 10 tasks, defects and their status (not a user study) |
| [real-device-test-script.md](real-device-test-script.md) | The first real-phone and real-people checks: iPhone and Android, airplane mode, 5 fishers and 5 researchers (moderated script and results sheet) |
| [integration-66.md](integration-66.md) | v4 on main with Harbor Intelligence v3 (#66): merge checks, rollback references, the species-line schema change and the small-sample disclosure |

The audits were written by reading code (A–D) or scripting the running apps (E); each row cites
file and line. Paths are relative to the repository.

## From 273 to v4's browser tests

v2 released with 262 browser tests; Harbor Intelligence v3 (#66) adds 11 (`evidence-v3`), 273 in
all. v4 deleted 15 v2 spec files (180 tests) whose surfaces it replaced, and first shipped with 119.

The reconciliation classified every deleted test. Of the 64 rated critical, 3 were still covered;
the rest were partial, uncovered or removed, and in a number of cases the audit found behaviour v4
had actually lost, not only its tests. Restored, each with an end-to-end test (`trust`, `briefing-trust`, `navigation`,
`states`, `sync`, `ergonomics`, and tightened `discrepancy`, `regulatory`, `research`):

- official notices: every record, the verification reasons and ages, the expected-end-passed
  warning, the agency's words, "does not mean an area is open", the phone lines, focus on Escape;
- freshness: stale and historical runs, views and hours are said and never "Ready"; failed updates;
- caveats: not a closure decision; a low probability is not safe; the inferred issue date;
- gaps: a clouded satellite day, the regions the radar did not see, "nothing current";
- values by click and keyboard (not hover only), and v2 `inspect=` links that open them;
- the research-preview and test-data disclosure on every workspace; county lines' control;
- the agency's own quote on the season status; the dialable closure line; at most five checks;
- currents in knots and the 24-hour mean's tidal note; skip links; the about and welcome links.

Replaced on purpose (each replacement is tested): the guided tour and welcome card (start card,
search with the common places, readiness); the region-scoped landing ("Go to Monterey Bay");
the v2 layer dock (layer panel, phone chips); the notices drawer (notices panel); My Coast's map
column (a locator and "Show on map"); the Coastal/Research mode switch (workspaces).

Removed at the first acceptance pass, and **restored in the final pass** (see feature-parity.md): home-port set and clear; choosing waters by clicking My Coast's map; the
combined currents and chlorophyll view (`chl=1`); per-region satellite coverage percentages;
multi-sensor agreement and VIIRS-age lines; the per-harbour notice count chip; Explore's
per-variable port medians; the briefing checklist's tick boxes.

## Gate

Every push of this branch passed lint, typecheck, the generated-types check, the unit suite and
the browser suite in the Linux Playwright container, the same steps as CI, on the exact tree.
