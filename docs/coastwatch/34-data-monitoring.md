# 34 — Data monitoring: silent outages, stale-but-published data, per-source freshness

**Status:** in review (this PR). Changes only the health check (`pipeline/coastwatch_pipeline/health.py`), which writes `health.json` and the health issue. The web does not read `health.json`; no manifest field and no published dataset changes.

## 1. The problem

The 2026-10-13 00:42Z run had three source failures and raised nothing. C-HARM, HF radar and VIIRS all got HTTP 403 from `coastwatch.pfeg.noaa.gov`:
- each was on its first failure, and alerts needed 3 in a row, which takes 1–2 days at the drifting 8–10 h schedule;
- the 2-day-old C-HARM run that the app labelled "Stale" was under the 3-day `charm_missing` threshold.

## 2. What alerts now

| Alert | When | Why |
|---|---|---|
| `source_not_updating:<source>` | no successful update within the source's own time budget (C-HARM 30 h, HF radar 18 h, …) | counting runs fails when runs drift apart |
| `upstream_outage:<host>` | ≥ 2 sources failing against one host in this run and the previous one | one alert per host, absorbing those sources' own alerts |
| `upstream_intermittent:<host>` | ≥ 2 sources failing against one host in ≥ 3 of the last 10 runs, recovering in between | found by the replay below; clears only after a clean window |
| `data_stale:charm` | the newest published C-HARM run is older than its own freshness policy, exactly what the app labels "Stale" | says whether CoastWatch could not reach NOAA or NOAA published nothing newer |
| `charm_input_changed`, `charm_input_ageing` | NOAA's declared C-HARM input changes, or still says S-NPP after 2026-11-02 | see [33 §5](33-suomi-npp-continuity.md#5-c-harm-in-this-pr) |

**No noisy duplicates:**
- one alert per source, root cause first;
- an alert that is already open keeps its key and its "since", so the health issue is commented only when something new appears or clears;
- intermittent alerts have hysteresis.

**Freshness table:**
- every run writes, per source: last attempt, last success, newest data date and its basis (issued, observed, valid or reviewed), age, state (current, stale or historical) and coverage;
- HF radar's age is in hours;
- dates are the sources' own, and nothing published is relabelled.

## 3. Validation against real outages (replay of every published run)

[`research/health-replay`](../../research/health-replay) downloads each data run's own `manifest.json` and `health.json` from its workflow artifact. It then replays the current rules over the 20 `main` runs from 2026-10-08 19:20Z to 2026-10-13 14:14Z, in order. This is read-only and publishes nothing.

| Run | What happened | Alerts then | Alerts now |
|---|---|---|---|
| 10-09 22:40 | pfeg 403: C-HARM failed; Sentinel-3 failed | none | Sentinel-3 not in dataset (rollout run) |
| 10-10 03:16 to 19:57 | C-HARM issue date stuck at 10-08 (NOAA late); the app showed "Stale" | none | **`data_stale:charm`**: one alert, held 4 runs |
| 10-10 12:34 to 21:46 | HF radar `ucsdHfrW2` 404, three runs | `source_failing` on the 3rd | same |
| 10-10 19:57, 10-11 06:59 | pfeg 403 again: C-HARM, HF radar, VIIRS | `hfr_stale` | **`upstream_intermittent:pfeg`** on the third burst, held open |
| 10-13 00:42 | pfeg 403: C-HARM, HF radar, VIIRS (the silent run) | **none** | `data_stale:charm` (the reason given: the 403) and the intermittent alert still open |
| 10-13 14:14 | HF radar 404 again; newest hour 28 h old | `hfr_stale` | `hfr_stale`; table row "stale, 28 h" |

- **Over 5 days:** 6 alert openings, every one a real problem, and no flapping.
- **The worst gap:** the 00:42Z run would no longer be silent.
- **The pattern the earlier version still missed:** pfeg refused three sources at once in 4 single runs in 3.5 days. It looks like a block or rate limit on GitHub's runner addresses. Retries already exist (4 attempts), so the next step, if it continues, is to ask NOAA PFEG. That would be outreach, so it needs your approval.

## 4. Not changed

- Thresholds for satellite latency (OLCI 4 d, VIIRS 8 d), `charm_missing` and the HF radar coverage checks. The replay shows VIIRS ages of 6–8 days, which are normal for S-NPP near-real time.
- How data publish: a failed source still keeps the last good data, with its own dates.
