"""Pipeline health: alerts for problems a person should act on, without alerting on the
ordinary behaviour of the sources.

Evaluated after each run from the new manifest and a small history (health.json, carried
in the published dataset like everything else). What alerts, and why that threshold:

- source_failing      a NOAA source failed MIN_CONSECUTIVE_FAILURES runs in a row (~18 h at
                      the 6-hourly schedule). NOAA's ERDDAP refuses single runs (HTTP 403/502)
                      and serves the next one; one failure is not news.
- source_degraded     the same for a source that partly fails (e.g. Sentinel-3 down, VIIRS up).
- satellite_stale     newest Sentinel-3 pixel older than OLCI_MAX_AGE_DAYS, or newest VIIRS
                      pixel older than VIIRS_MAX_AGE_DAYS. Both are latency-aware (OLCI ~1-2 d;
                      VIIRS ~1.5 d, measured 2026-10-14 from ERDDAP's /files/ listings: the
                      "3-8 days" seen before was an ERDDAP index stuck for 9 days). Statewide some
                      pixel is almost always clear, so this means the feed stopped, not clouds.
- product_failing     one satellite product (OLCI or VIIRS) failed MIN_CONSECUTIVE_FAILURES runs
                      in a row while the other kept the source "partial". Each product has its
                      own key, so a failing VIIRS never hides behind, or masks, an OLCI alert.
- platform_silent     a platform of a product (Sentinel-3A/3B; for VIIRS each satellite) has no
                      scene newer than PLATFORM_MAX_AGE_DAYS while the product itself is current
                      (Sentinel-3B stopped after 2026-10-01 while 3A carried on: nothing alerted).
- coverage_collapse   a product's latest clear view covers under COVERAGE_DROP of its median over
                      the last runs (at least COVERAGE_MIN_RUNS of them): masking or an upstream
                      processing change, not weather (a 7-day composite rarely halves).
- port_chl_empty      no port has a satellite chlorophyll value (0 of 22 on 2026-10-13..14, when
                      the ports' dataset host refused the runners; nothing alerted).
- charm_missing       newest C-HARM issue date older than CHARM_MAX_AGE_DAYS (daily product).
- hfr_stale           newest HF-radar hour older than HFR_MAX_AGE_HOURS (latency is ~4-15 h).
- hfr_coverage_drop   newest hour has fewer than HFR_COVERAGE_DROP of the window's median
                      valid cells (a network outage, not hour-to-hour variation).
- hfr_region_lost     a region that has had radar coverage before (per health.json history)
                      has none in the 24-hour window. Regions never covered do not alert.
- publish_stale       the previous published manifest is older than PUBLISH_MAX_AGE_HOURS when
                      this run starts (missed runs).
- source_not_updating a source has not succeeded for longer than its time budget
                      (SUCCESS_BUDGET_HOURS). Counting failed runs alone missed outages when the
                      6-hourly schedule drifted to 8-10 h apart (three failures took >1 day).
- upstream_outage     two or more sources failed against the same host in this run and the
                      previous one (the 2026-10-13 00:42Z run: C-HARM, HF radar and VIIRS all
                      got HTTP 403 from coastwatch.pfeg.noaa.gov, each on its first failure, so
                      nothing alerted). One alert per host, not one per source.
- upstream_intermittent  two or more sources failed against the same host in at least
                      INTERMITTENT_MIN_RUNS of the last INTERMITTENT_WINDOW runs, never two in a row
                      (a replay of every published run, 2026-10-08..13: coastwatch.pfeg.noaa.gov
                      refused C-HARM, HF radar and VIIRS together in 4 single runs within 3.5 days;
                      each recovered by the next run, so no other check fired). One alert per host;
                      it clears only after a whole window without such a failure.
- data_stale          the newest published C-HARM run is older than its own freshness policy
                      allows (what the app labels "Stale"), whether or not the run succeeded. The message says which: CoastWatch could not reach the
                      source (last error), or the source published nothing newer.

- charm_input_changed NOAA's own metadata for the newest C-HARM run declares a different satellite
                      input, platform, product name or version than the previous run did (NASA
                      ends Suomi-NPP data on 2026-11-02; C-HARM v3.1 names S-NPP VIIRS as its
                      ocean-colour input). Raised once; forecasts either side may not compare.
- charm_input_ageing  after SNPP_END, new C-HARM runs still declare S-NPP as their input.

Every run also writes a per-source freshness and coverage table (health["freshness"]) with
each source's own dates; nothing is relabelled. Alerts are de-duplicated by key: the health
issue is commented only when an alert appears or clears.
Publication failures of this workflow itself are reported by the workflow (a failed job opens
or updates an issue); this module cannot run if the build fails.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

from .models import Manifest

MIN_CONSECUTIVE_FAILURES = 3
OLCI_MAX_AGE_DAYS = 4
VIIRS_MAX_AGE_DAYS = 5
PLATFORM_MAX_AGE_DAYS = 5
COVERAGE_DROP = 0.3
COVERAGE_MIN_RUNS = 4
COVERAGE_HISTORY = 10
# satellite products: latest-clear-view layer, age limit, label
SATELLITE_PRODUCTS = (("olci300", "olci300_chl_latest", OLCI_MAX_AGE_DAYS, "Sentinel-3 OLCI 300 m"), ("viirs750", "viirs750_chl_latest", VIIRS_MAX_AGE_DAYS, "VIIRS 750 m"))
CHARM_MAX_AGE_DAYS = 2
HFR_MAX_AGE_HOURS = 18
HFR_COVERAGE_DROP = 0.5
PUBLISH_MAX_AGE_HOURS = 14
NOAA_SOURCES = ("charm", "satellite_chl", "hf_radar")
# hours without a successful update before a source alerts (about two missed daily updates for
# daily sources; HF radar is hourly upstream)
SUCCESS_BUDGET_HOURS = {"charm": 30, "satellite_chl": 30, "gibs_chl": 36, "hf_radar": 18, "official": 30, "calhabmap": 48, "port_intel": 30}
OUTAGE_MIN_SOURCES = 2
INTERMITTENT_WINDOW = 10
INTERMITTENT_MIN_RUNS = 3
# sources whose published data are checked against the same freshness policy the app uses to
# label them. C-HARM: the app says "Stale" from 2 days, but charm_missing waited for 3 (the
# 2026-10-13 run published a 2-day-old run, labelled Stale, with no alert). Satellite and HF
# radar keep their latency-aware checks (satellite_stale, hfr_stale) instead.
STALE_CHECK = {"charm": "C-HARM forecast"}
HEALTH_FILE = "health.json"
SNPP_END = date(2026, 11, 2)
CHARM_INPUT_KEYS = ("satellite", "platform", "product_name", "product_version")


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _product_datasets(key: str) -> set[str]:
    from .sources import satellite as sat

    prods = [p for p in (getattr(sat, "PRODUCTS", None) or (sat.OLCI, sat.VIIRS)) if p.key == key]
    return {sec[0] for p in prods for _, sectors in p.platforms for sec in sectors}


def _product_platforms(key: str) -> list[str]:
    """The platforms the product's primary definition expects (OLCI: Sentinel-3A and 3B)."""
    from .sources import satellite as sat

    p = next((x for x in (sat.OLCI, sat.VIIRS) if x.key == key), None)
    return [pl for pl, _ in p.platforms] if p and len(p.platforms) > 1 else []


def evaluate(manifest: Manifest, previous: dict | None, now: datetime, previous_generated_at: str | None = None, port_intel: dict | None = None) -> dict:
    prev = previous or {}
    prev_sources = prev.get("sources", {})
    prev_alerts = {a["key"]: a for a in prev.get("alerts", [])}
    now_iso = now.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sources: dict[str, dict] = {}
    alerts: list[dict] = []

    def alert(key: str, kind: str, message: str) -> None:
        alerts.append({"key": key, "kind": kind, "message": message, "since": prev_alerts.get(key, {}).get("since", now_iso)})

    def source_of(a: dict) -> str | None:
        k = a["key"]
        if ":" in k and a["kind"] in ("source_failing", "source_degraded", "source_not_updating", "data_stale"):
            return k.split(":", 1)[1]
        if a["kind"] == "platform_silent":  # its own slot per platform: never masks the product's alerts
            return "satellite_chl:" + k.split(":", 1)[1]
        if a["kind"] in ("satellite_stale", "product_failing", "coverage_collapse"):
            # one de-duplication slot per product (OLCI, VIIRS), never one for both
            return "satellite_chl:" + k.split(":", 1)[1].split("_", 1)[0]
        return {"charm_missing": "charm", "hfr_stale": "hf_radar", "hfr_coverage_drop": "hf_radar"}.get(a["kind"])

    # consecutive failures per source
    for st in manifest.sources:
        p = prev_sources.get(st.source_id, {})
        n = (p.get("consecutive_failures", 0) + 1) if st.outcome == "failed" else 0
        # partial with an error: part of a source down (e.g. Sentinel-3 while VIIRS still updates)
        d = (p.get("consecutive_degraded", 0) + 1) if st.outcome == "partial" and st.error else 0
        sources[st.source_id] = {"outcome": st.outcome, "consecutive_failures": n, "consecutive_degraded": d, "last_success_at": st.last_success_at, "error": (st.error or "")[:300] or None}
        if st.source_id in NOAA_SOURCES and n >= MIN_CONSECUTIVE_FAILURES:
            alert(f"source_failing:{st.source_id}", "source_failing", f"{st.title}: failed {n} runs in a row (last success {st.last_success_at or 'never'}). Last error: {(st.error or '')[:200]}")
        elif st.source_id in NOAA_SOURCES and st.source_id != "satellite_chl" and d >= MIN_CONSECUTIVE_FAILURES:
            # (satellite chlorophyll alerts per product instead: product_failing below)
            alert(f"source_degraded:{st.source_id}", "source_degraded", f"{st.title}: partly failing {d} runs in a row; the rest still updates. Last error: {(st.error or '')[:200]}")

    layers = {lyr.layer_id: lyr for lyr in manifest.layers}
    today = now.date()

    # satellite products, each on its own: freshness, failures, platforms, coverage
    sat_status = next((st for st in manifest.sources if st.source_id == "satellite_chl"), None)
    prev_products = prev.get("satellite_products", {})
    products: dict[str, dict] = {}
    for key, lid, limit, label in SATELLITE_PRODUCTS:
        lyr = layers.get(lid)
        pp = prev_products.get(key, {})
        rec: dict = {"consecutive_failures": 0, "platform_last_seen": dict(pp.get("platform_last_seen", {})), "coverage_history": list(pp.get("coverage_history", []))}
        products[key] = rec
        age = None
        if lyr and lyr.time.observed_date:
            age = (today - date.fromisoformat(lyr.time.observed_date)).days
            if age > limit:
                alert(f"satellite_stale:{lid}", "satellite_stale", f"{label}: newest observation {lyr.time.observed_date} is {age} days old (alert above {limit}).")
        elif "satellite_chl" in sources:
            alert(f"satellite_stale:{lid}", "satellite_stale", f"{label}: not in the published dataset.")
        # failures: this product's own datasets named in the source's error this run
        datasets = _product_datasets(key) | {"OLCI", "Sentinel-3"} if key == "olci300" else _product_datasets(key) | {"VIIRS", "S-NPP", "NOAA-20", "NOAA-21"}
        err = (sat_status.error or "") if sat_status and sat_status.outcome in ("failed", "partial") else ""
        mine = [e for e in re.split(r";\s+(?=[A-Z])", err) if any(ds in e for ds in datasets)]
        if mine:
            rec["consecutive_failures"] = pp.get("consecutive_failures", 0) + 1
            if rec["consecutive_failures"] >= MIN_CONSECUTIVE_FAILURES:
                alert(f"product_failing:{key}", "product_failing", f"{label}: could not be updated {rec['consecutive_failures']} runs in a row (newest published {lyr.time.observed_date if lyr else 'none'}). Last error: {mine[0][:200]}")
        # platforms: newest scene of each expected platform (from the composite's days)
        if lyr and lyr.composite:
            for d in lyr.composite.days:
                if d.pixels_used or d.observed_times:
                    for pl in d.platforms:
                        rec["platform_last_seen"][pl] = max(rec["platform_last_seen"].get(pl, ""), d.date)
        if age is not None and age <= limit:
            for pl in _product_platforms(key):
                seen_ = rec["platform_last_seen"].get(pl)
                p_age = (today - date.fromisoformat(seen_)).days if seen_ else None
                if p_age is None or p_age > PLATFORM_MAX_AGE_DAYS:
                    alert(f"platform_silent:{key}:{pl}", "platform_silent", f"{label}: no {pl} scene {'since ' + seen_ if seen_ else 'in the window'} (alert above {PLATFORM_MAX_AGE_DAYS} days) while the product is current from its other platform(s); coverage is reduced, nothing is filled.")
        # coverage of the latest clear view against its recent median
        if lyr and lyr.coverage is not None and lyr.provenance.pipeline_run_id == manifest.pipeline_run_id:
            frac = lyr.coverage.domain_observed_fraction
            hist_ = rec["coverage_history"]
            if len(hist_) >= COVERAGE_MIN_RUNS:
                med = statistics.median(hist_)
                if med > 0.02 and frac < COVERAGE_DROP * med:
                    alert(f"coverage_collapse:{key}", "coverage_collapse", f"{label}: latest clear view covers {frac:.1%} of the domain, under {COVERAGE_DROP:.0%} of its median over the last {len(hist_)} runs ({med:.1%}).")
            rec["coverage_history"] = (hist_ + [round(frac, 4)])[-COVERAGE_HISTORY:]

    # ports without any satellite chlorophyll
    if port_intel is not None:
        ports = port_intel.get("ports") or []
        with_chl = sum(1 for p in ports if (p.get("chlorophyll") or {}).get("latest") is not None)
        if ports and with_chl == 0:
            errs = sorted({str((p.get("chlorophyll") or {}).get("error") or "")[:160] for p in ports} - {""})
            alert("port_chl_empty", "port_chl_empty", f"No port has a satellite chlorophyll value (0 of {len(ports)}). {('Error: ' + errs[0]) if errs else ''}".strip())

    # C-HARM run
    run = next((r for r in manifest.forecast_runs if r.group_id == "charm"), None)
    if run:
        age = (today - date.fromisoformat(run.issued_date)).days
        if age > CHARM_MAX_AGE_DAYS:
            alert("charm_missing", "charm_missing", f"C-HARM: newest run issued {run.issued_date}, {age} days ago (alert above {CHARM_MAX_AGE_DAYS}).")
    elif "charm" in sources:
        alert("charm_missing", "charm_missing", "C-HARM: no forecast run in the published dataset.")

    # HF radar
    hourly = sorted((lyr for lyr in manifest.layers if lyr.layer_id.startswith("hfr2km_currents_2") and lyr.qc), key=lambda x: x.time.valid_time or "")
    region_cov: dict[str, bool] = {}
    if hourly:
        newest = hourly[-1]
        hours = (now - _dt(newest.time.valid_time or now_iso)).total_seconds() / 3600
        if hours > HFR_MAX_AGE_HOURS:
            alert("hfr_stale", "hfr_stale", f"HF radar: newest hour {newest.time.valid_time} is {hours:.0f} h old (alert above {HFR_MAX_AGE_HOURS} h).")
        counts = [h.qc.n_valid for h in hourly if h.qc]
        med = statistics.median(counts) if counts else 0
        if med and newest.qc and newest.qc.n_valid < HFR_COVERAGE_DROP * med:
            alert("hfr_coverage_drop", "hfr_coverage_drop", f"HF radar: newest hour has {newest.qc.n_valid} valid cells, under {HFR_COVERAGE_DROP:.0%} of the 24 h median ({med:.0f}).")
        for h in hourly:
            for rc in (h.coverage.regions if h.coverage else []):
                region_cov[rc.region_id] = region_cov.get(rc.region_id, False) or rc.observed_fraction > 0
    elif "hf_radar" in sources:
        alert("hfr_stale", "hfr_stale", "HF radar: no hourly currents in the published dataset.")
    seen = dict(prev.get("hfr_regions_last_covered", {}))
    for rid, covered in region_cov.items():
        if covered:
            seen[rid] = now_iso
        elif rid in seen:
            alert(f"hfr_region_lost:{rid}", "hfr_region_lost", f"HF radar: no coverage in {rid} for the last 24 hours (last covered {seen[rid]}).")

    # no success for too long (time-based: robust to a drifting schedule)
    for st in manifest.sources:
        budget = SUCCESS_BUDGET_HOURS.get(st.source_id)
        if budget is None:
            continue
        since = (now - _dt(st.last_success_at)).total_seconds() / 3600 if st.last_success_at else None
        if since is None or since > budget:
            when = f"{since:.0f} h ago ({st.last_success_at})" if since is not None else "never"
            alert(f"source_not_updating:{st.source_id}", "source_not_updating", f"{st.title}: last successful update {when}, over the {budget} h budget. Last error: {(st.error or 'none')[:200]}")

    # one upstream host failing for several sources, in this run and the previous one
    hosts: dict[str, list[str]] = {}
    for st in manifest.sources:
        if st.outcome in ("failed", "partial") and st.error:
            for h in sorted(set(re.findall(r"https?://([^/\s]+)", st.error))):
                hosts.setdefault(h, []).append(st.source_id)
    prev_hosts = prev.get("failing_hosts", {})
    failing_hosts = {h: sorted(set(ids)) for h, ids in hosts.items() if len(set(ids)) >= OUTAGE_MIN_SOURCES}
    for h, ids in failing_hosts.items():
        if h in prev_hosts:
            alert(f"upstream_outage:{h}", "upstream_outage", f"{h}: {len(ids)} sources failed against it in this run and the previous one ({', '.join(ids)}). The last published data stay live with their own dates.")

    # the same host refusing several sources again and again, one run at a time
    history = (prev.get("failing_hosts_history") or [])[-(INTERMITTENT_WINDOW - 1):] + [{"at": now_iso, "hosts": sorted(failing_hosts)}]
    for h in sorted({h for run_ in history for h in run_["hosts"]}):
        n = sum(h in run_["hosts"] for run_ in history)
        # once open, it stays open until the host has a clean window (no flapping, no repeat comment)
        need = 1 if f"upstream_intermittent:{h}" in prev_alerts else INTERMITTENT_MIN_RUNS
        if n >= need and f"upstream_outage:{h}" not in {a["key"] for a in alerts}:
            alert(f"upstream_intermittent:{h}", "upstream_intermittent", f"{h}: several sources failed against it in {n} of the last {len(history)} runs (recovering in between). Each failed run keeps the last good data, which age meanwhile; a block on the runners' addresses or a rate limit would look like this.")

    # stale-but-published: the newest data of a daily source against its own policy
    freshness: dict[str, dict] = {}
    for st in manifest.sources:
        basis_date = {"issued_date": st.latest_issued_date, "valid_date": st.latest_valid_date}.get(st.freshness.basis if st.freshness else "", None) or st.latest_valid_date or st.latest_issued_date
        age = (today - date.fromisoformat(basis_date)).days if basis_date and re.match(r"\d{4}-\d{2}-\d{2}$", basis_date) else None
        state = None
        if age is not None and st.freshness:
            state = "current" if age <= st.freshness.current_max_age_days else "stale" if age <= st.freshness.stale_max_age_days else "historical"
        freshness[st.source_id] = {
            "title": st.title,
            "outcome": st.outcome,
            "last_attempt_at": st.last_attempt_at,
            "last_success_at": st.last_success_at,
            "basis": st.freshness.basis if st.freshness else None,
            "newest_date": basis_date,
            "age_days": age,
            "state": state,
        }
        if st.source_id in STALE_CHECK and state in ("stale", "historical"):
            why = (
                f"Neritic's last update failed ({(st.error or '')[:160]}); a newer one may exist upstream"
                if st.outcome in ("failed", "partial") and st.error
                else "the source has published nothing newer"
            )
            alert(f"data_stale:{st.source_id}", "data_stale", f"{STALE_CHECK[st.source_id]}: newest {st.freshness.basis.replace('_', ' ')} {basis_date} is {age} days old, {state} by its own policy (current up to {st.freshness.current_max_age_days} d); {why}.")

    # coverage beside freshness (what share of the domain the newest data cover)
    run_layers = [lyr for lyr in manifest.layers if lyr.group_id == "charm" and lyr.qc]
    if run_layers and "charm" in freshness:
        freshness["charm"]["coverage"] = {"valid_fraction": min(lyr.qc.valid_fraction for lyr in run_layers if lyr.qc and lyr.qc.valid_fraction is not None)}
    for lid in ("olci300_chl_latest", "viirs750_chl_latest"):
        lyr = layers.get(lid)
        if lyr and lyr.coverage and "satellite_chl" in freshness:
            freshness["satellite_chl"].setdefault("coverage", {})[lid] = {"observed_fraction": lyr.coverage.domain_observed_fraction, "observed_date": lyr.time.observed_date}
    # each satellite product its own row: one product updating must not make the other look current
    for key, lid, limit, label in SATELLITE_PRODUCTS:
        lyr = layers.get(lid)
        if lyr is None or "satellite_chl" not in freshness:
            continue
        age = (today - date.fromisoformat(lyr.time.observed_date)).days if lyr.time.observed_date else None
        fresh = lyr.freshness
        freshness[f"satellite_chl:{key}"] = {
            "title": label + (f" ({', '.join(lyr.platforms)})" if lyr.platforms else ""),
            "outcome": "failed" if products[key]["consecutive_failures"] else "updated" if lyr.provenance.pipeline_run_id == manifest.pipeline_run_id else "kept",
            "last_attempt_at": freshness["satellite_chl"]["last_attempt_at"],
            "last_success_at": lyr.provenance.retrieved_at,
            "basis": "observed_date",
            "newest_date": lyr.time.observed_date,
            "age_days": age,
            "state": None if age is None or fresh is None else "current" if age <= fresh.current_max_age_days else "stale" if age <= fresh.stale_max_age_days else "historical",
            "coverage": {"observed_fraction": lyr.coverage.domain_observed_fraction if lyr.coverage else None, "platforms_last_seen": products[key]["platform_last_seen"]},
        }
    if hourly and "hf_radar" in freshness:
        newest = hourly[-1]
        freshness["hf_radar"]["coverage"] = {"newest_hour": newest.time.valid_time, "n_valid": newest.qc.n_valid if newest.qc else None}
        # an hourly source: its date alone reads "1 d, current" for an hour 28 h old
        hours = (now - _dt(newest.time.valid_time or now_iso)).total_seconds() / 3600
        freshness["hf_radar"]["age_hours"] = round(hours)
        if hours > HFR_MAX_AGE_HOURS:
            freshness["hf_radar"]["state"] = "stale"

    # one alert per source (the root cause first), so one outage is one message, not five:
    # an upstream outage covers the sources failing against that host; stale data outranks a
    # failing count, which outranks "not updating"
    rank = {"data_stale": 0, "charm_missing": 1, "satellite_stale": 1, "hfr_stale": 1, "hfr_coverage_drop": 2, "product_failing": 2, "coverage_collapse": 2, "platform_silent": 3, "source_failing": 3, "source_degraded": 3, "source_not_updating": 4}
    outage_sources = {sid for a in alerts if a["kind"] == "upstream_outage" for sid in failing_hosts.get(a["key"].split(":", 1)[1], [])}
    best: dict[str, dict] = {}
    for a in alerts:
        sid = source_of(a)
        if sid is None:
            continue
        if sid in outage_sources and a["kind"] in ("source_failing", "source_degraded", "source_not_updating"):
            continue
        # an alert already open for this source stays the one reported (stable key and "since",
        # no extra comment when its cause also turns into stale data)
        def score(x: dict) -> tuple[int, int]:
            return (0 if x["key"] in prev_alerts else 1, rank.get(x["kind"], 9))

        if sid not in best or score(a) < score(best[sid]):
            best[sid] = a
    alerts[:] = [a for a in alerts if source_of(a) is None or best.get(source_of(a)) is a]

    # C-HARM's declared input (NOAA's metadata on the newest run's layers)
    charm_input: dict[str, str] = {}
    for lyr in manifest.layers:
        if lyr.group_id == "charm" and lyr.provenance and lyr.provenance.upstream_metadata:
            md = lyr.provenance.upstream_metadata
            charm_input = {k: str(md[k]) for k in CHARM_INPUT_KEYS if k in md}
            break
    prev_input = prev.get("charm_input") or {}
    changed = {k: (prev_input.get(k), v) for k, v in charm_input.items() if k in prev_input and prev_input[k] != v}
    if changed:
        what = "; ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items())
        alert("charm_input_changed", "charm_input_changed", f"C-HARM's declared input changed ({what}). Forecasts before and after may not be directly comparable; state this in the app.")
    elif "charm_input_changed" in prev_alerts and (now - _dt(prev_alerts["charm_input_changed"]["since"])).days < 7:
        alerts.append(prev_alerts["charm_input_changed"])  # stays open for a week, then clears by itself
    if today >= SNPP_END and "S-NPP" in charm_input.get("satellite", "") + charm_input.get("product_name", ""):
        alert("charm_input_ageing", "charm_input_ageing", f"C-HARM runs after {SNPP_END} still declare S-NPP as their input ({charm_input}); NASA ended S-NPP data then, so the input may be ageing or stale.")

    # missed runs
    if previous_generated_at:
        gap = (now - _dt(previous_generated_at)).total_seconds() / 3600
        if gap > PUBLISH_MAX_AGE_HOURS:
            alert("publish_stale", "publish_stale", f"The previously published dataset was {gap:.0f} h old when this run started (runs are every 6 h).")

    return {
        "schema": "coastwatch-health-1",
        "evaluated_at": now_iso,
        "pipeline_run_id": manifest.pipeline_run_id,
        "thresholds": {
            "min_consecutive_failures": MIN_CONSECUTIVE_FAILURES, "olci_max_age_days": OLCI_MAX_AGE_DAYS, "viirs_max_age_days": VIIRS_MAX_AGE_DAYS,
            "platform_max_age_days": PLATFORM_MAX_AGE_DAYS, "coverage_drop": COVERAGE_DROP, "charm_max_age_days": CHARM_MAX_AGE_DAYS, "hfr_max_age_hours": HFR_MAX_AGE_HOURS, "hfr_coverage_drop": HFR_COVERAGE_DROP,
            "publish_max_age_hours": PUBLISH_MAX_AGE_HOURS, "success_budget_hours": SUCCESS_BUDGET_HOURS, "outage_min_sources": OUTAGE_MIN_SOURCES,
            "intermittent_window": INTERMITTENT_WINDOW, "intermittent_min_runs": INTERMITTENT_MIN_RUNS,
        },
        "sources": sources,
        "freshness": freshness,
        "failing_hosts": failing_hosts,
        "failing_hosts_history": history,
        "charm_input": charm_input or prev_input,
        "hfr_regions_last_covered": seen,
        "satellite_products": products,
        "alerts": alerts,
        "new_alerts": [a["key"] for a in alerts if a["key"] not in prev_alerts],
        "cleared_alerts": [k for k in prev_alerts if k not in {a["key"] for a in alerts}],
    }


def markdown(h: dict, run_url: str = "") -> str:
    lines = [f"Pipeline health at {h['evaluated_at']} (run {h['pipeline_run_id']}{', ' + run_url if run_url else ''})", ""]
    if not h["alerts"]:
        lines.append("All checks clear.")
    for a in h["alerts"]:
        new = " **new**" if a["key"] in h["new_alerts"] else ""
        lines.append(f"- [{a['kind']}]{new} {a['message']} (since {a['since']})")
    if h["cleared_alerts"]:
        lines += ["", "Cleared since the last run: " + ", ".join(h["cleared_alerts"])]
    if h.get("freshness"):
        lines += ["", "| Source | Last attempt | Last success | Newest data | Age | State |", "|---|---|---|---|---|---|"]
        for sid, f in h["freshness"].items():
            lines.append(f"| {sid} | {f['outcome']} {f['last_attempt_at'] or ''} | {f['last_success_at'] or 'never'} | {f['basis'] or ''} {f['newest_date'] or '—'} | {str(f['age_hours']) + ' h' if f.get('age_hours') is not None else '' if f['age_days'] is None else str(f['age_days']) + ' d'} | {f['state'] or '—'} |")
    lines += ["", "Thresholds and rationale: pipeline/coastwatch_pipeline/health.py. The published data keep their own dates; nothing is relabelled."]
    return "\n".join(lines)


def run_health(out_dir: Path, previous_manifest_generated_at: str | None, now: datetime | None = None) -> dict:
    m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
    p = out_dir / HEALTH_FILE
    previous = json.loads(p.read_text()) if p.exists() else None
    pi = out_dir / m.port_intel_url if m.port_intel_url else None
    port_intel = json.loads(pi.read_text()) if pi and pi.exists() else None
    h = evaluate(m, previous, now or datetime.now(timezone.utc), previous_manifest_generated_at, port_intel)
    p.write_text(json.dumps(h, indent=1) + "\n")
    return h
