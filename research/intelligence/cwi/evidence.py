"""Evidence items: the only things an answer may state as fact.

Every item has a stable id, an authority tier and a plain-text body. Tiers
follow the CoastWatch hierarchy (GOALS.md §2): official notices outrank
measurements, which outrank model output, which outranks context and research.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum

from .snapshot import Snapshot


class Tier(IntEnum):
    OFFICIAL = 1      # CDFW / CDPH notices and statements
    MEASUREMENT = 2   # CalHABMAP shore-station samples (observations, never forecasts)
    MODEL = 3         # C-HARM official forecast/nowcast (water-column probabilities)
    CONTEXT = 4       # satellite chlorophyll, fisheries exposure, data status
    RESEARCH = 5      # offline research results; never operational


@dataclass
class Evidence:
    id: str
    tier: Tier
    kind: str
    title: str
    text: str
    as_of: str | None = None              # ISO date the content refers to
    source_url: str | None = None
    port_codes: tuple[int, ...] = ()
    species: tuple[str, ...] = ()
    flags: dict = field(default_factory=dict)  # restrictive, disputed, unverified, stale, unavailable
    numbers: tuple[str, ...] = ()          # extra numeric facts not in text

    def numeric_facts(self) -> set[float]:
        vals = set()
        for s in (self.text, self.title, *self.numbers):
            vals |= numbers_in(s)
        return vals

    def card(self) -> dict:
        """Compact form handed to a model as a tool result."""
        d = {"evidence_id": self.id, "tier": self.tier.name.lower(), "kind": self.kind,
             "title": self.title, "text": self.text}
        if self.as_of:
            d["as_of"] = self.as_of
        if self.source_url:
            d["source_url"] = self.source_url
        if self.flags:
            d["flags"] = self.flags
        return d


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:]+Z?)?\b")


def numbers_in(s: str) -> set[float]:
    """Numbers in ``s``, ignoring ISO dates (their parts are not numeric facts)."""
    out = set()
    for m in _NUM.findall(ISO_DATE.sub(" ", s or "")):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.{nd}g}"


SPECIES_ALIASES = {
    "mussels": ["mussel", "mussels"],
    "clams": ["clam", "clams"],
    "razor clam": ["razor clam", "razor clams"],
    "oysters": ["oyster", "oysters"],
    "scallops": ["scallop", "scallops"],
    "bivalves": ["bivalve", "bivalves", "shellfish"],
    "northern anchovy": ["anchovy", "anchovies"],
    "rock crab": ["rock crab", "rock crabs"],
    "dungeness crab": ["dungeness", "dungeness crab"],
    "spiny lobster": ["lobster", "spiny lobster"],
}
BIVALVES = {"mussels", "clams", "razor clam", "oysters", "scallops", "bivalves"}


def canonical_species(text: str) -> set[str]:
    t = text.lower()
    out = {k for k, al in SPECIES_ALIASES.items() if any(re.search(rf"\b{re.escape(a)}\b", t) for a in al)}
    if "razor clam" in out:
        out.discard("clams")
    return out


def _record_species(rec: dict) -> tuple[str, ...]:
    s = canonical_species(" ".join(rec.get("species", [])) + " " + rec["title"])
    if s & BIVALVES:
        s.add("bivalves")
    return tuple(sorted(s))


RESTRICTIVE_ACTIONS = {"quarantine", "consumption_advisory", "take_restriction", "fishery_closure", "special_advisory"}


def build_corpus(snap: Snapshot) -> dict[str, Evidence]:
    ev: dict[str, Evidence] = {}
    disputed = snap.disputed_record_ids()
    review = snap.review
    unverified = review.get("status") != "verified"
    relations: dict[str, list[int]] = {}
    for p in snap.ports:
        for rel in p["official_relations"]:
            relations.setdefault(rel["record_id"], []).append(p["port_code"])

    # ---- tier 1: official records and statements -------------------------
    for rec in snap.records:
        flags = {}
        if rec["action"] in RESTRICTIVE_ACTIONS and rec["status"] == "active":
            flags["restrictive"] = True
        if rec["id"] in disputed:
            flags["disputed"] = disputed[rec["id"]]
        if unverified:
            flags["unverified"] = f"registry review status: {review.get('status')}"
        status = "DISPUTED (an agency release appears to lift it; not yet reviewed by a person)" \
            if rec["id"] in disputed else rec["status"]
        text = (f"{rec['agency']} {rec['action'].replace('_', ' ')} — {rec['title']}. "
                f"Area: {rec['area']['description']}. Status: {status}. "
                f"Effective {rec.get('effective_date') or 'date not stated'}"
                + (f"; expected end {rec['expected_end_date']}" if rec.get("expected_end_date") else "")
                + f". Official text: \"{rec['official_text']}\"")
        ev[f"official:{rec['id']}"] = Evidence(
            id=f"official:{rec['id']}", tier=Tier.OFFICIAL, kind="official_notice",
            title=rec["title"], text=text, as_of=rec.get("effective_date"),
            source_url=(rec.get("sources") or [{}])[0].get("url"),
            port_codes=tuple(relations.get(rec["id"], ())), species=_record_species(rec), flags=flags)
    for st in snap.statements:
        flags = {"unverified": f"registry review status: {review.get('status')}"} if unverified else {}
        if re.search(r"\b(season is closed|is closed|closures? (?:is|are) in effect)\b", st["statement"], re.I):
            flags["restrictive"] = True
        ev[f"official:{st['id']}"] = Evidence(
            id=f"official:{st['id']}", tier=Tier.OFFICIAL, kind="official_statement",
            title=st["topic"], text=f"{st['source']['label']}: \"{st['statement']}\"",
            source_url=st["source"].get("url"),
            port_codes=tuple(p["port_code"] for p in snap.ports),
            species=tuple(sorted(canonical_species(st["topic"]))), flags=flags)

    # ---- tier 2: CalHABMAP shore-station measurements ---------------------
    var_meta = {v["id"]: v for v in snap.observations["variables"]}
    for stn in snap.stations:
        for sm in stn["summaries"]:
            if not sm.get("last_date"):
                continue
            meta = var_meta.get(sm["variable"], {})
            age = sm.get("days_since_last")
            flags = {}
            if age is not None and age > 30:
                flags["stale"] = f"last sample {age} days before the snapshot"
            q = " (reported 0: not quantified, not 'absent')" if sm["last_value"] == 0 else ""
            text = (f"CalHABMAP {stn['name']} ({meta.get('label', sm['variable'])}, {meta.get('matrix', 'seawater')}): "
                    f"last sample {sm['last_date']}, value {fmt(sm['last_value'])} {meta.get('units', '')}{q}; "
                    f"{age} days old at snapshot time; {sm['n_last_365d']} samples in the last 365 days"
                    + (f", max {fmt(sm['max_last_365d'])}" if sm.get("max_last_365d") is not None else "")
                    + f". Station is {stn['nearest_port_km']} km from {stn['nearest_port_name']}. "
                    "A single water sample at the pier; it is an observation, not a forecast, and seawater "
                    "toxin is not toxin in seafood.")
            eid = f"obs:{stn['station_id']}:{sm['variable']}"
            ev[eid] = Evidence(
                id=eid, tier=Tier.MEASUREMENT, kind="measurement", title=f"{stn['name']} {meta.get('label', sm['variable'])}",
                text=text, as_of=sm["last_date"], source_url=stn.get("source_url"),
                port_codes=(stn["nearest_port_code"],), flags=flags,
                numbers=(str(sm["last_value"]), str(age), str(stn["nearest_port_km"])))

    # ---- tier 3 / 4: per-port C-HARM and satellite chlorophyll ------------
    src = {s["source_id"]: s for s in snap.manifest["sources"]}
    for p in snap.ports:
        code, name, ch = p["port_code"], p["display_name"], p["charm"]
        if ch.get("cells_in_radius"):
            parts = []
            for lead in ch["leads"]:
                vs = lead["variables"]
                bits = ", ".join(f"{k.replace('_', ' ')} median {fmt(v['median'])} (range {fmt(v['min'])}–{fmt(v['max'])}, n={v['n']})"
                                 for k, v in vs.items())
                parts.append(f"lead {lead['lead_days']} d (valid {lead['valid_date']}): {bits}")
            flags = {}
            if src.get("charm", {}).get("outcome") == "failed":
                flags["stale"] = f"latest C-HARM fetch failed; values are from the run issued {ch['issued_date']}"
            text = (f"C-HARM v3.1 run issued {ch['issued_date']}, cells within {ch['radius_km']} km of {name} "
                    f"({ch['cells_in_radius']} cells; nearest {ch['nearest_cell_km']} km). " + "; ".join(parts)
                    + ". Values are model probabilities for the water column, not toxin in seafood; low values do not mean safe.")
            flags_un = flags
        else:
            text = (f"No C-HARM model cell within {ch['radius_km']} km of {name}; nearest cell {ch['nearest_cell_km']} km away.")
            flags_un = {"unavailable": "no C-HARM value near this port"}
        ev[f"charm:{code}"] = Evidence(
            id=f"charm:{code}", tier=Tier.MODEL, kind="charm_forecast", title=f"C-HARM near {name}", text=text,
            as_of=ch.get("issued_date"), source_url="https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.html",
            port_codes=(code,), flags=flags_un)

        chl = p["chlorophyll"]
        if chl.get("latest") is not None:
            text = (f"VIIRS 8-day chlorophyll-a median within {chl['radius_km']} km of {name}: {fmt(chl['latest'])} mg/m³ "
                    f"(composite centred {chl['latest_center_date']}). Chlorophyll is algal biomass, not toxin.")
            flags = {}
        else:
            text = (f"Satellite chlorophyll for {name} is unavailable in this snapshot (the fetch failed). "
                    "Chlorophyll measures algal biomass, not toxin.")
            flags = {"unavailable": (chl.get("error") or "")[:160]}
        ev[f"chl:{code}"] = Evidence(
            id=f"chl:{code}", tier=Tier.CONTEXT, kind="satellite_chlorophyll", title=f"Satellite chlorophyll near {name}",
            text=text, as_of=chl.get("latest_center_date"), source_url=chl.get("source_url"), port_codes=(code,), flags=flags)

    # ---- tier 4: data status ---------------------------------------------
    for s in snap.manifest["sources"]:
        flags = {"unavailable": "latest fetch failed"} if s["outcome"] == "failed" else {}
        ev[f"status:{s['source_id']}"] = Evidence(
            id=f"status:{s['source_id']}", tier=Tier.CONTEXT, kind="source_status", title=s["title"],
            text=(f"{s['title']}: last attempt outcome '{s['outcome']}'; last success {s.get('last_success_at')}; "
                  f"latest issued {s.get('latest_issued_date')}; latest valid {s.get('latest_valid_date')}."),
            as_of=(s.get("last_success_at") or "")[:10] or None, flags=flags)

    # ---- tier 4: fisheries exposure (historical, statewide) ---------------
    for g in snap.fisheries["groups"]:
        yrs = [a for a in g["annual"] if a.get("dollars_real")]
        if not yrs:
            continue
        last = yrs[-1]
        ev[f"fisheries:{g['id']}"] = Evidence(
            id=f"fisheries:{g['id']}", tier=Tier.CONTEXT, kind="fisheries_exposure", title=f"{g['label']} landings",
            text=(f"Statewide California commercial landings of {g['label']} in {last['year']}: "
                  f"{fmt(last['pounds'])} lb, ${fmt(last['dollars_real'])} (2024 dollars), NOAA FOSS. "
                  "Historical exposure, not a forecast of losses."),
            as_of=f"{last['year']}-12-31", species=tuple(sorted(canonical_species(g["label"]))))

    # ---- caveats published with the data (citable hedges) ---------------
    for i, c in enumerate(snap.observations.get("caveats", [])):
        ev[f"caveat:observations:{i}"] = Evidence(
            id=f"caveat:observations:{i}", tier=Tier.CONTEXT, kind="caveat", title="CalHABMAP data caveat", text=c)
    for i, c in enumerate(snap.ports[0].get("caveats", []) if snap.ports else []):
        ev[f"caveat:port:{i}"] = Evidence(
            id=f"caveat:port:{i}", tier=Tier.CONTEXT, kind="caveat", title="Port summary caveat", text=c)

    # ---- tier 5: research results (same pipeline only) --------------------
    ev["research:pinn-skill"] = Evidence(
        id="research:pinn-skill", tier=Tier.RESEARCH, kind="research",
        title="Physics-guided chlorophyll forecasting research (offline)",
        text=("Offline research, 2003–2021 MODIS 4 km 8-day composites; not used by CoastWatch. On the same evaluation "
              "pipeline the physics-informed model (PINN) matches but does not beat the ConvLSTM: test RMSE 0.801 vs 0.801 "
              "ln(mg/m³), against 0.889 for persistence (about 9.9% skill). It forecasts chlorophyll (biomass), not toxin. "
              "An internal audit of the evaluation protocol is in progress (research/pinn_audit)."),
        source_url="https://doi.org/10.33422/ccgconf.v2i2.1619")
    return ev


def find_caveat(corpus: dict[str, Evidence], pattern: str) -> str | None:
    """Id of the first published caveat whose text matches ``pattern``."""
    rx = re.compile(pattern, re.I)
    return next((e.id for e in corpus.values() if e.kind == "caveat" and rx.search(e.text)), None)


def is_fresh(e: Evidence, today: date, max_days: int) -> bool:
    if not e.as_of:
        return True
    try:
        return (today - date.fromisoformat(e.as_of[:10])).days <= max_days
    except ValueError:
        return True
