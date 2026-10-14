"""Read-only tools over a recorded snapshot.

Each tool returns ``{"evidence": [card, ...], ...}``. Cards carry evidence ids;
an answer may cite only ids returned by a tool in the same conversation.
Schemas are JSON Schema in the shape the Claude Messages API expects for
user-defined tools (``name``/``description``/``input_schema``, ``strict``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from .evidence import BIVALVES, Evidence, Tier, build_corpus, canonical_species, find_caveat
from .retrieval import BM25
from .snapshot import Snapshot

MEASUREMENT_RADIUS_KM = 25.0
EXTRA_ALIASES = {
    452: ["half moon bay", "princeton", "pillar point"],
    602: ["avila", "port san luis"],
    592: ["moss landing"],
    440: ["sf", "san francisco"],
    770: ["san pedro", "los angeles", "la harbor", "long beach"],
    611: ["santa barbara"],
    883: ["oceanside"],
}


def _km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., dict]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema, "strict": True}


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


class Toolbox:
    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.corpus: dict[str, Evidence] = build_corpus(snap)
        self.index = BM25(list(self.corpus.values()))
        self.ports = {p["port_code"]: p for p in snap.ports}
        self.caveats = {
            "seawater_not_seafood": find_caveat(self.corpus, r"seawater is not domoic acid in seafood"),
            "zero": find_caveat(self.corpus, r"reported 0"),
            "chlorophyll": find_caveat(self.corpus, r"^Chlorophyll measures algae biomass\. High"),
            "charm": find_caveat(self.corpus, r"C-HARM probabilities"),
            "port_radius": find_caveat(self.corpus, r"within 15 km of the CDFW port"),
        }
        self.tools: dict[str, Tool] = {t.name: t for t in self._define()}

    # ------------------------------------------------------------------ #
    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> dict:
        if name not in self.tools:
            return {"error": f"unknown tool {name!r}", "evidence": []}
        try:
            return self.tools[name].fn(**args)
        except TypeError as exc:  # bad arguments from the model
            return {"error": f"invalid arguments: {exc}", "evidence": []}

    # ------------------------------------------------------------------ #
    def resolve_port(self, query: str) -> list[dict]:
        q = query.lower()
        hits = []
        for code, p in self.ports.items():
            names = [p["display_name"].lower(), *EXTRA_ALIASES.get(code, [])]
            names += [n.strip() for n in p["display_name"].lower().split("/")]
            if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names if n):
                hits.append(p)
        if not hits:  # fall back to county
            hits = [p for p in self.ports.values() if re.search(rf"\b{re.escape(p['county'].lower())}\b", q)]
        # prefer the longest matching name ("Monterey" vs "Monterey Bay" ambiguity)
        return hits

    def _cards(self, ids) -> list[dict]:
        return [self.corpus[i].card() for i in ids if i in self.corpus]

    def _define(self) -> list[Tool]:
        snap = self.snap

        def find_port(query: str) -> dict:
            hits = self.resolve_port(query)
            return {"ports": [{"port_code": p["port_code"], "name": p["display_name"], "county": p["county"],
                               "region": p["region"]} for p in hits], "evidence": []}

        def get_official_notices(port_code: int, species: str = "") -> dict:
            if port_code not in self.ports:
                return {"error": f"unknown port_code {port_code}", "evidence": []}
            want = canonical_species(species) if species else set()
            if want & BIVALVES:
                want |= {"bivalves"}
            ids = []
            for e in self.corpus.values():
                if e.tier != Tier.OFFICIAL or port_code not in e.port_codes:
                    continue
                if want and e.species and not (set(e.species) & want):
                    continue
                if e.kind == "official_statement" and not want:
                    continue  # species-specific statements only when a species is asked about
                ids.append(e.id)
            ids.sort(key=lambda i: (not self.corpus[i].flags.get("restrictive"), i))
            rv = snap.review
            return {"port": self.ports[port_code]["display_name"], "evidence": self._cards(ids),
                    "registry_review": {"status": rv.get("status"), "reviewed_at": rv.get("reviewed_at")},
                    "note": ("Only CDFW/CDPH decide what may be harvested or eaten. Registry is "
                             f"{rv.get('status')}; records are NOT human-verified.")}

        def get_measurements(port_code: int, variable: str = "") -> dict:
            if port_code not in self.ports:
                return {"error": f"unknown port_code {port_code}", "evidence": []}
            p = self.ports[port_code]
            ids = []
            for st in snap.stations:
                d = _km(p["lat"], p["lon"], st["lat"], st["lon"])
                if d > MEASUREMENT_RADIUS_KM:
                    continue
                for sm in st["summaries"]:
                    if sm.get("last_date") and (not variable or sm["variable"] == variable):
                        ids.append(f"obs:{st['station_id']}:{sm['variable']}")
            ids += [self.caveats["seawater_not_seafood"], self.caveats["zero"]]
            return {"port": p["display_name"], "radius_km": MEASUREMENT_RADIUS_KM, "evidence": self._cards(ids),
                    "note": "Observations of seawater at shore stations. Not forecasts; not toxin in seafood."
                    + ("" if ids else f" No CalHABMAP station within {MEASUREMENT_RADIUS_KM:g} km.")}

        def get_charm(port_code: int) -> dict:
            if port_code not in self.ports:
                return {"error": f"unknown port_code {port_code}", "evidence": []}
            return {"evidence": self._cards([f"charm:{port_code}", "status:charm", self.caveats["charm"],
                                             self.caveats["port_radius"]])}

        def get_satellite_chlorophyll(port_code: int) -> dict:
            if port_code not in self.ports:
                return {"error": f"unknown port_code {port_code}", "evidence": []}
            return {"evidence": self._cards([f"chl:{port_code}", "status:satellite_chl", self.caveats["chlorophyll"]])}

        def get_data_status() -> dict:
            return {"generated_at": snap.manifest["generated_at"],
                    "evidence": self._cards(sorted(i for i in self.corpus if i.startswith("status:")))}

        def get_fisheries_exposure(species: str) -> dict:
            want = canonical_species(species)
            ids = [e.id for e in self.corpus.values() if e.kind == "fisheries_exposure" and set(e.species) & want]
            return {"evidence": self._cards(ids)}

        def get_research_results() -> dict:
            return {"evidence": self._cards(["research:pinn-skill"])}

        def search_evidence(query: str, k: int = 5) -> dict:
            return {"evidence": [e.card() for e, _ in self.index.search(query, k=min(int(k), 10))]}

        P = {"port_code": {"type": "integer", "description": "CDFW port code from find_port"}}
        return [
            Tool("find_port", "Resolve a place name, port or county in the user's question to CoastWatch port codes.",
                 _schema({"query": {"type": "string"}}, ["query"]), find_port),
            Tool("get_official_notices",
                 "Official CDFW/CDPH closures, advisories and statements that may apply near a port, optionally for one "
                 "species. Always call this before discussing harvest, eating or fishing.",
                 _schema({**P, "species": {"type": "string", "description": "species named by the user, or empty"}},
                         ["port_code", "species"]), get_official_notices),
            Tool("get_measurements",
                 "Latest CalHABMAP shore-station seawater measurements (domoic acid, Pseudo-nitzschia, chlorophyll, "
                 "temperature) within 25 km of a port, with sample dates.",
                 _schema({**P, "variable": {"type": "string", "description": "pDA, tDA, dDA, pn_seriata, "
                         "pn_delicatissima, chl_extracted, temp, or empty for all"}}, ["port_code", "variable"]),
                 get_measurements),
            Tool("get_charm", "C-HARM v3.1 model probabilities (Pseudo-nitzschia bloom, particulate and cellular domoic "
                 "acid) near a port for leads 0–3 days, plus source status.", _schema(P, ["port_code"]), get_charm),
            Tool("get_satellite_chlorophyll", "Latest satellite chlorophyll (algal biomass) near a port.",
                 _schema(P, ["port_code"]), get_satellite_chlorophyll),
            Tool("get_data_status", "When each CoastWatch source last updated and whether the last fetch failed.",
                 _schema({}, []), get_data_status),
            Tool("get_fisheries_exposure", "Historical statewide commercial landings for a species group.",
                 _schema({"species": {"type": "string"}}, ["species"]), get_fisheries_exposure),
            Tool("get_research_results", "Offline research results on chlorophyll forecasting (not operational).",
                 _schema({}, []), get_research_results),
            Tool("search_evidence", "Keyword search over all evidence when no structured tool fits.",
                 _schema({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query", "k"]), search_evidence),
        ]
