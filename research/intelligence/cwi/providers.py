"""LLM providers behind one interface.

``complete(system, messages, tools) -> {"content": [...blocks], "stop_reason": ...}``
uses the Claude Messages API block shapes (``text``, ``tool_use``,
``tool_result``), so a real provider is a thin adapter.

* ``MockProvider``      deterministic planner + template writer. Stands in for an
                        LLM so the pipeline, tools, guard and evals run offline.
* ``ScriptedProvider``  replays fixed responses (adversarial guard tests, or
                        recorded transcripts from a real model).
* ``AnthropicDryRun``   builds the exact request body a Claude call would send
                        and refuses to send it. No paid API is ever invoked here.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Protocol

from .evidence import canonical_species, fmt
from .tools import Toolbox

MODEL_ID = "claude-opus-5-5"


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict: ...


# --------------------------------------------------------------------------- #
# Intent detection shared by the mock planner and the eval harness
# --------------------------------------------------------------------------- #
INTENTS = {
    "harvest": r"\b(eat|eating|harvest\w*|safe|collect|gather|pick|dig|digging|catch|fish(?:ing)?|take|keep|cook|buy)\b",
    "forecast": r"\b(forecast\w*|tomorrow|next (?:week|few days)|this week|coming|outlook|c-?harm|probabilit\w*|predict\w*)\b",
    "measurement": r"\b(measur\w*|sample\w*|domoic|toxin\w*|pseudo-?nitzschia|cells?|tested|levels?|pier data)\b",
    "chlorophyll": r"\b(chlorophyll|satellite|green water|algae)\b",
    "research": r"\b(pinn|physics|research|rmse|convlstm|your model|model accuracy|skill)\b",
    "status": r"\b(fresh|updated?|up to date|stale|status|outage|down|data (?:is|are) (?:old|current)|last update\w*|current)\b",
    "fisheries": r"\b(worth|value|landings|economic\w*|revenue|dollars|money)\b",
}


def detect_intents(q: str) -> set[str]:
    ql = q.lower()
    return {k for k, rx in INTENTS.items() if re.search(rx, ql)}


def cite_each(text: str, eid: str) -> str:
    """Append ``[eid]`` to every sentence so no sentence of a quote is left uncited."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.strip())
    return " ".join(f"{p.rstrip()} [{eid}]" for p in parts if p.strip())


class MockProvider:
    """Deterministic stand-in for an LLM. Turn 1: plan tool calls. Turn 2: write."""

    name = "mock"

    def __init__(self, toolbox: Toolbox):
        self.tb = toolbox
        self._ids = itertools.count(1)

    # ---- helpers ------------------------------------------------------------
    def _question(self, messages):
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], str):
                return m["content"]
        return ""

    def _results(self, messages) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        out.append(json.loads(b["content"]))
        return out

    def _call(self, name, **inp):
        return {"type": "tool_use", "id": f"toolu_mock_{next(self._ids)}", "name": name, "input": inp}

    # ---- planner ------------------------------------------------------------
    def complete(self, system, messages, tools):
        q = self._question(messages)
        if not any(m["role"] == "assistant" for m in messages):
            return self._plan(q)
        revise = [m for m in messages if m["role"] == "user" and isinstance(m["content"], str)
                  and m["content"].startswith("GUARD:")]
        text = self._write(q, self._results(messages))
        if revise:
            flagged = [ln.split(" :: ", 1)[1] for ln in revise[-1]["content"].splitlines() if " :: " in ln]
            text = "\n".join(ln for ln in text.splitlines()
                             if not any(f and f[:60] in ln for f in flagged))
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

    def _plan(self, q):
        intents = detect_intents(q)
        ports = self.tb.resolve_port(q)
        species = canonical_species(q)
        calls = []
        if intents & {"research"} and not ports:
            calls.append(self._call("get_research_results"))
        if "status" in intents:
            calls.append(self._call("get_data_status"))
        if "fisheries" in intents and species:
            calls.append(self._call("get_fisheries_exposure", species=" ".join(sorted(species))))
        if ports:
            code = ports[0]["port_code"]
            calls.append(self._call("find_port", query=q))
            calls.append(self._call("get_official_notices", port_code=code, species=" ".join(sorted(species))))
            if intents & {"measurement", "harvest"} or not intents - {"status"}:
                calls.append(self._call("get_measurements", port_code=code, variable=""))
            if intents & {"forecast", "harvest"} or not intents - {"status"}:
                calls.append(self._call("get_charm", port_code=code))
            if "chlorophyll" in intents:
                calls.append(self._call("get_satellite_chlorophyll", port_code=code))
        if not calls:
            return {"content": [{"type": "text", "text": self._abstain(q, intents, species)}], "stop_reason": "end_turn"}
        return {"content": calls, "stop_reason": "tool_use"}

    def _abstain(self, q, intents, species):
        if intents & {"harvest", "measurement", "forecast", "chlorophyll"} or species:
            return ("I couldn't tell which port or stretch of the California coast you mean. Name a port "
                    "(for example Bodega Bay, Santa Cruz or Morro Bay) and I'll look up the official notices, "
                    "recent measurements and C-HARM values for it.")
        return ("I can only answer from CoastWatch's recorded official notices, shore-station measurements, "
                "C-HARM values and data status, and I don't have information on that.")

    # ---- writer -------------------------------------------------------------
    def _write(self, q, results):
        cards = {}
        notes = []
        for r in results:
            for c in r.get("evidence", []):
                cards[c["evidence_id"]] = c
            if r.get("note"):
                notes.append(r["note"])
        if not cards:
            return self._abstain(q, detect_intents(q), canonical_species(q))
        ev = {i: self.tb.corpus[i] for i in cards}
        port = next((r["port"] for r in results if r.get("port")), None)
        lines: list[str] = []

        official = [e for e in ev.values() if e.tier == 1]
        restrictive = [e for e in official if e.flags.get("restrictive")]
        statements = [e for e in official if e.kind == "official_statement"]
        if port is not None:
            if restrictive:
                lines.append(f"Official notices that may apply near {port}:")
                for e in restrictive:
                    if e.kind == "official_statement":
                        quote = e.text.split(': "', 1)[-1].rstrip('"')
                        lines.append("- CDFW states: " + cite_each(quote, e.id))
                    elif e.flags.get("disputed"):
                        lines.append(f"- DISPUTED: {e.title}. An agency release appears to lift it, but a person has not "
                                     f"yet confirmed this; treat it as possibly still in effect and check the agency page [{e.id}]")
                    else:
                        lines.append(f"- {e.title} (effective {e.as_of or 'date not stated'}) [{e.id}]")
            elif not statements:
                lines.append(f"CoastWatch has no record of an official closure or advisory for this species near {port}; "
                             "that is not the same as an all-clear, so check CDPH and CDFW directly.")
            for e in statements:
                if e in restrictive:
                    continue
                quote = e.text.split(': "', 1)[-1].rstrip('"')
                lines.append("- CDFW states: " + cite_each(quote, e.id))
            if official:
                lines.append("These notice records are transcriptions that have not yet been verified by a person; "
                             "the agency pages decide.")

        meas = sorted((e for e in ev.values() if e.tier == 2), key=lambda e: (e.as_of or ""), reverse=True)
        keep_vars = {"pDA", "tDA", "pn_seriata", "pn_delicatissima"}
        meas = [e for e in meas if e.id.rsplit(":", 1)[-1] in keep_vars][:4]
        if meas:
            lines.append("Most recent shore-station measurements (seawater, past samples):")
            for e in meas:
                stn, var = e.title, e.id.rsplit(":", 1)[-1]
                val, age = e.numbers[0], e.numbers[1]
                v = fmt(float(val))
                units = "ng/mL" if var.endswith("DA") else "cells/L"
                extra = " (reported 0, which means not quantified, not absent)" if float(val) == 0 else ""
                lines.append(f"- {stn}: {v} {units}{extra}, sampled {e.as_of}, {age} days before this snapshot [{e.id}]")
            lines.append("These are past observations of seawater, not a forecast. "
                         f"Domoic acid in seawater is not domoic acid in seafood [{self.tb.caveats['seawater_not_seafood']}].")
        elif port is not None and any(r.get("radius_km") for r in results):
            lines.append(f"CoastWatch has no CalHABMAP shore-station measurement on file near {port}.")

        for e in ev.values():
            if e.kind != "charm_forecast":
                continue
            if e.flags.get("unavailable"):
                lines.append(f"C-HARM has no model value near {port}, so there is no model estimate for this port [{e.id}].")
                continue
            code = int(e.id.split(":")[1])
            lead0 = self.tb.ports[code]["charm"]["leads"][0]
            pn = lead0["variables"].get("pseudo_nitzschia", {}).get("median")
            pda = lead0["variables"].get("particulate_domoic", {}).get("median")
            lines.append(f"C-HARM (run issued {e.as_of}, valid {lead0['valid_date']}) gives a median Pseudo-nitzschia bloom "
                         f"probability of {fmt(pn)} and a median particulate domoic acid probability of {fmt(pda)} near {port} [{e.id}].")
            lines.append("C-HARM probabilities describe the water column, not toxin in seafood; low values do not mean "
                         f"an area is safe [{self.tb.caveats['charm']}].")
            if e.flags.get("stale") and "status:charm" in ev:
                lines.append(f"The latest C-HARM fetch failed, so these values are from the run issued {e.as_of} [status:charm].")

        for e in ev.values():
            if e.kind == "satellite_chlorophyll":
                if e.flags.get("unavailable"):
                    lines.append(f"Satellite chlorophyll near {port} is unavailable in this snapshot because the fetch failed [{e.id}].")
                else:
                    lines.append(cite_each(e.text, e.id))
                lines.append(cite_each("Chlorophyll measures algae biomass. High chlorophyll is not a toxic bloom, and low "
                                       "chlorophyll does not rule one out.", self.tb.caveats["chlorophyll"]))

        status = [e for e in ev.values() if e.kind == "source_status"]
        if status and "status" in detect_intents(q):
            failed = [e for e in status if e.flags.get("unavailable")]
            for e in failed:
                lines.append(cite_each(e.text, e.id))
            if not failed:
                lines.append("All sources updated on their last attempt [" + status[0].id + "].")

        for e in ev.values():
            if e.kind in ("fisheries_exposure", "research"):
                lines.append(cite_each(e.text, e.id))
        return "\n".join(lines)


class ScriptedProvider:
    """Replays a fixed list of responses (one per ``complete`` call)."""

    name = "scripted"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    def complete(self, system, messages, tools):
        if not self.responses:
            return {"content": [{"type": "text", "text": ""}], "stop_reason": "end_turn"}
        return self.responses.pop(0)


class PaidCallBlocked(RuntimeError):
    pass


class AnthropicDryRun:
    """Builds the Claude Messages API request body; never sends it.

    The body is what ``anthropic.Anthropic().messages.create(**body)`` would take.
    Sending requires explicit approval and is deliberately not implemented here.
    """

    name = "anthropic-dry-run"

    def __init__(self, model: str = MODEL_ID, effort: str = "medium"):
        self.model, self.effort, self.requests = model, effort, []

    def build(self, system, messages, tools) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "tool_choice": {"type": "auto"},          # forced tool_choice is rejected on this model
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "messages": messages,
        }

    def complete(self, system, messages, tools):
        self.requests.append(self.build(system, messages, tools))
        raise PaidCallBlocked("AnthropicDryRun never sends requests; paid API calls need explicit approval.")
