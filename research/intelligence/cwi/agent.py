"""Tool-calling loop with a verify-then-answer gate.

provider -> tool calls -> toolbox -> provider -> draft -> guard
  pass  -> answer + fixed footer
  fail  -> one revision request listing the violations -> guard again
  fail  -> deterministic fallback built only from official evidence
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .evidence import Evidence, Tier
from .guard import GuardReport, check
from .tools import Toolbox

SYSTEM_PROMPT = """You answer questions about harmful algal blooms on the California coast for fishermen and shellfish harvesters, using only CoastWatch's recorded data, which you reach through tools.

Rules:
1. Lead with official CDFW/CDPH notices when any may apply. Official notices outrank measurements, measurements outrank model output, and model output outranks context and research.
2. Every factual sentence ends with the evidence ids it relies on, in square brackets, e.g. [official:cdph-2026-annual-mussel-quarantine]. Cite only ids that a tool returned. Copy numbers and dates exactly from the evidence.
3. Never say or imply that seafood or an area is safe. Only CDPH and CDFW decide that.
4. Shore-station measurements are past observations of seawater, not forecasts and not toxin in seafood. Give their sample date or age.
5. Chlorophyll is algal biomass. Never present it as toxin or as evidence of toxin.
6. If a notice is flagged disputed, say it is disputed and not yet confirmed; never call it lifted. Say that notice records are not yet verified by a person when they are flagged unverified.
7. Research results are offline studies, not forecasts. The physics-informed model matches, and does not beat, the ConvLSTM.
8. If the tools return nothing relevant, say you don't have that information.
"""

FOOTER = ("CoastWatch is not an official source. Only CDPH and CDFW decide what may be harvested or eaten; "
          "check their pages or call the CDPH biotoxin line before you go.")


@dataclass
class Turn:
    question: str
    answer: str
    outcome: str                          # "passed", "revised", "fallback", "abstained"
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: dict[str, Evidence] = field(default_factory=dict)
    first_report: GuardReport | None = None
    final_report: GuardReport | None = None
    transcript: list[dict] = field(default_factory=list)


def fallback_answer(retrieved: dict[str, Evidence]) -> str:
    official = sorted((e for e in retrieved.values() if e.tier == Tier.OFFICIAL and e.flags.get("restrictive")),
                      key=lambda e: e.id)
    lines = ["I couldn't produce an answer I could check against CoastWatch's evidence."]
    if official:
        lines.append("Official notices that may apply (records not yet verified by a person):")
        for e in official:
            tag = "DISPUTED, not yet confirmed: " if e.flags.get("disputed") else ""
            lines.append(f"- {tag}{e.title} [{e.id}]")
    return "\n".join(lines)


def run(question: str, provider, toolbox: Toolbox, max_steps: int = 6, revise: bool = True) -> Turn:
    tools = toolbox.specs()
    messages: list[dict] = [{"role": "user", "content": question}]
    retrieved: dict[str, Evidence] = {}
    calls: list[dict] = []
    draft = ""
    for _ in range(max_steps):
        resp = provider.complete(SYSTEM_PROMPT, messages, tools)
        messages.append({"role": "assistant", "content": resp["content"]})
        uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
        if not uses:
            draft = "\n".join(b["text"] for b in resp["content"] if b.get("type") == "text").strip()
            break
        results = []
        for u in uses:
            out = toolbox.call(u["name"], u["input"])
            calls.append({"name": u["name"], "input": u["input"], "error": out.get("error")})
            for c in out.get("evidence", []):
                retrieved[c["evidence_id"]] = toolbox.corpus[c["evidence_id"]]
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": json.dumps(out),
                            **({"is_error": True} if out.get("error") else {})})
        messages.append({"role": "user", "content": results})   # all results in one message

    first = check(draft, retrieved, question)
    final, outcome, answer = first, "passed", draft
    if not retrieved and first.passed:
        outcome = "abstained"
    elif not first.passed and revise:
        feedback = "GUARD: revise the answer. Violations:\n" + "\n".join(
            f"- {v.code}: {v.detail} :: {v.sentence[:160]}" for v in first.violations)
        messages.append({"role": "user", "content": feedback})
        resp = provider.complete(SYSTEM_PROMPT, messages, tools)
        messages.append({"role": "assistant", "content": resp["content"]})
        answer = "\n".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()
        final = check(answer, retrieved, question)
        outcome = "revised"
    if not final.passed or not answer.strip():
        answer, outcome = fallback_answer(retrieved), "fallback"
        final = check(answer, retrieved, question)
    return Turn(question=question, answer=f"{answer}\n\n{FOOTER}", outcome=outcome, tool_calls=calls,
                retrieved=retrieved, first_report=first, final_report=final, transcript=messages)
