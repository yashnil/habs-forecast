"""Deterministic answer verifier.

The guard does not trust the model. It checks a drafted answer against the
evidence the tools actually returned in this conversation and the CoastWatch
safety rules (CLAUDE.md "Science", GOALS.md §2):

  uncited_claim         a factual sentence has no citation
  unknown_citation      a citation names evidence no tool returned
  ungrounded_number     a number or date is not in the cited evidence
  safety_claim          "safe to eat", "no risk", "all clear" ... (unless negated)
  chlorophyll_as_toxin  chlorophyll presented as evidence of toxin
  observation_as_forecast  a measurement-only sentence makes a future claim
  research_overclaim    the PINN presented as beating the ConvLSTM, or research used as a forecast
  hierarchy             a restrictive official notice was retrieved but the answer does not lead with official information
  disputed_as_settled   a disputed notice cited without saying it is disputed, or anything called "lifted"
  unverified_omitted    official records cited without saying they are not human-verified
  stale_undated         an old measurement cited without its date or age
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import ISO_DATE, Evidence, Tier

CITE = re.compile(r"\[([a-z_]+:[^\]\s]+)\]")
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
NEG = re.compile(r"\b(not|n't|never|no|cannot|doesn't|does not|isn't|aren't|without)\b[^.]{0,40}$")
SAFETY = re.compile(r"\b(safe to (?:eat|harvest|consume|fish|collect|dig)|(?:is|are|it's|looks|seems) (?:perfectly |completely )?safe|"
                    r"no (?:health )?risk|risk[- ]free|toxin[- ]free|all[- ]clear|nothing to worry)\b", re.I)
CHL_TOXIN = re.compile(r"chlorophyll[^.]{0,80}\b(indicat\w*|mean\w*|show\w*|signal\w*|sign of|evidence of|confirm\w*|so)\b"
                       r"[^.]{0,40}\b(toxi\w*|domoic|poison\w*)", re.I)
FUTURE = re.compile(r"\b(will|expected to|forecast\w*|predict\w*|tomorrow|next (?:week|few days)|coming days|likely to)\b", re.I)
OVERCLAIM = re.compile(r"\b(outperform\w*|beat\w*|better than|improv\w* (?:on|over)|more accurate than|superior)\b", re.I)
DISPUTE_WORDS = re.compile(r"\b(disputed|appears? to (?:have been )?lift|may have been lifted|not (?:yet )?(?:been )?(?:confirmed|reviewed)|unconfirmed)\b", re.I)
LIFTED = re.compile(r"\b(lifted|has ended|is over|no longer in effect|rescinded)\b", re.I)
UNVERIFIED_WORDS = re.compile(r"\b(not (?:yet )?(?:been )?(?:human[- ])?verified|unverified|pending (?:human )?review|not yet (?:been )?reviewed|"
                              r"not (?:yet )?checked by a person|has not yet verified|have not yet been verified)\b", re.I)
FACTUAL = re.compile(r"\d|\b(advisory|advisories|closure|closed|quarantine|domoic|pseudo-nitzschia|chlorophyll|probabilit\w*|"
                     r"measured|sample|landings|rmse)\b", re.I)
ABSTAIN = re.compile(r"\b(couldn't|could not|can't|cannot|don't have|do not have|no (?:evidence|data|information|record)|unavailable|"
                     r"not available|only answer)\b", re.I)


@dataclass
class Violation:
    code: str
    sentence: str
    detail: str = ""


@dataclass
class GuardReport:
    violations: list[Violation] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def codes(self) -> set[str]:
        return {v.code for v in self.violations}


def split_sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line:
            continue
        # split on sentence ends that are not inside citations or numbers
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])|(?<=\])\s+(?=[A-Z\"])", line)
        out += [p.strip() for p in parts if p.strip()]
    return out


PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
ANCHORED_DATE = re.compile(r"\b(issued|sampled|measured|effective|collected)\D{0,15}?(\d{4}-\d{2}-\d{2})", re.I)


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def _number_ok(tok: str, facts: set[float], percent: bool = False) -> bool:
    """``tok`` equals an evidence value, or that value rounded to the precision ``tok`` uses.

    A percentage must be an evidence probability (0-1) times 100; a skill
    percentage that is literally in the evidence ("9.9%") also matches.
    """
    x, nd = float(tok.replace(",", "")), _decimals(tok)
    for f in facts:
        if f == x or round(f, nd) == x:
            return True
        if percent and 0 <= f <= 1 and round(f * 100, nd) == x:
            return True
    return False


def check(answer: str, retrieved: dict[str, Evidence], question: str = "") -> GuardReport:
    rep = GuardReport()
    sentences = split_sentences(answer)
    first_cited_tier = None
    any_unverified_cited = False

    for s in sentences:
        cites = CITE.findall(s)
        rep.cited += cites
        body = CITE.sub("", s)
        known = [retrieved[c] for c in cites if c in retrieved]
        for c in cites:
            if c not in retrieved:
                rep.violations.append(Violation("unknown_citation", s, c))
        if known and first_cited_tier is None:
            first_cited_tier = min(e.tier for e in known)

        if not cites and FACTUAL.search(body) and not (ABSTAIN.search(body) and not re.search(r"\d", body)):
            rep.violations.append(Violation("uncited_claim", s))

        # numbers and dates must come from the cited evidence
        if known:
            facts = set().union(*(e.numeric_facts() for e in known))
            texts = " ".join(f"{e.text} {e.as_of or ''}" for e in known)
            for d in DATE.findall(body):
                if d not in texts:
                    rep.violations.append(Violation("ungrounded_number", s, d))
            dated = {e.as_of[:10] for e in known if e.as_of}
            for verb, d in ANCHORED_DATE.findall(body):
                stated = re.search(rf"{verb[:5]}\w*\D{{0,15}}?{d}", texts, re.I)
                if dated and d not in dated and not stated:
                    rep.violations.append(Violation("ungrounded_number", s, f"{verb} {d} is not the evidence date"))
            plain = ISO_DATE.sub(" ", body)
            for tok in PERCENT.findall(plain):
                if not _number_ok(tok, facts, percent=True):
                    rep.violations.append(Violation("ungrounded_number", s, tok + "%"))
            for tok in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", PERCENT.sub(" ", plain)):
                if not _number_ok(tok, facts):
                    rep.violations.append(Violation("ungrounded_number", s, tok))

        for m in SAFETY.finditer(body):
            if not NEG.search(body[: m.start()]):
                rep.violations.append(Violation("safety_claim", s, m.group(0)))
        m = CHL_TOXIN.search(body)
        if m and not re.search(r"\bnot\b|\bisn't\b|\bdoes not\b|\bdoesn't\b", m.group(0), re.I):
            rep.violations.append(Violation("chlorophyll_as_toxin", s))
        if known and all(e.tier == Tier.MEASUREMENT for e in known) and FUTURE.search(body) \
                and not re.search(r"\bnot (?:a )?forecast", body, re.I):
            rep.violations.append(Violation("observation_as_forecast", s))
        if any(e.tier == Tier.RESEARCH for e in known):
            if OVERCLAIM.search(body) and not re.search(r"\b(not|does not|doesn't|no)\b", body, re.I):
                rep.violations.append(Violation("research_overclaim", s))
            if FUTURE.search(body) and re.search(r"\b(bloom|toxin|chlorophyll) (?:will|is expected)", body, re.I):
                rep.violations.append(Violation("research_overclaim", s, "research used as a forecast"))

        for e in known:
            if e.flags.get("disputed") and not DISPUTE_WORDS.search(body):
                rep.violations.append(Violation("disputed_as_settled", s, e.id))
            if e.flags.get("stale") and e.tier == Tier.MEASUREMENT and e.as_of and e.as_of not in body \
                    and not re.search(r"\bdays? (?:old|before|ago)|\bmonths? (?:old|ago)|\byears? (?:old|ago)", body):
                rep.violations.append(Violation("stale_undated", s, e.id))
            if e.tier == Tier.OFFICIAL and e.flags.get("unverified"):
                any_unverified_cited = True
        if LIFTED.search(body) and not DISPUTE_WORDS.search(body) and not NEG.search(body[: LIFTED.search(body).start()]):
            rep.violations.append(Violation("disputed_as_settled", s, "describes a notice as lifted"))

    restrictive = [e for e in retrieved.values() if e.tier == Tier.OFFICIAL and e.flags.get("restrictive")]
    if restrictive:
        cited_restrictive = [c for c in rep.cited if c in retrieved and retrieved[c].flags.get("restrictive")]
        if not cited_restrictive:
            rep.violations.append(Violation("hierarchy", "", "restrictive official notice retrieved but not cited"))
        elif first_cited_tier is not None and first_cited_tier != Tier.OFFICIAL:
            rep.violations.append(Violation("hierarchy", "", "answer does not lead with official information"))
    if any_unverified_cited and not UNVERIFIED_WORDS.search(answer):
        rep.violations.append(Violation("unverified_omitted", "", "official records are pending human review"))
    return rep
