"""CDPH domoic-acid test results -> `v1/biotoxin/index.json` (BiotoxinIndex, pilot sidecar).

Sources (both rolling PDFs on CDPH's Domoic Acid page, read as untrusted data):
- crab: `CrabDAResults.pdf` (fixed name). One row per sampling set: BMA/port, collection site,
  routine (R) or retest (RT), collection date, crab type, six individual results (ppm), the average
  ("information only") and the percent of samples over the action level. A paired value
  "37 | 6.6" is "elevated viscera | meat".
- seafood: `SeafoodDAResults<period>.pdf` (the name changes with the period; found on the page).
  One row per composite sample: date of catch, species, tissue type, sample size, CDFW block, ppm.

Rules: values are transcribed, never invented. "ND" and "<2.5" become null and are counted in
nd_count, the printed text stays in values_raw. A row the parser cannot read completely goes to
parse_warnings and is left out. Site locations and the pilot places a site informs come from a small
hand-checked gazetteer (SITES); a site not in it gets no location and informs no place.

These are measured test results (official, published by CDPH), not a forecast, and say nothing about
other species or places. CDPH's TLS chain is incomplete; http.tls_context() adds the missing public
intermediate (certs/), verification is never disabled.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

from ..context import RunContext
from ..pilot_models import ActionLevel, BiotoxinIndex, BiotoxinSet, BiotoxinSite, BiotoxinSource

SOURCE_ID = "cdph_biotoxin"
PAGE_URL = "https://www.cdph.ca.gov/Programs/CEH/DFDCS/Pages/FDBPrograms/FoodSafetyProgram/DomoicAcid.aspx"
DOC_BASE = "https://www.cdph.ca.gov/Programs/CEH/DFDCS/CDPH%20Document%20Library/FDB/FoodSafetyProgram/DomoicAcid/"
CRAB_URL = DOC_BASE + "CrabDAResults.pdf"
OUT_REL = "biotoxin/index.json"

PILOT_MONTEREY_BAY = ["port-592", "port-550", "port-593"]  # Moss Landing, Monterey, Santa Cruz
PILOT_HALF_MOON_BAY = ["port-452"]  # Princeton / Half Moon Bay

# CDPH collection sites -> approximate location of the named place and the pilot places it informs.
# Conservative: only sites inside the bay or stretch of coast a pilot port fishes. BMA G is Monterey
# Bay; BMA F is San Francisco/San Mateo, and of its sites only Pillar Point and Pigeon Point (San
# Mateo coast) inform Princeton/Half Moon Bay; Duxbury Reef (Marin, north of the Golden Gate) informs
# no pilot place. Coordinates are the named feature's, to two decimals; they are not sampling stations.
SITES: dict[str, tuple[float, float, list[str]]] = {
    "monterey bay": (36.80, -121.90, PILOT_MONTEREY_BAY),
    "pillar point": (37.49, -122.50, PILOT_HALF_MOON_BAY),
    "pigeon point": (37.18, -122.39, PILOT_HALF_MOON_BAY),
    "duxbury reef": (37.89, -122.70, []),
    "klamath river": (41.55, -124.09, []),
    "george reef": (41.84, -124.37, []),
    "usal": (39.83, -123.85, []),
}

SPECIES = {"dungeness crab": "Dungeness crab", "rock crab": "Rock crab", "n. anchovies": "Northern anchovy", "n. anchovy": "Northern anchovy", "northern anchovies": "Northern anchovy", "pacific sardines": "Pacific sardine", "sardines": "Pacific sardine"}
TISSUES = {"viscera": "viscera", "whole body": "whole", "meat": "meat"}

_NUM = r"\d+(?:\.\d+)?"
_ONE = rf"(?:ND|<\s?{_NUM}|{_NUM})"
VALUE = re.compile(rf"^{_ONE}(?:\s?\|\s?{_ONE})?$")
DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


class ParseError(ValueError):
    pass


def slug(*parts: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", " ".join(p for p in parts if p).lower()).strip("-")


def us_date(s: str) -> str:
    m = DATE.match(s.strip())
    if not m:
        raise ParseError(f"not a date: {s!r}")
    return date(int(m.group(3)), int(m.group(1)), int(m.group(2))).isoformat()


def pdf_text(blob: bytes) -> str:
    """`pdftotext -layout` (poppler-utils). Raises if the tool is missing or the PDF is unreadable."""
    exe = shutil.which("pdftotext") or next((p for p in ("/opt/homebrew/bin/pdftotext", "/usr/bin/pdftotext") if Path(p).exists()), None)
    if not exe:
        raise RuntimeError("pdftotext not found (install poppler-utils)")
    if not blob.startswith(b"%PDF"):
        raise ParseError("not a PDF (no %PDF signature)")
    r = subprocess.run([exe, "-layout", "-enc", "UTF-8", "-", "-"], input=blob, capture_output=True, timeout=60)
    if r.returncode != 0:
        raise ParseError(f"pdftotext failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout.decode("utf-8", "replace")


def value(tok: str) -> tuple[float | None, float | None, bool]:
    """'4.0' -> (4.0, None, False); '37 | 6.6' -> (37, 6.6, False); 'ND'/'<2.5' -> (None, None, True)."""
    parts = [p.strip() for p in tok.split("|")]

    def one(p: str) -> tuple[float | None, bool]:
        if p.upper() == "ND" or p.startswith("<"):
            return None, True
        return float(p), False

    v, nd = one(parts[0])
    meat = one(parts[1])[0] if len(parts) > 1 else None
    return v, meat, nd


def as_of(text: str) -> str | None:
    m = re.search(r"As of\s+(\d{1,2}/\d{1,2}/\d{4})", text)
    return us_date(m.group(1)) if m else None


def period(text: str) -> str | None:
    m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4}\s*[–-]\s*[A-Z][a-z]+ \d{1,2}, \d{4})", text)
    return re.sub(r"\s+", " ", m.group(1)) if m else None


# ---------------------------------------------------------------- crab
@dataclass
class CrabRow:
    bma: str
    bma_label: str
    site: str
    kind: str
    date: str
    species: str
    raw: list[str]
    avg: float | None
    pct: float | None


@dataclass
class Parsed:
    rows: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    as_of: str | None = None
    period: str | None = None
    action_quoted: str | None = None
    action_ppm: float | None = None


def _species_around(lines: list[str], i: int) -> str | None:
    """A crab type printed on two lines around the data line ('Dungeness' above, 'Crab' below)."""
    above = lines[i - 1].strip() if i > 0 else ""
    below = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if above and below and len(above.split()) == 1 and len(below.split()) == 1 and above.isalpha() and below.isalpha():
        return f"{above} {below}"
    return None


def parse_crab(text: str) -> Parsed:
    out = Parsed(as_of=as_of(text), period=period(text))
    m = re.search(r"\(FDA Action Level\s*>?\s*(\d+(?:\.\d+)?)\s*PPM\)", text, re.I)
    if m:
        out.action_quoted = re.sub(r"[()]", "", m.group(0)).strip()
        out.action_ppm = float(m.group(1))
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not re.match(r"^BMA\s+[A-Z]\b", s):
            continue
        toks = [t.strip() for t in re.split(r"\s{2,}", s) if t.strip()]
        try:
            mb = re.match(r"^BMA\s+([A-Z])\s*-\s*(.+)$", toks[0])
            if not mb or len(toks) < 5:
                raise ParseError("unexpected columns")
            site, kind_tok, date_tok = toks[1], toks[2], toks[3]
            kind = {"R": "routine", "RT": "retest"}.get(kind_tok.upper())
            if kind is None:
                raise ParseError(f"routine/retest column reads {kind_tok!r}")
            d = us_date(date_tok)
            rest = toks[4:]
            if rest and not VALUE.match(rest[0]):
                species, rest = rest[0], rest[1:]
            else:
                species = _species_around(lines, i) or ""
            if not species:
                raise ParseError("crab type unreadable")
            if len(rest) != 8:
                raise ParseError(f"expected 6 results, an average and a percent; read {len(rest)} cells")
            raw, avg_tok, pct_tok = rest[:6], rest[6], rest[7]
            if not all(VALUE.match(t) for t in raw):
                raise ParseError(f"unreadable result among {raw}")
            for t in raw:
                value(t)
            avg = float(avg_tok) if re.fullmatch(_NUM, avg_tok) else None
            if avg is None and avg_tok.upper() not in ("ND",):
                raise ParseError(f"average reads {avg_tok!r}")
            mp = re.fullmatch(r"(\d+(?:\.\d+)?)\s?%", pct_tok)
            if not mp:
                raise ParseError(f"percent over action level reads {pct_tok!r}")
            out.rows.append(CrabRow(mb.group(1), mb.group(2).strip(), site, kind, d, species, raw, avg, float(mp.group(1))))
        except (ParseError, ValueError, IndexError) as e:
            out.warnings.append(f"crab PDF line {i + 1} not read ({e}): {s[:160]}")
    if not out.rows:
        out.warnings.append("crab PDF: no result rows found")
    return out


# ---------------------------------------------------------------- seafood (anchovy, sardine)
@dataclass
class SeafoodRow:
    date: str
    species: str
    tissue: str
    size: int
    block: str
    raw: str
    note: str | None = None


SEAFOOD_LINE = re.compile(
    rf"^(?:(?P<date>\d{{1,2}}/\d{{1,2}}/\d{{4}})\s{{2,}})?(?P<species>[A-Z][A-Za-z.]*(?: [A-Za-z.]+)*)\s{{2,}}(?P<tissue>Viscera|Whole Body|Meat)\s{{2,}}(?P<size>\d+)\s{{2,}}(?P<block>\d+)\s{{2,}}(?P<val>{_ONE})$"
)


def parse_seafood(text: str) -> Parsed:
    out = Parsed(as_of=as_of(text), period=period(text))
    # the header cell "FDA Action Level:" and "20 ppm or greater" print on different layout lines
    m = re.search(r"(\d+(?:\.\d+)?)\s*ppm or greater", text, re.I)
    if m and re.search(r"FDA Action Level", text, re.I):
        out.action_quoted = f"FDA Action Level: {m.group(1)} ppm or greater"
        out.action_ppm = float(m.group(1))
    pending_date: str | None = None
    last: SeafoodRow | None = None
    in_table = False
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith("Date of Catch"):
            in_table = True
            continue
        if not in_table:
            continue
        if s.startswith("*") or s.startswith("Abbreviations") or s.startswith("As of"):
            break
        if DATE.match(s):
            pending_date = us_date(s)
            continue
        if re.match(r"^RT\b", s):
            if last is not None and last.note is None:
                last.note = s
            else:
                out.warnings.append(f"seafood PDF line {i + 1}: retest note with no row: {s[:80]}")
            continue
        mm = SEAFOOD_LINE.match(s)
        if not mm:
            if re.search(r"\d", s) and not s.startswith("20 ppm"):
                out.warnings.append(f"seafood PDF line {i + 1} not read: {s[:160]}")
            continue
        d = us_date(mm.group("date")) if mm.group("date") else pending_date
        if d is None:
            out.warnings.append(f"seafood PDF line {i + 1} has no catch date: {s[:160]}")
            continue
        pending_date = None
        sp = SPECIES.get(mm.group("species").lower())
        if sp is None:
            out.warnings.append(f"seafood PDF line {i + 1}: unknown species {mm.group('species')!r}")
            continue
        last = SeafoodRow(d, sp, TISSUES[mm.group("tissue").lower()], int(mm.group("size")), mm.group("block"), mm.group("val"))
        out.rows.append(last)
    if not out.rows:
        out.warnings.append("seafood PDF: no result rows found")
    return out


# ---------------------------------------------------------------- page: current seafood file
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}


def _end_date(name: str) -> date | None:
    """End of the period in a seafood file name: 'SeafoodDAResults082826to102226.pdf' (MMDDYY) or
    'SeafoodDAResultsJune052026toJuly232026.pdf'."""
    m = re.search(r"to(\d{2})(\d{2})(\d{2})\.pdf$", name, re.I)
    if m:
        return date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = re.search(r"to([A-Za-z]+)(\d{1,2})(\d{4})\.pdf$", name)
    if m and m.group(1).lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def seafood_url(page_html: str) -> str | None:
    """The seafood results PDF with the latest period end linked from CDPH's Domoic Acid page."""
    best: tuple[date, str] | None = None
    for href in set(re.findall(r'href="([^"]*/DomoicAcid/SeafoodDAResults[^"]*\.pdf)"', page_html, re.I)):
        end = _end_date(href)
        if end and (best is None or end > best[0]):
            best = (end, href)
    return urljoin(PAGE_URL, best[1]) if best else None


# ---------------------------------------------------------------- index
def _site_meta(site: str) -> tuple[float | None, float | None, list[str]]:
    lat, lon, places = SITES.get(site.lower().strip(), (None, None, []))
    return lat, lon, list(places)


def crab_sites(p: Parsed, url: str) -> list[BiotoxinSite]:
    sites: dict[str, BiotoxinSite] = {}
    for r in p.rows:
        species = SPECIES.get(r.species.lower(), r.species)
        sid = slug("bma", r.bma, r.site, species)
        vals, meats, nd = [], [], 0
        for t in r.raw:
            v, meat, is_nd = value(t)
            vals.append(v)
            meats.append(meat)
            nd += int(is_nd)
        nums = [v for v in vals if v is not None]
        st = BiotoxinSet(
            date=r.date, kind=r.kind, tissue="viscera", values_ppm=vals, nd_count=nd, avg_ppm=r.avg,
            max_ppm=max(nums) if nums else None, pct_over_action=r.pct, action_level_ppm=p.action_ppm,
            values_raw=list(r.raw), meat_values_ppm=meats if any(m is not None for m in meats) else None,
        )
        if sid not in sites:
            lat, lon, places = _site_meta(r.site)
            sites[sid] = BiotoxinSite(id=sid, bma=r.bma, bma_label=r.bma_label, port_label=r.bma_label, site_label=r.site, species=species, lat=lat, lon=lon, informs_places=places, sets=[], source_url=url)
        sites[sid].sets.append(st)
    return list(sites.values())


def seafood_sites(p: Parsed, url: str) -> list[BiotoxinSite]:
    sites: dict[str, BiotoxinSite] = {}
    for r in p.rows:
        sid = slug("block", r.block, r.species)
        v, _, nd = value(r.raw)
        kind = "retest" if r.note and r.note.upper().startswith("RT") else "unknown"
        st = BiotoxinSet(
            date=r.date, kind=kind, tissue=r.tissue, values_ppm=[v], nd_count=int(nd), avg_ppm=None, max_ppm=v,
            pct_over_action=None, action_level_ppm=p.action_ppm, values_raw=[r.raw], composite_of=r.size, note=r.note,
        )
        if sid not in sites:
            # CDFW fishing blocks are not mapped to places here (no block geometry in the repo)
            sites[sid] = BiotoxinSite(id=sid, bma=None, bma_label=None, port_label=None, site_label=f"CDFW block {r.block}", species=r.species, lat=None, lon=None, informs_places=[], sets=[], source_url=url)
        sites[sid].sets.append(st)
    return list(sites.values())


def build_index(now_iso: str, crab: tuple[bytes, str, str] | None, seafood: tuple[bytes, str, str] | None, errors: list[str] | None = None) -> BiotoxinIndex:
    """crab/seafood: (pdf bytes, url, fetched_at). Either may be None (fetch failed: in errors)."""
    errors = list(errors or [])
    warnings: list[str] = []
    sites: list[BiotoxinSite] = []
    levels: list[ActionLevel] = []
    sources: list[BiotoxinSource] = []
    for name, doc, parse, make, species, tissue in (
        ("CDPH Summary of Domoic Acid Levels in Crab", crab, parse_crab, crab_sites, "Dungeness crab and rock crab", "viscera"),
        ("CDPH Summary of Domoic Acid Levels in Seafood", seafood, parse_seafood, seafood_sites, "Northern anchovy and Pacific sardine", "viscera or whole body"),
    ):
        if doc is None:
            continue
        blob, url, fetched = doc
        src = BiotoxinSource(name=name, url=url, as_of=None, fetched_at=fetched, sha256=hashlib.sha256(blob).hexdigest())
        try:
            p = parse(pdf_text(blob))
        except Exception as e:
            errors.append(f"{name}: {e}")
            sources.append(src)
            continue
        src.as_of, src.period = p.as_of, p.period
        sources.append(src)
        warnings.extend(p.warnings)
        if p.action_ppm is not None:
            levels.append(ActionLevel(species=species, tissue=tissue, ppm=p.action_ppm, quoted=p.action_quoted))
        sites.extend(make(p, url))
    for s in sites:
        s.sets.sort(key=lambda x: x.date, reverse=True)
    sites.sort(key=lambda s: (s.informs_places == [], s.id))
    if not sources:
        raise RuntimeError("; ".join(errors) or "no CDPH document")
    return BiotoxinIndex(
        version=1, generated_at=now_iso, source=sources[0], additional_sources=sources[1:], action_levels=levels,
        sites=sites, parse_warnings=warnings, errors=errors,
    )


def run(ctx: RunContext) -> BiotoxinIndex:
    """Fetch both PDFs and the page, build the index. A document that cannot be fetched is an error
    in the index (the other still publishes); nothing at all raises (the previous file stays)."""
    from ..http import FetchError

    errors: list[str] = []
    crab = seafood = None
    try:
        r = ctx.fetcher(CRAB_URL)
        crab = (r.body, CRAB_URL, ctx.now_iso)
    except FetchError as e:
        errors.append(f"crab PDF: {e}"[:300])
    try:
        url = seafood_url(ctx.fetcher(PAGE_URL).body.decode("utf-8", "replace"))
        if url is None:
            errors.append("seafood PDF: no SeafoodDAResults link found on the CDPH Domoic Acid page")
        else:
            seafood = (ctx.fetcher(url).body, url, ctx.now_iso)
    except FetchError as e:
        errors.append(f"seafood PDF: {e}"[:300])
    return build_index(ctx.now_iso, crab, seafood, errors)


def write(ctx: RunContext, idx: BiotoxinIndex) -> Path:
    from ..publish.files import write_json

    path = ctx.out_dir / OUT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, idx.model_dump(mode="json"))
    return path


def is_newer_or_equal(new: BiotoxinIndex, old_path: Path) -> bool:
    """Never replace a readable index with one built from an older crab PDF, or with one that has
    no crab results at all (crab PDF unreachable or unreadable while the seafood PDF was read: the
    new index would lead with the seafood document and silently drop every crab site)."""
    if not old_path.exists():
        return True
    try:
        old = BiotoxinIndex.model_validate_json(old_path.read_text())
    except Exception:
        return True
    if new.source.name != old.source.name:
        return False
    return (new.source.as_of or "") >= (old.source.as_of or "")

