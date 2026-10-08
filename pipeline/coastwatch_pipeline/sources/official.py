"""Official closures and advisories (CDFW, CDPH).

Human-reviewed by design (docs/coastwatch/04-architecture.md, 05-science-and-safety.md):

* `data/curated/official_notices.json` is the only place regulatory records come from. It is
  edited by people, through pull requests.
* This module validates that registry, fetches official geometry, and *watches* the official
  pages. The watcher reports changes since the last human review; it never creates, edits
  or lifts a record.
* The browser decides whether the registry counts as verified (review status, review age,
  and whether the watcher saw changes or failed). A missing record never means "open".
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..context import RunContext
from ..models import (
    OfficialDataset,
    OfficialRecord,
    OfficialRegistry,
    Provenance,
    VerificationPolicy,
    WatchResult,
)

SOURCE_ID = "official"
REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "data" / "curated" / "official_notices.json"
OCEAN_MASK = REPO_ROOT / "data" / "reference" / "charm_ocean_mask.json"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0 Safari/537.36 CoastWatch-watcher"
)
CDPH_LAYERS = "https://services2.arcgis.com/wi1yEacfYjH5viqb/arcgis/rest/services"
COUNTY_LAYER = f"{CDPH_LAYERS}/California_Coastal_Counties/FeatureServer/3"
NCI_LAYER = f"{CDPH_LAYERS}/All_Bivalve_Shellfish_Health_Advisory_Special_Advisory/FeatureServer/3"
KNOWN_COUNTIES = {
    "Del Norte", "Humboldt", "Mendocino", "Sonoma", "Marin", "San Francisco", "San Mateo",
    "Santa Cruz", "Monterey", "San Luis Obispo", "Santa Barbara", "Ventura", "Los Angeles",
    "Orange", "San Diego", "Alameda", "Contra Costa", "Solano",
}
CA_LAT = (32.4, 42.1)
BAND_OFFSHORE_KM = 20.0

POLICY = VerificationPolicy(
    verified_max_age_days=3,
    aging_max_age_days=7,
    note=(
        "Verified: a person reviewed every record against the official pages within 3 days and the "
        "watcher has seen no change since. Ageing: 4-7 days since review. Older reviews, pending "
        "reviews, detected source changes, or a failed watcher mean the records are NOT verified."
    ),
)


class RegistryError(Exception):
    pass


# ---------------------------------------------------------------- registry validation
def _is_date(s: str | None) -> bool:
    if s is None:
        return True
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def validate_registry(reg: OfficialRegistry) -> tuple[list[str], list[str]]:
    """Return (errors, conflicts). Errors make the registry unpublishable; conflicts are
    published and make the whole registry 'not verified' in the app."""
    errors: list[str] = []
    ids = [r.id for r in reg.records] + [s.id for s in reg.statements]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append(f"duplicate ids: {dupes}")
    for r in reg.records:
        tag = f"{r.id}:"
        for name in ("effective_date", "expected_end_date", "lifted_date"):
            if not _is_date(getattr(r, name)):
                errors.append(f"{tag} {name} is not an ISO date")
        if r.status == "lifted" and not r.lifted_date:
            errors.append(f"{tag} status 'lifted' requires lifted_date")
        if r.status == "active" and r.lifted_date:
            errors.append(f"{tag} status 'active' must not have lifted_date (contradictory)")
        if r.effective_date is None and not r.effective_date_note:
            errors.append(f"{tag} missing effective_date requires effective_date_note explaining why")
        if r.effective_date and r.expected_end_date and r.expected_end_date < r.effective_date:
            errors.append(f"{tag} expected_end_date before effective_date")
        if r.expected_end_date and not r.expected_end_note:
            errors.append(f"{tag} expected_end_date requires expected_end_note (end dates can be extended)")
        a = r.area
        if a.type == "lat_band":
            if a.lat_north is None or a.lat_south is None:
                errors.append(f"{tag} lat_band needs lat_north and lat_south")
            elif not (CA_LAT[0] <= a.lat_south < a.lat_north <= CA_LAT[1]):
                errors.append(f"{tag} lat_band {a.lat_south}..{a.lat_north} is not a valid California range")
        if a.type == "county":
            if not a.counties:
                errors.append(f"{tag} county area needs counties")
            unknown = sorted(set(a.counties) - KNOWN_COUNTIES)
            if unknown:
                errors.append(f"{tag} unknown counties {unknown}")
        if a.geometry_basis == "official_polygon" and a.type not in ("county", "named_area"):
            errors.append(f"{tag} official_polygon basis only for county or named areas")
        if a.geometry_basis == "derived_from_official_latitudes" and a.type != "lat_band":
            errors.append(f"{tag} derived geometry needs a lat_band area")
        if a.geometry_basis != "none" and not a.geometry_note:
            errors.append(f"{tag} mapped areas need a geometry_note explaining the drawing")
        for s in r.sources:
            if not (s.url.startswith("https://") or s.url.startswith("tel:")):
                errors.append(f"{tag} source URL must be https or tel: ({s.url})")
        if not r.official_text.strip():
            errors.append(f"{tag} official_text must quote the agency")
    for s in reg.statements:
        if re.search(r"\b(safe|open for|all clear)\b", s.statement, re.I):
            errors.append(f"{s.id}: statements must not characterise areas as safe or open")
    if not reg.watched_sources:
        errors.append("registry must list watched official sources")
    return errors, find_conflicts(reg)


def _overlaps(a: OfficialRecord, b: OfficialRecord) -> bool:
    if a.area.type == "statewide" or b.area.type == "statewide":
        return True
    if a.area.type == b.area.type == "county":
        return bool(set(a.area.counties) & set(b.area.counties))
    if a.area.type == b.area.type == "lat_band":
        return min(a.area.lat_north, b.area.lat_north) > max(a.area.lat_south, b.area.lat_south)  # type: ignore[type-var]
    return a.area.description == b.area.description


def find_conflicts(reg: OfficialRegistry) -> list[str]:
    restrictive = {"fishery_closure", "take_restriction", "consumption_advisory", "quarantine", "special_advisory"}
    permissive = {"reopening", "advisory_lifted"}
    active = [r for r in reg.records if r.status == "active"]
    out = []
    for a in active:
        if a.action not in permissive:
            continue
        for b in active:
            if b.action in restrictive and set(a.species) & set(b.species) and _overlaps(a, b):
                out.append(
                    f"Active '{a.action}' record {a.id} overlaps active '{b.action}' record {b.id} for "
                    f"{sorted(set(a.species) & set(b.species))}; both cannot be current."
                )
    return out


def load_registry(path: Path = REGISTRY) -> OfficialRegistry:
    return OfficialRegistry.model_validate_json(path.read_text())


# ---------------------------------------------------------------- page watcher
def normalise_html(raw: str) -> str:
    s = re.sub(r"<!--.*?-->", "", raw, flags=re.S)  # CDFW keeps stale sections in comments
    s = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</tr>|</td>|</div>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("​", "").replace("⁰", "°").replace("⁰", "°")
    s = re.sub(r"[ \t\xa0]+", " ", s)
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def main_text(raw: str) -> str:
    """Normalised page text from the H1 to the end of the main content, so navigation,
    banners and footers do not trigger change alarms."""
    t = normalise_html(raw)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", raw, flags=re.S | re.I)
    if m:
        title = normalise_html(m.group(1)).strip()
        i = t.find(title)
        if i >= 0:
            t = t[i:]
    for end in ("Back to Top", "Back To Top"):
        j = t.find(end)
        if j > 0:
            t = t[:j]
    return t


def cdph_release_items(raw: str) -> list[str]:
    seen: dict[str, str] = {}
    for href, title in re.findall(r'href="([^"]*SN\d\d-\d{3}\.aspx)"[^>]*>(.*?)</a>', raw, flags=re.S | re.I):
        rid = re.search(r"(SN\d\d-\d{3})", href).group(1)  # type: ignore[union-attr]
        t = normalise_html(title).strip()
        if t and rid not in seen:
            seen[rid] = t
    return [f"{k} | {v}" for k, v in sorted(seen.items(), reverse=True)]


def fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def watch(ctx: RunContext, reg: OfficialRegistry) -> list[WatchResult]:
    out = []
    for src in reg.watched_sources:
        try:
            r = ctx.fetcher(src.url) if not _uses_default_fetch(ctx) else _browser_fetch(ctx, src.url)
            raw = r.body.decode("utf-8", "replace")
            if src.parser == "cdph_release_list":
                items = cdph_release_items(raw)
                if not items:
                    raise ValueError("no CDPH release links found; page layout may have changed")
                fp = fingerprint("\n".join(items))
                ids = [i.split(" | ")[0] for i in items]
                new = [i for i in ids if i not in set(reg.review.cdph_release_ids_reviewed)]
                seen = ids[:30]
            else:
                text = main_text(raw)
                if len(text) < 200:
                    raise ValueError("page text unexpectedly short; layout may have changed or access was blocked")
                fp = fingerprint(text)
                new, seen = [], []
            reviewed = reg.review.source_fingerprints.get(src.id)
            out.append(
                WatchResult(
                    source_id=src.id,
                    url=src.url,
                    checked_at=ctx.now_iso,
                    ok=True,
                    http_status=r.status,
                    content_hash=fp,
                    matches_review=(fp == reviewed) if reviewed else None,
                    items_seen=seen,
                    new_items=new,
                )
            )
        except Exception as e:
            out.append(WatchResult(source_id=src.id, url=src.url, checked_at=ctx.now_iso, ok=False, error=str(e)[:300]))
    return out


def _uses_default_fetch(ctx: RunContext) -> bool:
    from ..http import fetch

    return ctx.fetcher is fetch


def _browser_fetch(ctx: RunContext, url: str):
    from ..http import fetch

    return fetch(url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html"})


# ---------------------------------------------------------------- geometry
@dataclass
class OceanMask:
    lat_first: float
    lat_step: float
    lon_first: float
    lon_step: float
    ocean: np.ndarray = field(repr=False)

    @classmethod
    def load(cls, path: Path = OCEAN_MASK) -> "OceanMask":
        d = json.loads(path.read_text())
        m = np.zeros((d["height"], d["width"]), dtype=bool)
        for r, runs in enumerate(d["rows_rle"]):
            for s, n in runs:
                m[r, s : s + n] = True
        return cls(d["lat_first"], d["lat_step"], d["lon_first"], d["lon_step"], m)


def lat_band_feature(mask: OceanMask, north: float, south: float, offshore_km: float = BAND_OFFSHORE_KM) -> dict:
    """Ocean cells within `offshore_km` of the coast between two official latitudes, as a
    MultiPolygon of row runs. Cells are clipped exactly to the band latitudes."""
    mid = math.radians((north + south) / 2)
    dy_km = abs(mask.lat_step) * 111.32
    dx_km = abs(mask.lon_step) * 111.32 * math.cos(mid)
    land_dist = ndimage.distance_transform_edt(mask.ocean, sampling=(dy_km, dx_km))
    near = mask.ocean & (land_dist <= offshore_km)
    polys = []
    h, w = mask.ocean.shape
    for r in range(h):
        c_lat = mask.lat_first + r * mask.lat_step
        top, bot = c_lat + abs(mask.lat_step) / 2, c_lat - abs(mask.lat_step) / 2
        top, bot = min(top, north), max(bot, south)
        if top <= bot:
            continue
        c = 0
        while c < w:
            if near[r, c]:
                s = c
                while c < w and near[r, c]:
                    c += 1
                west = mask.lon_first + (s - 0.5) * mask.lon_step
                east = mask.lon_first + (c - 0.5) * mask.lon_step
                ring = [[west, bot], [east, bot], [east, top], [west, top], [west, bot]]
                polys.append([[[round(x, 5), round(y, 5)] for x, y in ring]])
            else:
                c += 1
    if not polys:
        raise ValueError(f"no ocean cells between {south} and {north}")
    return {"type": "MultiPolygon", "coordinates": polys}


def _round_geom(g: dict) -> dict:
    def rnd(x):
        if isinstance(x, list) and x and isinstance(x[0], (int, float)):
            return [round(x[0], 5), round(x[1], 5)]
        return [rnd(y) for y in x] if isinstance(x, list) else x

    return {"type": g["type"], "coordinates": rnd(g["coordinates"])}


def _bounds(g: dict) -> tuple[float, float, float, float]:
    xs, ys = [], []

    def walk(x):
        if isinstance(x, list) and x and isinstance(x[0], (int, float)):
            xs.append(x[0])
            ys.append(x[1])
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(g["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def _arcgis_geojson(ctx: RunContext, layer: str, where: str) -> list[dict]:
    from urllib.parse import quote

    url = f"{layer}/query?where={quote(where)}&outFields=TITLE&outSR=4326&f=geojson"
    data = json.loads(ctx.fetcher(url).body)
    feats = data.get("features") or []
    if not feats:
        raise ValueError(f"no features for {where} from {layer}")
    return feats


def build_geometry(ctx: RunContext, reg: OfficialRegistry) -> tuple[dict, list[str]]:
    features: list[dict] = []
    errors: list[str] = []
    active = [r for r in reg.records if r.status == "active"]
    # official county polygons (CDPH advisory map layer)
    for county in sorted({c for r in active if r.area.type == "county" for c in r.area.counties}):
        ids = [r.id for r in active if r.area.type == "county" and county in r.area.counties]
        try:
            f = _arcgis_geojson(ctx, COUNTY_LAYER, f"TITLE = '{county} County'")[0]
            g = _round_geom(f["geometry"])
            w, s, e, n = _bounds(g)
            if not (-125.5 < w < e < -114 and 32 < s < n < 42.5):
                raise ValueError(f"{county} polygon bounds {w, s, e, n} outside California")
            features.append(_feature(g, ids, "official_polygon", f"{county} County (CDPH advisory map county polygon)", "county"))
        except Exception as e:
            errors.append(f"{county} County polygon unavailable: {e}")
    # named areas with an official polygon (Northern Channel Islands special advisory)
    for r in active:
        if r.area.type == "named_area" and r.area.geometry_basis == "official_polygon":
            try:
                f = _arcgis_geojson(ctx, NCI_LAYER, "1=1")[0]
                g = _round_geom(f["geometry"])
                w, s, e, n = _bounds(g)
                if not (-121.0 < w < e < -118.5 and 33.5 < s < n < 34.5):
                    raise ValueError(f"polygon bounds {w, s, e, n} not at the Northern Channel Islands")
                features.append(_feature(g, [r.id], "official_polygon", r.area.description, "named_area"))
            except Exception as e:
                errors.append(f"{r.id}: official polygon unavailable: {e}")
    # latitude bands derived from official latitudes over nearshore ocean
    try:
        mask = OceanMask.load()
    except Exception as e:
        mask = None
        errors.append(f"ocean mask unavailable, latitude bands not drawn: {e}")
    bands: dict[tuple[float, float], list[OfficialRecord]] = {}
    for r in active:
        if r.area.type == "lat_band" and r.area.geometry_basis == "derived_from_official_latitudes":
            bands.setdefault((round(r.area.lat_north, 3), round(r.area.lat_south, 3)), []).append(r)  # type: ignore[arg-type]
    for (n, s), recs in sorted(bands.items(), reverse=True):
        if mask is None:
            continue
        try:
            g = lat_band_feature(mask, max(r.area.lat_north for r in recs), min(r.area.lat_south for r in recs))  # type: ignore[type-var]
            label = recs[0].area.description
            features.append(
                _feature(
                    g,
                    [r.id for r in recs],
                    "derived_from_official_latitudes",
                    f"{label}. Shaded within ~{BAND_OFFSHORE_KM:.0f} km of the coast for display; the official wording controls.",
                    "lat_band",
                )
            )
        except Exception as e:
            errors.append(f"band {s}..{n}: {e}")
    return {"type": "FeatureCollection", "features": features}, errors


def _feature(g: dict, ids: list[str], basis: str, note: str, kind: str) -> dict:
    return {"type": "Feature", "geometry": g, "properties": {"record_ids": ids, "basis": basis, "note": note, "kind": kind}}


# ---------------------------------------------------------------- run
@dataclass
class OfficialResult:
    dataset: OfficialDataset | None
    errors: list[str]
    notes: list[str]


def run(ctx: RunContext, path: Path = REGISTRY) -> OfficialResult:
    try:
        reg = load_registry(path)
    except Exception as e:
        return OfficialResult(None, [f"registry failed schema validation: {e}"], [])
    errors, conflicts = validate_registry(reg)
    if errors:
        return OfficialResult(None, [f"registry invalid: {'; '.join(errors)}"], [])
    watch_results = watch(ctx, reg)
    geometry, geo_errors = build_geometry(ctx, reg)
    notes = []
    for w in watch_results:
        if not w.ok:
            notes.append(f"watcher could not read {w.source_id}: {w.error}")
        elif w.new_items:
            notes.append(f"{w.source_id}: official items not yet reviewed: {', '.join(w.new_items)}")
        elif w.matches_review is False:
            notes.append(f"{w.source_id}: page content changed since the last review")
    ds = OfficialDataset(
        generated_at=ctx.now_iso,
        registry=reg,
        watch=watch_results,
        conflicts=conflicts,
        geometry=geometry,
        geometry_errors=geo_errors,
        policy=POLICY,
        provenance=Provenance(
            source_id=SOURCE_ID,
            source_name="CoastWatch official notices registry (human-reviewed transcription of CDFW and CDPH notices)",
            source_url="https://github.com/yashnil/habs-forecast/blob/main/data/curated/official_notices.json",
            institution="Records: California Department of Fish and Wildlife; California Department of Public Health",
            license="Official notices are public information from CDFW and CDPH; the agencies' own pages control.",
            retrieved_at=ctx.now_iso,
            request_urls=[w.url for w in reg.watched_sources],
            upstream_metadata={"review_status": reg.review.status, "reviewed_at": reg.review.reviewed_at},
            pipeline_version=ctx.pipeline_version,
            pipeline_run_id=ctx.run_id,
        ),
    )
    return OfficialResult(ds, [], notes + geo_errors + conflicts)


def record_review(path: Path, ctx: RunContext, reviewer: str, status: str, method: str) -> dict:
    """Used by a person after checking every record against the official pages. Stores the
    current fingerprints of the watched pages so later changes are detected."""
    reg = load_registry(path)
    errors, conflicts = validate_registry(reg)
    if errors:
        raise RegistryError("; ".join(errors))
    results = watch(ctx, reg)
    failed = [w.source_id for w in results if not w.ok]
    if failed:
        raise RegistryError(f"cannot record a review while sources are unreadable: {failed}")
    raw = json.loads(path.read_text())
    raw["review"] = {
        "status": status,
        "reviewed_at": ctx.now_iso,
        "reviewed_by": reviewer,
        "method": method,
        "source_fingerprints": {w.source_id: w.content_hash for w in results},
        "cdph_release_ids_reviewed": sorted(
            {i for w in results for i in w.items_seen} | set(reg.review.cdph_release_ids_reviewed), reverse=True
        ),
    }
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    return raw["review"]
