"""NASA GIBS satellite chlorophyll browse tiles (visual layer only).

Verified 2026-10-08 (docs/coastwatch/evidence/sources-satellite.md): tiles for the
newest date advertised in GetCapabilities (<Default>) came back as HTTP 500 or as
corrupt RGBA images, while the previous day was clean. So a date is only used after
probe tiles over the California coast come back as palette-mode PNGs. Legend URLs are
read from GetCapabilities and checked, not hard-coded.
"""

from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from PIL import Image

from ..context import RunContext
from ..models import FreshnessPolicy, LayerArtifact, Provenance, TileLayer, TimeInfo

SOURCE_ID = "gibs_chl"
CAPS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml"
TILE_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/{tms}/{{z}}/{{y}}/{{x}}.png"
)
# z=6 tiles covering central California water (row, col), verified ocean coverage
PROBE_TILES = ((6, 24, 9), (6, 24, 10), (6, 25, 10))
MAX_LOOKBACK_DAYS = 6
# A date whose probe tiles are almost entirely transparent has not been populated yet
# (or is fully clouded); skip it rather than show an empty layer as "latest".
MIN_OPAQUE_FRACTION = 0.02
LICENSE = "NASA EOSDIS GIBS imagery: open, no restrictions on reuse; attribute NASA GIBS."

FRESHNESS = FreshnessPolicy(
    basis="observed_date",
    current_max_age_days=4,
    stale_max_age_days=10,
    note=(
        "Daily satellite composites usually appear 1-3 days after observation. Current: "
        "observed within 4 days. Stale: 5-10 days. Historical: older."
    ),
)


@dataclass(frozen=True)
class GibsLayer:
    layer_id: str
    artifact_id: str
    title: str
    short_title: str
    sensor: str


LAYERS = (
    GibsLayer(
        "VIIRS_NOAA20_Chlorophyll_a",
        "gibs_viirs_noaa20_chl",
        "Satellite chlorophyll-a (VIIRS, NOAA-20)",
        "Chlorophyll · VIIRS",
        "VIIRS on NOAA-20",
    ),
    GibsLayer(
        "OCI_PACE_Chlorophyll_a",
        "gibs_pace_oci_chl",
        "Satellite chlorophyll-a (PACE OCI)",
        "Chlorophyll · PACE",
        "OCI on PACE",
    ),
)

CAVEATS = [
    "Chlorophyll measures algae biomass. It does not measure toxins and does not predict where fish are.",
    "Clouds and fog leave gaps; a gap means no observation, not low chlorophyll.",
    "Browse imagery for visual context. Colours follow NASA's legend; values are not read from these tiles.",
]


@dataclass
class LayerCaps:
    default: str | None
    tms: str | None
    legend_urls: list[str]
    available: list[tuple[date, date]]

    def has(self, d: date) -> bool:
        return any(a <= d <= b for a, b in self.available)


def parse_capabilities(xml: str, layer_id: str) -> LayerCaps:
    m = re.search(rf"<ows:Identifier>{re.escape(layer_id)}</ows:Identifier>", xml)
    if not m:
        return LayerCaps(None, None, [], [])
    start = xml.rfind("<Layer>", 0, m.start())
    end = xml.find("</Layer>", m.end())
    block = xml[max(start, 0) : end if end > 0 else m.end() + 40000]
    dm = re.search(r"<ows:Identifier>Time</ows:Identifier>.*?<Default>([^<]+)</Default>", block, re.S)
    tm = re.search(r"<TileMatrixSet>([^<]+)</TileMatrixSet>", block)
    legends = re.findall(r"<LegendURL[^>]*xlink:href=['\"]([^'\"]+)['\"]", block)
    available: list[tuple[date, date]] = []
    for v in re.findall(r"<Value>([^<]+)</Value>", block):
        parts = v.strip().split("/")
        try:
            if len(parts) >= 2:
                available.append((date.fromisoformat(parts[0][:10]), date.fromisoformat(parts[1][:10])))
            else:
                d = date.fromisoformat(parts[0][:10])
                available.append((d, d))
        except ValueError:
            continue
    return LayerCaps(dm.group(1).strip() if dm else None, tm.group(1).strip() if tm else None, legends, available)


def png_color_type(body: bytes) -> int | None:
    """PNG IHDR colour type (3 = palette). None if not a PNG."""
    if len(body) < 29 or body[:8] != b"\x89PNG\r\n\x1a\n" or body[12:16] != b"IHDR":
        return None
    return struct.unpack(">B", body[25:26])[0]


def tile_opaque_fraction(body: bytes) -> float:
    with Image.open(io.BytesIO(body)) as im:
        a = np.asarray(im.convert("RGBA"))[..., 3]
    return float((a > 0).mean())


def tile_check(status: int, body: bytes) -> tuple[bool, float]:
    """(structurally valid, opaque fraction).

    Healthy GIBS chlorophyll tiles are palette PNGs. Two failure modes were observed on
    the newest advertised date: HTTP 500 / striped RGBA noise (2026-10-08 16:30Z), and
    well-formed but completely transparent tiles (2026-10-08 17:02Z)."""
    if status != 200 or png_color_type(body) != 3:
        return False, 0.0
    try:
        return True, tile_opaque_fraction(body)
    except Exception:
        return False, 0.0


def select_date(ctx: RunContext, layer: GibsLayer, caps: LayerCaps) -> tuple[str | None, list[str]]:
    notes: list[str] = []
    if not caps.default or not caps.tms:
        return None, [f"{layer.layer_id}: not found in GetCapabilities"]
    d0 = date.fromisoformat(caps.default)
    for back in range(0, MAX_LOOKBACK_DAYS + 1):
        dd = d0 - timedelta(days=back)
        d = dd.isoformat()
        if caps.available and not caps.has(dd):
            notes.append(f"{layer.layer_id}: {d} not listed as available in GetCapabilities; skipped.")
            continue
        tmpl = TILE_TEMPLATE.format(layer=layer.layer_id, date=d, tms=caps.tms)
        results: list[bool] = []
        coverage: list[float] = []
        for z, y, x in PROBE_TILES:
            url = tmpl.replace("{z}", str(z)).replace("{y}", str(y)).replace("{x}", str(x))
            try:
                r = ctx.fetcher(url)
                ok, frac = tile_check(r.status, r.body)
            except Exception:
                ok, frac = False, 0.0
            results.append(ok)
            coverage.append(frac)
        mean_cov = sum(coverage) / len(coverage)
        if all(results) and mean_cov >= MIN_OPAQUE_FRACTION:
            if back:
                notes.append(f"{layer.layer_id}: newest advertised date {caps.default} failed tile checks; using {d}.")
            return d, notes
        notes.append(
            f"{layer.layer_id}: {d} rejected ({sum(results)}/{len(results)} probe tiles well-formed, "
            f"{mean_cov:.1%} of probe pixels with data; need {MIN_OPAQUE_FRACTION:.0%})."
        )
    return None, notes


def check_legend(ctx: RunContext, urls: list[str]) -> tuple[str | None, bool]:
    # Prefer the horizontal SVG legend
    ordered = sorted(urls, key=lambda u: (not u.endswith("_H.svg"), u))
    for u in ordered:
        try:
            r = ctx.fetcher(u)
            if r.status == 200 and b"<svg" in r.body[:2000]:
                return u, True
        except Exception:
            continue
    return (ordered[0] if ordered else None), False


@dataclass
class GibsResult:
    layers: list[LayerArtifact]
    errors: list[str]
    notes: list[str]
    latest_date: str | None


def run(ctx: RunContext) -> GibsResult:
    try:
        xml = ctx.fetcher(CAPS_URL).body.decode("utf-8", "replace")
    except Exception as e:
        return GibsResult([], [f"GetCapabilities failed: {e}"], [], None)
    layers: list[LayerArtifact] = []
    errors: list[str] = []
    notes: list[str] = []
    for gl in LAYERS:
        caps = parse_capabilities(xml, gl.layer_id)
        chosen, n = select_date(ctx, gl, caps)
        notes.extend(n)
        if not chosen:
            errors.append(f"{gl.layer_id}: no date passed tile validation")
            continue
        legend, legend_ok = check_legend(ctx, caps.legend_urls)
        layers.append(
            LayerArtifact(
                layer_id=gl.artifact_id,
                group_id="satellite_chlorophyll",
                product_class="observation",
                title=gl.title,
                short_title=gl.short_title,
                variable="chlorophyll_a",
                units="mg m-3 (NASA legend)",
                description=(
                    f"Daily ocean-colour chlorophyll-a from {gl.sensor}, rendered by NASA GIBS. "
                    "A measure of algae biomass near the surface."
                ),
                time=TimeInfo(observed_date=chosen, valid_date=chosen),
                freshness=FRESHNESS,
                tiles=TileLayer(
                    url_template=TILE_TEMPLATE.format(layer=gl.layer_id, date=chosen, tms=caps.tms),
                    max_native_zoom=7 if caps.tms and caps.tms.endswith("Level7") else 7,
                    legend_url=legend,
                    legend_verified=legend_ok,
                    date_selection=(
                        f"Newest date advertised by GetCapabilities was {caps.default}. Chosen date is the "
                        f"most recent of up to {MAX_LOOKBACK_DAYS + 1} days whose {len(PROBE_TILES)} probe tiles "
                        "over central California returned HTTP 200 palette PNGs with data in at least "
                        f"{MIN_OPAQUE_FRACTION:.0%} of pixels."
                    ),
                ),
                caveats=CAVEATS,
                provenance=Provenance(
                    source_id=SOURCE_ID,
                    source_name="NASA GIBS (Global Imagery Browse Services)",
                    source_url="https://nasa-gibs.github.io/gibs-api-docs/",
                    dataset_id=gl.layer_id,
                    institution="NASA EOSDIS",
                    license=LICENSE,
                    retrieved_at=ctx.now_iso,
                    request_urls=[CAPS_URL],
                    upstream_metadata={"capabilities_default_date": caps.default or "", "tile_matrix_set": caps.tms or ""},
                    pipeline_version=ctx.pipeline_version,
                    pipeline_run_id=ctx.run_id,
                ),
            )
        )
    latest = max((lyr.time.observed_date for lyr in layers if lyr.time.observed_date), default=None)
    return GibsResult(layers, errors, notes, latest)
