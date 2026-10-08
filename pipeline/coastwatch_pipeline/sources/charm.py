"""C-HARM v3.1 (California Harmful Algae Risk Mapping) from NOAA CoastWatch West Coast ERDDAP.

Verified 2026-10-08 (docs/coastwatch/evidence/sources-charm.md):
- datasets wvcharmV3_{0,1,2,3}day (0-360 longitudes), product_version 3.1
- `time` is the VALID day (12:00Z), not the issue time
- one daily run issues all four leads; issue date = nowcast valid date + 1 (inferred)
- runs are frequently missing; never back-fill a missing lead from an older run
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
from scipy.io import netcdf_file

from ..context import RunContext
from ..http import FetchError
from ..models import (
    ForecastRun,
    FreshnessPolicy,
    LayerArtifact,
    Provenance,
    QCCheck,
    QualityControl,
    RasterImage,
    TimeInfo,
    ValueGrid,
)
from ..process import grid as gridcodec
from ..process.mercator import SourceGrid, plan_image, resample_to_mercator
from ..process.palette import PROBABILITY, apply_palette
from ..publish.files import write_bytes, write_png

SOURCE_ID = "charm"
GROUP_ID = "charm"
SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
LEADS = (0, 1, 2, 3)
EXPECTED_VERSION = "3.1"
STEP = 0.03
# Full published domain (0-360 longitudes), verified 2026-10-08
DOMAIN = {"lat_min": 31.3, "lat_max": 43.0, "lon_min": 232.5, "lon_max": 243.0}
MIN_VALID_FRACTION = 0.10
LICENSE = (
    "NOAA CoastWatch: may be used and redistributed for free but is not intended for "
    "legal use, since it may contain inaccuracies. No warranty or legal liability."
)
CITATION = (
    "C-HARM v3.1, NOAA NMFS SWFSC ERD CoastWatch West Coast (creator C. Anderson, "
    "SCCOOS/UCSD). Methods: Anderson et al. 2016, Harmful Algae 59:1-18, "
    "doi:10.1016/j.hal.2016.08.006 (assessment of v1)."
)


def dataset_id(lead: int) -> str:
    return f"wvcharmV3_{lead}day"


@dataclass(frozen=True)
class Variable:
    name: str
    short_title: str
    title: str
    threshold_text: str
    description: str


VARIABLES: tuple[Variable, ...] = (
    Variable(
        name="pseudo_nitzschia",
        short_title="Pseudo-nitzschia bloom",
        title="Probability of a Pseudo-nitzschia bloom",
        threshold_text="Probability that Pseudo-nitzschia exceeds 10,000 cells per litre",
        description=(
            "Chance that cells of Pseudo-nitzschia, the diatom that can produce domoic acid, "
            "exceed 10,000 per litre in surface water."
        ),
    ),
    Variable(
        name="particulate_domoic",
        short_title="Particulate domoic acid",
        title="Probability of elevated particulate domoic acid",
        threshold_text="Probability that particulate domoic acid exceeds 500 ng per litre",
        description="Chance that domoic acid in the plankton exceeds 500 nanograms per litre of seawater.",
    ),
    Variable(
        name="cellular_domoic",
        short_title="Cellular domoic acid",
        title="Probability of elevated cellular domoic acid",
        threshold_text="Probability that cellular domoic acid exceeds 10 pg per cell",
        description="Chance that toxin per Pseudo-nitzschia cell exceeds 10 picograms.",
    ),
)

CAVEATS_COMMON = [
    "Forecast of water-column conditions. It does not measure toxin in seafood and is not a closure or health decision; CDFW and CDPH decide closures and advisories.",
    "A low probability does not mean an area is safe.",
    "Published skill assessment covers C-HARM v1 (Anderson et al. 2016); no published skill assessment for v3.1 was found.",
    "The producer notes salinity errors can affect domoic acid predictions and that bloom predictions include many false positives.",
]
CAVEAT_TOXIN_MASK = (
    "Toxin probabilities are not provided for many cells within about 3-6 km of shore, "
    "including piers and harbours."
)
CAVEAT_ISSUE = (
    "Issue date is inferred (nowcast valid day + 1). C-HARM does not publish an issue time."
)

FRESHNESS = FreshnessPolicy(
    basis="issued_date",
    current_max_age_days=1,
    stale_max_age_days=7,
    note=(
        "C-HARM runs daily but runs are often missing. Current: issued today or yesterday (UTC). "
        "Stale: 2-7 days old. Historical: older than 7 days; not a current forecast."
    ),
)


class ValidationFailure(Exception):
    def __init__(self, message: str, checks: list[QCCheck]):
        super().__init__(message)
        self.checks = checks


# ---------------------------------------------------------------- time helpers
def parse_time(s: str) -> datetime:
    s = s.strip()
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def issue_date_for(valid: date, lead: int) -> date:
    """Nowcast (lead 0) for day D is produced by the run on D+1, which also produces
    leads 1..3 valid D+1..D+3. So issue = valid - lead + 1."""
    return valid - timedelta(days=lead) + timedelta(days=1)


# ---------------------------------------------------------------- upstream access
def latest_time_url(lead: int) -> str:
    return f"{SERVER}/griddap/{dataset_id(lead)}.csv0?time%5Blast%5D"


def data_url(lead: int, valid_time: datetime) -> str:
    t = valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    sel = (
        f"[({t})]"
        f"[({DOMAIN['lat_min']}):({DOMAIN['lat_max']})]"
        f"[({DOMAIN['lon_min']}):({DOMAIN['lon_max']})]"
    )
    q = ",".join(f"{v.name}{sel}" for v in VARIABLES)
    return f"{SERVER}/griddap/{dataset_id(lead)}.nc?" + quote(q, safe=",():")


def probe_latest(ctx: RunContext, lead: int) -> datetime:
    r = ctx.fetcher(latest_time_url(lead))
    text = r.body.decode("utf-8", "replace").strip().splitlines()
    if not text:
        raise FetchError(f"empty time response for {dataset_id(lead)}")
    return parse_time(text[-1])


@dataclass
class LeadData:
    lead: int
    valid_time: datetime
    lat: np.ndarray
    lon: np.ndarray  # converted to -180..180
    values: dict[str, np.ndarray]
    attrs: dict[str, str]
    request_url: str


def parse_netcdf(blob: bytes) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, str]]:
    """Read an ERDDAP NetCDF-3 response. Returns (arrays, per-variable attrs, global attrs)."""
    if not blob.startswith(b"CDF"):
        raise ValueError("response is not NetCDF-3 (missing CDF signature)")
    f = netcdf_file(io.BytesIO(blob), mmap=False)
    try:
        arrays: dict[str, np.ndarray] = {}
        vattrs: dict[str, dict[str, object]] = {}
        for name, var in f.variables.items():
            arrays[name] = np.array(var.data, copy=True)
            vattrs[name] = {k: _decode(v) for k, v in var._attributes.items()}
        gattrs = {k: str(_decode(v)) for k, v in f._attributes.items()}
    finally:
        f.close()
    return arrays, vattrs, gattrs


def _decode(v: object) -> object:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.item()
    return v


def load_lead(blob: bytes, lead: int, expected_time: datetime, request_url: str) -> tuple[LeadData, list[QCCheck]]:
    """Parse and validate one lead. Raises ValidationFailure on any hard failure."""
    checks: list[QCCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(QCCheck(name=name, passed=bool(ok), detail=detail))

    try:
        arrays, vattrs, gattrs = parse_netcdf(blob)
    except Exception as e:  # malformed body, HTML error page, truncated file...
        check("parse_netcdf", False, str(e))
        raise ValidationFailure(f"lead {lead}: unreadable response: {e}", checks) from e
    check("parse_netcdf", True, "NetCDF-3 parsed")

    missing = [n for n in ("time", "latitude", "longitude", *(v.name for v in VARIABLES)) if n not in arrays]
    check("required_variables", not missing, f"missing: {missing}" if missing else "all present")
    if missing:
        raise ValidationFailure(f"lead {lead}: missing variables {missing}", checks)

    version = gattrs.get("product_version", "")
    check("product_version", version == EXPECTED_VERSION, f"product_version={version!r}, expected {EXPECTED_VERSION!r}")

    t = np.atleast_1d(arrays["time"]).astype(np.float64)
    check("single_time_step", t.size == 1, f"{t.size} time steps")
    vt = datetime.fromtimestamp(float(t[0]), tz=timezone.utc) if t.size else None
    check(
        "valid_time_matches_probe",
        vt is not None and vt == expected_time,
        f"file time {vt.isoformat() if vt else None}, probed {expected_time.isoformat()}",
    )
    check(
        "valid_time_is_noon_utc",
        vt is not None and (vt.hour, vt.minute, vt.second) == (12, 0, 0),
        f"{vt.time() if vt else None}",
    )

    lat = arrays["latitude"].astype(np.float64)
    lon = arrays["longitude"].astype(np.float64)
    dlat, dlon = np.diff(lat), np.diff(lon)
    uniform = (
        lat.ndim == 1
        and lon.ndim == 1
        and lat.size > 1
        and lon.size > 1
        and np.allclose(np.abs(dlat), STEP, atol=1e-4)
        and np.allclose(np.abs(dlon), STEP, atol=1e-4)
        and (np.all(dlat > 0) or np.all(dlat < 0))
        and np.all(dlon > 0)
    )
    check("regular_grid_0p03", uniform, f"lat step {dlat[:1]}, lon step {dlon[:1]}, sizes {lat.size}x{lon.size}")
    in_domain = bool(
        lat.size
        and lon.size
        and lat.min() >= DOMAIN["lat_min"] - 1e-6
        and lat.max() <= DOMAIN["lat_max"] + 1e-6
        and lon.min() >= DOMAIN["lon_min"] - 1e-6
        and lon.max() <= DOMAIN["lon_max"] + 1e-6
    )
    check("within_published_domain", in_domain, f"lat {lat.min():.3f}..{lat.max():.3f}, lon {lon.min():.3f}..{lon.max():.3f}")

    values: dict[str, np.ndarray] = {}
    n_valid_total = 0
    for v in VARIABLES:
        a = np.squeeze(arrays[v.name].astype(np.float64), axis=0) if arrays[v.name].ndim == 3 else arrays[v.name].astype(np.float64)
        ok_shape = a.shape == (lat.size, lon.size)
        check(f"{v.name}_shape", ok_shape, f"{a.shape} vs {(lat.size, lon.size)}")
        if not ok_shape:
            raise ValidationFailure(f"lead {lead}: {v.name} has shape {a.shape}", checks)
        for key in ("_FillValue", "missing_value"):
            fv = vattrs[v.name].get(key)
            if isinstance(fv, (int, float)) and math.isfinite(float(fv)):
                a[np.isclose(a, float(fv))] = np.nan
        finite = a[np.isfinite(a)]
        in_range = bool(finite.size == 0 or (finite.min() >= -1e-6 and finite.max() <= 1 + 1e-6))
        check(
            f"{v.name}_probability_range",
            in_range,
            f"min {finite.min() if finite.size else None}, max {finite.max() if finite.size else None}",
        )
        frac = finite.size / a.size if a.size else 0.0
        check(f"{v.name}_valid_fraction", frac >= MIN_VALID_FRACTION, f"{frac:.3f} (min {MIN_VALID_FRACTION})")
        values[v.name] = np.clip(a, 0.0, 1.0)
        n_valid_total += finite.size

    if not all(c.passed for c in checks):
        failed = [c.name for c in checks if not c.passed]
        raise ValidationFailure(f"lead {lead}: failed checks {failed}", checks)

    lon180 = np.where(lon > 180, lon - 360, lon)
    return (
        LeadData(
            lead=lead,
            valid_time=vt,  # type: ignore[arg-type]
            lat=lat,
            lon=lon180,
            values=values,
            attrs=gattrs,
            request_url=request_url,
        ),
        checks,
    )


# ---------------------------------------------------------------- artifacts
def _qc(values: np.ndarray, checks: list[QCCheck]) -> QualityControl:
    finite = values[np.isfinite(values)]
    return QualityControl(
        checks=checks,
        n_cells=int(values.size),
        n_valid=int(finite.size),
        valid_fraction=round(finite.size / values.size, 6) if values.size else 0.0,
        value_min=float(finite.min()) if finite.size else None,
        value_max=float(finite.max()) if finite.size else None,
    )


def _provenance(ctx: RunContext, ld: LeadData, probe_url: str) -> Provenance:
    keep = ("title", "product_version", "date_created", "creator_name", "creator_email", "source")
    meta = {k: ld.attrs[k] for k in keep if k in ld.attrs}
    hist = ld.attrs.get("history", "")
    if hist:
        meta["history_first_line"] = hist.strip().splitlines()[0][:400]
    return Provenance(
        source_id=SOURCE_ID,
        source_name="C-HARM v3.1 — California Harmful Algae Risk Mapping",
        source_url=f"{SERVER}/griddap/{dataset_id(ld.lead)}.html",
        dataset_id=dataset_id(ld.lead),
        product_version=ld.attrs.get("product_version"),
        institution=ld.attrs.get("institution", "NOAA NMFS SWFSC ERD (CoastWatch West Coast)"),
        license=LICENSE,
        citation=CITATION,
        retrieved_at=ctx.now_iso,
        request_urls=[probe_url, ld.request_url],
        upstream_metadata=meta,
        pipeline_version=ctx.pipeline_version,
        pipeline_run_id=ctx.run_id,
    )


def build_lead_artifacts(ctx: RunContext, ld: LeadData, checks: list[QCCheck], issued: date) -> list[LayerArtifact]:
    lat_first, lat_step = float(ld.lat[0]), float(ld.lat[1] - ld.lat[0])
    lon_first, lon_step = float(ld.lon[0]), float(ld.lon[1] - ld.lon[0])
    src = SourceGrid(lat_first, lat_step, lon_first, lon_step, ld.lat.size, ld.lon.size)
    img = plan_image(src, upsample=4)
    base = f"charm/{issued.isoformat()}/lead{ld.lead}"
    valid = ld.valid_time.date()
    artifacts: list[LayerArtifact] = []
    for v in VARIABLES:
        vals = ld.values[v.name]
        blob, scale, offset, qerr = gridcodec.encode(vals, 0.0, 1.0)
        grid_rel = f"{base}/{v.name}.u16.gz"
        write_bytes(ctx.out_dir / grid_rel, blob)
        rgba = apply_palette(resample_to_mercator(vals, src, img), PROBABILITY)
        img_rel = f"{base}/{v.name}.png"
        write_png(ctx.out_dir / img_rel, rgba)
        caveats = list(CAVEATS_COMMON) + [CAVEAT_ISSUE]
        if v.name != "pseudo_nitzschia":
            caveats.insert(2, CAVEAT_TOXIN_MASK)
        artifacts.append(
            LayerArtifact(
                layer_id=f"charm_{v.name}_lead{ld.lead}",
                group_id=GROUP_ID,
                product_class="official_forecast",
                title=v.title,
                short_title=v.short_title,
                variable=v.name,
                units="probability (0-1)",
                threshold_text=v.threshold_text,
                description=v.description,
                resolution_deg=STEP,
                time=TimeInfo(
                    issued_date=issued.isoformat(),
                    issued_date_derived=True,
                    issued_date_method="nowcast valid day + 1 day (one run issues all leads)",
                    valid_date=valid.isoformat(),
                    valid_time=ld.valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    lead_days=ld.lead,
                ),
                freshness=FRESHNESS,
                image=RasterImage(
                    url=img_rel,
                    width=img.width,
                    height=img.height,
                    bounds_lnglat=(img.west, img.south, img.east, img.north),
                    corners_lnglat=img.corners,
                ),
                grid=ValueGrid(
                    url=grid_rel,
                    width=src.width,
                    height=src.height,
                    lat_first=lat_first,
                    lat_step=lat_step,
                    lon_first=lon_first,
                    lon_step=lon_step,
                    scale_factor=scale,
                    add_offset=offset,
                    max_quantization_error=qerr,
                ),
                palette=PROBABILITY,
                caveats=caveats,
                qc=_qc(vals, checks),
                provenance=_provenance(ctx, ld, latest_time_url(ld.lead)),
            )
        )
    return artifacts


@dataclass
class CharmResult:
    layers: list[LayerArtifact]
    run: ForecastRun | None
    errors: list[str]
    latest_issued: date | None
    latest_valid: date | None


def run(ctx: RunContext) -> CharmResult:
    """Fetch the newest complete-or-partial run. Leads that belong to an older run are
    reported missing; they are never mixed into the newest run."""
    errors: list[str] = []
    probed: dict[int, datetime] = {}
    for lead in LEADS:
        try:
            probed[lead] = probe_latest(ctx, lead)
        except Exception as e:
            errors.append(f"lead {lead}: probe failed: {e}")
    if not probed:
        return CharmResult([], None, errors, None, None)

    issues = {lead: issue_date_for(t.date(), lead) for lead, t in probed.items()}
    newest = max(issues.values())
    notes: list[str] = []
    stale_leads = sorted(lead for lead, d in issues.items() if d != newest)
    for lead in stale_leads:
        notes.append(
            f"Lead {lead}: newest available valid day {probed[lead].date()} belongs to the run issued "
            f"{issues[lead]}, not {newest}; not shown."
        )

    layers: list[LayerArtifact] = []
    available: list[int] = []
    for lead in LEADS:
        if lead not in probed or issues[lead] != newest:
            continue
        url = data_url(lead, probed[lead])
        try:
            r = ctx.fetcher(url)
            ld, checks = load_lead(r.body, lead, probed[lead], url)
            layers.extend(build_lead_artifacts(ctx, ld, checks, newest))
            available.append(lead)
        except ValidationFailure as e:
            failed = "; ".join(f"{c.name}: {c.detail}" for c in e.checks if not c.passed)
            errors.append(f"{e} ({failed})")
        except Exception as e:
            errors.append(f"lead {lead}: {e}")

    missing = [lead for lead in LEADS if lead not in available]
    run_info = (
        ForecastRun(
            group_id=GROUP_ID,
            source_id=SOURCE_ID,
            issued_date=newest.isoformat(),
            issued_date_derived=True,
            leads_available=available,
            leads_missing=missing,
            notes=notes + [f"Lead {lead} could not be published this run." for lead in missing if lead not in stale_leads],
        )
        if available
        else None
    )
    latest_valid = max((probed[lead].date() for lead in available), default=None)
    return CharmResult(layers, run_info, errors, newest if available else None, latest_valid)
