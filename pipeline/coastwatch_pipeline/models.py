"""Versioned artifact schemas shared by the pipeline and the web app.

These Pydantic models are the single source of truth. `cwp schema` exports them
to JSON Schema (`schemas/v1/*.schema.json`), from which the web app's
TypeScript types are generated. Bump SCHEMA_VERSION on breaking changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

ProductClass = Literal[
    "official_regulatory",
    "official_forecast",
    "observation",
    "experimental_model",
    "historical_context",
    "reference",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(_Model):
    source_id: str = Field(description="Stable id of the upstream source, e.g. 'charm'")
    source_name: str
    source_url: str = Field(description="Dataset or page URL a person can open")
    dataset_id: str | None = None
    product_version: str | None = None
    institution: str | None = None
    license: str
    citation: str | None = None
    retrieved_at: str = Field(description="UTC ISO-8601 time the pipeline fetched the data")
    request_urls: list[str] = Field(default_factory=list)
    upstream_metadata: dict[str, str] = Field(
        default_factory=dict, description="Selected upstream attributes kept verbatim"
    )
    pipeline_version: str
    pipeline_run_id: str


class FreshnessPolicy(_Model):
    """How the client classifies age. Computed in the browser so a dead pipeline
    still degrades to 'stale' / 'historical' instead of looking current."""

    basis: Literal["issued_date", "valid_date", "observed_date"]
    current_max_age_days: int = Field(ge=0)
    stale_max_age_days: int = Field(ge=0, description="Older than this is 'historical'")
    note: str


class TimeInfo(_Model):
    issued_date: str | None = Field(None, description="UTC calendar date the product was issued")
    issued_date_derived: bool = Field(
        False, description="True when the issue date is inferred, not published upstream"
    )
    issued_date_method: str | None = None
    valid_date: str | None = Field(None, description="UTC calendar date the value represents")
    valid_time: str | None = Field(None, description="Upstream time stamp, verbatim")
    observed_date: str | None = None
    lead_days: int | None = Field(None, ge=0)


class QCCheck(_Model):
    name: str
    passed: bool
    detail: str


class QualityControl(_Model):
    checks: list[QCCheck]
    n_cells: int
    n_valid: int
    valid_fraction: float = Field(ge=0, le=1)
    value_min: float | None
    value_max: float | None


class PaletteStop(_Model):
    value: float
    color: str = Field(pattern=r"^#[0-9a-f]{6}$")


class Palette(_Model):
    id: str
    domain: tuple[float, float]
    stops: list[PaletteStop]
    interpolation: Literal["linear"] = "linear"
    nodata: Literal["transparent"] = "transparent"


class RasterImage(_Model):
    """A PNG already reprojected to EPSG:3857 so it can be placed by its corners."""

    url: str
    crs: Literal["EPSG:3857"] = "EPSG:3857"
    width: int
    height: int
    bounds_lnglat: tuple[float, float, float, float] = Field(
        description="west, south, east, north of the image edges"
    )
    corners_lnglat: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ] = Field(description="top-left, top-right, bottom-right, bottom-left (MapLibre order)")
    resampling: Literal["nearest"] = "nearest"


class ValueGrid(_Model):
    """Source values on the source (equirectangular) grid, quantized to uint16 and
    gzip-compressed. value = raw * scale_factor + add_offset; raw == nodata -> no value."""

    url: str
    encoding: Literal["uint16le+gzip"] = "uint16le+gzip"
    width: int
    height: int
    lat_first: float = Field(description="Latitude of the first row's cell centre")
    lat_step: float = Field(description="Signed step between rows")
    lon_first: float
    lon_step: float
    scale_factor: float
    add_offset: float
    nodata: int = 65535
    max_quantization_error: float


class TileLayer(_Model):
    """Third-party pre-rendered tiles (e.g. NASA GIBS)."""

    url_template: str
    tile_size: int = 256
    max_native_zoom: int
    legend_url: str | None
    legend_verified: bool
    date_selection: str = Field(description="How the tile date was chosen and verified")


class LayerArtifact(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    layer_id: str
    group_id: str
    product_class: ProductClass
    title: str
    short_title: str
    variable: str
    units: str
    threshold_text: str | None = None
    description: str
    resolution_deg: float | None = None
    time: TimeInfo
    freshness: FreshnessPolicy
    image: RasterImage | None = None
    grid: ValueGrid | None = None
    tiles: TileLayer | None = None
    palette: Palette | None = None
    caveats: list[str]
    qc: QualityControl | None = None
    provenance: Provenance


class ForecastRun(_Model):
    group_id: str
    source_id: str
    issued_date: str
    issued_date_derived: bool
    leads_available: list[int]
    leads_missing: list[int]
    notes: list[str]


class SourceStatus(_Model):
    source_id: str
    title: str
    product_class: ProductClass
    last_attempt_at: str
    last_success_at: str | None
    outcome: Literal["updated", "unchanged", "failed", "partial"]
    error: str | None = None
    latest_issued_date: str | None = None
    latest_valid_date: str | None = None
    freshness: FreshnessPolicy


class PortProperties(_Model):
    port_code: int
    name: str
    display_name: str
    port_area: str
    port_area_code: int


class PortFeature(_Model):
    type: Literal["Feature"] = "Feature"
    geometry: dict
    properties: PortProperties


class PortsCollection(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[PortFeature]
    caveats: list[str]
    provenance: Provenance


class Manifest(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: str
    pipeline_version: str
    pipeline_run_id: str
    layers: list[LayerArtifact]
    forecast_runs: list[ForecastRun]
    sources: list[SourceStatus]
    ports_url: str | None


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "manifest": Manifest,
    "ports": PortsCollection,
}
