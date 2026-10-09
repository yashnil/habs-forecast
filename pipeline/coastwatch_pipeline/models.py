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
    "derived_summary",
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

    basis: Literal["issued_date", "valid_date", "observed_date", "reviewed_date"]
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
    domain: list[float] = Field(min_length=2, max_length=2)
    stops: list[PaletteStop]
    interpolation: Literal["linear"] = "linear"
    nodata: Literal["transparent"] = "transparent"


class RasterImage(_Model):
    """A PNG already reprojected to EPSG:3857 so it can be placed by its corners."""

    url: str
    crs: Literal["EPSG:3857"] = "EPSG:3857"
    width: int
    height: int
    bounds_lnglat: list[float] = Field(
        min_length=4, max_length=4, description="west, south, east, north of the image edges"
    )
    corners_lnglat: list[list[float]] = Field(
        min_length=4,
        max_length=4,
        description="[lng, lat] of top-left, top-right, bottom-right, bottom-left (MapLibre order)",
    )
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
    notes: list[str] = Field(default_factory=list, description="Informational messages, e.g. fallbacks taken")
    latest_issued_date: str | None = None
    latest_valid_date: str | None = None
    freshness: FreshnessPolicy


class PortProperties(_Model):
    port_code: int
    name: str
    display_name: str
    port_area: str
    port_area_code: int
    # Added in M2. Optional so ports files written by the M1 pipeline (same schema
    # version) stay valid; the pipeline always fills them now.
    county: str | None = None
    region: str | None = None


class Region(_Model):
    id: str
    label: str
    bounds: list[list[float]] = Field(min_length=2, max_length=2, description="[[west, south], [east, north]]")


class PortFeature(_Model):
    type: Literal["Feature"] = "Feature"
    geometry: dict
    properties: PortProperties


class PortsCollection(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[PortFeature]
    regions: list[Region] = Field(default_factory=list)
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
    official_url: str | None = None
    port_intel_url: str | None = None
    # Added in M3 (additive; absent in manifests written by earlier pipelines).
    observations_url: str | None = None
    fisheries_url: str | None = None


# ---------------------------------------------------------------- official notices (M2)
Agency = Literal["CDFW", "CDPH", "OEHHA"]


class SourceRef(_Model):
    label: str
    url: str
    kind: Literal["press_release", "status_page", "legal_document", "map", "information_line"]
    published_date: str | None = None


class AreaSpec(_Model):
    type: Literal["statewide", "county", "lat_band", "named_area"]
    description: str = Field(description="Area as worded by the agency")
    counties: list[str] = Field(default_factory=list)
    lat_north: float | None = None
    lat_south: float | None = None
    geometry_basis: Literal["official_polygon", "derived_from_official_latitudes", "none"]
    geometry_note: str | None = None


class OfficialRecord(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    agency: Agency
    action: Literal[
        "fishery_closure",
        "take_restriction",
        "consumption_advisory",
        "quarantine",
        "special_advisory",
        "reopening",
        "advisory_lifted",
    ]
    status: Literal["active", "lifted", "superseded"]
    title: str
    summary: str = Field(description="Plain-language summary written by the reviewer")
    official_text: str = Field(description="Verbatim wording from the official source")
    fishery: Literal["commercial", "recreational", "commercial_and_recreational", "consumption", "sport_harvest"]
    species: list[str] = Field(min_length=1)
    toxins: list[Literal["domoic_acid", "psp"]] = Field(min_length=1)
    area: AreaSpec
    effective_date: str | None
    effective_date_note: str | None = None
    expected_end_date: str | None = None
    expected_end_note: str | None = None
    lifted_date: str | None = None
    sources: list[SourceRef] = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)


class OfficialStatement(_Model):
    """An official statement that does not restrict anything, quoted verbatim (e.g. 'There
    are currently no closures … due to naturally occurring marine toxins'). Never rendered
    as 'open' or 'safe'."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    agency: Agency
    topic: str
    statement: str
    source: SourceRef


class WatchedSource(_Model):
    id: str
    label: str
    url: str
    parser: Literal["cdph_release_list", "html_main_text"]


class ReviewInfo(_Model):
    status: Literal["human_verified", "pending_human_review"]
    reviewed_at: str
    reviewed_by: str
    method: str
    source_fingerprints: dict[str, str] = Field(description="watched source id -> content hash at review")
    cdph_release_ids_reviewed: list[str]


class OfficialRegistry(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    records: list[OfficialRecord]
    statements: list[OfficialStatement]
    watched_sources: list[WatchedSource]
    review: ReviewInfo
    hotlines: list[dict[str, str]] = Field(default_factory=list)


class WatchResult(_Model):
    source_id: str
    url: str
    checked_at: str
    ok: bool
    http_status: int | None = None
    error: str | None = None
    content_hash: str | None = None
    matches_review: bool | None = None
    items_seen: list[str] = Field(default_factory=list)
    new_items: list[str] = Field(default_factory=list, description="Items not present at the last review")


class VerificationPolicy(_Model):
    verified_max_age_days: int
    aging_max_age_days: int
    note: str


class OfficialDataset(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: str
    registry: OfficialRegistry
    watch: list[WatchResult]
    conflicts: list[str]
    geometry: dict = Field(description="GeoJSON FeatureCollection; properties.record_ids, basis, note")
    geometry_errors: list[str]
    policy: VerificationPolicy
    provenance: Provenance


# ---------------------------------------------------------------- port intelligence (M2)
class Stat(_Model):
    n: int
    median: float | None
    min: float | None
    max: float | None


class PortLeadSummary(_Model):
    lead_days: int
    valid_date: str
    variables: dict[str, Stat]


class SeriesPoint(_Model):
    date: str
    value: float | None
    n: int


class PortCharm(_Model):
    issued_date: str | None
    radius_km: float
    cells_in_radius: int
    nearest_cell_km: float | None
    leads: list[PortLeadSummary]
    history: dict[str, list[SeriesPoint]] = Field(default_factory=dict, description="nowcast median by valid date")
    history_error: str | None = None


class PortChlorophyll(_Model):
    dataset_id: str
    source_url: str
    radius_km: float
    composite_days: int
    latest_center_date: str | None
    latest: Stat | None
    latest_valid_fraction: float | None
    history: list[SeriesPoint] = Field(default_factory=list)
    error: str | None = None


class PortOfficialRelation(_Model):
    record_id: str
    relation: Literal["statewide", "same_county", "port_latitude_within_stated_range", "named_area_nearby"]
    note: str


class PortIntel(_Model):
    port_code: int
    display_name: str
    county: str
    region: str
    lon: float
    lat: float
    charm: PortCharm | None
    chlorophyll: PortChlorophyll | None
    official_relations: list[PortOfficialRelation] = Field(
        description="Official records whose stated area may include waters near this port, and why"
    )
    caveats: list[str]


class PortIntelCollection(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: str
    ports: list[PortIntel]
    method: dict[str, str]
    provenance: list[Provenance]


# ---------------------------------------------------------------- measured observations (M3)
ObsVariableId = Literal["pDA", "tDA", "dDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]
Qualifier = Literal["reported_zero", "rejected_negative", "flag_high"]


class ObsVariable(_Model):
    """How one measured quantity is defined. Values of different variables are never
    combined: fractions (particulate/dissolved/total), matrices and units stay separate."""

    id: ObsVariableId
    source_variable: str = Field(description="Upstream column name, verbatim")
    label: str
    upstream_long_name: str | None = None
    units: str = Field(description="Units as published upstream, verbatim")
    kind: Literal["toxin", "cell_abundance", "pigment", "physical"]
    matrix: Literal["seawater"]
    fraction: str | None = Field(None, description="e.g. particulate, dissolved, total; null if not applicable")
    method: str = Field(description="Analytical method, or a statement that it is not published")
    detection_limit: float | None = Field(None, description="Only when published upstream")
    detection_limit_note: str
    zero_policy: str = Field(description="How a reported 0 is interpreted")
    plausible_max: float | None = Field(None, description="Values above this are kept but flagged for review")


class ObsSeries(_Model):
    """Values aligned with the station's sample_times. null = not measured in that sample
    (absence of a measurement, never zero). qualifiers maps a sample index (as a string)
    to a qualifier code."""

    variable: ObsVariableId
    values: list[float | None]
    qualifiers: dict[str, Qualifier] = Field(default_factory=dict)


class ObsVariableSummary(_Model):
    variable: ObsVariableId
    n_measured: int = Field(description="Samples with a reported value (including reported zeros)")
    n_reported_zero: int
    n_rejected: int
    first_date: str | None
    last_date: str | None
    last_value: float | None
    last_qualifier: Qualifier | None = None
    days_since_last: int | None = Field(None, description="Days from the last measurement to the run date")
    n_last_365d: int
    median_interval_days_365d: float | None = Field(None, description="Median days between measurements in the last 365 days")
    max_last_365d: float | None


class StationCharm(_Model):
    """C-HARM nowcast near the station: a model probability, shown beside (never compared
    numerically with) the measurements."""

    radius_km: float
    nearest_cell_km: float | None
    history_days: int
    history: dict[str, list[SeriesPoint]] = Field(default_factory=dict)
    error: str | None = None


class ObsStation(_Model):
    station_id: str = Field(description="Upstream dataset id, e.g. HABs-SantaCruzWharf")
    name: str
    location_code: str | None = None
    lat: float | None = Field(description="Most recent sampling position; null only when the station has never been retrieved")
    lon: float | None
    region: str | None = None
    nearest_port_code: int | None = None
    nearest_port_name: str | None = None
    nearest_port_km: float | None = None
    status: Literal["updated", "carried_forward", "failed"]
    error: str | None = None
    retrieved_at: str | None = Field(None, description="When this station's data were fetched")
    source_url: str
    request_url: str | None = None
    sample_times: list[str] = Field(description="UTC ISO-8601 sample times, ascending")
    depths_m: list[float | None]
    series: list[ObsSeries]
    summaries: list[ObsVariableSummary]
    qc: list[QCCheck]
    charm: StationCharm | None = None


class ObservationDataset(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: str
    window_start: str = Field(description="Earliest sample date requested")
    program: str
    variables: list[ObsVariable]
    stations: list[ObsStation]
    method: dict[str, str]
    caveats: list[str]
    provenance: Provenance


# ---------------------------------------------------------------- fisheries exposure (M3)
class Deflator(_Model):
    series_id: str
    title: str
    source_url: str
    base_year: int
    annual_index: dict[str, float] = Field(description="year -> annual average index")
    months_used: dict[str, int] = Field(description="year -> number of monthly values averaged")
    method: str
    notes: list[str] = Field(default_factory=list)


class YearValue(_Model):
    year: int
    pounds: float | None
    dollars_nominal: float | None
    dollars_real: float | None = Field(description="In base-year dollars; null when nominal is null")
    n_rows: int = Field(description="Source rows aggregated")
    n_rows_without_value: int = Field(0, description="Source rows with no published value (not zero)")


class SpeciesGroup(_Model):
    id: str
    label: str
    tier: Literal[1, 2]
    tier_basis: str
    official_record_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(description="Upstream species names assigned to this group")
    annual: list[YearValue]


class SuppressedValue(_Model):
    year: int
    pounds: float | None
    dollars_nominal: float | None
    dollars_real: float | None
    note: str


class ExcludedRow(_Model):
    year: int
    source_name: str
    duplicate_of: str
    pounds: float | None
    dollars_nominal: float | None
    reason: str


class PortLevelStatus(_Model):
    status: Literal["available", "unavailable"]
    reasons: list[str]
    adapters: list[dict[str, str]] = Field(description="id, status, note for each port-level source considered")


class FisheriesDataset(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: str
    scope: str
    years: list[int]
    years_requested_unavailable: list[int] = Field(default_factory=list)
    deflator: Deflator
    groups: list[SpeciesGroup]
    statewide_total: list[YearValue] = Field(description="All commercial rows including the withheld row (as in NOAA's state totals), duplicates counted once")
    withheld: list[SuppressedValue]
    excluded_rows: list[ExcludedRow]
    port_level: PortLevelStatus
    terminology: str
    method: dict[str, str]
    caveats: list[str]
    provenance: list[Provenance]


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "manifest": Manifest,
    "ports": PortsCollection,
    "official": OfficialDataset,
    "port_intel": PortIntelCollection,
    "observations": ObservationDataset,
    "fisheries": FisheriesDataset,
}
