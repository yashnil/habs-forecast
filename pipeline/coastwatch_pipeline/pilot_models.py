"""Sidecar products of the Neritic private pilot (2026-11): `v1/biotoxin/index.json` and
`v1/fisheries-status/index.json`.

They mirror `coastwatch-web/src/pilot/contracts.ts` (BiotoxinIndex, FisheriesStatusIndex) exactly;
fields marked "optional extension" are additions the contract allows (documented for the integrator).
They are published OUTSIDE the manifest: the deployed apps reject unknown manifest fields, so the
manifest never references them (docs/coastwatch/pilot/README.md).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ biotoxin (CDPH PDFs)
class BiotoxinSource(_M):
    name: str
    url: str
    as_of: str | None = Field(description="the 'As of' date printed on the PDF, YYYY-MM-DD; null if unreadable")
    fetched_at: str
    sha256: str
    # optional extension: the period printed in the PDF's title line, as written
    period: str | None = None


class ActionLevel(_M):
    species: str
    tissue: str
    ppm: float
    # optional extension: the action-level text as printed (e.g. "FDA Action Level >30 PPM")
    quoted: str | None = None


class BiotoxinSet(_M):
    date: str = Field(description="collection date (Pacific), YYYY-MM-DD")
    kind: Literal["routine", "retest", "unknown"]
    tissue: Literal["viscera", "meat", "whole", "unknown"]
    values_ppm: list[float | None] = Field(description="individual samples as published; null = ND, below the reporting limit, or unreadable")
    nd_count: int = Field(ge=0, description="samples reported ND (not detected) or below a reporting limit ('<2.5')")
    avg_ppm: float | None
    max_ppm: float | None
    pct_over_action: float | None
    action_level_ppm: float | None
    # optional extensions
    values_raw: list[str] = Field(default_factory=list, description="each sample exactly as printed, e.g. '37 | 6.6', 'ND', '<2.5'")
    meat_values_ppm: list[float | None] | None = Field(default=None, description="crab: the meat value of a paired 'viscera | meat' result, aligned with values_ppm; null where not printed")
    composite_of: int | None = Field(default=None, description="seafood: number of fish in the composite sample")
    note: str | None = Field(default=None, description="annotation printed with the row, e.g. 'RT 1-1st set'")


class BiotoxinSite(_M):
    id: str
    bma: str | None
    bma_label: str | None
    port_label: str | None
    site_label: str
    species: str
    lat: float | None
    lon: float | None
    informs_places: list[str]
    sets: list[BiotoxinSet]
    # optional extension: which CDPH document the site's sets come from
    source_url: str | None = None


class BiotoxinIndex(_M):
    version: Literal[1]
    generated_at: str
    source: BiotoxinSource
    action_levels: list[ActionLevel]
    sites: list[BiotoxinSite]
    parse_warnings: list[str]
    # optional extension: the other CDPH documents read (the seafood/anchovy PDF)
    additional_sources: list[BiotoxinSource] = Field(default_factory=list)
    # optional extension: honest status of each document this run (fetch/parse failures stay visible)
    errors: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ fisheries status (curated YAML)
FisheryId = Literal[
    "dungeness_commercial",
    "dungeness_recreational",
    "rock_crab",
    "salmon_commercial",
    "groundfish",
    "anchovy_sardine",
    "market_squid",
    "bivalve_shellfish",
    "spiny_lobster",
    "other",
]
StatusLevel = Literal["closed", "delayed", "restricted", "watch", "no_restriction_on_record"]
ReviewState = Literal["pending_human_review", "human_verified"]


class FisheryStatus(_M):
    id: str
    fishery: FisheryId
    label: str
    level: StatusLevel
    cause: Literal["whale_entanglement", "domoic_acid", "psp", "regulation", "season", "quality", "other"]
    headline: str = Field(max_length=110)
    detail: str
    area_label: str
    applies_to_places: list[str]
    effective: str | None
    until: str | None
    next_review: str | None
    source_name: str
    source_url: str
    checked_at: str
    review_state: ReviewState
    # optional extensions: who reviewed (set only by the human review command) and other sources read
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    other_sources: list[str] = Field(default_factory=list)


class RampZone(_M):
    zone: str
    label: str
    action: str
    depth_limit_fathoms: float | None
    gear_reduction_pct: float | None
    # optional extensions: latitude bounds as defined by 14 CCR 132.8(a)(7) / CDFW BIOS ds3120
    lat_north: float | None = None
    lat_south: float | None = None
    applies_to_places: list[str] = Field(default_factory=list)


class RampState(_M):
    season: str
    zones: list[RampZone]
    last_assessment: str | None
    next_assessment: str | None
    source_url: str
    review_state: ReviewState
    # optional extensions
    checked_at: str | None = None
    next_assessment_window: str | None = Field(default=None, description="as published, e.g. 'December 15-18'")
    zone_source_url: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None


class FisheriesStatusIndex(_M):
    version: Literal[1]
    generated_at: str
    records: list[FisheryStatus]
    ramp: RampState | None
