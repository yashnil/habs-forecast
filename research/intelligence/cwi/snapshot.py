"""Load a recorded CoastWatch data snapshot (the published ``v1/`` files).

A snapshot is a directory holding ``manifest.json`` plus the files it names
(official notices, CalHABMAP observations, port intel, fisheries, ports).
Nothing here touches the network or writes anything.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import date, datetime

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = HERE.parent / "snapshots" / "2026-10-13"


@dataclass
class Snapshot:
    root: pathlib.Path
    manifest: dict
    official: dict
    observations: dict
    port_intel: dict
    fisheries: dict
    ports_geo: dict
    generated_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.generated_at = datetime.fromisoformat(self.manifest["generated_at"].replace("Z", "+00:00"))

    @property
    def as_of(self) -> date:
        return self.generated_at.date()

    # ---- convenience views -------------------------------------------------
    @property
    def records(self) -> list[dict]:
        return self.official["registry"]["records"]

    @property
    def statements(self) -> list[dict]:
        return self.official["registry"]["statements"]

    @property
    def review(self) -> dict:
        return self.official["registry"]["review"]

    @property
    def ports(self) -> list[dict]:
        return self.port_intel["ports"]

    @property
    def stations(self) -> list[dict]:
        return self.observations["stations"]

    def disputed_record_ids(self) -> dict[str, str]:
        """Record id -> conflict text, for records an agency source appears to contradict."""
        out: dict[str, str] = {}
        for text in self.official.get("conflicts", []):
            for rec in self.records:
                if rec["id"] in text:
                    out[rec["id"]] = text
        return out


def load_snapshot(root: str | pathlib.Path = DEFAULT_SNAPSHOT) -> Snapshot:
    root = pathlib.Path(root)
    manifest = json.loads((root / "manifest.json").read_text())

    def load(key: str) -> dict:
        return json.loads((root / manifest[key]).read_text())

    return Snapshot(
        root=root,
        manifest=manifest,
        official=load("official_url"),
        observations=load("observations_url"),
        port_intel=load("port_intel_url"),
        fisheries=load("fisheries_url"),
        ports_geo=load("ports_url"),
    )
