"""Shared helpers for map overlay PNG + manifest (dashboard)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def viewer_time_explanation(data_source: str, time_label: Optional[str]) -> Tuple[str, str]:
    """
    Return (short headline, plain-language detail) for non-technical users.

    headline: what to show under "When is this from?"
    detail:   what the timestamp means / caveats
    """
    ds = (data_source or "").lower()
    tl = (time_label or "").strip()

    if ds == "synthetic_demo" or tl.lower() in ("demo-static", "demo", ""):
        return (
            "Sample map — not today’s ocean",
            "This is a built-in pattern so you can see how the map works. "
            "It is **not** a real satellite image and **not** current conditions. "
            "For real use, export a recent layer from the project NetCDF (see “Technical details”).",
        )

    # numpy / ISO datetime strings
    pretty = tl
    try:
        t = np.datetime64(tl)
        day = str(t.astype("datetime64[D]"))
        dt = datetime.strptime(day, "%Y-%m-%d")
        pretty = f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except Exception:
        try:
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", tl)
            if m:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                pretty = f"{y}-{mo:02d}-{d:02d}"
        except Exception:
            pass

    headline = f"Data snapshot: {pretty}"
    detail = (
        "This is **one time step** from your file (the date above). "
        "In the research dataset, each step is often an **~8‑day ocean‑color composite**, not a single afternoon snapshot. "
        "Warmer colors mean **more chlorophyll (algae biomass proxy)** in the water — **not** domoic acid or shellfish safety."
    )
    return headline, detail


def logchl_to_rgba(
    z: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str = "YlGnBu",
    nan_alpha: int = 0,
) -> np.ndarray:
    """Map scalar grid to HxWx4 uint8 RGBA; NaNs transparent."""
    z = np.asarray(z, dtype=np.float64)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(np.nan_to_num(z, nan=np.nan)))
    rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
    bad = ~np.isfinite(z)
    rgba[..., 3] = np.where(bad, nan_alpha, 255)
    return rgba


def write_overlay_png(rgba: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
    except ImportError as e:
        raise ImportError("Pillow is required to write overlay PNGs") from e


def build_manifest(
    *,
    bounds: Tuple[float, float, float, float],
    time_label: Optional[str],
    variable: str,
    data_source: str,
    vmin: float,
    vmax: float,
    cmap: str,
    notes: str = "",
    overlay_filename: str = "overlay.png",
    composite_days_hint: Optional[int] = None,
    regional_algae: Optional[Dict[str, Any]] = None,
    land_mask: str = "natural_earth_geopandas",
) -> Dict[str, Any]:
    south, west, north, east = bounds
    headline, time_detail = viewer_time_explanation(data_source, time_label)
    out: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "california_focus": True,
        "data_source": data_source,
        "variable": variable,
        "time": time_label,
        "viewer": {
            "time_headline": headline,
            "time_detail": time_detail,
            "what_map_shows": (
                "Colored shading is clipped to the **ocean** using a global land outline (Natural Earth via GeoPandas when available). "
                "Colors rank **relative algae abundance** (chlorophyll) compared with other **water** pixels on this map — "
                "useful as **bloom context**, not as a fishing guarantee or health warning."
            ),
        },
        "bounds": {"south": south, "west": west, "north": north, "east": east},
        "colormap": {"name": cmap, "vmin": vmin, "vmax": vmax},
        "overlay_file": overlay_filename,
        "notes": notes,
        "land_mask": land_mask,
        "refresh_policy": {
            "recommended_every_days": 7,
            "what_to_refresh": ["overlay.png", "snapshot.json"],
            "how": (
                "Run `python dashboard/scripts/export_map_snapshot.py` on your latest NetCDF (cron / GitHub Actions / cloud scheduler). "
                "Daily–weekly is typical for satellite composites."
            ),
        },
    }
    if regional_algae is not None:
        out["regional_algae"] = regional_algae
    if composite_days_hint is not None:
        out["composite_days_hint"] = int(composite_days_hint)
        out["viewer"]["composite_note"] = (
            f"Each step in this dataset is often about **{int(composite_days_hint)} days** of blended satellite "
            "ocean color (not a single snapshot in one hour)."
        )
    return out


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
