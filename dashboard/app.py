"""
California nearshore HAB context map — decision-support dashboard (not regulatory).

Run from repo root:
  pip install -r dashboard/requirements.txt
  streamlit run dashboard/app.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_DASH_DIR = Path(__file__).resolve().parent
if str(_DASH_DIR) not in sys.path:
    sys.path.insert(0, str(_DASH_DIR))

import folium
import folium.plugins
from folium.raster_layers import ImageOverlay
import streamlit as st
from streamlit_folium import st_folium

from snapshot_utils import load_manifest, viewer_time_explanation

DASH_DIR = _DASH_DIR
DATA_DIR = DASH_DIR / "data"
FISHERIES_JSON = DASH_DIR / "fisheries_context.json"
DEFAULT_MANIFEST = DATA_DIR / "snapshot.json"
DEFAULT_OVERLAY = DATA_DIR / "overlay.png"

LANDMARKS = [
    (-124.067, 41.546, "Klamath River"),
    (-124.356, 40.718, "Eel River"),
    (-123.743, 39.128, "Navarro River"),
    (-122.974, 38.020, "Point Reyes"),
    (-122.419, 37.775, "San Francisco"),
    (-122.030, 36.974, "Santa Cruz"),
    (-121.894, 36.600, "Monterey"),
    (-120.855, 35.365, "Morro Bay"),
    (-120.471, 34.449, "Point Conception"),
]

OFFICIAL_LINKS = [
    ("Marine biotoxin updates (CDPH)", "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/MarineBiotech.aspx"),
    ("Domoic acid info (OEHHA)", "https://oehha.ca.gov/fish/general-info/domoic-acid"),
    ("West Coast harmful algae (NOAA)", "https://www.fisheries.noaa.gov/west-coast/ecosystems/harmful-algal-blooms-west-coast"),
    ("CDFW — Ocean fishing", "https://wildlife.ca.gov/Fishing/Ocean"),
]

_DEFAULT_WHAT = (
    "Colored shading is drawn **only over the ocean** (Natural Earth land mask). "
    "Colors show **relative algae abundance** (chlorophyll proxy) compared with other **water** on this map."
)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; max-width: 1280px; }
          h1 { letter-spacing: -0.02em; font-weight: 700 !important; }
          .dash-hero {
            background: linear-gradient(135deg, rgba(14,165,233,0.14) 0%, rgba(15,23,42,0.35) 55%, rgba(15,23,42,0.2) 100%);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 14px;
            padding: 1rem 1.15rem 1.05rem 1.15rem;
            margin-bottom: 0.9rem;
          }
          div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_generated_at(iso_s: str | None) -> str:
    if not iso_s:
        return "unknown"
    try:
        s = iso_s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%b %d, %Y · %H:%M UTC")
    except Exception:
        return str(iso_s)[:19] + " UTC"


def _load_fisheries() -> dict:
    if not FISHERIES_JSON.is_file():
        return {"regions": {}, "disclaimer": ""}
    try:
        return json.loads(FISHERIES_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"regions": {}, "disclaimer": ""}


def _overlay_data_uri(path: Path, strength: float) -> str:
    """Bake overlay strength into pixel alpha so Folium can use opacity=1 (avoids land tinting)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img, dtype=np.uint16)
    arr[..., 3] = np.minimum(255, (arr[..., 3].astype(np.float32) * strength).astype(np.int32))
    out = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _resolve_viewer(manifest: dict) -> dict:
    v = dict(manifest.get("viewer") or {})
    ds = manifest.get("data_source") or ""
    tl = manifest.get("time")
    if not v.get("time_headline") or not v.get("time_detail"):
        tl_str = tl if isinstance(tl, str) or tl is None else str(tl)
        h, d = viewer_time_explanation(ds, tl_str)
        v.setdefault("time_headline", h)
        v.setdefault("time_detail", d)
    v.setdefault("what_map_shows", _DEFAULT_WHAT)
    hint = manifest.get("composite_days_hint")
    if hint is not None and not v.get("composite_note"):
        n = int(hint)
        v["composite_note"] = (
            f"Each step in this dataset is often about **{n} days** of blended satellite ocean color "
            "(not a single snapshot in one hour)."
        )
    return v


def _fisherman_legend_md() -> str:
    return (
        "- **Cooler blues** — less algae signal **relative to other water on this map**\n"
        "- **Mid tones** — about average for **this** snapshot\n"
        "- **Warmer yellow–greens** — more algae biomass proxy — **not** “toxic” or “closed”\n\n"
        "**Shellfish / human health:** CDPH + OEHHA in the sidebar."
    )


@st.cache_data(show_spinner=False)
def _read_manifest(path_str: str) -> dict:
    return load_manifest(Path(path_str))


@st.cache_data(show_spinner=False)
def _cached_overlay_uri(path_str: str, strength: float) -> str:
    return _overlay_data_uri(Path(path_str), strength)


def build_map(
    manifest: dict,
    overlay_uri: str | None,
    *,
    show_landmarks: bool,
    tile_style: str,
    show_footprint: bool,
) -> folium.Map:
    b = manifest["bounds"]
    south, north = b["south"], b["north"]
    west, east = b["west"], b["east"]
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)

    tiles = "CartoDB positron" if tile_style == "Light" else "OpenStreetMap"
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=tiles, control_scale=True)

    if show_footprint:
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#38bdf8",
            weight=1,
            fill=False,
            dash_array="5 5",
            tooltip="Image extent (colors are ocean-only)",
        ).add_to(m)

    if overlay_uri:
        ImageOverlay(
            image=overlay_uri,
            bounds=[[south, west], [north, east]],
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=1,
            name="Ocean algae context",
        ).add_to(m)

    if show_landmarks:
        for lo, la, name in LANDMARKS:
            folium.CircleMarker(
                location=[la, lo],
                radius=5,
                color="#0f172a",
                weight=1,
                fill=True,
                fill_color="#f8fafc",
                fill_opacity=0.95,
                tooltip=name,
            ).add_to(m)

    folium.LatLngPopup().add_to(m)
    folium.plugins.Fullscreen(position="topright").add_to(m)
    return m


def main() -> None:
    st.set_page_config(page_title="CA Coast — Algae & trip context", layout="wide", initial_sidebar_state="expanded")
    _inject_style()

    st.markdown('<div class="dash-hero">', unsafe_allow_html=True)
    st.title("California coast — algae snapshot & trip context")
    st.caption(
        "Ocean-only map layer plus **your area** summary. "
        "Official biotoxin and fishing rules always win — this is extra context from the research grid."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    manifest_path = DEFAULT_MANIFEST
    overlay_path = DEFAULT_OVERLAY
    env_override = Path(__file__).resolve().parent.parent / ".dashboard_paths.json"
    if env_override.is_file():
        try:
            cfg = json.loads(env_override.read_text(encoding="utf-8"))
            if cfg.get("manifest"):
                manifest_path = Path(cfg["manifest"]).expanduser()
            if cfg.get("overlay"):
                overlay_path = Path(cfg["overlay"]).expanduser()
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    if not manifest_path.is_file():
        st.error("No `snapshot.json` found. Run `python dashboard/scripts/make_demo_snapshot.py` once.")
        st.stop()

    manifest = _read_manifest(str(manifest_path.resolve()))
    if not overlay_path.is_file():
        overlay_path = manifest_path.parent / manifest.get("overlay_file", "overlay.png")

    viewer = _resolve_viewer(manifest)
    fisheries = _load_fisheries()
    refresh = manifest.get("refresh_policy") or {}
    rec_days = int(refresh.get("recommended_every_days", 7))

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.metric("Map file generated", _fmt_generated_at(manifest.get("generated_at")))
    with c1:
        st.metric("Refresh cadence (typical)", f"Every {rec_days} days")
    with c2:
        lm = manifest.get("land_mask", "unknown")
        st.metric("Land mask", "Natural Earth" if "geopandas" in lm else "Coastline fallback")
    with c3:
        ds = manifest.get("data_source", "—")
        st.metric("Data layer", "Demo pattern" if ds == "synthetic_demo" else "Project export")

    with st.sidebar:
        st.markdown("### Display")
        strength = st.slider("Shading strength", 0.0, 1.0, 0.72, 0.02)
        show_landmarks = st.checkbox("Ports & river mouths", value=True)
        show_footprint = st.checkbox("Show image outline", value=False)
        tile_style = st.selectbox("Basemap", ["Light", "Default OSM"], index=0)
        st.divider()
        st.markdown("### Official sources")
        for label, url in OFFICIAL_LINKS:
            st.markdown(f"- [{label}]({url})")
        st.divider()
        with st.expander("Technical metadata"):
            st.json(
                {
                    "generated_at_utc": manifest.get("generated_at"),
                    "data_source": manifest.get("data_source"),
                    "variable": manifest.get("variable"),
                    "time_coordinate": manifest.get("time"),
                    "land_mask": manifest.get("land_mask"),
                    "bounds": manifest.get("bounds"),
                }
            )
        with st.expander("Operators — how to refresh"):
            st.markdown(refresh.get("how", "Re-run `export_map_snapshot.py` or `make_demo_snapshot.py` on a schedule."))

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown("#### Map")
        overlay_uri = None
        if overlay_path.is_file():
            overlay_uri = _cached_overlay_uri(str(overlay_path.resolve()), strength)
        m = build_map(
            manifest,
            overlay_uri,
            show_landmarks=show_landmarks,
            tile_style=tile_style,
            show_footprint=show_footprint,
        )
        st_folium(m, width=None, height=520, returned_objects=[], key="hab_map")

    with right:
        with st.container(border=True):
            st.markdown("##### When is this from?")
            st.write(viewer["time_headline"])
            st.caption(viewer["time_detail"])
            if viewer.get("composite_note"):
                st.caption(viewer["composite_note"])

        with st.container(border=True):
            st.markdown("##### What the colors mean")
            st.caption(viewer.get("what_map_shows", _DEFAULT_WHAT))
            st.markdown(_fisherman_legend_md())

        with st.container(border=True):
            st.markdown("##### Your area (pick a region)")
            reg_block = manifest.get("regional_algae") or {}
            regions_meta = reg_block.get("regions") or {}
            fish_regs = fisheries.get("regions") or {}
            keys = list(regions_meta.keys())
            if not keys:
                st.caption("No regional summary in this snapshot — re-export with the latest dashboard scripts.")
            else:
                choice = st.selectbox(
                    "Region",
                    keys,
                    format_func=lambda k: fish_regs.get(k, {}).get("headline")
                    or regions_meta.get(k, {}).get("label", k.replace("_", " ").title()),
                    label_visibility="collapsed",
                )
                rm = regions_meta.get(choice, {})
                fr = fish_regs.get(choice, {})
                if rm.get("status") == "not_enough_water_pixels":
                    st.caption(
                        "Not enough ocean pixels in this band for this file — try another region or a full coastal export."
                    )
                else:
                    tier = rm.get("tier_short") or "—"
                    st.write(f"**Algae signal vs the rest of this map:** {tier}")
                    med = rm.get("median_log_chl")
                    if med is not None and np.isfinite(med):
                        st.caption(
                            f"Median of ocean cells in this band: **{float(med):.3f}** "
                            f"(project log-chlorophyll axis — optional detail)."
                        )
                if fr:
                    st.markdown(f"**Common targets (general):** {fr.get('typical_targets', '')}")
                    st.caption(fr.get("operational_note", ""))
                    for link in fr.get("links") or []:
                        st.markdown(f"- [{link.get('label')}]({link.get('url')})")

        with st.container(border=True):
            st.markdown("##### Before you leave the dock (checklist)")
            st.markdown(
                "1. **Biotoxins / quarantines** — CDPH + OEHHA\n"
                "2. **Seasons, bag limits, gear rules** — CDFW\n"
                "3. **Weather & sea state** — NWS / local harbor\n"
                "4. **Market / price risk** — this app does **not** optimize earnings; use buyers & landing reports separately\n"
                "5. **Map** — if your patch looks **higher** than nearby water, treat it as a cue to dig deeper, not a forecast"
            )
            if fisheries.get("disclaimer"):
                st.caption(fisheries["disclaimer"])

        st.markdown(
            "**Legal / health:** not for compliance or human-health decisions. "
            "No toxin or closure prediction. You remain responsible at sea."
        )


if __name__ == "__main__":
    main()
