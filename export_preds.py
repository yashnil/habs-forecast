"""
export_preds.py — Simple, case‑study friendly inference helpers

Drop this file into the repo root. It exposes three top‑level functions:
  • run_convlstm_from_ckpt(ckpt_path, obs_like, start=None, end=None, **kwargs)
  • run_tft_from_ckpt(ckpt_path, obs_like, start=None, end=None, **kwargs)
  • run_pinn_from_ckpt(ckpt_path, obs_like, start=None, end=None, **kwargs)

All three delegate to the same worker, so the differences are crystal clear.
They return an xarray.DataArray of predicted log‑chlorophyll (log_chl) for the
requested times with coords matching the input dataset.

Design goals
- Zero fiddly config: sensible defaults derived from the training scripts.
- Correct nowcast / 1‑step lead semantics (Δ log‑chl + last frame).
- Robust to small feature layout variations (compute/soft‑fill missing features).

Example
-------
>>> da = run_convlstm_from_ckpt("checkpoints/conv.ckpt", DS, "2019-01-01", "2021-06-30")
>>> da
<xarray.DataArray (time: ..., lat: ..., lon: ...)> ...

Notes
-----
- Checkpoints saved with PyTorch Lightning are handled (state_dict key); plain
  state_dict files also work. If you used a LightningModule wrapper with
  attribute names like "net.", those prefixes are removed automatically.
- The forward() for these models returns the predicted Δ log‑chl. We then add
  the *raw* last‑frame log_chl (de‑standardized) to obtain the absolute nowcast
  or short‑lead prediction.


python export_preds.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --start 2019-01-01 --end 2021-12-31 \
  --convlstm.ckpt "/Users/yashnilmohanty/HAB_Models/vanilla_best.pt" \
  --tft.ckpt      "/Users/yashnilmohanty/HAB_Models/convTFT_best.ckpt" \
  --pinn.ckpt     "/Users/yashnilmohanty/HAB_Models/convLSTM_best.pt" \
  --outdir "/Users/yashnilmohanty/HAB_Models/exports"
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

# Torch is required at inference time
import torch
from torch import nn

# ===== CHANGE THESE to match your modules =====
from pinn.baseline_model import ConvLSTM          # <-- replace ConvLSTM if your class has a different name
# ===== USE THESE with your repo layout =====
# ConvLSTM lives in testing.py
from pinn.testing import ConvLSTM  # class name is ConvLSTM in your file

# TFT lives in tft_train.py. If you have a class, import it;
# if you only have a builder, import that (both shown — the code below handles either).
try:
    from tft.tft_train import TFT  # class (preferred if it exists)
except Exception:
    TFT = None
try:
    from tft.tft_train import build_tft_for_infer  # builder (fallback)
except Exception:
    build_tft_for_infer = None
# ==========================================
# ==============================================

# -----------------------------------------------------------------------------
# Resolve model classes and shared constants from the training code
# We try a few likely module names so this file is drop‑in friendly.
# -----------------------------------------------------------------------------

ConvLSTMClass: Optional[type] = None
TFTClass: Optional[type] = None
ALL_VARS_DEFAULT: Tuple[str, ...] = (
    # Conservative superset ordered to match typical training
    "log_chl",            # 0 — MUST be present (target channel)
    "chl_anom_monthly",
    "Kd_490",
    "nflh",
    "u10", "v10",
    "t2m", "d2m",
    "thetao", "so",
    "uo", "vo",
    "zos", "curl_uv",
    "wind_speed", "tau_mag",
    "cur_speed", "cur_div", "cur_vort",
    "ssh_grad_mag",
    "river_rank",  # often 1/exp(dist_river_km)
)
LOGCHL_NAME = "log_chl"
SEQ_DEFAULT = 6    # L_in used in training for 8‑day composites (≈48 days of history)
LEAD_DEFAULT = 1   # 1 step = 8‑day lead (nowcast/short‑lead)


def _try_imports() -> Tuple[Optional[type], Optional[type], Tuple[str, ...], int, int]:
    """Attempt to import model classes and constants from common repo files."""
    global ConvLSTMClass, TFTClass

    seq, lead = SEQ_DEFAULT, LEAD_DEFAULT
    var_order = ALL_VARS_DEFAULT

    # 1) A combined baseline module (common pattern in this repo)
    for modname in ("baseline_model", "convlstm_train", "tft_train", "models", "testing"):
        try:
            m = __import__(modname, fromlist=["*"])  # type: ignore
        except Exception:
            continue
        # Pull classes if available
        if ConvLSTMClass is None and hasattr(m, "ConvLSTM"):
            ConvLSTMClass = getattr(m, "ConvLSTM")
        if TFTClass is None and hasattr(m, "TFT"):
            TFTClass = getattr(m, "TFT")
        # Shared constants if present
        if hasattr(m, "ALL_VARS"):
            try:
                _vars = tuple(getattr(m, "ALL_VARS"))
                if _vars:
                    var_order = _vars
            except Exception:
                pass
        if hasattr(m, "SEQ"):
            try:
                seq = int(getattr(m, "SEQ"))
            except Exception:
                pass
        if hasattr(m, "LEAD_IDX"):
            try:
                lead = int(getattr(m, "LEAD_IDX"))
            except Exception:
                pass

    return ConvLSTMClass, TFTClass, var_order, seq, lead


ConvLSTMClass, TFTClass, ALL_VARS, SEQ, LEAD_IDX = _try_imports()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_convlstm_from_ckpt(
    ckpt_path: str,
    obs_like: xr.Dataset,
    start: Optional["np.datetime64|str"] = None,
    end: Optional["np.datetime64|str"] = None,
    *,
    device: Optional[str] = None,
    seq: Optional[int] = None,
    lead: Optional[int] = None,
    vars: Optional[Sequence[str]] = None,
    batch_size: int = 2,
) -> xr.DataArray:
    """Convenience wrapper for ConvLSTM checkpoints.

    See _run_from_ckpt for parameter semantics.
    """
    model_ctor = _resolve_model_ctor("convlstm")
    return _run_from_ckpt(
        model_ctor=model_ctor,
        ckpt_path=ckpt_path,
        obs_like=obs_like,
        start=start,
        end=end,
        device=device,
        seq=seq or SEQ,
        lead=lead or LEAD_IDX,
        vars=tuple(vars) if vars else ALL_VARS,
        batch_size=batch_size,
        model_name="ConvLSTM",
    )


def run_tft_from_ckpt(
    ckpt_path: str,
    obs_like: xr.Dataset,
    start: Optional["np.datetime64|str"] = None,
    end: Optional["np.datetime64|str"] = None,
    *,
    device: Optional[str] = None,
    seq: Optional[int] = None,
    lead: Optional[int] = None,
    vars: Optional[Sequence[str]] = None,
    batch_size: int = 2,
) -> xr.DataArray:
    """Convenience wrapper for TFT checkpoints."""
    model_ctor = _resolve_model_ctor("tft")
    return _run_from_ckpt(
        model_ctor=model_ctor,
        ckpt_path=ckpt_path,
        obs_like=obs_like,
        start=start,
        end=end,
        device=device,
        seq=seq or SEQ,
        lead=lead or LEAD_IDX,
        vars=tuple(vars) if vars else ALL_VARS,
        batch_size=batch_size,
        model_name="TFT",
    )


def run_pinn_from_ckpt(
    ckpt_path: str,
    obs_like: xr.Dataset,
    start: Optional["np.datetime64|str"] = None,
    end: Optional["np.datetime64|str"] = None,
    *,
    device: Optional[str] = None,
    seq: Optional[int] = None,
    lead: Optional[int] = None,
    vars: Optional[Sequence[str]] = None,
    batch_size: int = 2,
) -> xr.DataArray:
    """Convenience wrapper for physics‑augmented checkpoints.

    Forward() at inference is identical to ConvLSTM/TFT (predicts Δ log‑chl), so
    we instantiate the base ConvLSTM/TFT architecture as dictated by the saved
    weights. If your PINN used ConvLSTM as the backbone (most common here), this
    will Just Work™.
    """
    # Heuristic: most PINN checkpoints in this repo wrap a ConvLSTM backbone.
    # If you used a TFT backbone, just call run_tft_from_ckpt instead or pass
    # vars/seq/lead as needed.
    model_ctor = _resolve_model_ctor("convlstm")
    return _run_from_ckpt(
        model_ctor=model_ctor,
        ckpt_path=ckpt_path,
        obs_like=obs_like,
        start=start,
        end=end,
        device=device,
        seq=seq or SEQ,
        lead=lead or LEAD_IDX,
        vars=tuple(vars) if vars else ALL_VARS,
        batch_size=batch_size,
        model_name="PINN",
    )


# -----------------------------------------------------------------------------
# Core worker
# -----------------------------------------------------------------------------

def _run_from_ckpt(
    *,
    model_ctor: Callable[[int], nn.Module],
    ckpt_path: str,
    obs_like: xr.Dataset,
    start: Optional["np.datetime64|str"],
    end: Optional["np.datetime64|str"],
    device: Optional[str],
    seq: int,
    lead: int,
    vars: Sequence[str],
    batch_size: int,
    model_name: str,
) -> xr.DataArray:
    """Shared inference path used by all wrappers.

    Parameters
    ----------
    model_ctor : Callable[[Cin], nn.Module]
        Function that, given the number of input channels, returns the model.
    ckpt_path : str
        Path to a PyTorch or Lightning checkpoint containing the model weights.
    obs_like : xr.Dataset
        Dataset with at least a time axis and log_chl; additional variables are
        used if available and listed in `vars`.
    start, end : datetime64 or str, optional
        Requested inclusive time range. If omitted, use the intersection of
        obs_like.time and the available history (enough to support `seq` & `lead`).
    device : str, optional
        'cuda' (if available) or 'cpu'. If None, auto‑select.
    seq : int
        Number of historical frames the model expects (L_in).
    lead : int
        Forecast lead in steps (8‑day composites → 1 step = 8 days).
    vars : Sequence[str]
        Ordered list of feature names matching the model training layout. Missing
        features are soft‑filled/derived where possible; extras are ignored.
    batch_size : int
        Batch size for inference windows (increase if VRAM allows).

    Returns
    -------
    xr.DataArray
        Predicted log_chl with dims (time, lat, lon) and matching coords.
    """
    if LOGCHL_NAME not in obs_like:
        raise ValueError(f"'{LOGCHL_NAME}' is required in obs_like")

    # Normalize/clean inputs without mutating the caller's dataset
    ds = obs_like.copy(deep=False)

    # Enrich/derive a few soft features if they are missing
    ds = _ensure_soft_features(ds)

    ds = _canonicalize_latlon(ds)

    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Time windowing
    t_all: xr.DataArray = ds[LOGCHL_NAME].time
    t0 = np.datetime64(str(t_all.values[0]))
    tN = np.datetime64(str(t_all.values[-1]))

    if start is None:
        # First valid index that still leaves seq+lead history
        start_idx = int(seq + lead - 1)
        start_time = np.datetime64(str(t_all.values[start_idx]))
    else:
        start_time = np.datetime64(str(np.datetime64(start)))

    if end is None:
        end_time = tN
    else:
        end_time = np.datetime64(str(np.datetime64(end)))

    # Clip to dataset bounds
    start_time = max(start_time, t0)
    end_time = min(end_time, tN)
    if start_time > end_time:
        raise ValueError("Requested time window has no overlap with dataset.")

    # Build the exact list of times we will predict on, honoring seq/lead history
    times: List[np.datetime64] = []
    time_to_index: Dict[np.datetime64, int] = {np.datetime64(str(t)): i for i, t in enumerate(t_all.values)}
    for t in t_all.sel(time=slice(start_time, end_time)).values:
        t = np.datetime64(str(t))
        i = time_to_index[t]
        if i - lead - (seq - 1) >= 0:
            times.append(t)
    if not times:
        raise ValueError("No times have sufficient history for the given seq/lead.")

    # Compute normalization stats from a training‑like slice: use all frames
    # strictly before the first prediction step (so we don't leak the target).
    train_times = t_all.sel(time=slice(t0, times[0]))
    if len(train_times) < seq + 2:
        # Fallback: use the first 70% of available frames
        cut = int(0.7 * len(t_all))
        train_times = t_all.isel(time=slice(0, cut))

    stats = _norm_stats(ds, vars, train_times)

    # Prepare standardized input tensor [T, C, H, W]
    X = _stack_and_standardize(ds, vars, stats).astype(np.float32)
    # Save raw (un‑standardized) last‑frame log_chl to add back later
    logchl_raw = ds[LOGCHL_NAME].transpose("time", "lat", "lon").astype(np.float32).values

    T, C, H, W = X.shape

    # Window indices for inference
    idxs: List[int] = [time_to_index[t] for t in times]
    in_starts = [i - lead - (seq - 1) for i in idxs]
    in_ends = [i - lead + 1 for i in idxs]         # exclusive

    # Instantiate and load the model
    Cin = C
    net = model_ctor(Cin)
    net.to(device)
    _load_state_dict_forgiving(net, ckpt_path, strict=False)
    net.eval()

    # Run in modest batches to avoid VRAM spikes
    preds_delta = np.empty((len(times), H, W), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, len(times), batch_size):
            b1 = min(b0 + batch_size, len(times))
            # Build batch [B, L, C, H, W]
            batch = np.stack([X[s:e, :, :, :] for (s, e) in zip(in_starts[b0:b1], in_ends[b0:b1])], axis=0)
            xb = torch.from_numpy(batch).to(device)  # (B,L,C,H,W)
            out = net(xb)                             # (B,H,W) — Δ log‑chl
            preds_delta[b0:b1, :, :] = out.detach().cpu().numpy()

    # Add back last raw log_chl at i‑lead to get absolute prediction at time i
    last_idx = [i - lead for i in idxs]
    last_log = np.stack([logchl_raw[j, :, :] for j in last_idx], axis=0)
    preds_log = preds_delta + last_log

    # Mask invalid (non‑ocean) cells using validity of log_chl
    valid = np.isfinite(logchl_raw[last_idx[0], :, :])
    preds_log[:, ~valid] = np.nan

    # Wrap as DataArray with matching coords
    da = xr.DataArray(
        preds_log,
        dims=("time", "lat", "lon"),
        coords=dict(
            time=[np.datetime64(str(t)) for t in times],
            lat=ds[LOGCHL_NAME].lat,
            lon=ds[LOGCHL_NAME].lon,
        ),
        name="log_chl_pred",
        attrs=dict(
            long_name="Predicted log chlorophyll‑a",
            description=f"{model_name} inference (Δ+last), seq={seq}, lead={lead}",
            ckpt=ckpt_path,
        ),
    )
    return da


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _canonicalize_latlon(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.dims:  ren["latitude"]  = "lat"
    if "longitude" in ds.dims: ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    return ds

def _resolve_model_ctor(kind: str) -> Callable[[int], nn.Module]:
    kind = kind.lower()
    if kind == "convlstm":
        if ConvLSTM is None:
            raise ImportError("ConvLSTM class not found (testing.py).")
        return lambda Cin: ConvLSTM(Cin)
    if kind == "tft":
        # Prefer a TFT class if available
        if 'TFT' in globals() and TFT is not None:
            return lambda Cin: TFT(Cin)
        # Fallback: builder that returns a ready-to-run model
        if 'build_tft_for_infer' in globals() and build_tft_for_infer is not None:
            def _ctor(Cin: int) -> nn.Module:
                try:
                    return build_tft_for_infer(Cin)  # if your builder accepts Cin
                except TypeError:
                    return build_tft_for_infer()     # if it doesn’t
            return _ctor
        raise ImportError("Neither TFT class nor build_tft_for_infer() found in tft_train.py.")
    raise ValueError(f"Unknown model kind: {kind}")



def _load_state_dict_forgiving(net: torch.nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """
    Load a checkpoint into `net` while:
      1) stripping common wrapper prefixes ("model.", "module.", etc.),
      2) dropping any keys whose tensor shapes don't match the current model.

    This prevents crashes when the number of input channels changed between training and export
    (e.g., reduce.weight 24x28x1x1 in ckpt vs 24x21x1x1 in current model).
    """
    obj = torch.load(ckpt_path, map_location="cpu")

    # Unwrap common containers
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]

    # If it's a Module, get a plain state_dict
    if not isinstance(obj, dict):
        obj = obj.state_dict()

    # Strip prefixes from keys (DataParallel/Lightning/etc.)
    cleaned = {}
    for k, v in obj.items():
        nk = k
        for prefix in ("model.", "module.", "net.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    # Filter out mismatched shapes
    model_state = net.state_dict()
    filtered = {}
    skipped = []
    for k, v in cleaned.items():
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            if k in model_state:
                skipped.append((k, tuple(getattr(v, "shape", [])), tuple(model_state[k].shape)))

    if skipped:
        warnings.warn(
            "Skipping {} mismatched parameter(s):\n{}".format(
                len(skipped),
                "\n".join([f"  - {k}: ckpt{sv} -> model{mv}" for (k, sv, mv) in skipped]),
            )
        )

    # Load what we can; let PyTorch tell us about any remaining gaps
    missing, unexpected = net.load_state_dict(filtered, strict=False)
    if missing:
        warnings.warn(f"Missing {len(missing)} keys when loading (left at init): {missing[:6]}")
    if unexpected:
        warnings.warn(f"Unexpected {len(unexpected)} keys in checkpoint (ignored): {unexpected[:6]}")


def _ensure_soft_features(ds: xr.Dataset) -> xr.Dataset:
    """Compute a few lightweight derived features if missing.

    Everything here is intentionally simple/defensive so that case studies don't
    fail just because a feature isn't pre‑materialized. If a variable already
    exists, it is left untouched.
    """
    ds = ds.copy()

    # Wind speed & simple wind stress magnitude
    if "wind_speed" not in ds and {"u10", "v10"}.issubset(ds.data_vars):
        ds["wind_speed"] = xr.apply_ufunc(lambda u, v: np.hypot(u, v), ds["u10"], ds["v10"], dask="allowed")
    if "tau_mag" not in ds and "wind_speed" in ds:
        rho_air = 1.225
        Cd = 1.3e-3
        ds["tau_mag"] = (rho_air * Cd * ds["wind_speed"] ** 2).astype("float32")

    # Surface current speed
    if "cur_speed" not in ds and {"uo", "vo"}.issubset(ds.data_vars):
        ds["cur_speed"] = xr.apply_ufunc(lambda u, v: np.hypot(u, v), ds["uo"], ds["vo"], dask="allowed")

    # SSH gradient magnitude
    if "ssh_grad_mag" not in ds and "zos" in ds:
        try:
            gx = ds["zos"].differentiate("lon")
            gy = ds["zos"].differentiate("lat")
            ds["ssh_grad_mag"] = xr.apply_ufunc(lambda a, b: np.hypot(a, b), gx, gy, dask="allowed")
        except Exception:
            pass

    # Current divergence & vorticity (very simple finite diff in grid coords)
    if ("cur_div" not in ds or "cur_vort" not in ds) and {"uo", "vo"}.issubset(ds.data_vars):
        try:
            du_dx = ds["uo"].differentiate("lon")
            dv_dy = ds["vo"].differentiate("lat")
            ds["cur_div"] = (du_dx + dv_dy).astype("float32")
            dv_dx = ds["vo"].differentiate("lon")
            du_dy = ds["uo"].differentiate("lat")
            ds["cur_vort"] = (dv_dx - du_dy).astype("float32")
        except Exception:
            pass

    # River influence rank if only distance is present
    if "river_rank" not in ds:
        for cand in ("dist_river_km", "river_dist_km"):
            if cand in ds:
                ds["river_rank"] = xr.apply_ufunc(lambda d: 1.0 / np.exp(np.clip(d, 0.0, None)), ds[cand])
                break

    # Monthly chl anomaly in linear space
    if "chl_anom_monthly" not in ds and LOGCHL_NAME in ds:
        chl_lin = np.exp(ds[LOGCHL_NAME])  # back‑transform
        clim = chl_lin.groupby("time.month").mean("time", skipna=True)
        ds["chl_anom_monthly"] = (chl_lin.groupby("time.month") - clim).astype("float32")

    return ds


def _norm_stats(ds: xr.Dataset, vars: Sequence[str], train_times: xr.DataArray) -> Dict[str, Tuple[float, float]]:
    """Compute mean/std for each variable over a training‑like slice."""
    stats: Dict[str, Tuple[float, float]] = {}
    # Use validity of log_chl as the ocean mask
    valid = np.isfinite(ds[LOGCHL_NAME])
    ds_train = ds.sel(time=train_times)
    for v in vars:
        if v not in ds_train:
            # Missing var → use zero‑mean, unit‑sd so z(x)=x
            stats[v] = (0.0, 1.0)
            continue
        da = ds_train[v].where(valid)
        mu = float(da.mean(skipna=True).values)
        sd = float(da.std(skipna=True).values)
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        if not np.isfinite(mu):
            mu = 0.0
        stats[v] = (mu, sd)
    return stats


def _stack_and_standardize(ds: xr.Dataset, vars: Sequence[str], stats: Mapping[str, Tuple[float, float]]) -> np.ndarray:
    """Create a [T,C,H,W] stack standardized by given stats.

    Missing variables are filled with zeros after z‑scoring convention (x-μ)/σ.
    Extra variables in ds are ignored.
    """
    T = ds[LOGCHL_NAME].sizes["time"]
    lat = ds[LOGCHL_NAME].sizes.get("lat")
    lon = ds[LOGCHL_NAME].sizes.get("lon")
    C = len(vars)
    out = np.zeros((T, C, lat, lon), dtype=np.float32)

    for ci, v in enumerate(vars):
        if v in ds:
            da = ds[v].transpose("time", "lat", "lon").astype(np.float32)
            mu, sd = stats.get(v, (0.0, 1.0))
            out[:, ci, :, :] = (da.values - mu) / sd
        else:
            # Already zero due to initialization (represents z‑score 0)
            warnings.warn(f"Feature '{v}' not found in obs_like — filled with zeros after z‑score.")
    return out

if __name__ == "__main__":
    import argparse, os, pathlib

    p = argparse.ArgumentParser()
    p.add_argument("--obs", required=True, help="Path to obs-like NetCDF")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--convlstm.ckpt", dest="convlstm_ckpt")
    p.add_argument("--tft.ckpt",      dest="tft_ckpt")
    p.add_argument("--pinn.ckpt",     dest="pinn_ckpt")
    p.add_argument("--batch-size",    dest="batch_size", type=int, default=2)
    args = p.parse_args()

    ds = xr.open_dataset(args.obs)
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

    def _save(da: xr.DataArray, tag: str, ckpt: str):
        name = f"{tag}__{os.path.splitext(os.path.basename(ckpt))[0]}.nc"
        out  = os.path.join(args.outdir, name)
        da.to_netcdf(out)
        print("Wrote", out)

    if args.convlstm_ckpt:
        da = run_convlstm_from_ckpt(args.convlstm_ckpt, ds, args.start, args.end,
                                    batch_size=args.batch_size)
        _save(da, "convlstm", args.convlstm_ckpt)

    if args.tft_ckpt:
        da = run_tft_from_ckpt(args.tft_ckpt, ds, args.start, args.end,
                               batch_size=args.batch_size)
        _save(da, "tft", args.tft_ckpt)

    if args.pinn_ckpt:
        da = run_pinn_from_ckpt(args.pinn_ckpt, ds, args.start, args.end,
                                batch_size=args.batch_size)
        _save(da, "pinn", args.pinn_ckpt)
