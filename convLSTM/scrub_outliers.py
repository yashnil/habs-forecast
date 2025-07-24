#!/usr/bin/env python
"""
scrub_outliers.py
-----------------
Replace extreme outliers in HAB_convLSTM_core_v1.nc with the spatial median
of the surrounding 3×3 neighbourhood (per time-step).

Variables cleaned:
    * log_chl   – outliers detected in *linear* chlorophyll space
    * Kd_490
    * nflh

Outputs:
    * HAB_convLSTM_core_v1_clean.nc  (default)
    * *_outmask data variables marking replaced pixels
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# ------------------------ configuration ---------------------------------- #

CLEAN_VARS = ["log_chl", "Kd_490", "nflh"]   # edit if you add more
CHUNK_T    = 64                              # good enough for a one-off pass
OUT_SUFFIX = "_clean"

# -------------------- helper functions ----------------------------------- #

_FLOOR = 0.056616  # detection floor used in log_chl creation

def _to_linear_chl(log_chl):
    return np.exp(log_chl) - _FLOOR

def _from_linear_chl(chl_lin):
    return np.log(chl_lin + _FLOOR)

def _compute_thresh(da, q):
    """Return the q-quantile (e.g. 0.999) as a scalar."""
    return da.quantile(q).compute().item()

def _scrub_var(ds, var, q):
    is_log_chl = var == "log_chl"
    raw = ds[var]

    # ---- convert *once* (with Dask enabled) ------------------------------
    if is_log_chl:
        lin = xr.apply_ufunc(
            _to_linear_chl,
            raw,
            dask="parallelized",
            vectorize=True,
            output_dtypes=[raw.dtype],
        )
    else:
        lin = raw  # Kd_490, nflh: use native units

    # ---- detect outliers -------------------------------------------------
    thresh = _compute_thresh(lin, q)
    mask   = lin > thresh

    # ---- 3×3 spatial median replacement ---------------------------------
    median3 = raw.rolling(lat=3, lon=3, center=True).median()
    cleaned = raw.where(~mask, median3).where(~np.isnan(median3), raw)

    return cleaned, mask

# ----------------------------- main --------------------------------------- #

def main(infile, outfile, q):
    ds = xr.open_dataset(infile, chunks={"time": CHUNK_T})

    updates, masks = {}, {}
    for var in CLEAN_VARS:
        cleaned, mask = _scrub_var(ds, var, q)
        updates[var] = cleaned
        masks[f"{var}_outmask"] = mask.astype("uint8")  # 1 = outlier replaced

        n_replaced = int(mask.sum().compute())
        print(f"{var:8s}: replaced {n_replaced:,} pixels (q>{q})")

    ds_clean = ds.assign(**updates, **masks)
    ds_clean.to_netcdf(outfile)
    print(f"\n✅  Cleaned file written to {outfile}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("infile",
                   nargs="?",
                   default="HAB_convLSTM_core_v1.nc",
                   help="input NetCDF file")
    p.add_argument("-o", "--outfile",
                   default=None,
                   help="output NetCDF (default: <infile>_clean.nc)")
    p.add_argument("--quantile",
                   type=float,
                   default=0.999,
                   help="upper quantile defining outliers (default 0.999)")
    args = p.parse_args()

    infile  = Path(args.infile)
    outfile = Path(args.outfile or infile.with_stem(infile.stem + OUT_SUFFIX))
    main(infile, outfile, args.quantile)

'''
To run the code:

python notebooks/scrub_outliers.py \
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1.nc" \
    -o "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc"
'''