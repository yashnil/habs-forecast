"""Extract WCOFS surface fields for California from NOAA's public S3 bucket.

Reads only the surface level of the 4 km regular-grid output with HTTP byte ranges
(fsspec + h5netcdf), so a 582 MB file costs a few MB. Logs bytes transferred and time.

usage: python fetch_wcofs.py <out-dir> <YYYYMMDD> [hours...]
"""
import json, sys, time
from pathlib import Path
import fsspec, numpy as np, xarray as xr

BUCKET = "https://noaa-nos-ofs-pds.s3.amazonaws.com/wcofs/netcdf"
LAT, LON = (32.0, 42.6), (-126.5, -117.0)

def main(out, day, hours):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    log = []
    fs = fsspec.filesystem("https", block_size=2 * 2**20)
    for h in hours:
        tag = f"n{h:03d}" if h <= 0 else f"f{h:03d}"
        if h <= 0: tag = f"n{24 + h:03d}" if h < 0 else "n024"
        url = f"{BUCKET}/{day[:4]}/{day[4:6]}/{day[6:]}/wcofs.t03z.{day}.regulargrid.{tag}.nc"
        t0 = time.time(); rec = dict(url=url)
        try:
            with fs.open(url) as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                lat, lon = ds.Latitude.values, ds.Longitude.values
                rows = np.where((lat[:, 0] >= LAT[0]) & (lat[:, 0] <= LAT[1]))[0]
                cols = np.where((lon[0] >= LON[0]) & (lon[0] <= LON[1]))[0]
                sl = dict(ny=slice(rows[0], rows[-1] + 1), nx=slice(cols[0], cols[-1] + 1))
                sub = ds[["u_eastward", "v_northward", "temp", "salt"]].isel(Depth=0, **sl).load()
                sub = sub.assign_coords(lat=("ny", lat[sl["ny"], 0]), lon=("nx", lon[0, sl["nx"]]))
                sub.attrs.update(source=url, valid_time=str(ds.time.values[0]))
                p = out / f"wcofs_{day}_{tag}.nc"; sub.to_netcdf(p)
            rec.update(valid_time=sub.attrs["valid_time"], seconds=round(time.time() - t0, 1), shape=list(sub.u_eastward.shape),
                       bytes_read=getattr(fs, "_bytes", None), out_bytes=p.stat().st_size,
                       ocean_cells=int(np.isfinite(sub.u_eastward.values).sum()))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        log.append(rec); print(json.dumps(rec), flush=True)
    (out / f"wcofs-log-{day}.json").write_text(json.dumps(log, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])
