"""Nearshore coverage: share of water, by distance from shore, that has a value in
C-HARM (3 km) versus a clear-day OLCI overpass (300 m).

Water mask = OLCI pixels valid on at least one day of the 14-day sample (so it is the
water OLCI can see, including the nearshore strip). Distance to shore = Euclidean
distance transform on that mask with ~250 m pixels. Approximate by construction.

usage: python nearshore_coverage.py <samples-dir>
"""
import glob, sys
import numpy as np, xarray as xr
from scipy.ndimage import distance_transform_edt

CASES = [("monterey", "olci300_s3a_CI", "2026-09-24"), ("socal_bight", "olci300_s3a_DI", "2026-10-06"), ("north_coast", "olci300_s3a_CI", "2026-10-01")]
BANDS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 99)]

def main(S):
    print("region        band_km   water_px  C-HARM_has_value  OLCI_clear_day_has_value")
    for reg, lid, day in CASES:
        fs = sorted(glob.glob(f"{S}/{reg}/{lid}_*.nc"))
        stack = []
        for f in fs:
            with xr.open_dataset(f) as d:
                stack.append(d.chlor_a.values.squeeze()); lat = d.latitude.values; lon = d.longitude.values
        water = np.any(np.isfinite(stack), axis=0)
        with xr.open_dataset(f"{S}/{reg}/{lid}_{day}.nc") as d:
            today = np.isfinite(d.chlor_a.values.squeeze())
        dist_km = distance_transform_edt(water, sampling=(0.278, 0.278 * np.cos(np.radians(lat.mean())))) # km
        ch = sorted(glob.glob(f"{S}/{reg}/charm_nowcast_*.nc"))[-1]
        with xr.open_dataset(ch) as d:
            cv = d.particulate_domoic.values.squeeze(); clat = d.latitude.values; clon = d.longitude.values - 360
        ri = np.clip(np.round((lat - clat[0]) / (clat[1] - clat[0])).astype(int), 0, len(clat) - 1)
        ci = np.clip(np.round((lon - clon[0]) / (clon[1] - clon[0])).astype(int), 0, len(clon) - 1)
        charm_ok = np.isfinite(cv[ri[:, None], ci[None, :]])
        for b0, b1 in BANDS:
            m = water & (dist_km > b0) & (dist_km <= b1)
            n = int(m.sum())
            if n:
                print(f"{reg:13s} {b0:>2}-{b1:<3}    {n:8d}   {100*charm_ok[m].mean():6.0f}%            {100*today[m].mean():6.0f}%  ({day})")
if __name__ == "__main__":
    main(sys.argv[1])
