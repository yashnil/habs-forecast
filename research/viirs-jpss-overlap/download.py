"""Download S-NPP / NOAA-20 / NOAA-21 VIIRS chl subsets over California (stride 4).

usage: download.py <outdir> <start YYYY-MM-DD> <end YYYY-MM-DD> [snpp_only_dates_mode]
Sequential, >=1.5 s between requests, exponential backoff on 403/429/5xx.
"""
import csv, os, sys, time, datetime as dt
from urllib.parse import quote
import requests

OUT = sys.argv[1]
START = dt.date.fromisoformat(sys.argv[2])
END = dt.date.fromisoformat(sys.argv[3])
REQUIRE_SNPP = len(sys.argv) > 4 and sys.argv[4] == "require_snpp"

# Lattice-aligned bounds (cell centres verified identical on both servers)
LAT0, LAT1 = 42.49875, 32.00625   # descending on both servers
LON0, LON1 = -126.99375, -116.00625
S = 4

PRODUCTS = {
    "snpp": ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/", "erdVHNchla1day", ["chla"]),
    "n20": ("https://coastwatch.noaa.gov/erddap/griddap/", "noaacwN20VIIRSchlaSectorUSDaily", ["chlor_a"]),
    "n21": ("https://coastwatch.noaa.gov/erddap/griddap/", "noaacwN21VIIRSchlaSectorUSDaily", ["chlor_a"]),
}
UA = {"User-Agent": "research-overlap-validation/1.0 (polite; sequential)"}
sess = requests.Session()
last = [0.0]


def get(url, timeout=180):
    for attempt in range(7):
        wait = 1.5 - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        t0 = time.time()
        try:
            r = sess.get(url, headers=UA, timeout=timeout)
            code, body = r.status_code, r.content
        except Exception as e:  # network error
            code, body = -1, str(e).encode()
        last[0] = time.time()
        if code == 200:
            return code, body, attempt, time.time() - t0
        if code in (400, 404) or (code == 500 and b"index" in body):
            return code, body, attempt, time.time() - t0
        back = min(120, 5 * 2 ** attempt)
        print(f"  retry {attempt+1} after HTTP {code}; sleeping {back}s", flush=True)
        time.sleep(back)
    return code, body, attempt, time.time() - t0


def time_axis(prod):
    base, ds, _ = PRODUCTS[prod]
    code, body, *_ = get(f"{base}{ds}.csv0?time")  # full axis; filter locally (avoids 404 when START < axis min)
    if code != 200:
        raise SystemExit(f"time axis {prod} failed {code}: {body[:300]}")
    m = {}
    for line in body.decode().split():
        d = dt.date.fromisoformat(line[:10])
        if START <= d <= END:
            if d in m:
                print(f"WARNING duplicate date {prod} {d}: {m[d]} {line}")
            m.setdefault(d, line.strip())
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    axes = {p: time_axis(p) for p in PRODUCTS}
    with open(os.path.join(OUT, f"time_map_{START}_{END}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "snpp_time", "n20_time", "n21_time"])
        d = START
        while d <= END:
            w.writerow([d] + [axes[p].get(d, "") for p in PRODUCTS])
            d += dt.timedelta(days=1)
    man_path = os.path.join(OUT, "manifest.csv")
    new = not os.path.exists(man_path)
    man = open(man_path, "a", newline="")
    mw = csv.writer(man)
    if new:
        mw.writerow(["product", "date", "time_stamp", "url", "http_status", "bytes", "retries", "seconds", "retrieved_utc", "path"])
    d = START
    while d <= END:
        if REQUIRE_SNPP and d not in axes["snpp"]:
            d += dt.timedelta(days=1)
            continue
        for p, (base, ds, vars_) in PRODUCTS.items():
            ts = axes[p].get(d)
            path = os.path.join(OUT, "raw", p, f"{d}.nc")
            if ts is None:
                mw.writerow([p, d, "", "", "not_in_time_axis", 0, 0, 0, dt.datetime.utcnow().isoformat() + "Z", ""])
                continue
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                continue
            q = ",".join(
                quote(f"{v}[({ts})][(0.0)][({LAT0}):{S}:({LAT1})][({LON0}):{S}:({LON1})]", safe="():,")
                for v in vars_)
            url = f"{base}{ds}.nc?{q}"
            code, body, att, secs = get(url)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if code == 200:
                with open(path, "wb") as f:
                    f.write(body)
            mw.writerow([p, d, ts, url, code, len(body) if code == 200 else 0, att, round(secs, 1),
                         dt.datetime.utcnow().isoformat() + "Z", path if code == 200 else ""])
            man.flush()
            print(p, d, ts, code, len(body), flush=True)
        d += dt.timedelta(days=1)
    man.close()


if __name__ == "__main__":
    main()
