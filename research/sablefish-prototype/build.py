import json, pathlib, pandas as pd, numpy as np
D = pathlib.Path(__file__).parent
h = pd.DataFrame(json.loads((D/"raw/haul.json").read_text()))
c = pd.DataFrame(json.loads((D/"raw/catch_sablefish.json").read_text()))
print("haul rows", len(h), "catch rows", len(c), "dup trawl_id in catch", c.trawl_id.duplicated().sum(), "dup in haul", h.trawl_id.duplicated().sum())
print(h.performance.value_counts(dropna=False)); print(h.station_invalid.value_counts(dropna=False))
print(c.scientific_name.value_counts())
steps = {}
steps["all hauls 2003-2025"] = len(h)
h = h[h.latitude_dd.notna()]; steps["non-null lat"] = len(h)
h = h[h.latitude_dd < 42]; steps["California (lat<42)"] = len(h)
h = h[h.performance == "Satisfactory"]; steps["performance==Satisfactory"] = len(h)
h = h[h.station_invalid == 0]; steps["station_invalid==0"] = len(h)
h = h[h.area_swept_ha_der.notna() & (h.area_swept_ha_der > 0)]; steps["area_swept>0"] = len(h)
c = c.groupby("trawl_id", as_index=False).agg(total_catch_wt_kg=("total_catch_wt_kg","sum"), total_catch_numbers=("total_catch_numbers","sum"), cpue_kg_per_ha_der=("cpue_kg_per_ha_der","sum"))
df = h.merge(c, on="trawl_id", how="left")
for k in ["total_catch_wt_kg","total_catch_numbers","cpue_kg_per_ha_der"]: df[k] = df[k].fillna(0.0)
df["cpue_check"] = df.total_catch_wt_kg / df.area_swept_ha_der
print("cpue vs wt/area max abs diff", (df.cpue_check - df.cpue_kg_per_ha_der).abs().max())
df["date"] = pd.to_datetime(df.datetime_utc_iso)
df["doy"] = df.date.dt.dayofyear
df["present"] = (df.total_catch_wt_kg > 0).astype(int)
df["log1p_cpue"] = np.log1p(df.cpue_kg_per_ha_der)
df = df.drop(columns=["cpue_check","project"])
df.to_parquet(D/"data/ca_sablefish_tows.parquet", index=False)
print(steps)
yr = df.groupby("year").agg(tows=("trawl_id","size"), positive=("present","sum"), vessels=("vessel","nunique"),
   temp_missing=("temperature_at_gear_c_der", lambda s: int(s.isna().sum())), mean_cpue=("cpue_kg_per_ha_der","mean"),
   lat_min=("latitude_dd","min"), lat_max=("latitude_dd","max"))
yr["pct_pos"] = (100*yr.positive/yr.tows).round(1)
print(yr.to_string())
print("depth range", df.depth_hi_prec_m.min(), df.depth_hi_prec_m.max(), "doy range", df.doy.min(), df.doy.max())
print(df.vessel.value_counts())
json.dump({"filter_steps": steps, "per_year": yr.reset_index().to_dict(orient="records")}, open(D/"data/summary.json","w"), indent=1, default=float)
