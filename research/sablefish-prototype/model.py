"""Sablefish survey-density (habitat) model, California WCGBTS 2003-2025.

Protocol
  * tuning:  fit 2003-2016, select hyperparameters on 2017-2019 (validation)
  * final:   refit 2003-2019 with selected hyperparameters, predict 2021-2025 ONCE (test)
  * geographic and vessel holdouts use 2003-2019 only (test years untouched), fixed hyperparameters
Run: uv run --python 3.12 --with pandas --with numpy --with scipy==1.14.1 --with scikit-learn \
       --with pyarrow --with matplotlib --with pygam python -I model.py
"""
import json, pathlib, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, LinearGAM, s, te
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
D = pathlib.Path(__file__).parent
OUT = D / "results"; OUT.mkdir(exist_ok=True); FIG = D / "figs"
RNG = np.random.default_rng(20261013)
df = pd.read_parquet(D / "data/ca_sablefish_tows.parquet")
df["logdepth"] = np.log(df.depth_hi_prec_m)
df["temp"] = df.temperature_at_gear_c_der
DEPTH_EDGES = [0, 100, 150, 200, 300, 400, 550, 700, 900, 1300]
df["dbin"] = pd.cut(df.depth_hi_prec_m, DEPTH_EDGES, labels=False)
df["lband"] = np.floor(df.latitude_dd).astype(int).clip(32, 41)
df["bin"] = df.dbin.astype(str) + "_" + df.lband.astype(str)
df["region"] = np.where(df.latitude_dd >= 40, "north(>=40N)", np.where(df.latitude_dd < 34.5, "south(<34.5N)", "central"))
df["cluster"] = df.year.astype(str) + "_" + df.lband.astype(str)   # bootstrap resampling unit

FEAT = ["logdepth", "latitude_dd", "longitude_dd", "temp", "doy"]
FEAT_NOTEMP = ["logdepth", "latitude_dd", "longitude_dd", "doy"]
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40); GH_W = GH_W / GH_W.sum()

# ----------------------------------------------------------------------------- helpers
def impute_temp(train, other):
    """GAM cannot take NaN: impute bottom temp from TRAINING medians by depth-bin x lat-band (fallback depth bin)."""
    m1 = train.groupby("bin").temp.median(); m2 = train.groupby("dbin").temp.median(); g = train.temp.median()
    t = other.temp.copy()
    fill = other.bin.map(m1).fillna(other.dbin.map(m2)).fillna(g)
    return t.fillna(fill)

def hurdle_stats(p, mu, sigma):
    """p = P(present); log(cpue | present) ~ N(mu, sigma^2).  Returns mean density and E[log1p(y)]."""
    mean = p * np.exp(mu + sigma**2 / 2)
    z = mu[:, None] + sigma * GH_X[None, :]
    elog = p * (np.log1p(np.exp(z)) * GH_W[None, :]).sum(1)
    return mean, elog

def bin_baseline(train, test, keys=("bin", "dbin")):
    """Climatology: per-bin presence rate, mean kg/ha, mean log1p, with depth-bin then global fallback."""
    out = {}
    for stat, col in [("p", "present"), ("mean", "cpue_kg_per_ha_der"), ("elog", "log1p_cpue")]:
        v = pd.Series(np.nan, index=test.index)
        for k in keys:
            if k not in test: continue
            v = v.fillna(test[k].map(train.groupby(k)[col].mean()))
        out[stat] = v.fillna(train[col].mean()).values
    return out

def persistence(hist, test, rolling):
    """Last-available-year in each depth x lat bin.  rolling=False: only hist (pre-test years).
    rolling=True: for test year Y also uses observed test years < Y (operationally available)."""
    res = {k: np.full(len(test), np.nan) for k in ("p", "mean", "elog")}
    clim = bin_baseline(hist, test)
    for Y in sorted(test.year.unique()):
        pool = pd.concat([hist, test[test.year < Y]]) if rolling else hist
        last = pool.groupby("bin").year.max().rename("ly")
        pl = pool.merge(last, left_on="bin", right_index=True); pl = pl[pl.year == pl.ly]
        g = pl.groupby("bin").agg(p=("present", "mean"), mean=("cpue_kg_per_ha_der", "mean"), elog=("log1p_cpue", "mean"))
        idx = np.where(test.year.values == Y)[0]
        for k in res:
            res[k][idx] = test.iloc[idx]["bin"].map(g[k]).values
    for k in res:
        miss = np.isnan(res[k]); res[k][miss] = clim[k][miss]
    return res

# ----------------------------------------------------------------------------- models
def hgb_params_grid():
    for lr, leaves, msl in itertools.product([0.03, 0.1], [7, 15, 31], [20, 50]):
        yield dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl)

def fit_gbm_hurdle(tr, feats, pc, pr):
    c = HGBC(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **pc).fit(tr[feats], tr.present)
    pos = tr[tr.present == 1]
    r = HGBR(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **pr).fit(pos[feats], np.log(pos.cpue_kg_per_ha_der))
    return c, r

def fit_gam_hurdle(tr, lam_c, lam_r):
    X = np.c_[tr.logdepth, tr.latitude_dd, tr.longitude_dd, tr.temp_imp, tr.doy]
    terms = s(0, n_splines=12) + s(1, n_splines=12) + s(2, n_splines=10) + s(3, n_splines=10) + s(4, n_splines=8) + te(0, 1, n_splines=[8, 8])
    c = LogisticGAM(terms, lam=lam_c).fit(X, tr.present.values)
    pos = tr.present.values == 1
    terms2 = s(0, n_splines=12) + s(1, n_splines=12) + s(2, n_splines=10) + s(3, n_splines=10) + s(4, n_splines=8) + te(0, 1, n_splines=[8, 8])
    r = LinearGAM(terms2, lam=lam_r).fit(X[pos], np.log(tr.cpue_kg_per_ha_der.values[pos]))
    return c, r

def gam_X(d): return np.c_[d.logdepth, d.latitude_dd, d.longitude_dd, d.temp_imp, d.doy]

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def tune(tr, va):
    """Select hyperparameters on validation years only.  Returns choices + out-of-sample log-residual SDs."""
    ch = {}
    for name, feats in [("gbm", FEAT), ("gbm_notemp", FEAT_NOTEMP)]:
        best_c = min(hgb_params_grid(), key=lambda p: brier_score_loss(va.present,
                     HGBC(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **p).fit(tr[feats], tr.present).predict_proba(va[feats])[:, 1]))
        pos_tr, pos_va = tr[tr.present == 1], va[va.present == 1]
        best_r = min(hgb_params_grid(), key=lambda p: rmse(np.log(pos_va.cpue_kg_per_ha_der),
                     HGBR(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **p).fit(pos_tr[feats], np.log(pos_tr.cpue_kg_per_ha_der)).predict(pos_va[feats])))
        r = HGBR(max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **best_r).fit(pos_tr[feats], np.log(pos_tr.cpue_kg_per_ha_der))
        sig = float(np.std(np.log(pos_va.cpue_kg_per_ha_der) - r.predict(pos_va[feats])))
        ch[name] = dict(clf=best_c, reg=best_r, sigma=sig)
    best_p = min(hgb_params_grid(), key=lambda p: poisson_dev(va.cpue_kg_per_ha_der.values,
                 HGBR(loss="poisson", max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **p).fit(tr[FEAT], tr.cpue_kg_per_ha_der).predict(va[FEAT])))
    ch["gbm_poisson"] = dict(params=best_p)
    lams = [0.3, 3, 30, 300]
    Xva = gam_X(va); pos = va.present.values == 1
    bc = min(lams, key=lambda l: brier_score_loss(va.present, fit_gam_hurdle(tr, l, 1)[0].predict_proba(Xva)))
    br = min(lams, key=lambda l: rmse(np.log(va.cpue_kg_per_ha_der.values[pos]), fit_gam_hurdle(tr, 1, l)[1].predict(Xva[pos])))
    _, r = fit_gam_hurdle(tr, bc, br)
    ch["gam"] = dict(lam_c=bc, lam_r=br, sigma=float(np.std(np.log(va.cpue_kg_per_ha_der.values[pos]) - r.predict(Xva[pos]))))
    return ch

def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))

def predict_all(tr, te_, ch):
    """Fit every method on tr, return dict method -> {p, mean, elog} on te_."""
    P = {}
    P["clim"] = bin_baseline(tr, te_)
    for name, feats in [("gbm", FEAT), ("gbm_notemp", FEAT_NOTEMP)]:
        c, r = fit_gbm_hurdle(tr, feats, ch[name]["clf"], ch[name]["reg"])
        p = c.predict_proba(te_[feats])[:, 1]; mu = r.predict(te_[feats])
        m, e = hurdle_stats(p, mu, ch[name]["sigma"]); P[name + "_hurdle"] = dict(p=p, mean=m, elog=e)
    pm = HGBR(loss="poisson", max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0, **ch["gbm_poisson"]["params"]).fit(tr[FEAT], tr.cpue_kg_per_ha_der).predict(te_[FEAT])
    P["gbm_poisson"] = dict(p=None, mean=pm, elog=np.log1p(pm))   # elog is log1p(mean): Jensen-biased, flagged
    c, r = fit_gam_hurdle(tr, ch["gam"]["lam_c"], ch["gam"]["lam_r"])
    X = gam_X(te_); p = c.predict_proba(X); mu = r.predict(X)
    m, e = hurdle_stats(p, mu, ch["gam"]["sigma"]); P["gam_hurdle"] = dict(p=p, mean=m, elog=e)
    return P

# ----------------------------------------------------------------------------- metrics
def metrics(d, pr):
    y = d.cpue_kg_per_ha_der.values; o = d.present.values
    out = dict(n=len(d), spearman=float(spearmanr(pr["mean"], y).correlation),
               rmse_log1p=rmse(pr["elog"], np.log1p(y)), poisson_dev=poisson_dev(y, pr["mean"]),
               mean_pred=float(np.mean(pr["mean"])), mean_obs=float(np.mean(y)))
    ws = [spearmanr(pr["mean"][d.year.values == Y], y[d.year.values == Y]).correlation for Y in np.unique(d.year)]
    out["spearman_within_year_mean"] = float(np.nanmean(ws))
    if pr.get("p") is not None:
        p = np.clip(pr["p"], 1e-4, 1 - 1e-4)
        out.update(auc=float(roc_auc_score(o, p)), brier=float(brier_score_loss(o, p)))
        lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p))[:, None], o)
        out.update(cal_slope=float(lr.coef_[0, 0]), cal_intercept=float(lr.intercept_[0]))
    return out

def boot_skill(d, pa, pb, B=1000):
    """Cluster bootstrap (year x 1-deg lat band) of skill of model a relative to baseline b."""
    y = d.cpue_kg_per_ha_der.values; o = d.present.values; ly = np.log1p(y)
    cl = d.cluster.values; uc = np.unique(cl); idx_by = {c: np.where(cl == c)[0] for c in uc}
    def stat(ix):
        r = dict(d_spearman=spearmanr(pa["mean"][ix], y[ix]).correlation - spearmanr(pb["mean"][ix], y[ix]).correlation,
                 msess_log1p=1 - np.mean((pa["elog"][ix] - ly[ix]) ** 2) / np.mean((pb["elog"][ix] - ly[ix]) ** 2),
                 poisson_dev_skill=1 - poisson_dev(y[ix], pa["mean"][ix]) / poisson_dev(y[ix], pb["mean"][ix]))
        if pa.get("p") is not None and pb.get("p") is not None and len(np.unique(o[ix])) == 2:
            r["d_auc"] = roc_auc_score(o[ix], pa["p"][ix]) - roc_auc_score(o[ix], pb["p"][ix])
            r["brier_ss"] = 1 - brier_score_loss(o[ix], pa["p"][ix]) / brier_score_loss(o[ix], pb["p"][ix])
        return r
    point = stat(np.arange(len(d)))
    reps = []
    for _ in range(B):
        ix = np.concatenate([idx_by[c] for c in RNG.choice(uc, len(uc), replace=True)])
        reps.append(stat(ix))
    R = pd.DataFrame(reps)
    return {k: dict(est=float(point[k]), lo=float(R[k].quantile(0.025)), hi=float(R[k].quantile(0.975))) for k in point}

# ============================================================================= main
train = df[df.year <= 2016].copy(); val = df[(df.year >= 2017) & (df.year <= 2019)].copy()
trval = df[df.year <= 2019].copy(); test = df[df.year >= 2021].copy()
train["temp_imp"] = impute_temp(train, train); val["temp_imp"] = impute_temp(train, val)
trval["temp_imp"] = impute_temp(trval, trval); test["temp_imp"] = impute_temp(trval, test)

print("tuning on 2017-2019 ...")
CH = tune(train, val)
json.dump(CH, open(OUT / "hyperparameters.json", "w"), indent=1)
print(json.dumps(CH, indent=1))

results = {}
# validation-period scores (models fit on 2003-2016) -- for information; used for selection
Pv = predict_all(train, val, CH)
Pv["persist_frozen"] = persistence(train, val, rolling=False)
Pv["persist_rolling"] = persistence(train, val, rolling=True)
results["validation_2017_2019"] = {k: metrics(val, v) for k, v in Pv.items()}

# ---------------------------------------------------------------- TEST (run once)
print("final refit 2003-2019, test 2021-2025 ...")
P = predict_all(trval, test, CH)
P["persist_frozen"] = persistence(trval, test, rolling=False)
P["persist_rolling"] = persistence(trval, test, rolling=True)
results["test_2021_2025"] = {k: metrics(test, v) for k, v in P.items()}
models = ["gbm_hurdle", "gam_hurdle", "gbm_notemp_hurdle", "gbm_poisson"]
bases = ["clim", "persist_frozen", "persist_rolling"]
results["test_skill_bootstrap"] = {f"{m}_vs_{b}": boot_skill(test, P[m], P[b]) for m in models for b in bases}
results["test_skill_bootstrap"]["gbm_hurdle_vs_gam_hurdle"] = boot_skill(test, P["gbm_hurdle"], P["gam_hurdle"])
results["test_skill_bootstrap"]["gbm_hurdle_vs_gbm_notemp_hurdle"] = boot_skill(test, P["gbm_hurdle"], P["gbm_notemp_hurdle"])
# per-year: observed vs predicted mean density (index-level check)
py = []
for Y in sorted(test.year.unique()):
    m = test.year.values == Y
    row = dict(year=int(Y), n=int(m.sum()), obs_mean=float(test.cpue_kg_per_ha_der.values[m].mean()))
    for k in ["clim", "persist_rolling", "gbm_hurdle", "gam_hurdle"]:
        row[k + "_mean"] = float(P[k]["mean"][m].mean())
        row[k + "_spearman"] = float(spearmanr(P[k]["mean"][m], test.cpue_kg_per_ha_der.values[m]).correlation)
    py.append(row)
results["test_per_year"] = py

# reliability tables
rel = {}
for k in ["gbm_hurdle", "gam_hurdle", "clim", "persist_rolling"]:
    p = P[k]["p"]; b = np.clip((p * 10).astype(int), 0, 9)
    t = pd.DataFrame(dict(b=b, p=p, o=test.present.values)).groupby("b").agg(n=("o", "size"), mean_pred=("p", "mean"), obs_rate=("o", "mean")).reset_index()
    rel[k] = t.to_dict(orient="records")
results["test_reliability"] = rel

# ---------------------------------------------------------------- prediction intervals
def quantile_models(tr, feats, alphas):
    return {a: HGBR(loss="quantile", quantile=a, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=0,
                    **CH["gbm"]["reg"]).fit(tr[feats], tr.log1p_cpue) for a in alphas}
A = [0.025, 0.1, 0.9, 0.975]
qm = quantile_models(trval, FEAT, A)
Q = {a: np.clip(qm[a].predict(test[FEAT]), 0, None) for a in A}
# conformalized QR: quantile models fit 2003-2016, conformity scores on 2017-2019, applied to test
qm0 = quantile_models(train, FEAT, A)
cqr = {}
for lo, hi, lvl in [(0.1, 0.9, 0.8), (0.025, 0.975, 0.95)]:
    sv = np.maximum(qm0[lo].predict(val[FEAT]) - val.log1p_cpue, val.log1p_cpue - qm0[hi].predict(val[FEAT]))
    n = len(sv); qhat = np.quantile(sv, min(1, np.ceil((n + 1) * lvl) / n))
    cqr[lvl] = (np.clip(qm0[lo].predict(test[FEAT]) - qhat, 0, None), qm0[hi].predict(test[FEAT]) + qhat, float(qhat))
ly = test.log1p_cpue.values; pos = test.present.values == 1
cov = {}
for lvl, (lo, hi) in [(0.8, (0.1, 0.9)), (0.95, (0.025, 0.975))]:
    inside = (ly >= Q[lo] - 1e-9) & (ly <= Q[hi] + 1e-9)
    ci = (ly >= cqr[lvl][0] - 1e-9) & (ly <= cqr[lvl][1])
    cov[f"{int(lvl*100)}%"] = dict(
        qr_coverage=float(inside.mean()), qr_cov_positive=float(inside[pos].mean()), qr_cov_zero=float(inside[~pos].mean()),
        qr_median_width_kgha=float(np.median(np.expm1(Q[hi]) - np.expm1(Q[lo]))),
        cqr_coverage=float(ci.mean()), cqr_cov_positive=float(ci[pos].mean()), cqr_cov_zero=float(ci[~pos].mean()), cqr_qhat_log1p=cqr[lvl][2],
        cqr_median_width_kgha=float(np.median(np.expm1(cqr[lvl][1]) - np.expm1(cqr[lvl][0]))),
        qr_by_year={int(Y): float(inside[test.year.values == Y].mean()) for Y in sorted(test.year.unique())},
        cqr_by_year={int(Y): float(ci[test.year.values == Y].mean()) for Y in sorted(test.year.unique())})
results["test_interval_coverage"] = cov

# ---------------------------------------------------------------- geographic + vessel holdouts (2003-2019 only)
def holdout(split_col, groups):
    out = {}
    for g in groups:
        tr = trval[trval[split_col] != g].copy(); ho = trval[trval[split_col] == g].copy()
        tr["temp_imp"] = impute_temp(tr, tr); ho["temp_imp"] = impute_temp(tr, ho)
        Ph = predict_all(tr, ho, CH)
        if split_col == "region":   # held-out lat bands unseen -> depth-only climatology
            Ph["clim"] = bin_baseline(tr, ho, keys=("dbin",))
        r = {k: metrics(ho, v) for k, v in Ph.items()}
        r["skill_gbm_vs_clim"] = boot_skill(ho, Ph["gbm_hurdle"], Ph["clim"], B=500)
        r["skill_gam_vs_clim"] = boot_skill(ho, Ph["gam_hurdle"], Ph["clim"], B=500)
        out[g] = r
    return out
print("geographic holdout ...")
results["geo_holdout_2003_2019"] = holdout("region", ["north(>=40N)", "central", "south(<34.5N)"])
print("vessel holdout ...")
results["vessel_holdout_2003_2019"] = holdout("vessel", ["Excalibur", "Noahs Ark", "Ms. Julie", "Last Straw", "Raven"])

json.dump(results, open(OUT / "results.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="perfect")
for k, c in [("gbm_hurdle", "#1f6feb"), ("gam_hurdle", "#d97706"), ("clim", "#6b7280"), ("persist_rolling", "#9333ea")]:
    t = pd.DataFrame(rel[k]); t = t[t.n >= 10]
    ax.plot(t.mean_pred, t.obs_rate, marker="o", color=c, label=k)
ax.set_xlabel("Predicted P(sablefish present)"); ax.set_ylabel("Observed fraction of tows with sablefish")
ax.set_title("Occurrence reliability, test 2021-2025 (bins n>=10)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "reliability_occurrence_test.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 5))
for k, c in [("gbm_hurdle", "#1f6feb"), ("gam_hurdle", "#d97706"), ("clim", "#6b7280")]:
    q = pd.qcut(P[k]["mean"], 10, labels=False, duplicates="drop")
    t = pd.DataFrame(dict(q=q, p=P[k]["mean"], y=test.cpue_kg_per_ha_der.values)).groupby("q").mean()
    ax.loglog(t.p, t.y, marker="o", color=c, label=k)
lim = [0.05, 100]; ax.plot(lim, lim, color="#888", ls="--", lw=1, label="1:1")
ax.set_xlabel("Predicted mean density (kg/ha), decile mean"); ax.set_ylabel("Observed mean density (kg/ha)")
ax.set_title("Density calibration by predicted decile, test 2021-2025"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "calibration_density_test.png", dpi=150); plt.close(fig)

t = pd.DataFrame(py)
fig, ax = plt.subplots(figsize=(6, 3.8))
allyr = df.groupby("year").cpue_kg_per_ha_der.mean()
ax.plot(allyr.index, allyr.values, "o-", color="#111", label="observed (all years)")
ax.plot(t.year, t.gbm_hurdle_mean, "s--", color="#1f6feb", label="GBM hurdle predicted")
ax.plot(t.year, t.gam_hurdle_mean, "^--", color="#d97706", label="GAM hurdle predicted")
ax.plot(t.year, t.clim_mean, "x--", color="#6b7280", label="climatology")
ax.axvspan(2020.5, 2025.5, color="#eee"); ax.set_ylabel("Mean tow density (kg/ha)"); ax.set_title("Annual mean density: observed vs frozen models (test shaded)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "annual_mean_density.png", dpi=150); plt.close(fig)

# GAM partial dependence of occurrence on depth & temp (fit 2003-2019)
c, r = fit_gam_hurdle(trval, CH["gam"]["lam_c"], CH["gam"]["lam_r"])
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
for i, (term, lab, tr_) in enumerate([(0, "Depth (m)", np.exp), (3, "Bottom temperature (C)", lambda x: x)]):
    XX = c.generate_X_grid(term=term); pdp, ci = c.partial_dependence(term=term, X=XX, width=0.95)
    axs[i].plot(tr_(XX[:, term]), pdp, color="#1f6feb"); axs[i].fill_between(tr_(XX[:, term]), ci[:, 0], ci[:, 1], color="#1f6feb", alpha=0.2)
    axs[i].set_xlabel(lab); axs[i].set_ylabel("partial effect (logit occurrence)")
axs[0].set_xscale("log"); fig.suptitle("GAM occurrence partial effects, 2003-2019 fit (depth also enters a depth x lat tensor term)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "gam_partial_effects.png", dpi=150); plt.close(fig)
print("done")
