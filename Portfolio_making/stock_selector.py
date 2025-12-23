#!/usr/bin/env python3
"""

Hyperparameter search for stock-selection scoring weights.

- Samples N random weight combinations for (Sharpe, Vol, Return, Correlation) that sum to 1.
- For each sampled combo:
    * compute normalized metrics per stock (sharpe, ann return, ann vol, corr diversification score)
    * composite = w_sharpe*sharpe_norm + w_vol*vol_score + w_ret*ret_norm + w_corr*corr_score
    * select top-2 stocks per sector by composite
    * compute max-sharpe portfolio (long-only) on those 10 stocks using historical returns up to cutoff_date
    * record portfolio annual return, annual vol, and Sharpe
- Save all trials to CSV and write best trial selection.

Edit USER PARAMETERS as needed.
"""
import os
import math
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------- USER PARAMETERS ----------------
price_root = Path("data/stock_data")      # location of historical CSVs: price_root/<Sector>/<Stock>.csv
output_dir = Path("Portfolio_making/stock_selector_h")     # outputs will be written here

# Candidate universe (5 sectors, 5 each)
candidate_universe = {
    "Bank":     ["AXISBANK", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN"],
    "FMCG":     ["BRITANNIA", "HINDUNILVR", "ITC", "NESTLEIND", "TATACONSUM"],
    "IT":       ["HCLTECH", "INFY", "TCS", "TECHM", "WIPRO"],
    "OilGas":   ["BPCL", "GAIL", "IOC", "ONGC", "RELIANCE"],
    "Pharma":   ["CIPLA", "DIVISLAB", "DRREDDY", "LUPIN", "SUNPHARMA"]
}

date_col = "date"
close_col = "close"

# Sampling & optimization
N_TRIALS = 1000                   # number of random weight combos to try
random_seed = 42
cutoff_date = pd.Timestamp("2025-05-01")   # use history up to & incl. this date to compute mu & cov
return_freq = 'D'                 # 'D' = daily returns (default). Use 'W' or 'M' to switch.
risk_free = 0.01
trading_days = 252

# preferred vol center for vol_score (tunable)
preferred_vol = 0.18
vol_sigma = 0.12

# bounds/constraints for portfolio optimizer (long-only)
allow_short = False

# -------------------------------------------------
np.random.seed(random_seed)
random.seed(random_seed)
os.makedirs(output_dir, exist_ok=True)

# ---------------- Helpers ----------------
def find_price_file(sector, stock):
    # try exact path first
    p = price_root / sector / f"{stock}.csv"
    if p.exists():
        return p
    # recursive search fallback
    for f in price_root.rglob("*.csv"):
        if f.stem.upper().startswith(stock.upper()):
            return f
    return None

def load_close_series(path):
    df = pd.read_csv(path, parse_dates=[date_col], usecols=[date_col, close_col])
    df = df.dropna(subset=[close_col]).sort_values(by=date_col).drop_duplicates(subset=[date_col], keep='last')
    ser = pd.Series(df[close_col].values, index=pd.to_datetime(df[date_col]))
    ser.name = path.stem
    return ser

def ann_metrics_from_price_series(price_ser):
    rets = price_ser.pct_change().dropna()
    if rets.empty:
        return None
    # resample if return_freq not daily
    if return_freq == 'W':
        rets = price_ser.pct_change().dropna().resample('W').apply(lambda x: (x+1.0).prod()-1.0)
    elif return_freq == 'M':
        rets = price_ser.pct_change().dropna().resample('M').apply(lambda x: (x+1.0).prod()-1.0)
    mean_daily = rets.mean()
    vol_daily = rets.std()
    # annualize (approx)
    ann_ret = mean_daily * trading_days
    ann_vol = vol_daily * math.sqrt(trading_days)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "n_obs": len(rets)}

# normalize helper
def safe_minmax(series):
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if np.isnan(mn) or np.isnan(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

# optimization: max Sharpe (long-only)
def max_sharpe_weights(mu_vec, cov_mat, rf=risk_free):
    n = len(mu_vec)
    x0 = np.ones(n) / n
    # bounds = [(0.0, 1.0) for _ in range(n)] if not allow_short else [(-1.0, 1.0) for _ in range(n)]
    bounds = [(0.02, 1.0) for _ in range(n)] if not allow_short else [(-1.0, 1.0) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    def neg_sharpe(w):
        port_ret = float(np.dot(w, mu_vec))
        port_vol = float(np.sqrt(w.T @ cov_mat @ w))
        if port_vol <= 0:
            return 1e6
        return - (port_ret - rf) / port_vol
    try:
        res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':500})
        if res.success:
            return res.x, None
        else:
            return None, res.message
    except Exception as e:
        return None, str(e)

# small jitter to covariance to make invertible
def stabilize_cov(cov, eps=1e-6):
    cov = cov.copy()
    cov.values[np.diag_indices_from(cov)] += eps
    return cov

# ---------------- Precompute metrics for all candidates ----------------
rows = []
price_series_map = {}
for sector, stocks in candidate_universe.items():
    for stock in stocks:
        f = find_price_file(sector, stock)
        if f is None:
            print(f"[WARN] {stock} file not found under {price_root}/{sector}; skipping.")
            rows.append({"Sector": sector, "Stock": stock, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "n_obs":0})
            continue
        ser = load_close_series(f)
        price_series_map[stock] = ser
        metrics = ann_metrics_from_price_series(ser.loc[:cutoff_date])
        if metrics is None:
            rows.append({"Sector": sector, "Stock": stock, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "n_obs": 0})
        else:
            rows.append({"Sector": sector, "Stock": stock, "ann_return": metrics["ann_return"], "ann_vol": metrics["ann_vol"], "sharpe": metrics["sharpe"], "n_obs": metrics["n_obs"]})

metrics_df = pd.DataFrame(rows).set_index("Stock")

# ---------------- Helper to compute correlation/diversification per sector ---------------
def compute_corr_score(sector, stocks):
    # compute pairwise daily return correlations (use data up to cutoff_date)
    rets = {}
    for s in stocks:
        ser = price_series_map.get(s)
        if ser is None:
            rets[s] = None
            continue
        rets[s] = ser.loc[:cutoff_date].pct_change().dropna()
    # build correlation matrix (pairwise using intersection)
    corr = pd.DataFrame(index=stocks, columns=stocks, dtype=float)
    for i in stocks:
        for j in stocks:
            ri, rj = rets.get(i), rets.get(j)
            if ri is None or rj is None:
                corr.loc[i,j] = np.nan
            else:
                idx = ri.index.intersection(rj.index)
                if len(idx) < 20:
                    corr.loc[i,j] = np.nan
                else:
                    corr.loc[i,j] = float(ri.loc[idx].corr(rj.loc[idx]))
    # average absolute correlation
    avg_abs = corr.abs().mean(axis=1, skipna=True)
    # normalize and invert (lower avg_abs => higher corr_score)
    if avg_abs.isna().all():
        corr_score = pd.Series(0.5, index=stocks)
    else:
        corr_score = 1.0 - safe_minmax(avg_abs.fillna(avg_abs.mean()))
    return corr_score

# Precompute per-sector corr_scores
sector_corr_scores = {}
for sector, stocks in candidate_universe.items():
    sector_corr_scores[sector] = compute_corr_score(sector, stocks)

# ---------------- Random search over scoring weights ----------------
def sample_weights():
    # sample 4 positive numbers and normalize
    v = np.random.rand(4)
    v = v / v.sum()
    return {"w_sharpe": float(v[0]), "w_vol": float(v[1]), "w_ret": float(v[2]), "w_corr": float(v[3])}

trial_results = []
best_sharpe = -np.inf
best_trial = None

# We'll also need mu and cov computed from returns up to cutoff_date for the selected stocks per trial
# Start trials
for t in range(1, N_TRIALS+1):
    weights = sample_weights()
    # build composite per stock
    composite_scores = {}
    # for each sector compute normalized metrics and combine
    for sector, stocks in candidate_universe.items():
        # subset metrics
        sub = metrics_df.loc[stocks].copy()
        # normalized sharpe and return (higher better)
        sharpe_norm = safe_minmax(sub["sharpe"].fillna(0))
        ret_norm = safe_minmax(sub["ann_return"].fillna(0))
        # volatility score: prefer moderate vol; gaussian centered at preferred_vol
        vol = sub["ann_vol"].fillna(preferred_vol)
        vol_score_raw = np.exp(-0.5 * ((vol - preferred_vol) / vol_sigma) ** 2)
        vol_score = safe_minmax(vol_score_raw)
        # corr score from precomputed
        corr_score = sector_corr_scores[sector].reindex(stocks).fillna(0.5)
        # composite
        comp = (weights["w_sharpe"] * sharpe_norm) + (weights["w_vol"] * vol_score) + (weights["w_ret"] * ret_norm) + (weights["w_corr"] * corr_score)
        for s in stocks:
            composite_scores[s] = float(comp.loc[s])

    # select top 2 per sector
    selected = []
    for sector, stocks in candidate_universe.items():
        # sort sector stocks by composite
        ranked = sorted(stocks, key=lambda x: composite_scores.get(x, -999), reverse=True)
        # take up to 2
        chosen = [r for r in ranked[:2] if not np.isnan(composite_scores.get(r, np.nan))]
        selected.extend(chosen)

    # Ensure we have 10 selected, else skip
    if len(selected) < 2 * len(candidate_universe):
        # skip trial (not enough valid stocks)
        continue

    # Now compute mu & cov for selected stocks using returns up to cutoff_date
    returns_map = {}
    for s in selected:
        ser = price_series_map.get(s)
        if ser is None:
            returns_map[s] = None
        else:
            returns_map[s] = ser.loc[:cutoff_date].pct_change().dropna()
    # build returns DataFrame aligned by union of dates, then drop columns with too few observations
    rets_df = pd.DataFrame(returns_map).dropna(axis=1, how='all')
    # drop columns with fewer than 20 obs
    rets_df = rets_df.loc[:, rets_df.count() >= 20]
    if rets_df.shape[1] < len(selected):
        # some stocks missing returns; if less than 10 left, skip
        if rets_df.shape[1] < 10:
            continue

    mu = rets_df.mean() * trading_days
    cov = rets_df.cov() * trading_days
    cov = stabilize_cov(cov, eps=1e-6)

    # compute max-sharpe weights long-only
    try:
        w, err = max_sharpe_weights(mu.values, cov.values, rf=risk_free)
        if w is None:
            # optimization failed
            continue
    except Exception:
        continue

    # portfolio stats
    port_ret = float(np.dot(w, mu.values))
    port_vol = float(np.sqrt(w.T @ cov.values @ w))
    port_sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else np.nan

    # record trial
    trial_results.append({
        "trial": t,
        "w_sharpe": weights["w_sharpe"],
        "w_vol": weights["w_vol"],
        "w_ret": weights["w_ret"],
        "w_corr": weights["w_corr"],
        "selected": ";".join(selected),
        "n_selected": len(selected),
        "port_ann_return": port_ret,
        "port_ann_vol": port_vol,
        "port_sharpe": port_sharpe
    })

    # update best
    if np.isfinite(port_sharpe) and port_sharpe > best_sharpe:
        best_sharpe = port_sharpe
        best_trial = trial_results[-1]

    # progress print every 100 trials
    if t % 100 == 0:
        print(f"Trial {t}/{N_TRIALS} — current best Sharpe: {best_sharpe:.4f}")

# ---------------- Save results ----------------
trials_df = pd.DataFrame(trial_results).sort_values("port_sharpe", ascending=False)
trials_df.to_csv(output_dir / "hyperopt_results.csv", index=False, float_format="%.6f")
print(f"[DONE] trials saved to {output_dir/'hyperopt_results.csv'}  (tried {len(trial_results)} valid trials)")

if best_trial is None:
    print("No successful trial found.")
else:
    print("Best trial summary:")
    print(best_trial)
    # save best selection details
    best_weights = {k: best_trial[k] for k in ["w_sharpe","w_vol","w_ret","w_corr"]}
    selected_list = best_trial["selected"].split(";")
    # build detailed metrics for selected stocks
    selected_metrics = []
    # recompute composite for that best weight combo to store
    w = best_weights
    for sector, stocks in candidate_universe.items():
        # recreate per-sector scores
        sub = metrics_df.loc[stocks].copy()
        sharpe_norm = safe_minmax(sub["sharpe"].fillna(0))
        ret_norm = safe_minmax(sub["ann_return"].fillna(0))
        vol = sub["ann_vol"].fillna(preferred_vol)
        vol_score_raw = np.exp(-0.5 * ((vol - preferred_vol) / vol_sigma) ** 2)
        vol_score = safe_minmax(vol_score_raw)
        corr_score = sector_corr_scores[sector].reindex(stocks).fillna(0.5)
        comp = (w["w_sharpe"] * sharpe_norm) + (w["w_vol"] * vol_score) + (w["w_ret"] * ret_norm) + (w["w_corr"] * corr_score)
        for s in stocks:
            if s in selected_list:
                selected_metrics.append({
                    "Stock": s,
                    "Sector": sector,
                    "ann_return": metrics_df.loc[s,"ann_return"],
                    "ann_vol": metrics_df.loc[s,"ann_vol"],
                    "sharpe": metrics_df.loc[s,"sharpe"],
                    "composite": float(comp.loc[s])
                })

    sel_df = pd.DataFrame(selected_metrics).sort_values(["Sector","composite"], ascending=[True,False])
    sel_df.to_csv(output_dir / "best_selection.csv", index=False, float_format="%.6f")
    # save best trial as json
    with open(output_dir / "best_portfolio_stats.json", "w") as f:
        json.dump(best_trial, f, indent=2)
    print(f"[DONE] best selection saved to {output_dir/'best_selection.csv'} and best trial json")

print("Finished.")
