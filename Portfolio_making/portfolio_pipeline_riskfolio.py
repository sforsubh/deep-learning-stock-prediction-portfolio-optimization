# portfolio_pipeline_riskfolio_randommsr.py
# Requirements: pandas, numpy, scipy, openpyxl, riskfolio-lib (optional),
#               plotly, kaleido (optional)
# pip install pandas numpy scipy openpyxl riskfolio-lib plotly kaleido

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

# plotly
import plotly.graph_objects as go

# ---------------- USER PARAMETERS ----------------
best_model_csv = "output/best_model_summary.csv"
stock_data_root = "output/stock_data"

tickers_10 = ['MARUTI', 'EICHERMOT', 'SBIN', 'UPL', 'NESTLEIND',
              'APOLLOHOSP', 'HINDALCO', 'JSWSTEEL', 'IOC', 'SUNPHARMA']

cutoff_date = pd.Timestamp("2025-05-01")
eval_date   = pd.Timestamp("2025-10-01")

initial_capital = 100000.0
risk_free = 0.01

output_root = "Portfolio_making/output_riskfolio"

# Random portfolio settings (improved stability)
N_RANDOM_PORTFOLIOS = 10000   # increased sampling for stability (you can lower if slow)
RANDOM_SEED = 42

# ---------------- Helper functions ----------------
def pick_pred_column(model_name):
    m = str(model_name).strip().upper()
    if m == "CNN":
        return "pred_cnn"
    if m == "LSTM":
        return "pred_lstm"
    if m == "HYBRID":
        return "pred_hybrid"
    return model_name

def find_stock_csv(stock_name):
    stock_name = stock_name.strip()
    for root, dirs, files in os.walk(stock_data_root):
        for d in dirs:
            if d.upper() == stock_name.upper():
                fpath = os.path.join(root, d, f"{stock_name}_NS_pred.csv")
                if os.path.exists(fpath):
                    return fpath
    for root, dirs, files in os.walk(stock_data_root):
        for f in files:
            if f.upper().startswith(stock_name.upper()) and f.upper().endswith("_NS_PRED.CSV"):
                return os.path.join(root, f)
    return None

def last_available(series, date):
    if date in series.index:
        return series.loc[date]
    idx = series.index[series.index <= date]
    return series.loc[idx[-1]] if len(idx) else np.nan

def expand_weights(opt_w, all_tickers, clean_list):
    s = pd.Series(0.0, index=all_tickers)
    try:
        arr = np.asarray(opt_w).flatten()
        if arr.size == len(clean_list):
            s.loc[clean_list] = arr
            return s.values
    except Exception:
        pass
    # fallback: try pandas alignment
    try:
        tmp = pd.Series(opt_w).squeeze()
        tmp = tmp.reindex(clean_list).fillna(0.0)
        s.loc[clean_list] = tmp.values.flatten()
        return s.values
    except Exception:
        # last fallback: zeros
        return s.values

# ---------------- Load best-model summary ----------------
if not os.path.exists(best_model_csv):
    raise FileNotFoundError(best_model_csv)

bm = pd.read_csv(best_model_csv)
required_cols = ["Sector", "Stock", "Best_By_RMSE_Mean_Model"]
for c in required_cols:
    if c not in bm.columns:
        raise ValueError(f"best_model_summary.csv must contain column '{c}'")
bm = bm.drop_duplicates(subset=["Stock"], keep="first").set_index("Stock")

missing = [s for s in tickers_10 if s not in bm.index]
if missing:
    raise ValueError(f"Missing stocks in best_model_summary.csv: {missing}")

# ---------------- Read price CSVs ----------------
actual_prices = pd.DataFrame()
selected_preds = pd.DataFrame()
sector_map = {}

for s in tickers_10:
    csv = find_stock_csv(s)
    if csv is None:
        raise FileNotFoundError(f"No CSV found for {s}")
    df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
    if "actual" not in df.columns:
        raise ValueError(f"actual column missing in {csv}")
    model = bm.loc[s, "Best_By_RMSE_Mean_Model"]
    pred_col = pick_pred_column(model)
    if pred_col not in df.columns:
        raise ValueError(f"Pred column {pred_col} not found in {csv}")
    actual_prices = actual_prices.join(df[["actual"]].rename(columns={"actual": s}), how="outer")
    selected_preds = selected_preds.join(df[[pred_col]].rename(columns={pred_col: s}), how="outer")
    sector_map[s] = bm.loc[s, "Sector"]

# ensure columns in same order
actual_prices = actual_prices.reindex(columns=tickers_10)
selected_preds = selected_preds.reindex(columns=tickers_10)

# ---------------- Build historical returns (Option 2 forward-fill: internal gaps only) ----------------
hist_actual = actual_prices.loc[:cutoff_date].copy()

# check each ticker has at least one real price before cutoff_date
no_data_tickers = [col for col in hist_actual.columns if hist_actual[col].dropna().empty]
if len(no_data_tickers) > 0:
    raise ValueError("Some tickers have no price data before cutoff_date: " + ", ".join(no_data_tickers))

# forward-fill internal gaps without backfilling before first available price
hist_ffill = hist_actual.copy()
for col in hist_ffill.columns:
    first_idx = hist_ffill[col].first_valid_index()
    if first_idx is None:
        continue
    hist_ffill.loc[first_idx:, col] = hist_ffill.loc[first_idx:, col].ffill()

# compute returns (pct_change). Leading NaNs remain.
returns_hist = hist_ffill.pct_change()

# ensure every ticker has at least one return data point
tickers_no_returns = [col for col in returns_hist.columns if returns_hist[col].dropna().empty]
if len(tickers_no_returns) > 0:
    raise ValueError("Some tickers have no valid returns before cutoff_date (after pct_change): " + ", ".join(tickers_no_returns))

# Create an intersection window where all tickers have non-NaN returns (needed by cov calculations & solver)
returns_hist_intersection = returns_hist.dropna(axis=0, how='any').copy()

if returns_hist_intersection.shape[0] < 5:
    print("Warning: intersection window (rows where all tickers have returns) is small:",
          returns_hist_intersection.shape[0], "rows. Results may be unstable.")

# Annualize using 252 trading days
mu = returns_hist_intersection.mean() * 252.0
cov = returns_hist_intersection.cov() * 252.0

# tiny regularization for numerical stability
eps_cov = 1e-8
cov += np.eye(cov.shape[0]) * eps_cov

# returns_hist_clean for riskfolio/port usage
returns_hist_clean = returns_hist_intersection.copy()
clean_tickers = returns_hist_clean.columns.tolist()

for col in clean_tickers:
    n_pts = returns_hist_clean[col].dropna().shape[0]
    if n_pts < 20:
        print(f"Warning: few historical return points for {col} (n={n_pts}) — consider longer history.")

mu_clean = mu.reindex(clean_tickers)
cov_clean = cov.reindex(index=clean_tickers, columns=clean_tickers)

# ---------------- Random Portfolio Sampling (REPLACES convex optimization in your old flow) ----------------
print("Generating random portfolios (using cleaned tickers)...")
np.random.seed(RANDOM_SEED)
n_clean = len(clean_tickers)

random_results = {"weights": [], "ret": [], "vol": [], "sharpe": []}

mu_vec_clean = mu_clean.reindex(clean_tickers).values
cov_mat_clean = cov_clean.reindex(index=clean_tickers, columns=clean_tickers).values

# add small regularization for sampling stability
cov_mat_clean += np.eye(n_clean) * 1e-10

for i in range(N_RANDOM_PORTFOLIOS):
    w = np.random.random(n_clean)
    w = w / w.sum()
    port_ret = float(np.dot(w, mu_vec_clean))
    port_vol = float(np.sqrt(max(w.T @ cov_mat_clean @ w, 0.0)))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0.0
    random_results["weights"].append(w)
    random_results["ret"].append(port_ret)
    random_results["vol"].append(port_vol)
    random_results["sharpe"].append(sharpe)

rand_df = pd.DataFrame({
    "return": random_results["ret"],
    "volatility": random_results["vol"],
    "sharpe": random_results["sharpe"]
})
weights_arr_clean = np.vstack(random_results["weights"])
for idx, t in enumerate(clean_tickers):
    rand_df[t] = weights_arr_clean[:, idx]

# pick sampled min variance and max sharpe (OLD logic)
min_var_idx = rand_df["volatility"].idxmin()
max_sharpe_idx = rand_df["sharpe"].idxmax()

w_min_clean = rand_df.loc[min_var_idx, clean_tickers].values
w_max_clean = rand_df.loc[max_sharpe_idx, clean_tickers].values

# expand to full tickers_10 (retain zeros for tickers not in clean_tickers — but we kept all tickers)
w_mvp = expand_weights(w_min_clean, tickers_10, clean_tickers)
w_msr = expand_weights(w_max_clean, tickers_10, clean_tickers)

w_mvp_series = pd.Series(np.asarray(w_mvp, dtype=float).flatten(), index=tickers_10)
w_msr_series = pd.Series(np.asarray(w_msr, dtype=float).flatten(), index=tickers_10)

# ---------------- Portfolio Stats (use mu & cov from returns_hist_intersection) ----------------
mu_all = mu  # annualized
cov_all = cov  # annualized

def port_stats_arr(w, mu_vec, cov_mat, rf=0.01):
    w = np.asarray(w, dtype=float).flatten()
    ret = float(np.dot(w, mu_vec))
    vol = float(np.sqrt(max(w.T @ cov_mat @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret * 100.0, vol * 100.0, sharpe

mvp_ret_pct, mvp_vol_pct, mvp_sh = port_stats_arr(w_mvp, mu_all.values, cov_all.values, rf=risk_free)
msr_ret_pct, msr_vol_pct, msr_sh = port_stats_arr(w_msr, mu_all.values, cov_all.values, rf=risk_free)

# ---------------- Amount invested, fractional shares at cutoff
price_on_cutoff = pd.Series({s: last_available(actual_prices[s], cutoff_date) for s in tickers_10})
if price_on_cutoff.isna().any():
    print("Warning: some stocks missing prices on or before cutoff_date; using last available or NaN")
    if price_on_cutoff.isna().any():
        raise ValueError("Some stocks have no historical price before cutoff_date: " + ", ".join(price_on_cutoff[price_on_cutoff.isna()].index.tolist()))

invested_mvp = w_mvp_series * initial_capital
invested_msr = w_msr_series * initial_capital

shares_mvp = invested_mvp / price_on_cutoff
shares_msr = invested_msr / price_on_cutoff

shares_mvp = shares_mvp.replace([np.inf, -np.inf], 0).fillna(0)
shares_msr = shares_msr.replace([np.inf, -np.inf], 0).fillna(0)

cash_left_mvp = initial_capital - (shares_mvp * price_on_cutoff).sum()
cash_left_msr = initial_capital - (shares_msr * price_on_cutoff).sum()

# ---------------- Prices on evaluation date
price_on_eval_actual = pd.Series({s: last_available(actual_prices[s], eval_date) for s in tickers_10})
price_on_eval_pred   = pd.Series({s: last_available(selected_preds[s], eval_date) for s in tickers_10})

actual_value_mvp = shares_mvp * price_on_eval_actual + cash_left_mvp * (1.0 if not np.isnan(cash_left_mvp) else 0.0)
pred_value_mvp   = shares_mvp * price_on_eval_pred   + cash_left_mvp * (1.0 if not np.isnan(cash_left_mvp) else 0.0)

actual_value_msr = shares_msr * price_on_eval_actual + cash_left_msr * (1.0 if not np.isnan(cash_left_msr) else 0.0)
pred_value_msr   = shares_msr * price_on_eval_pred   + cash_left_msr * (1.0 if not np.isnan(cash_left_msr) else 0.0)

# ---------------- Build Weight Allocation table (Option A style)
weight_table = pd.DataFrame({
    "Stock": tickers_10,
    "Minimum Risk": w_mvp_series.round(6).values,
    "Mean Risk": w_msr_series.round(6).values
})

weight_table = pd.concat([
    weight_table,
    pd.DataFrame([{
        "Stock": "Annual Return (%)",
        "Minimum Risk": round(mvp_ret_pct, 2),
        "Mean Risk": round(msr_ret_pct, 2)
    }])
], ignore_index=True)

weight_table = pd.concat([
    weight_table,
    pd.DataFrame([{
        "Stock": "Annual Risk (%)",
        "Minimum Risk": round(mvp_vol_pct, 2),
        "Mean Risk": round(msr_vol_pct, 2)
    }])
], ignore_index=True)

# ---------------- Build Stock-level tables for MVP & OPR
rows_mvp = []
rows_msr = []

for s in tickers_10:
    rows_mvp.append({
        "Stock": s,
        "Amt_Invstd": round(invested_mvp[s], 2),
        "ActualPrice_1May": round(price_on_cutoff[s], 2) if not np.isnan(price_on_cutoff[s]) else np.nan,
        "Shares": float(shares_mvp[s]),
        "ActualPrice_1Oct": round(price_on_eval_actual[s], 2) if not np.isnan(price_on_eval_actual[s]) else np.nan,
        "ActualValue": round(float(shares_mvp[s] * price_on_eval_actual[s]), 2),
        "PredPrice_1Oct": round(price_on_eval_pred[s], 2) if not np.isnan(price_on_eval_pred[s]) else np.nan,
        "PredValue": round(float(shares_mvp[s] * price_on_eval_pred[s]), 2)
    })
    rows_msr.append({
        "Stock": s,
        "Amt_Invstd": round(invested_msr[s], 2),
        "ActualPrice_1May": round(price_on_cutoff[s], 2) if not np.isnan(price_on_cutoff[s]) else np.nan,
        "Shares": float(shares_msr[s]),
        "ActualPrice_1Oct": round(price_on_eval_actual[s], 2) if not np.isnan(price_on_eval_actual[s]) else np.nan,
        "ActualValue": round(float(shares_msr[s] * price_on_eval_actual[s]), 2),
        "PredPrice_1Oct": round(price_on_eval_pred[s], 2) if not np.isnan(price_on_eval_pred[s]) else np.nan,
        "PredValue": round(float(shares_msr[s] * price_on_eval_pred[s]), 2)
    })

stock_table_minimumVP = pd.DataFrame(rows_mvp)
stock_table_meanVP = pd.DataFrame(rows_msr)

stock_table_minimumVP.loc[len(stock_table_minimumVP)] = {
    "Stock": "TOTAL",
    "Amt_Invstd": round(invested_mvp.sum(), 2),
    "ActualValue": round(actual_value_mvp.sum(), 2),
    "PredValue": round(pred_value_mvp.sum(), 2)
}
stock_table_meanVP.loc[len(stock_table_meanVP)] = {
    "Stock": "TOTAL",
    "Amt_Invstd": round(invested_msr.sum(), 2),
    "ActualValue": round(actual_value_msr.sum(), 2),
    "PredValue": round(pred_value_msr.sum(), 2)
}

# ---------------- Portfolio summary table
def pct_return(start, end):
    return (end - start) / start * 100.0

summary_rows = []
act_end = actual_value_mvp.sum()
pred_end = pred_value_mvp.sum()
summary_rows.append({
    "Portfolio": "Minimum Variance",
    "StartValue": initial_capital,
    "ActualEndValue": round(float(act_end), 2),
    "PredEndValue": round(float(pred_end), 2),
    "ActualReturn_pct": round(pct_return(initial_capital, act_end), 2),
    "PredReturn_pct": round(pct_return(initial_capital, pred_end), 2),
    "Error_pct": round(pct_return(act_end, pred_end), 2)
})
act_end = actual_value_msr.sum()
pred_end = pred_value_msr.sum()
summary_rows.append({
    "Portfolio": "Mean Variance",
    "StartValue": initial_capital,
    "ActualEndValue": round(float(act_end), 2),
    "PredEndValue": round(float(pred_end), 2),
    "ActualReturn_pct": round(pct_return(initial_capital, act_end), 2),
    "PredReturn_pct": round(pct_return(initial_capital, pred_end), 2),
    "Error_pct": round(pct_return(act_end, pred_end), 2)
})

summary_table = pd.DataFrame(summary_rows)

# ---------------- Save outputs
os.makedirs(output_root, exist_ok=True)
# os.makedirs("output", exist_ok=True)

weight_table.to_csv(os.path.join(output_root, "weight_allocation_table.csv"), index=False)
stock_table_minimumVP.to_csv(os.path.join(output_root, "stock_table_minimumVP.csv"), index=False)
stock_table_meanVP.to_csv(os.path.join(output_root, "stock_table_meanVP.csv"), index=False)
summary_table.to_csv(os.path.join(output_root, "portfolio_summary.csv"), index=False)

# legacy output copies
weight_table.to_csv("output/weight_allocation_table.csv", index=False)
stock_table_minimumVP.to_csv("output/stock_table_minimumVP.csv", index=False)
stock_table_meanVP.to_csv("output/stock_table_meanVP.csv", index=False)
summary_table.to_csv("output/portfolio_summary.csv", index=False)

with pd.ExcelWriter(os.path.join(output_root, "portfolio_results_clean.xlsx"), engine="openpyxl") as writer:
    weight_table.to_excel(writer, sheet_name="Weight_Allocation", index=False)
    stock_table_minimumVP.to_excel(writer, sheet_name="Stock_Table_MinimumVP", index=False)
    stock_table_meanVP.to_excel(writer, sheet_name="Stock_Table_MeanVP", index=False)
    summary_table.to_excel(writer, sheet_name="Portfolio_Summary", index=False)

with pd.ExcelWriter("output/portfolio_results_clean.xlsx", engine="openpyxl") as writer:
    weight_table.to_excel(writer, sheet_name="Weight_Allocation", index=False)
    stock_table_minimumVP.to_excel(writer, sheet_name="Stock_Table_MinimumVP", index=False)
    stock_table_meanVP.to_excel(writer, sheet_name="Stock_Table_MeanVP", index=False)
    summary_table.to_excel(writer, sheet_name="Portfolio_Summary", index=False)

# ---------------- Efficient frontier (compute via quadratic solves) ----------------
present_tickers = [t for t in tickers_10 if t in mu.index]
if len(present_tickers) == 0:
    raise ValueError("No tickers present in mu for frontier plotting")

mu_f = mu.reindex(present_tickers).values
cov_f = cov.reindex(index=present_tickers, columns=present_tickers).values
n_f = len(present_tickers)

# bounds and simplex constraint
bnds = tuple((0.0, 1.0) for _ in range(n_f))
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

def minvar_for_target(target_ret):
    x0 = np.ones(n_f) / n_f
    def obj(w):
        return float(w.T @ cov_f @ w)
    cons_target = cons + ({'type': 'eq', 'fun': lambda w, target_ret=target_ret: float(np.dot(w, mu_f) - target_ret)},)
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons_target, options={'ftol':1e-12,'maxiter':1000})
    if not res.success:
        return None
    w = res.x
    ret = float(np.dot(w, mu_f))
    vol = float(np.sqrt(w.T @ cov_f @ w))
    return {'w': w, 'ret': ret, 'vol': vol}

# compute global minvar
def compute_minvar_global():
    x0 = np.ones(n_f)/n_f
    def obj(w): return float(w.T @ cov_f @ w)
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'ftol':1e-12,'maxiter':1000})
    if not res.success:
        return None
    w = res.x
    return {'w': w, 'ret': float(np.dot(w, mu_f)), 'vol': float(np.sqrt(w.T @ cov_f @ w))}

minvar_glob = compute_minvar_global()
if minvar_glob is None:
    ew = np.ones(n_f)/n_f
    min_ret = float(np.dot(ew, mu_f))
    min_vol = float(np.sqrt(ew.T @ cov_f @ ew))
else:
    min_ret = minvar_glob['ret']
    min_vol = minvar_glob['vol']

max_ret = float(np.max(mu_f))

n_points = 50
targets = np.linspace(min_ret, max_ret, n_points)
frontier_rets = []
frontier_vols = []
frontier_weights = []

for t in targets:
    res = minvar_for_target(t)
    if res is None:
        continue
    frontier_rets.append(res['ret'])
    frontier_vols.append(res['vol'])
    frontier_weights.append(res['w'])

# ---------------- Plot EF scatter + curve + markers (Plotly) ----------------
rand_df_plot = rand_df.copy()
rand_df_plot["return_pct"] = rand_df_plot["return"] * 100.0
rand_df_plot["vol_pct"] = rand_df_plot["volatility"] * 100.0

frontier_rets_pct = [r * 100.0 for r in frontier_rets]
frontier_vols_pct = [v * 100.0 for v in frontier_vols]

# equal weight (on present_tickers)
w_eq = np.array([1.0 / len(present_tickers)] * len(present_tickers))
eq_ret = float(np.dot(w_eq, mu_f)) * 100.0
eq_vol = float(np.sqrt(w_eq.T @ cov_f @ w_eq)) * 100.0

fig_frontier = go.Figure()

# Random portfolios cloud
fig_frontier.add_trace(go.Scatter(
    x=rand_df_plot["vol_pct"],
    y=rand_df_plot["return_pct"],
    mode="markers",
    marker=dict(
        size=6,
        color=rand_df_plot.get("sharpe", rand_df_plot["return"] / (rand_df_plot["volatility"] + 1e-9)),
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Sharpe"),
        opacity=0.55
    ),
    name="Random Portfolios"
))

# # EF curve
# fig_frontier.add_trace(go.Scatter(
#     x=frontier_vols_pct,
#     y=frontier_rets_pct,
#     mode="lines",
#     line=dict(width=3, color="black"),
#     name="Efficient Frontier"
# ))

# MVP (from random sampling)
fig_frontier.add_trace(go.Scatter(
    x=[mvp_vol_pct],
    y=[mvp_ret_pct],
    mode="markers+text",
    marker=dict(size=14, color="red", symbol="star"),
    text=["Min Var"],
    textposition="top center",
    name="Minimum Variance"
))

# MSR (from random sampling)
fig_frontier.add_trace(go.Scatter(
    x=[msr_vol_pct],
    y=[msr_ret_pct],
    mode="markers+text",
    marker=dict(size=14, color="green", symbol="star"),
    text=["Mean Var"],
    textposition="top center",
    name="Mean Variance"
))

# EW
fig_frontier.add_trace(go.Scatter(
    x=[eq_vol],
    y=[eq_ret],
    mode="markers+text",
    marker=dict(size=12, color="black", symbol="x"),
    text=["EW"],
    textposition="top center",
    name="Equal Weight"
))

fig_frontier.update_layout(
    title="Efficient Frontier Plot",
    xaxis_title="Annual Volatility (%)",
    yaxis_title="Annual Return (%)",
    template="plotly_white",
    legend=dict(x=1.2,y=1,xanchor="left",yanchor="top"),
    hovermode="closest",
    width=1200,
    height=600
)

png_path = os.path.join(output_root, "efficient_frontier_scatter.png")
html_path = os.path.join(output_root, "efficient_frontier_scatter.html")
try:
    fig_frontier.write_image(png_path, engine="kaleido", scale=2)
    print(f"Saved scatter EF PNG: {png_path}")
except Exception as e:
    print("Could not save frontier PNG:", e)
    fig_frontier.write_html(html_path)
    print(f"Saved interactive frontier HTML: {html_path}")

# ---------------- Allocation pie (MSR from random sampling) ----------------
alloc = pd.Series(w_msr_series.values, index=tickers_10)
alloc_nonzero = alloc[alloc > 0]

fig_pie = go.Figure(data=[go.Pie(labels=alloc_nonzero.index, values=alloc_nonzero.values, hole=0.4)])
fig_pie.update_layout(title="Asset Allocation (Mean Variance)")

pie_png = os.path.join(output_root, "allocation_pie_mean_variance.png")
pie_html = os.path.join(output_root, "allocation_pie_mean_variance.html")
try:
    fig_pie.write_image(pie_png, engine="kaleido", scale=2)
    print(f"Saved allocation pie PNG: {pie_png}")
except Exception as e:
    print("Could not save pie PNG:", e)
    fig_pie.write_html(pie_html)
    print(f"Saved allocation pie HTML: {pie_html}")

# ---------------- Risk Contribution % (MSR - sampled) ----------------
# Build present_tickers-aligned vector for MSR sampled weights
w_msr_present_vec = np.array([w_msr_series.get(t, 0.0) for t in present_tickers], dtype=float)
marginal = cov_f.dot(w_msr_present_vec)
rc = w_msr_present_vec * marginal
total_var = float(w_msr_present_vec.T @ cov_f @ w_msr_present_vec)
if total_var > 0:
    rc_pct = 100.0 * rc / total_var
else:
    rc_pct = np.zeros_like(rc)

rc_series = pd.Series(rc_pct, index=present_tickers).sort_values(ascending=False)

fig_rc = go.Figure()
fig_rc.add_trace(go.Bar(x=rc_series.index, y=rc_series.values))
fig_rc.update_layout(title="Risk Contribution % (Mean Variance)", yaxis_title="Risk Contribution (%)")

rc_png = os.path.join(output_root, "risk_contribution_meanVP.png")
rc_html = os.path.join(output_root, "risk_contribution_meanVP.html")
try:
    fig_rc.write_image(rc_png, engine="kaleido", scale=2)
    print(f"Saved risk contribution PNG: {rc_png}")
except Exception as e:
    print("Could not save risk contribution PNG:", e)
    fig_rc.write_html(rc_html)
    print(f"Saved risk contribution HTML: {rc_html}")

# ---------------- Final prints
print("\nSaved to folder:", output_root)
print(" -", os.path.join(output_root, "weight_allocation_table.csv"))
print(" -", os.path.join(output_root, "stock_table_minimumVP.csv"))
print(" -", os.path.join(output_root, "stock_table_meanVP.csv"))
print(" -", os.path.join(output_root, "portfolio_summary.csv"))
print(" -", os.path.join(output_root, "portfolio_results_clean.xlsx"))
print(" - EF Plot:", png_path if os.path.exists(png_path) else html_path)
print(" - MSR Pie (sampled):", pie_png if os.path.exists(pie_png) else pie_html)
print(" - Risk Contribution (sampled):", rc_png if os.path.exists(rc_png) else rc_html)

print("\nQuick stats (Annual):")
print(pd.DataFrame([{
    "Portfolio": "MVP",
    "AnnualReturn_pct": round(mvp_ret_pct,2),
    "AnnualRisk_pct": round(mvp_vol_pct,2),
    "Sharpe": round(mvp_sh,3)
},{
    "Portfolio": "OPR",
    "AnnualReturn_pct": round(msr_ret_pct,2),
    "AnnualRisk_pct": round(msr_vol_pct,2),
    "Sharpe": round(msr_sh,3)
}]))

# ---------------- Cleanup legacy files in output/ ----------------
files_to_delete = [
    r"output\portfolio_results_clean.xlsx",
    r"output\portfolio_summary.csv",
    r"output\stock_table_meanVP.csv",
    r"output\stock_table_minimumVP.csv",
    r"output\weight_allocation_table.csv"
]

for file_path in files_to_delete:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
# ---------------- Done ----------------

print()
print("Historical data used from:", returns_hist_intersection.index.min() if not returns_hist_intersection.empty else "N/A")
print("Historical data used until:", returns_hist_intersection.index.max() if not returns_hist_intersection.empty else "N/A")