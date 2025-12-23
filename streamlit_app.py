# streamlit_app.py
# Stock Prediction & Portfolio Dashboard - Streamlit (supports folder structure:
# output/stock_data/<SECTOR>/<STOCK>/... )
#
# Save at repo root (same level as output/ and data/). Run:
# pip install -r requirements.txt
# streamlit run streamlit_app.py

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import textwrap
import os
from scipy.optimize import minimize
import plotly.graph_objects as go
import time

# ----------------------
# Paths (relative)
# ----------------------
ROOT = Path(__file__).parent
OUTPUT_STOCK_DIR = ROOT / "output" / "stock_data"
BEST_MODEL_CSV = ROOT / "output" / "best_model_summary.csv"   # expected by the pipeline
TOP_STOCKS_FILE = ROOT / "data" / "top_stocks_code.xlsx"  # optional mapping file

# ----------------------
# Fixed dates for pipeline (user requested fixed)
# ----------------------
CUTOFF_DATE = pd.Timestamp("2025-05-01")
EVAL_DATE = pd.Timestamp("2025-10-01")

# ----------------------
# Fixed risk-free rate (do not let user change)
# ----------------------
RISK_FREE_RATE = 0.01

# ----------------------
# Helper functions for file structure
# ----------------------
def get_all_sectors():
    """Return sorted list of sector folder names inside output/stock_data"""
    if not OUTPUT_STOCK_DIR.exists():
        return []
    return sorted([p.name for p in OUTPUT_STOCK_DIR.iterdir() if p.is_dir()])


def get_stocks_in_sector(sector_name: str):
    """Return sorted list of stock folder names inside a given sector"""
    sector_path = OUTPUT_STOCK_DIR / sector_name
    if not sector_path.exists():
        return []
    return sorted([p.name for p in sector_path.iterdir() if p.is_dir()])


def build_sector_mapping():
    """Return dict mapping sector -> [stocks], plus 'All' combining all stocks."""
    mapping = {}
    sectors = get_all_sectors()
    for sec in sectors:
        mapping[sec] = get_stocks_in_sector(sec)
    # Remove duplicates while preserving alphabetical order
    unique_stocks = sorted(list(dict.fromkeys(sum(mapping.values(), []))))
    mapping["All"] = unique_stocks

    return mapping


def list_stock_files(stock_folder: Path):
    """Return dictionary of files inside a stock folder keyed by lowercase stem."""
    files = {}
    if not stock_folder.exists():
        return files
    for p in stock_folder.iterdir():
        if p.is_file():
            files[p.stem.lower()] = p
    return files


def load_image_safe(img_path: Path):
    try:
        return Image.open(img_path)
    except Exception:
        return None


def find_sector_for_stock(stock_name: str, sectors_map: dict):
    """Return the sector name that contains stock_name (or None)."""
    for sec, stocks in sectors_map.items():
        if sec == "All":
            continue
        if stock_name in stocks:
            return sec
    return None


# Used by pipeline: find csv for a stock name (case-insensitive)
def find_stock_csv_dynamic(stock_name: str):
    stock_name = stock_name.strip()
    if not OUTPUT_STOCK_DIR.exists():
        return None
    for root, dirs, files in os.walk(str(OUTPUT_STOCK_DIR)):
        # check folder names
        for d in dirs:
            if d.upper() == stock_name.upper():
                # try expected filename pattern
                candidate = os.path.join(root, d, f"{stock_name}_ns_pred.csv")
                if os.path.exists(candidate):
                    return candidate
        # check files directly
        for f in files:
            if f.upper().startswith(stock_name.upper()) and f.upper().endswith("_NS_PRED.CSV"):
                return os.path.join(root, f)
    return None

# ----------------------
# UI helpers & CSS (Light/Dark)
# ----------------------
BASE_CSS = """
<style>
/* General app padding */
.reportview-container .main .block-container{
    padding-top: 1rem !important;
    padding-left: 12rem !important;
    padding-right: 12rem !important;
    max-width: 1000px !important;     /* prevent too-wide stretching */
    margin-left: auto !important;
    margin-right: auto !important;
}


/* Top navbar card */
.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding: 12px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

/* Filter strip */
.filter-strip {
    display:flex;
    gap:12px;
    align-items:center;
    padding:10px;
    border-radius:10px;
    margin-bottom:12px;
}

/* Stock card */
.stock-card {
    padding:12px;
    border-radius:10px;
    margin-bottom:12px;
    box-shadow: 0 6px 18px rgba(5,10,30,0.06);
}

/* Small KPI chips */
.kpi {
    display:inline-block;
    padding:8px 10px;
    margin-right:8px;
    border-radius:8px;
    font-weight:600;
}

/* subtle hr replacement */
.hr-subtle{
    height:1px;
    border: none;
    margin: 6px 0 18px 0;
    opacity: 0.12;
    border-radius: 2px;
}

/* responsive image */
.stock-img {
    max-width:100%;
    height:auto;
    border-radius:6px;
}

</style>
"""

LIGHT_THEME_CSS = """
<style>
.stApp {background: linear-gradient(180deg,#f8fbff 0%, #ffffff 100%); color: #0b1b2b;}
.navbar { background: rgba(255,255,255,0.88); }
.filter-strip { background: rgba(255,255,255,0.88); }
.stock-card { background: #ffffff; }
.kpi { background: rgba(15,23,42,0.06); color: #0b1b2b; }
.hr-subtle { background: linear-gradient(90deg, rgba(15,23,42,0.06), rgba(15,23,42,0.03)); }
</style>
"""

DARK_THEME_CSS = """
<style>
.stApp {background: linear-gradient(180deg,#071023 0%, #071426 100%); color: #e6eef8;}
.navbar { background: rgba(8,12,20,0.6); }
.filter-strip { background: rgba(10,14,22,0.6); }
.stock-card { background: rgba(8,12,20,0.5); }
.kpi { background: rgba(255,255,255,0.03); color: #e6eef8; }
.hr-subtle { background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)); }
</style>
"""


def apply_theme_css(theme: str):
    """theme: 'Light' or 'Dark'"""
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if theme == "Dark":
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ----------------------
# Streamlit page config + state
# ----------------------
st.set_page_config(page_title="Stock Prediction & Portfolio Dashboard", layout="wide", initial_sidebar_state="auto")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"  # default

# ----------------------
# Top navigation (full-width title) + theme toggle + last updated
# ----------------------
apply_theme_css(st.session_state.theme)

nav_html = textwrap.dedent(f"""
<div class="navbar" style="
    display:flex;
    justify-content:center;
    align-items:center;
    height:80px;
">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="font-size:42px; font-weight:900;">📈 SmartInvest: AI-Driven Stock Forecasting & Portfolio Planner</div>
  </div>
</div>
""")

st.markdown(nav_html + "</div></div>", unsafe_allow_html=True)

# cols_theme = st.columns([1, 6, 1])
# with cols_theme[2]:
#     new_theme = st.selectbox("Theme", options=["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
#     if new_theme != st.session_state.theme:
#         st.session_state.theme = new_theme
#         apply_theme_css(st.session_state.theme)
#         # st.experimental_rerun()
#         st.rerun()


st.markdown('<div class="hr-subtle"></div>', unsafe_allow_html=True)

# ----------------------
# Mode selector sits at top always (simple horizontal control)
# ----------------------
# mode = st.radio("Mode", options=["Plots", "Directional Accuracy", "Portfolio Builder"], horizontal=True)
mode = st.radio("Mode", options=["Plots", "Portfolio Builder"], horizontal=True)


# ----------------------
# Filters strip (only for Plots and Directional Accuracy modes)
# ----------------------
sectors_map = build_sector_mapping()
sector_list = sorted([s for s in sectors_map.keys() if s != "All"]) 

if mode != "Portfolio Builder":

    # --------------------------
    # SESSION STATE INITIALIZATION
    # --------------------------
    if "selected_sectors" not in st.session_state:
        st.session_state.selected_sectors = []

    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = []

    filter_cols = st.columns([3, 5, 3, 1])

    with st.container():
        st.markdown('<div class="filter-strip">', unsafe_allow_html=True)

        # --------------------------
        # SECTOR MULTISELECT
        # --------------------------
        with filter_cols[0]:
            sectors = st.multiselect(
                "Sector (multiselect)",
                options=sector_list,
                default=st.session_state.selected_sectors,
                key="sector_selector_" + st.session_state.get("_force_new_sector_key", "0")
            )
            st.session_state.selected_sectors = sectors

        # --------------------------
        # BUILD ALLOWED STOCK LIST
        # --------------------------
        allowed_stocks = []
        for sec in st.session_state.selected_sectors:
            allowed_stocks.extend(sectors_map.get(sec, []))
        allowed_stocks = sorted(list(dict.fromkeys(allowed_stocks)))

        # Keep only valid stocks from previous selection
        st.session_state.selected_stocks = [
            s for s in st.session_state.selected_stocks if s in allowed_stocks
        ]

        # --------------------------
        # STOCK MULTISELECT
        # --------------------------
        with filter_cols[1]:
            selected_stocks = st.multiselect(
                "Stocks (filtered by selected sectors)",
                options=allowed_stocks,
                default=st.session_state.selected_stocks,
                key="stock_multiselect"
            )
            st.session_state.selected_stocks = selected_stocks

        with filter_cols[2]:
            st.write("")

        with filter_cols[3]:
            if st.button("Refresh", key="refresh_button"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state["_force_new_sector_key"] = str(time.time())
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


else:
    # For Portfolio Builder we'll show an inline compact filter bar below (simple horizontal layout - Option A)
    all_stocks = build_sector_mapping().get("All", [])
    if not all_stocks:
        st.warning("No stocks found under output/stock_data/. Portfolio Builder will be disabled until stock data is present.")

    top_cols = st.columns([4, 2, 2, 1])
    with top_cols[0]:
        pb_stocks = st.multiselect("Choose stocks for portfolio", options=all_stocks, default=[])

    with top_cols[1]:
        initial_capital = st.number_input(
            "Initial capital (INR)",
            value=100000.0,
            min_value=100000.0,
            step=1000.0,
            format="%.2f"
        )
    with top_cols[2]:
        # show fixed risk-free rate as small text
        st.markdown(f"<small>Risk-free rate fixed at <b>{RISK_FREE_RATE*100:.2f}%</b></small>", unsafe_allow_html=True)
        st.markdown(f"<small>Cutoff: <b>{CUTOFF_DATE.date()}</b> - Eval: <b>{EVAL_DATE.date()}</b></small>", unsafe_allow_html=True)
    with top_cols[3]:
        run_opt = st.button("Run Optimization")

# ----------------------
# Main content logic
# ----------------------
# MODE: PLOTS
# ----------------------
if mode == "Plots":
    st.markdown("### Plots - Actual vs Predicted")

    if 'selected_stocks' not in locals() or not selected_stocks:
        st.info("Select one or more stocks to view plots.")
    else:

        btn_cols = st.columns(len(selected_stocks)) if len(selected_stocks) <= 5 else st.columns(5)

        clicked_stock = None
        for i, stock in enumerate(selected_stocks):
            col = btn_cols[i % 5]
            with col:
                if st.button(stock):
                    clicked_stock = stock

        st.markdown('<div class="hr-subtle"></div>', unsafe_allow_html=True)

        if clicked_stock:
            assigned_sector = find_sector_for_stock(clicked_stock, sectors_map)
            if not assigned_sector:
                st.warning(f"Could not determine sector for {clicked_stock}.")
            else:
                stock_folder = OUTPUT_STOCK_DIR / assigned_sector / clicked_stock
                files = list_stock_files(stock_folder)

                st.markdown(f"### {clicked_stock} - {assigned_sector}")
                st.markdown('<div class="stock-card">', unsafe_allow_html=True)

                left, right = st.columns([2, 1])

                # ---------------- LEFT + RIGHT SIDE: Horizontal model cards with image + metrics ----------------
                model_patterns = {
                    "LSTM model 6 Month Prediction": "lstm",
                    "CNN model 6 Month Prediction": "cnn",
                    "Hybrid model 6 Month Prediction": "hybrid"
                }

                # Populate model_files list
                # Populate model_files list
                model_files = []
                for model_name, pat in model_patterns.items():
                    img_path = None
                    metrics_path = None

                    # Find image
                    for n, p in files.items():
                        if pat.lower() in n.lower() and p.suffix.lower() == ".png":
                            img_path = p
                            break

                    # Find metrics CSV (any file containing 'metrics')
                    for n, p in files.items():
                        if "metrics" in n.lower() and p.suffix.lower() == ".csv":
                            metrics_path = p
                            break

                    if img_path:
                        model_files.append((model_name, img_path, metrics_path))

                if model_files:
                    cols_models = st.columns(len(model_files), gap="large")
                    for i, (model_name, img_path, metrics_path) in enumerate(model_files):
                        with cols_models[i]:
                            # Show model image
                            img = load_image_safe(img_path)
                            if img:
                                st.image(img, caption=model_name, use_container_width=True)

                            # Show metrics nicely
                            if metrics_path:
                                try:
                                    metrics_df = pd.read_csv(metrics_path)
                                    # Pick only the row for this model
                                    row = metrics_df[metrics_df["Model"].str.lower() == model_name.split()[0].lower()]
                                    if not row.empty:
                                        st.markdown("**Metrics:**")
                                        st.markdown(f"- RMSE: {row['RMSE'].values[0]:.4f}")
                                        st.markdown(f"- RMSE/Mean: {row['RMSE/MEAN'].values[0]:.4f}")
                                        st.markdown(f"- Directional Accuracy: {row['DirectionalAcc'].values[0]:.2f}%")
                                        st.markdown(f"- MAPE: {row['MAPE'].values[0]:.2f}")
                                    else:
                                        st.write("No metrics found for this model.")
                                except Exception as e:
                                    st.write(f"Metrics could not be read: {e}")
                            else:
                                st.write("No metrics CSV found.")

                else:
                    st.info("No model images found for this stock.")



                # # ---------- LEFT SIDE: Images ----------
                # with left:
                #     png_candidates = []
                #     patterns = ["_lstm_6m", "_cnn_6m", "_hybrid_6m", "_ns_all_6m", "lstm", "cnn", "hybrid"]
                #     for pat in patterns:
                #         for name, p in files.items():
                #             if pat in name:
                #                 png_candidates.append(p)

                #     if not png_candidates:
                #         for name, p in files.items():
                #             if p.suffix.lower() == ".png":
                #                 png_candidates.append(p)

                #     cols_img = st.columns(3)
                #     for i, p in enumerate(png_candidates[:3]):
                #         img = load_image_safe(p)
                #         if img:
                #             with cols_img[i]:
                #                 st.image(img, caption=p.name, width=260)


                # # ---------- RIGHT SIDE: Metrics, Predictions ----------
                # with right:
                #     metrics_df = None
                #     for n, p in files.items():
                #         if "metrics" in n and p.suffix.lower() == ".csv":
                #             try:
                #                 metrics_df = pd.read_csv(p)
                #             except Exception:
                #                 metrics_df = None
                #             break

                #     if metrics_df is not None:
                #         desired_cols = ["Model", "RMSE", "RMSE_mean", "Directional_Accuracy", "MAPE"]
                #         available = [c for c in desired_cols if c in metrics_df.columns]
                #         clean_df = metrics_df[available].copy()
                #         st.write("**Metrics**")
                #         st.dataframe(clean_df, height=180)
                #     else:
                #         st.write("No metrics.csv found.")


                #     pred_df = None
                #     for n, p in files.items():
                #         if "pred" in n and p.suffix.lower() == ".csv":
                #             try:
                #                 pred_df = pd.read_csv(p)
                #             except Exception:
                #                 pred_df = None
                #             break
                #     if pred_df is not None:
                #         st.write("**Sample Predictions**")
                #         st.dataframe(pred_df.head(), height=180)
                #     else:
                #         st.write("No prediction file found.")

                    # Removed directional accuracy section from PLOTS mode

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("Click on a stock name above to view its plots.")


# MODE: DIRECTIONAL ACCURACY
elif mode == "Directional Accuracy":
    st.markdown("### Directional Accuracy - comparison across selected stocks")
    if 'selected_stocks' not in locals() or not selected_stocks:
        st.info("Select stocks to show directional accuracy charts.")
    else:
        cols = st.columns(2)
        idx = 0
        for stock in selected_stocks:
            assigned_sector = find_sector_for_stock(stock, sectors_map)
            if not assigned_sector:
                continue
            stock_folder = OUTPUT_STOCK_DIR / assigned_sector / stock
            files = list_stock_files(stock_folder)

            img_path = None
            for n, p in files.items():
                if "directional" in n and p.suffix.lower() == ".png":
                    img_path = p
                    break

            if img_path is None:
                for n, p in files.items():
                    if p.suffix.lower() == ".png":
                        img_path = p
                        break

            col = cols[idx % 2]
            with col:
                st.subheader(stock)
                if img_path:
                    img = load_image_safe(img_path)
                    if img:
                        st.image(img, caption=img_path.name, width=550)
                else:
                    st.write("No directional image found.")
            idx += 1


# --------------------------
# PORTFOLIO BUILDER (Option A: dynamic in-app pipeline)
# --------------------------
elif mode == "Portfolio Builder":
    st.markdown("### Portfolio Builder")

    # pb_stocks, initial_capital, run_opt come from the inline top bar defined earlier
    if not ('pb_stocks' in locals() and pb_stocks):
        st.info("Pick stocks from the top filter bar to start the portfolio builder.")
    else:
        st.write(f"Selected stocks: {', '.join(pb_stocks)}")
        if run_opt:
            @st.cache_data(ttl=600)
            def run_portfolio_pipeline(selected_stocks, initial_capital, risk_free):
                """Runs the portfolio engine for a list of selected_stocks.
                Returns a dict of tables and plotly figures."""
                result = {}
                # Validate best_model_summary exists
                if not BEST_MODEL_CSV.exists():
                    raise FileNotFoundError(f"Missing required file: {BEST_MODEL_CSV}")

                bm = pd.read_csv(BEST_MODEL_CSV)
                required_cols = ["Sector", "Stock", "Best_By_RMSE_Mean_Model"]
                for c in required_cols:
                    if c not in bm.columns:
                        raise ValueError(f"best_model_summary.csv must contain column '{c}'")

                bm = bm.drop_duplicates(subset=["Stock"], keep="first").set_index("Stock")

                # selected tickers as uppercase (match BM index case)
                tickers = [t.strip() for t in selected_stocks]
                missing_from_bm = [t for t in tickers if t not in bm.index]
                if missing_from_bm:
                    # allow lowercase/uppercase mismatch: attempt uppercase match
                    missing_from_bm_case = [t for t in tickers if t.upper() not in bm.index.map(str.upper).tolist()]
                    if missing_from_bm_case:
                        raise ValueError(f"Missing stocks in best_model_summary.csv for: {missing_from_bm_case}")

                # Read each stock CSV (actual + selected pred col)
                actual_prices = pd.DataFrame()
                selected_preds = pd.DataFrame()
                sector_map = {}

                def pick_pred_column(model_name):
                    m = str(model_name).strip().upper()
                    if m == "CNN":
                        return "pred_cnn"
                    if m == "LSTM":
                        return "pred_lstm"
                    if m == "HYBRID":
                        return "pred_hybrid"
                    return model_name

                for s in tickers:
                    csv_path = find_stock_csv_dynamic(s)
                    if csv_path is None:
                        raise FileNotFoundError(f"No CSV found for {s} under {OUTPUT_STOCK_DIR}.")
                    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
                    # try to make sure 'actual' column exists (case-insensitive)
                    cols_low = {c.lower(): c for c in df.columns}
                    if 'actual' in cols_low:
                        df = df.rename(columns={cols_low['actual']: 'actual'})
                    else:
                        raise ValueError(f"'actual' column missing in {csv_path}")
                    df.columns = [c.lower() for c in df.columns]

                    model = bm.loc[s, "Best_By_RMSE_Mean_Model"]
                    pred_col = pick_pred_column(model).lower()
                    if pred_col not in df.columns:
                        raise ValueError(f"Pred column {pred_col} not found in {csv_path}")

                    actual_prices = actual_prices.join(df[["actual"]].rename(columns={"actual": s}), how="outer")
                    selected_preds = selected_preds.join(df[[pred_col]].rename(columns={pred_col: s}), how="outer")
                    sector_map[s] = bm.loc[s, "Sector"]

                # reorder columns per selected tickers
                actual_prices = actual_prices.reindex(columns=tickers)
                selected_preds = selected_preds.reindex(columns=tickers)

                # Build historical returns up to cutoff (forward-fill internal gaps only)
                hist_actual = actual_prices.loc[:CUTOFF_DATE].copy()
                # ensure each ticker has at least one pre-cutoff price
                no_data_tickers = [col for col in hist_actual.columns if hist_actual[col].dropna().empty]
                if len(no_data_tickers) > 0:
                    raise ValueError("Some tickers have no price data before cutoff_date: " + ", ".join(no_data_tickers))

                # forward-fill internal gaps only (no backfill before first valid)
                hist_ffill = hist_actual.copy()
                for col in hist_ffill.columns:
                    first_idx = hist_ffill[col].first_valid_index()
                    if first_idx is None:
                        continue
                    hist_ffill.loc[first_idx:, col] = hist_ffill.loc[first_idx:, col].ffill()

                returns_hist = hist_ffill.pct_change()
                returns_hist_intersection = returns_hist.dropna(axis=0, how='any').copy()
                if returns_hist_intersection.shape[0] < 5:
                    # allow pipeline but warn later
                    pass

                mu = returns_hist_intersection.mean() * 252.0
                cov = returns_hist_intersection.cov() * 252.0
                # numerical regularization
                eps_cov = 1e-8
                cov += np.eye(cov.shape[0]) * eps_cov

                # Clean tickers (intersection)
                clean_tickers = returns_hist_intersection.columns.tolist()
                mu_clean = mu.reindex(clean_tickers)
                cov_clean = cov.reindex(index=clean_tickers, columns=clean_tickers)

                # Random portfolio sampling
                N_RANDOM_PORTFOLIOS = 10000  # reduce a bit to keep UI snappy; adjust as needed
                RANDOM_SEED = 42
                np.random.seed(RANDOM_SEED)
                n_clean = len(clean_tickers)
                random_results = {"weights": [], "ret": [], "vol": [], "sharpe": []}
                mu_vec_clean = mu_clean.reindex(clean_tickers).values
                cov_mat_clean = cov_clean.reindex(index=clean_tickers, columns=clean_tickers).values
                cov_mat_clean += np.eye(n_clean) * 1e-10  # numerical stability 1e-10

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

                # pick sampled min variance and max sharpe
                min_var_idx = rand_df["volatility"].idxmin()
                max_sharpe_idx = rand_df["sharpe"].idxmax()

                w_min_clean = rand_df.loc[min_var_idx, clean_tickers].values
                w_max_clean = rand_df.loc[max_sharpe_idx, clean_tickers].values

                # expand utilities
                def expand_weights(opt_w, all_tickers, clean_list):
                    s = pd.Series(0.0, index=all_tickers)
                    try:
                        arr = np.asarray(opt_w).flatten()
                        if arr.size == len(clean_list):
                            s.loc[clean_list] = arr
                            return s.values
                    except Exception:
                        pass
                    try:
                        tmp = pd.Series(opt_w).squeeze()
                        tmp = tmp.reindex(clean_list).fillna(0.0)
                        s.loc[clean_list] = tmp.values.flatten()
                        return s.values
                    except Exception:
                        return s.values

                # full tickers = selected tickers
                all_tickers = tickers
                w_mvp = expand_weights(w_min_clean, all_tickers, clean_tickers)
                w_msr = expand_weights(w_max_clean, all_tickers, clean_tickers)
                w_mvp_series = pd.Series(np.asarray(w_mvp, dtype=float).flatten(), index=all_tickers)
                w_msr_series = pd.Series(np.asarray(w_msr, dtype=float).flatten(), index=all_tickers)

                # portfolio stats util
                mu_all = mu.reindex(all_tickers).fillna(0.0)
                cov_all = cov.reindex(index=all_tickers, columns=all_tickers).fillna(0.0)

                def port_stats_arr(w, mu_vec, cov_mat, rf=0.01):
                    w = np.asarray(w, dtype=float).flatten()
                    ret = float(np.dot(w, mu_vec))
                    vol = float(np.sqrt(max(w.T @ cov_mat @ w, 0.0)))
                    sharpe = (ret - rf) / vol if vol > 0 else np.nan
                    return ret * 100.0, vol * 100.0, sharpe

                mvp_ret_pct, mvp_vol_pct, mvp_sh = port_stats_arr(w_mvp, mu_all.values, cov_all.values, rf=risk_free)
                msr_ret_pct, msr_vol_pct, msr_sh = port_stats_arr(w_msr, mu_all.values, cov_all.values, rf=risk_free)

                # prices on cutoff & eval
                def last_available(series, date):
                    if date in series.index:
                        return series.loc[date]
                    idx = series.index[series.index <= date]
                    return series.loc[idx[-1]] if len(idx) else np.nan

                price_on_cutoff = pd.Series({s: last_available(actual_prices[s], CUTOFF_DATE) for s in all_tickers})
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

                price_on_eval_actual = pd.Series({s: last_available(actual_prices[s], EVAL_DATE) for s in all_tickers})
                price_on_eval_pred   = pd.Series({s: last_available(selected_preds[s], EVAL_DATE) for s in all_tickers})

                actual_value_mvp = shares_mvp * price_on_eval_actual + cash_left_mvp * (1.0 if not np.isnan(cash_left_mvp) else 0.0)
                pred_value_mvp   = shares_mvp * price_on_eval_pred   + cash_left_mvp * (1.0 if not np.isnan(cash_left_mvp) else 0.0)
                actual_value_msr = shares_msr * price_on_eval_actual + cash_left_msr * (1.0 if not np.isnan(cash_left_msr) else 0.0)
                pred_value_msr   = shares_msr * price_on_eval_pred   + cash_left_msr * (1.0 if not np.isnan(cash_left_msr) else 0.0)

                # weight allocation table
                weight_table = pd.DataFrame({
                    "Stock": all_tickers,
                    "Minimum Risk": w_mvp_series.apply(lambda x: f"{x:.4f}"),
                    "Mean Risk": w_msr_series.apply(lambda x: f"{x:.4f}")
                })
                weight_table = pd.concat([
                    weight_table,
                    pd.DataFrame([{
                        "Stock": "Annual Return (%)",
                        "Minimum Risk": f"{mvp_ret_pct:.2f}",
                        "Mean Risk": f"{msr_ret_pct:.2f}"
                    }]),
                    pd.DataFrame([{
                        "Stock": "Annual Risk (%)",
                        "Minimum Risk": f"{mvp_vol_pct:.2f}",
                        "Mean Risk": f"{msr_vol_pct:.2f}"
                    }])
                ], ignore_index=True)

                # stock tables
                # Function to clean None/NaN into "-"
                def clean(x):
                    if pd.isna(x):
                        return "-"
                    return x

                # ---------- BUILD STOCK TABLES WITH NEW FORMAT ----------
                # ---------------------------
                # Build per-stock rows for MVP & MSR (populate rows_mvp / rows_msr)
                # ---------------------------
                rows_mvp = []
                rows_msr = []

                for s in all_tickers:
                    # safe access / convert to float where possible
                    price_cut = float(price_on_cutoff.get(s, np.nan)) if not pd.isna(price_on_cutoff.get(s, np.nan)) else np.nan
                    price_eval_act = float(price_on_eval_actual.get(s, np.nan)) if not pd.isna(price_on_eval_actual.get(s, np.nan)) else np.nan
                    price_eval_pred = float(price_on_eval_pred.get(s, np.nan)) if not pd.isna(price_on_eval_pred.get(s, np.nan)) else np.nan

                    rows_mvp.append({
                        "Stock": s,
                        "Amt_Invstd": round(float(invested_mvp.get(s, 0.0)), 2),
                        "ActualPrice_1May": round(price_cut, 2) if not pd.isna(price_cut) else np.nan,
                        "Shares": round(float(shares_mvp.get(s, 0.0)),2),
                        "ActualPrice_1Oct": round(price_eval_act, 2) if not pd.isna(price_eval_act) else np.nan,
                        "ActualValue": round(float(shares_mvp.get(s, 0.0) * (price_eval_act if not pd.isna(price_eval_act) else 0.0)), 2),
                        "PredPrice_1Oct": round(price_eval_pred, 2) if not pd.isna(price_eval_pred) else np.nan,
                        "PredValue": round(float(shares_mvp.get(s, 0.0) * (price_eval_pred if not pd.isna(price_eval_pred) else 0.0)), 2)
                    })

                    rows_msr.append({
                        "Stock": s,
                        "Amt_Invstd": round(float(invested_msr.get(s, 0.0)), 2),
                        "ActualPrice_1May": round(price_cut, 2) if not pd.isna(price_cut) else np.nan,
                        "Shares": round(float(shares_msr.get(s, 0.0)),2),
                        "ActualPrice_1Oct": round(price_eval_act, 2) if not pd.isna(price_eval_act) else np.nan,
                        "ActualValue": round(float(shares_msr.get(s, 0.0) * (price_eval_act if not pd.isna(price_eval_act) else 0.0)), 2),
                        "PredPrice_1Oct": round(price_eval_pred, 2) if not pd.isna(price_eval_pred) else np.nan,
                        "PredValue": round(float(shares_msr.get(s, 0.0) * (price_eval_pred if not pd.isna(price_eval_pred) else 0.0)), 2)
                    })

                start_date_str = CUTOFF_DATE.strftime("%b %d, %Y")
                end_date_str   = EVAL_DATE.strftime("%b %d, %Y")

                def pct_return(start, end):
                    return (end - start) / start * 100.0


                def build_stock_table(portfolio_name, rows, invested_sum, actual_sum, pred_sum,
                                    actual_roi, pred_roi):

                    df = pd.DataFrame(rows)

                    # Add TOTAL row (with "-" for missing)
                    df.loc[len(df)] = {
                        "Stock": "TOTAL",
                        "Amt_Invstd": invested_sum,
                        "ActualPrice_1May": "-",
                        "Shares": "-",
                        "ActualPrice_1Oct": "-",
                        "ActualValue": actual_sum,
                        "PredPrice_1Oct": "-",
                        "PredValue": pred_sum
                    }

                    # Apply cleaner
                    df = df.applymap(clean)

                    # Add ROI (%) row at bottom
                    df.loc[len(df)] = {
                        "Stock": "ROI (%)",
                        "Amt_Invstd": "",
                        "ActualPrice_1May": "",
                        "Shares": "",
                        "ActualPrice_1Oct": "",
                        "ActualValue": f"Actual: {actual_roi}",
                        "PredPrice_1Oct": "",
                        "PredValue": f"Predicted: {pred_roi}"
                    }

                    # Multi-index header like the image
                    df.columns = pd.MultiIndex.from_tuples([
                        ("Stock", "Stock"),
                        (f"Date: {start_date_str}", "Amount Invested (₹)"),
                        (f"Date: {start_date_str}", "Actual Price (₹)"),
                        (f"Date: {start_date_str}", "No. of Stocks"),
                        (f"Date: {end_date_str}", "Actual Price (₹)"),
                        (f"Date: {end_date_str}", "Actual Value (₹)"),
                        (f"Date: {end_date_str}", "Predicted Price (₹)"),
                        (f"Date: {end_date_str}", "Predicted Value (₹)")
                    ])

                    return df
                
                # Fetch ROI values from summary table logic
                mvp_actual_roi = round(pct_return(initial_capital, actual_value_mvp.sum()), 2)
                mvp_pred_roi   = round(pct_return(initial_capital, pred_value_mvp.sum()), 2)

                msr_actual_roi = round(pct_return(initial_capital, actual_value_msr.sum()), 2)
                msr_pred_roi   = round(pct_return(initial_capital, pred_value_msr.sum()), 2)

                # Build final pretty tables
                final_mvp_table = build_stock_table(
                    "MVP",
                    rows_mvp,
                    round(invested_mvp.sum(),2),
                    round(actual_value_mvp.sum(),2),
                    round(pred_value_mvp.sum(),2),
                    mvp_actual_roi,
                    mvp_pred_roi
                )

                final_msr_table = build_stock_table(
                    "MSR",
                    rows_msr,
                    round(invested_msr.sum(),2),
                    round(actual_value_msr.sum(),2),
                    round(pred_value_msr.sum(),2),
                    msr_actual_roi,
                    msr_pred_roi    
                )


                # Efficient frontier - compute on present tickers (clean_tickers)
                present_tickers = [t for t in all_tickers if t in mu.index]
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
                n_points = 40
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

                # Build frontier plot (plotly)
                rand_df_plot = rand_df.copy()
                rand_df_plot["return_pct"] = rand_df_plot["return"] * 100.0
                rand_df_plot["vol_pct"] = rand_df_plot["volatility"] * 100.0
                frontier_rets_pct = [r * 100.0 for r in frontier_rets]
                frontier_vols_pct = [v * 100.0 for v in frontier_vols]
                w_eq = np.array([1.0 / len(present_tickers)] * len(present_tickers))
                eq_ret = float(np.dot(w_eq, mu_f)) * 100.0
                eq_vol = float(np.sqrt(w_eq.T @ cov_f @ w_eq)) * 100.0

                fig_frontier = go.Figure()

                # Random portfolios scatter
                fig_frontier.add_trace(go.Scatter(
                    x=rand_df_plot["vol_pct"],
                    y=rand_df_plot["return_pct"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=rand_df_plot.get("sharpe", rand_df_plot["return"] / (rand_df_plot["volatility"] + 1e-9)),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Sharpe", font=dict(color="black")),
                            tickfont=dict(color="black")
                        ),
                        opacity=0.55
                    ),
                    name="Random Portfolios"
                ))

                # MVP marker
                fig_frontier.add_trace(go.Scatter(
                    x=[mvp_vol_pct],
                    y=[mvp_ret_pct],
                    mode="markers+text",
                    marker=dict(size=14, color="red", symbol="star"),
                    text=["Min Var"],
                    textposition="top center",
                    textfont=dict(
                        color="darkgreen",
                        size=14,
                        family="Arial Black"
                    ),
                    name="Minimum Variance"
                ))

                # MSR marker
                fig_frontier.add_trace(go.Scatter(
                    x=[msr_vol_pct],
                    y=[msr_ret_pct],
                    mode="markers+text",
                    marker=dict(size=14, color="green", symbol="star"),
                    text=["Mean Var"],
                    textposition="top center",
                    textfont=dict(
                        color="darkred",
                        size=14,
                        family="Arial Black"
                    ),
                    name="Mean Variance"
                ))

                # Equal weight marker
                fig_frontier.add_trace(go.Scatter(
                    x=[eq_vol],
                    y=[eq_ret],
                    mode="markers+text",
                    marker=dict(size=12, color="orange", symbol="x"),
                    text=["EW"],
                    textposition="top center",
                    textfont=dict(
                        color="blue",
                        size=14,
                        family="Arial Black"
                    ),
                    name="Equal Weight"
                ))

                # Reduce axis width (unchanged)
                try:
                    xmin = min(frontier_vols_pct + rand_df_plot["vol_pct"].tolist()) * 0.9
                    xmax = max(frontier_vols_pct + rand_df_plot["vol_pct"].tolist()) * 1.05
                    fig_frontier.update_xaxes(range=[xmin, xmax])
                except Exception:
                    pass

                # Layout (background, grid, colors unchanged)
                fig_frontier.update_layout(
                    xaxis=dict(
                        title=dict(text="Annual Volatility (%)", font=dict(color="black")),
                        tickfont=dict(color="black"),
                        linecolor="black",
                        showgrid=True,
                        gridcolor="lightgray"
                    ),
                    yaxis=dict(
                        title=dict(text="Annual Return (%)", font=dict(color="black")),
                        tickfont=dict(color="black"),
                        linecolor="black",
                        showgrid=True,
                        gridcolor="lightgray"
                    ),

                    # *** ONLY LEGEND UPDATED (copied from top code) ***
                    legend=dict(
                        x=1.2,
                        y=1,
                        xanchor="left",
                        yanchor="top",
                        font=dict(color="black", size=16)
                    ),

                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hovermode="closest",
                    height=420,
                    width=700,
                    margin=dict(l=40, r=40, t=50, b=40)
                )



                # Allocation pie (MSR)
                alloc = pd.Series(w_msr_series.values, index=all_tickers)
                alloc_nonzero = alloc[alloc > 0]
                fig_pie = go.Figure(data=[go.Pie(
                    labels=alloc_nonzero.index, 
                    values=alloc_nonzero.values, 
                    hole=0.6,  # bigger hole
                    marker=dict(line=dict(color='white', width=2)),  # white border between slices
                    domain=dict(x=[0.05, 0.95], y=[0.05, 0.95])  # inset from borders
                )])
                fig_pie.update_layout(
                    height=420,
                    width=350,  # slightly narrower
                    margin=dict(l=20, r=20, t=30, b=20),
                    shapes=[  # white border around the plot area
                        dict(
                            type="rect",
                            xref="paper",
                            yref="paper",
                            x0=0, y0=0,
                            x1=1, y1=1,
                            line=dict(color="white", width=3),
                            fillcolor="rgba(0,0,0,0)"  # transparent fill
                        )
                    ]
                )




                # Risk contribution for present tickers
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
                fig_rc.add_trace(go.Bar(
                    x=rc_series.index,
                    y=rc_series.values,
                    text=[f"{v:.1f}%" for v in rc_series.values],  # show % on bars
                    textposition="auto",  # place text outside bar
                    marker=dict(color="#1f77b4")  # optional: consistent color
                ))
                fig_rc.update_layout(
                    xaxis_title="",  # remove x-axis title
                    yaxis_title="Risk Contribution (%)",
                    height=420,  # same height as donut
                    width=350,   # slightly narrower
                    margin=dict(l=20, r=20, t=30, b=20),
                    shapes=[  # white border
                        dict(
                            type="rect",
                            xref="paper",
                            yref="paper",
                            x0=0, y0=0,
                            x1=1, y1=1,
                            line=dict(color="white", width=3),
                            fillcolor="rgba(0,0,0,0)"  # transparent inside
                        )
                    ]
                )

                # pack results
                result["weight_table"] = weight_table
                result["stock_table_mvp"] = final_mvp_table
                result["stock_table_msr"] = final_msr_table
                result["fig_frontier"] = fig_frontier
                result["fig_pie"] = fig_pie
                result["fig_rc"] = fig_rc
                return result

            # Execute pipeline (with spinner)
            try:
                with st.spinner("Running portfolio pipeline (this may take a few seconds)..."):
                    pipeline_results = run_portfolio_pipeline(pb_stocks, initial_capital, RISK_FREE_RATE)
            except Exception as e:
                st.error(f"Portfolio pipeline error: {e}")
                pipeline_results = None

            if pipeline_results:
                st.success("Portfolio pipeline completed.")
                # Plots
                st.markdown("#### Efficient Frontier")
                st.plotly_chart(pipeline_results["fig_frontier"], use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)  # Add a line break
                
                # Allocation and Risk Contribution side-by-side
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("#### Asset Allocation % (Mean Variance)")
                    st.plotly_chart(pipeline_results["fig_pie"], use_container_width=True)
                with colB:
                    st.markdown("#### Risk Contribution % (Mean Variance)")
                    st.plotly_chart(pipeline_results["fig_rc"], use_container_width=True)

                # Tables
                st.markdown("<br>", unsafe_allow_html=True)  # Add a line break
                st.markdown("#### Weight Allocation Table")
                df = pipeline_results["weight_table"]

                # Convert to HTML with centered headers + values
                html_table = df.to_html(index=False, justify="center", classes="center-table")

                st.markdown("""
                    <style>
                    table.center-table {
                        width: 800px !important;        /* makes the table expand inside the container */
                    }
                    table.center-table th {
                        text-align: center !important;
                    }
                    table.center-table td {
                        text-align: center !important;
                    }
                    </style>
                            
                    <div style="width: 1000px; margin-left: auto; margin-right: auto;">
                """, unsafe_allow_html=True)

                st.markdown(html_table, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)  # Add a line break

                st.markdown("#### Stock Table - Minimum Variance Portfolio")
                st.dataframe(pipeline_results["stock_table_mvp"])

                st.markdown("<br>", unsafe_allow_html=True)  # Add a line break

                st.markdown("#### Stock Table - Mean Variance Portfolio")
                st.dataframe(pipeline_results["stock_table_msr"])

# # Footer
# st.markdown("<div style='margin-top:18px;opacity:0.7;font-size:12px'>This dashboard reads relative paths from <code>output/stock_data/&lt;SECTOR&gt;/&lt;STOCK&gt;/</code>. Edit the constants at the top if your structure is different.</div>", unsafe_allow_html=True)