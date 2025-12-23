import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path("output/stock_data")
OUT_ALL_METRICS = Path("output/all_metrics.csv")
OUT_BEST_SUMMARY = Path("output/best_model_summary.csv")


def read_metrics(metrics_path):
    try:
        return pd.read_csv(metrics_path)
    except Exception as e:
        print(f"Error reading {metrics_path}: {e}")
        return None


def main():
    all_metrics_rows = []
    best_summary_rows = []

    # Loop through sectors
    for sector in sorted(os.listdir(BASE_DIR)):
        sector_path = BASE_DIR / sector
        if not sector_path.is_dir():
            continue

        # Loop through stocks
        for stock in sorted(os.listdir(sector_path)):
            stock_dir = sector_path / stock
            if not stock_dir.is_dir():
                continue

            # Expected metrics file
            metrics_files = list(stock_dir.glob("*_metrics.csv"))
            if not metrics_files:
                print(f"No metrics file found for {sector}/{stock}")
                continue

            df = read_metrics(metrics_files[0])
            if df is None or df.empty:
                continue

            # Add all rows to all_metrics.csv
            for _, row in df.iterrows():
                all_metrics_rows.append({
                    "Sector": sector,
                    "Stock": stock,
                    "Model": row["Model"],
                    "MSE": row["MSE"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "RMSE/MEAN": row["RMSE/MEAN"],
                    "DirectionalAcc": row["DirectionalAcc"],
                    "MAPE": row["MAPE"]
                })

            # --- Best by RMSE ---
            best_rmse_row = df.loc[df["RMSE"].idxmin()]
            best_rmse_model = best_rmse_row["Model"]
            best_rmse_value = best_rmse_row["RMSE"]

            # --- Best by RMSE/MEAN ---
            best_rmean_row = df.loc[df["RMSE/MEAN"].idxmin()]
            best_rmean_model = best_rmean_row["Model"]
            best_rmean_value = best_rmean_row["RMSE/MEAN"]

            # Add summary
            best_summary_rows.append({
                "Sector": sector,
                "Stock": stock,
                "Best_By_RMSE_Model": best_rmse_model,
                "Best_By_RMSE_Value": best_rmse_value,
                "Best_By_RMSE_Mean_Model": best_rmean_model,
                "Best_By_RMSE_Mean_Value": best_rmean_value
            })

    # Save results
    pd.DataFrame(all_metrics_rows).to_csv(OUT_ALL_METRICS, index=False)
    pd.DataFrame(best_summary_rows).to_csv(OUT_BEST_SUMMARY, index=False)

    print("\nFiles generated:")
    print(f" - {OUT_ALL_METRICS}")
    print(f" - {OUT_BEST_SUMMARY}")


if __name__ == "__main__":
    main()
