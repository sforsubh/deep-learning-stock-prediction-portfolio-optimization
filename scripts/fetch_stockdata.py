# fetch_stockdata.py

import os
import pandas as pd
import yfinance as yf

from scripts.config import STOCK_DATA_DIR, START_DATE, END_DATE, CLEAN_DATA_DIR


def fetch_stock_data(top_stock: dict, mapping_excel: str):
    # Load mapping file
    mapping = pd.read_excel(mapping_excel)
    mapping.columns = [c.strip() for c in mapping.columns]

    required_cols = ["Sector", "Company Name", "NSE Code", "Weight (%)"]
    for col in required_cols:
        if col not in mapping.columns:
            raise Exception(f"Column '{col}' missing in {mapping_excel}")

    # Remove NOT_FOUND rows
    mapping = mapping[mapping["NSE Code"] != "NOT_FOUND"].copy()

    # Iterate through each sector in top_stock
    for sector, companies in top_stock.items():
        print(f"\nSector: {sector}")

        # Create raw + clean folders
        sector_raw_dir = os.path.join(STOCK_DATA_DIR, sector)
        sector_clean_dir = os.path.join(CLEAN_DATA_DIR, sector)

        os.makedirs(sector_raw_dir, exist_ok=True)
        os.makedirs(sector_clean_dir, exist_ok=True)

        # List of just company names
        company_names = [name for name, _ in companies]

        df_sector = mapping[
            (mapping["Sector"] == sector) &
            (mapping["Company Name"].isin(company_names))
        ]

        if df_sector.empty:
            print(f"No matching companies found in Excel for sector: {sector}")
            continue

        for _, row in df_sector.iterrows():
            company = row["Company Name"]
            nse_code = row["NSE Code"]
            ticker = f"{nse_code}.NS"
            raw_path = os.path.join(sector_raw_dir, f"{nse_code}.csv")
            clean_path = os.path.join(sector_clean_dir, f"{nse_code}.csv")

            print(f"Updating {company} ({ticker}) ...")

            # -------------------------- DOWNLOAD RAW DATA --------------------------
            try:
                new_data = yf.download(
                    ticker,
                    start=START_DATE,
                    end=END_DATE,
                    progress=False,
                    auto_adjust=True
                )

                if new_data.empty:
                    print(f"No data returned for {ticker}")
                    continue

                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = [
                        "_".join([str(i) for i in col]).strip()
                        for col in new_data.columns.values
                    ]

                close_col = next((col for col in new_data.columns if "Close" in col), None)
                if not close_col:
                    print(f"'Close' column not found for {ticker}")
                    continue

                new_df = new_data[[close_col]].reset_index()
                new_df.rename(columns={'Date': 'date', close_col: 'close'}, inplace=True)

            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue

            new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
            new_df = new_df.sort_values(by='date')

            # -------------------------- SAVE / UPDATE RAW DATA --------------------------
            if os.path.exists(raw_path):
                old_df = pd.read_csv(raw_path, parse_dates=['date'])
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
                combined.sort_values(by='date', inplace=True)
                combined.to_csv(raw_path, index=False)
                print(f" Updated RAW file: {raw_path} ({len(combined)} rows)")
            else:
                new_df.to_csv(raw_path, index=False)
                print(f" Created RAW file: {raw_path}")

            # -------------------------- CREATE CLEANED VERSION --------------------------
            clean_df = pd.read_csv(raw_path, parse_dates=['date'])
            clean_df = clean_df.sort_values("date")

            # Reindex to business days
            all_days = pd.date_range(
                clean_df['date'].min(),
                clean_df['date'].max(),
                freq="B"
            )

            clean_df = clean_df.set_index("date").reindex(all_days)
            clean_df['close'] = clean_df['close'].ffill()

            clean_df.index.name = "date"
            clean_df = clean_df.reset_index()

            clean_df.to_csv(clean_path, index=False)
            print(f"   → Cleaned file created: {clean_path} ({len(clean_df)} rows)")

    print("\nAll stocks updated and cleaned successfully!")
