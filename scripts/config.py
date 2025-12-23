import os
from datetime import datetime

# --- Folder Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_data")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned_stock_data")

# --- Ensure folders exist ---
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(STOCK_DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# --- Sector PDFs ---
SECTOR_LINKS = {
    "Auto": "https://www.niftyindices.com/Factsheet/ind_nifty_auto.pdf",
    "Chemicals": "https://niftyindices.com/Factsheet/Factsheet_Nifty_Chemicals.pdf",
    "FMCG": "https://www.niftyindices.com/Factsheet/ind_nifty_FMCG.pdf",
    "Healthcare": "https://www.niftyindices.com/Factsheet/Factsheet_Nifty_Healthcare_Index.pdf",
    "IT": "https://www.niftyindices.com/Factsheet/ind_nifty_it.pdf",
    "Media": "https://www.niftyindices.com/Factsheet/ind_nifty_media.pdf",
    "Metal": "https://www.niftyindices.com/Factsheet/ind_nifty_metal.pdf",
    "Pharma": "https://www.niftyindices.com/Factsheet/ind_nifty_pharma.pdf",
    # "PrivateBank": "https://www.niftyindices.com/Factsheet/ind_nifty_private_bank.pdf",
    # "PSUBank": "https://www.niftyindices.com/Factsheet/ind_nifty_psu_bank.pdf",
    "OilGas": "https://www.niftyindices.com/Factsheet/Factsheet_nifty_oil_and_gas.pdf",
    "Bank": "https://www.niftyindices.com/Factsheet/ind_nifty_bank.pdf"
}

# --- Dates ---
START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
