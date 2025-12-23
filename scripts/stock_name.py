# stock_name.py

import os, requests, re, PyPDF2, csv, json
import pandas as pd
from scripts.config import SECTOR_LINKS, RAW_PDF_DIR, DATA_DIR


# Downloading PDF files of sectors.
def download_sector_pdfs():
    for sector, url in SECTOR_LINKS.items():
        filename = os.path.join(RAW_PDF_DIR, f"{sector}.pdf")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {sector}")
        else:
            print(f"Failed: {sector} ({response.status_code})")


# Extracting Stocks Name 
def extract_top_stocks(num_top=5):
    top_stocks = {}
    for pdf_file in os.listdir(RAW_PDF_DIR):
        sector = pdf_file.replace(".pdf", "")
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)

        extracted = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if not text:
                    continue
                text = text.replace("\r", "").replace("\t", " ")
                lines = text.split("\n")

                start_index = next((i for i, l in enumerate(lines)
                                    if re.search("Top constituents by weightage", l, re.I)), -1)
                if start_index == -1:
                    continue

                for line in lines[start_index + 1:]:
                    if re.search(r"(Sector|Index|Source|#|As on)", line, re.I):
                        break
                    match = re.match(r"^([\w&\.\'\s\-\(\)]+?)\s+([\d\.]+)\s*%?$", line)
                    if match:
                        name, wt = match.group(1).strip(), float(match.group(2))
                        extracted.append((name, wt))
                    if len(extracted) >= num_top:
                        break
                if extracted:
                    break
        top_stocks[sector] = extracted
    return top_stocks

#

def create_sector_csv(top_stocks):
    excel_path = os.path.join(DATA_DIR, "top_stocks_code.xlsx")

    # Ensure DATA_DIR exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # If file exists, load it; else create an empty DataFrame with headers
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
    else:
        existing_df = pd.DataFrame(columns=["Sector", "Company Name", "NSE Code", "Weight (%)"])
        existing_df.to_excel(excel_path, index=False)
        print(f"Created new Excel file: {excel_path}")

    # Convert existing company names to a set for quick lookup
    existing_companies = set(existing_df["Company Name"].astype(str))

    # Prepare new rows to add
    new_rows = []
    for sector, companies in top_stocks.items():
        for name, weight in companies:
            if name not in existing_companies:
                new_rows.append({
                    "Sector": sector,
                    "Company Name": name,
                    "NSE Code": "",   # or get code from JSON if needed
                    "Weight (%)": weight
                })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)
        print(f"Updated Excel file with {len(new_rows)} new entries.")
    else:
        print("No new companies to add — Excel already up to date.")

    print(f"Excel saved at: {excel_path}")


