import os
import re
import requests
import pandas as pd
from datetime import datetime

class Sec10KFetcher:
    def __init__(self, tickers, base_dir="download"):
        self.tickers = [t.upper() for t in tickers]
        self.base_dir = base_dir
        self.headers = {"User-Agent": "prakhar@example.com"}
        os.makedirs(os.path.join(self.base_dir, "file"), exist_ok=True)
        self.ticker_cik_map = self._load_ticker_cik_mapping()

    def _load_ticker_cik_mapping(self):
        """Download ticker-to-CIK mapping from SEC."""
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        mapping = {record["ticker"].upper(): str(record["cik_str"]).zfill(10)
                   for record in data.values()}
        return mapping

    def _get_10k_links_for_ticker(self, ticker):
        """Fetch all matching 10-K HTM links for a single ticker."""
        cik = self.ticker_cik_map.get(ticker)
        if not cik:
            print(f"CIK not found for ticker {ticker}")
            return []

        cik_int = str(int(cik))  # integer form for URLs
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(submissions_url, headers=self.headers)
        if resp.status_code == 404:
            print(f"⚠ No submissions found for {ticker}")
            return []
        resp.raise_for_status()
        sub_data = resp.json()

        results = []
        recent_filings = sub_data.get("filings", {}).get("recent", {})
        form_types = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])

        pattern = re.compile(rf"^{ticker.lower()}-\d{{8}}\.htm")

        for i, form in enumerate(form_types):
            if form == "10-K":
                acc_no_no_dashes = accession_numbers[i].replace("-", "")
                filing_index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_no_dashes}/index.json"
                filing_resp = requests.get(filing_index_url, headers=self.headers)
                if filing_resp.status_code != 200:
                    continue
                filing_data = filing_resp.json()
               
                for file_info in filing_data.get("directory", {}).get("item", []):
                    filename = file_info["name"]
                    if pattern.match(filename):
                        htm_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_no_dashes}/{filename}"
                        results.append({
                            "filing_date": filing_dates[i],
                            "form_type":form,
                            "cik": cik,
                            "ticker": ticker,
                            "url": htm_url
                        })
        return results

    def run(self):
        """Fetch data for all tickers and save combined CSV."""
        all_results = []
        for ticker in self.tickers:
            print(f"Fetching 10-K HTM links for {ticker}...")
            ticker_results = self._get_10k_links_for_ticker(ticker)
            all_results.extend(ticker_results)

        df = pd.DataFrame(all_results)
        if df.empty:
            print("⚠ No data found for any ticker.")
            return df

        # Save combined CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_csv = os.path.join(self.base_dir, "file", f"{timestamp}.csv")
        latest_csv = os.path.join(self.base_dir, "file", "latest.csv")

        df.to_csv(timestamped_csv, index=False)
        df.to_csv(latest_csv, index=False)

        print(f"✅ Saved timestamped CSV: {timestamped_csv}")
        print(f"✅ Saved latest CSV: {latest_csv}")
        return df
