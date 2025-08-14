import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from time import sleep

class SECFilerDownloader:
    def __init__(self, tickers,download_dir="download"):
        self.tickers = tickers
        self.download_dir =Path(download_dir)
        self.latest_file=os.path.join(self.download_dir,"file\\latest.csv")
        print(f"Reading data from {self.latest_file}")
    
        self.headers = {
            "User-Agent": "prakhar@gmail.com", 
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
 
    def download_html_files(self, ticker, df,form_types):
        """Download filing HTML files if not already downloaded."""
        ticker_dir = self.download_dir / "10k_html"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        df = df[df["ticker"] == ticker]

        for idx, row in df.iterrows():
            try:
                
                filing_date = row["filing_date"].replace("-", "")        
                url = row['url']
                filename = f"{ticker}-{filing_date}.htm"
                filepath = ticker_dir / filename
                
                if filepath.exists():
                    print(f"Already exists: {filename}")
                    downloaded_files.append(filepath)
                    return downloaded_files

                resp = requests.get(url, headers=self.headers)
                if resp.status_code == 404:
                    print(f"Not found: {url}")
                    continue
                elif resp.status_code == 403:
                    print(f"Access forbidden: {url}")
                    continue

                resp.raise_for_status()
                filepath.write_bytes(resp.content)
                print(f"Downloaded: {filename}")
                downloaded_files.append(filepath)
                sleep(0.2) 

            except Exception as e:
                print(f"Failed to download {ticker} {row.get('filing_date')}: {e}")

        return downloaded_files

    def run(self):
        filings_df = pd.read_csv(self.latest_file)
        filings_df["filing_date"] = pd.to_datetime(filings_df["filing_date"])
        filings_df = (
            filings_df.loc[filings_df.groupby(["ticker", "form_type"])["filing_date"].idxmax()]
            .reset_index(drop=True)
)
        filings_df["filing_date"] = filings_df["filing_date"].dt.strftime("%Y-%m-%d")
        print(filings_df)
        form_types = filings_df["form_type"].unique().tolist()

        # Save downloads
        for ticker in self.tickers:
            print(f"\nProcessing: {ticker}")
            for form_type in form_types:
                print(f"\nProcessing: {form_type}")
                downloaded_html_files=self.download_html_files(ticker, filings_df,form_type)

 