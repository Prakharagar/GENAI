
SEC Filings Downloader — Quick Start
====================================

1) Save tickers to a file (one per line). Example:
   AAPL
   MSFT
   TSLA

2) Run the script (replace the user agent with your info):
   python sec_downloader.py --tickers-file ./tickers_sample.txt --forms 10-K,8-K,DEF 14A --per-form 2 --out ./edgar_downloads --user-agent "Prakhar Agarwal Contact prakhar@mail.com"

3) Optional filters:
   --since 2023-01-01   # only filings on/after this date
   --until 2025-08-01   # only filings on/before this date

4) To download the complete submission text files too:
   --full-submission true

Outputs:
  ./edgar_downloads/{TICKER}/{FORM}/
    - <FORM>_<FILINGDATE>_<ACCESSION>_primary.html (or .htm/.html)
    - <FORM>_<FILINGDATE>_<ACCESSION>_full_submission.txt (if enabled)
    - <FORM>_<FILINGDATE>_<ACCESSION>_manifest.json
  ./edgar_downloads/download_summary.csv (run summary)
