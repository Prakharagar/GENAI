
#!/usr/bin/env python3
"""
SEC Filings Downloader
----------------------
Download filings (10-K, 8-K, DEF 14A, etc.) from the SEC EDGAR system for a list of tickers.

What this script does (step-by-step):
1) Loads a list of stock tickers (from CLI or a file) and normalizes them.
2) Downloads the official ticker->CIK mapping from the SEC once and caches it locally.
3) For each ticker, looks up its zero-padded CIK (10 digits).
4) Calls the SEC Submissions API (https://data.sec.gov/submissions/CIK##########.json)
   to get recent filings metadata (form types, accession numbers, primary document names, dates).
5) Filters filings to only include the requested form types (e.g., 10-K, 8-K, DEF 14A).
   - Flexible aliases supported: "10k" -> "10-K", "8k" -> "8-K", "def" -> "DEF 14A", "14a" -> "DEF 14A"
6) (Optional) Filters by "since" and/or "until" date (YYYY-MM-DD).
7) Limits to N filings PER FORM (default: 3).
8) Creates a folder structure: {out_dir}/{TICKER}/{FORM}/ and downloads for each filing:
    - Primary HTML document (if available)
    - (Optional) The "complete submission" text file with all exhibits
      (toggle with --full-submission true|false; default: false)
   Also writes a small JSON "manifest" capturing filing metadata and download paths.
9) Respects SEC's fair-use with a polite rate limit (sleep between requests) and a retry strategy.

Usage examples:
  - Simple (tickers inline):
    python sec_downloader.py --tickers AAPL,MSFT --forms 10-K,8-K,DEF 14A --out ./edgar_downloads

  - From a file (one ticker per line), last 5 per form since 2023-01-01:
    python sec_downloader.py --tickers-file ./tickers.txt --forms 10-K,8-K --per-form 5 --since 2023-01-01

  - Also store the complete submission .txt files:
    python sec_downloader.py --tickers TSLA --forms 10-K --full-submission true

Notes:
  * Provide a real, descriptive --user-agent per SEC guidance, e.g.:
     "Your Name Contact your.email@example.com"
  * Do NOT exceed SEC rate limits. This script sleeps between requests.
  * The "recent" submissions JSON typically contains up to ~100 of the most recent filings.
    If you need older filings beyond that, you'll need to paginate via filing detail indices,
    which is beyond this starter script.
"""
import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_no_pad}/{accession_no_dashes}"
DEFAULT_FORMS = ["10-K", "8-K", "DEF 14A","10-Q"]
SOURCE_DIR = Path("./edgar_downloads")
TARGET_DIR = Path("./data")

# ------------------------------ Utilities ------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

def zero_pad_cik(cik: int) -> str:
    return f"{int(cik):010d}"

def normalize_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-\.]", "", t or "").upper()

def normalize_form_alias(form: str) -> str:
    f = (form or "").strip().upper()
    # Common aliases mapping
    aliases = {
        "10K": "10-K",
        "10-Q": "10-Q", "10Q": "10-Q",
        "8K": "8-K",
        "DEF": "DEF 14A",  # most common "DEF" ask means proxy statement DEF 14A
        "14A": "DEF 14A",
        "DEFA14A": "DEFA14A",
        "DFAN14A": "DFAN14A",
        "SC 13D": "SC 13D", "SC13D": "SC 13D",
        "SC 13G": "SC 13G", "SC13G": "SC 13G",
    }
    return aliases.get(f, f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", s)

# ------------------------------ HTTP Session w/ Retries ------------------------------

def build_session(user_agent: str, max_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ------------------------------ CIK Mapping ------------------------------

def load_ticker_cik_map(session: requests.Session, cache_dir: Path, force: bool = False) -> Dict[str, str]:
    """
    Returns dict mapping normalized TICKER -> zero-padded CIK (string).
    Caches the SEC file under cache_dir/company_tickers.json
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "company_tickers.json"
    if not force and cache_file.exists() and cache_file.stat().st_mtime > (time.time() - 86400 * 3):
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        resp = session.get(SEC_TICKER_CIK_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")

    mapping = {}
    # The SEC JSON uses integer-like keys "0","1","2", each with {"ticker","cik_str",...}
    for _, row in data.items():
        tkr = normalize_ticker(row.get("ticker", ""))
        cik_padded = zero_pad_cik(int(row.get("cik_str", 0)))
        mapping[tkr] = cik_padded
    return mapping

# ------------------------------ Filings Query ------------------------------

@dataclass
class FilingMeta:
    form: str
    filingDate: str
    reportDate: Optional[str]
    accessionNumber: str
    primaryDocument: str
    cik: str
    ticker: str

    @property
    def accession_no_dashes(self) -> str:
        return self.accessionNumber.replace("-", "")

    @property
    def filing_folder_url(self) -> str:
        return SEC_ARCHIVES_BASE.format(cik_no_pad=str(int(self.cik)), accession_no_dashes=self.accession_no_dashes)

    @property
    def primary_doc_url(self) -> str:
        return f"{self.filing_folder_url}/{self.primaryDocument}"

    @property
    def full_submission_txt_url(self) -> str:
        base = SEC_ARCHIVES_BASE.format(cik_no_pad=str(int(self.cik)), accession_no_dashes=self.accession_no_dashes)
        return f"{base}/{self.accession_no_dashes}.txt"

    @property
    def index_html_url(self) -> str:
        return f"{self.filing_folder_url}-index.html"

def fetch_recent_filings(session: requests.Session, cik_padded: str) -> dict:
    url = SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=cik_padded)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

def filter_filings(payload: dict, forms_normalized: List[str], ticker: str, cik_padded: str,
                   since: Optional[str], until: Optional[str], per_form: int) -> List[FilingMeta]:
    # Convert since/until to date objects if provided
    def to_date(d: Optional[str]) -> Optional[datetime.date]:
        if not d:
            return None
        return datetime.strptime(d, "%Y-%m-%d").date()

    since_d = to_date(since)
    until_d = to_date(until)

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    accns = recent.get("accessionNumber", []) or []
    pdocs = recent.get("primaryDocument", []) or []
    fdates = recent.get("filingDate", []) or []
    rdates = recent.get("reportDate", []) or [None] * len(forms)

    candidates: List[FilingMeta] = []
    for form, acc, pd, fd, rd in zip(forms, accns, pdocs, fdates, rdates):
        form_norm = normalize_form_alias(form)
        if form_norm not in forms_normalized:
            continue
        # Date filters
        try:
            fd_date = datetime.strptime(fd, "%Y-%m-%d").date()
        except Exception:
            fd_date = None
        if since_d and fd_date and fd_date < since_d:
            continue
        if until_d and fd_date and fd_date > until_d:
            continue

        candidates.append(FilingMeta(
            form=form_norm,
            filingDate=fd,
            reportDate=rd if rd else None,
            accessionNumber=acc,
            primaryDocument=pd,
            cik=cik_padded,
            ticker=ticker,
        ))

    # Limit to N per form
    by_form: Dict[str, List[FilingMeta]] = {}
    for fm in candidates:
        by_form.setdefault(fm.form, []).append(fm)

    limited: List[FilingMeta] = []
    for ftype, lst in by_form.items():
        limited.extend(lst[:max(per_form, 0)])

    # Stable order: by filing date desc (as they come from SEC recent), then by form
    return limited

# ------------------------------ Downloading ------------------------------

def polite_sleep(last_call_ts: List[float], min_interval: float):
    """
    Ensures at least min_interval seconds between subsequent calls.
    """
    if last_call_ts and last_call_ts[0] is not None:
        elapsed = time.time() - last_call_ts[0]
        to_sleep = max(0.0, min_interval - elapsed)
        if to_sleep > 0:
            time.sleep(to_sleep)
    last_call_ts[:] = [time.time()]

def download_file(session: requests.Session, url: str, out_path: Path,
                  last_call_ts: List[float], min_interval: float) -> Tuple[bool, Optional[str]]:
    try:
        polite_sleep(last_call_ts, min_interval)
        resp = session.get(url, timeout=60)
        if resp.status_code == 404:
            return False, "404 Not Found"
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return True, None
    except Exception as e:
        return False, str(e)

# ------------------------------ Main Flow ------------------------------

def parse_tickers(tickers_csv: Optional[str], tickers_file: Optional[Path]) -> List[str]:
    result: List[str] = []
    if tickers_csv:
        parts = [p.strip() for p in tickers_csv.split(",") if p.strip()]
        result.extend(parts)
    if tickers_file and tickers_file.exists():
        for line in tickers_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                result.append(line)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in result:
        t_norm = normalize_ticker(t)
        if t_norm and t_norm not in seen:
            unique.append(t_norm)
            seen.add(t_norm)
    return unique

def reorganize():
    if not SOURCE_DIR.exists():
        print(f"Source dir {SOURCE_DIR} not found.")
        return
    
    for ticker_dir in SOURCE_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        dest_root = TARGET_DIR / f"dbe_{ticker}"
        dest_root.mkdir(parents=True, exist_ok=True)
        
        for form_dir in ticker_dir.iterdir():
            if not form_dir.is_dir():
                continue
            for file in form_dir.iterdir():
                if file.suffix.lower() in [".htm", ".html"]:
                    dest_file = dest_root / file.name
                    shutil.copy2(file, dest_file)  # use shutil.move if you want to move instead
                    print(f"Copied {file} -> {dest_file}")
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download SEC filings by ticker.")
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, e.g. AAPL,MSFT")
    p.add_argument("--tickers-file", type=Path, default=None, help="Path to a file with one ticker per line")
    p.add_argument("--forms", type=str, default=",".join(DEFAULT_FORMS),
                   help="Comma-separated form types (aliases allowed). Example: '10-K,8-K,DEF 14A'")
    p.add_argument("--since", type=str, default=None, help="Filter filings on/after this date (YYYY-MM-DD)")
    p.add_argument("--until", type=str, default=None, help="Filter filings on/before this date (YYYY-MM-DD)")
    p.add_argument("--per-form", type=int, default=3, help="Max filings per form to download (per ticker)")
    p.add_argument("--out", type=Path, default=Path("./edgar_downloads"), help="Output directory")
    p.add_argument("--user-agent", type=str, required=True,
                   help="Required: Identifying User-Agent per SEC guidance (e.g., 'Your Name Contact email@domain')")
    p.add_argument("--rate-limit", type=float, default=0.25,
                   help="Minimum seconds between HTTP requests (default: 0.25 = 4 req/s)")
    p.add_argument("--full-submission", type=str, choices=["true", "false"], default="false",
                   help="Also download the complete submission .txt file (default: false)")
    p.add_argument("--force-refresh-map", action="store_true",
                   help="Force refresh the ticker->CIK map cache")
    args = p.parse_args(argv)

    if not args.tickers and not args.tickers_file:
        p.error("Provide --tickers and/or --tickers-file")

    out_dir: Path = args.out
    ensure_dir(out_dir)
    cache_dir = out_dir / "_cache"
    ensure_dir(cache_dir)

    # Build session
    session = build_session(args.user_agent)

    # Ticker->CIK map
    try:
        ticker_cik_map = load_ticker_cik_map(session, cache_dir, force=args.force_refresh_map)
    except Exception as e:
        print(f"[{_now()}] ERROR: Failed to load ticker->CIK map: {e}", file=sys.stderr)
        return 2

    # Forms
    forms_req = [normalize_form_alias(f.strip()) for f in (args.forms or "").split(",") if f.strip()]
    if not forms_req:
        forms_req = DEFAULT_FORMS[:]

    # Tickers
    tickers = parse_tickers(args.tickers, args.tickers_file)
    if not tickers:
        print(f"[{_now()}] ERROR: No valid tickers provided.", file=sys.stderr)
        return 2

    print(f"[{_now()}] Starting download for {len(tickers)} tickers; forms={forms_req}; per_form={args.per_form}")
    last_call_ts = [None]  # for rate limiting

    summary_rows = []
    for tkr in tickers:
        cik = ticker_cik_map.get(normalize_ticker(tkr))
        if not cik:
            print(f"[{_now()}] WARNING: No CIK found for {tkr}; skipping.")
            continue

        tkr_dir = out_dir / tkr
        ensure_dir(tkr_dir)

        # Fetch recent filings JSON
        try:
            polite_sleep(last_call_ts, args.rate_limit)
            payload = fetch_recent_filings(session, cik)
        except Exception as e:
            print(f"[{_now()}] ERROR: Failed to fetch filings for {tkr} (CIK {cik}): {e}")
            continue

        filings = filter_filings(payload, forms_req, tkr, cik, args.since, args.until, args.per_form)
        if not filings:
            print(f"[{_now()}] INFO: No matching filings for {tkr}.")
            continue

        for fm in filings:
            form_dir = tkr_dir / fm.form.replace(" ", "_")
            ensure_dir(form_dir)

            base_name = f"{fm.form}_{fm.filingDate}_{fm.accession_no_dashes}"
            manifest = {
                "ticker": fm.ticker,
                "cik": fm.cik,
                "form": fm.form,
                "filingDate": fm.filingDate,
                "reportDate": fm.reportDate,
                "accessionNumber": fm.accessionNumber,
                "urls": {
                    "index_html": fm.index_html_url,
                    "primary_doc": fm.primary_doc_url,
                    "full_submission_txt": fm.full_submission_txt_url
                }
            }

            # Download primary document
            primary_ext = Path(fm.primaryDocument).suffix or ".html"
            primary_path = form_dir / sanitize_filename(f"{base_name}_primary{primary_ext}")
            ok_primary, err_primary = download_file(session, fm.primary_doc_url, primary_path, last_call_ts, args.rate_limit)

            # Optionally download the full submission txt (can be large)
            full_txt_path = None
            ok_full = False
            err_full = None
            if args.full_submission.lower() == "true":
                full_txt_path = form_dir / sanitize_filename(f"{base_name}_full_submission.txt")
                ok_full, err_full = download_file(session, fm.full_submission_txt_url, full_txt_path, last_call_ts, args.rate_limit)

            # Write manifest JSON
            manifest["downloads"] = {
                "primary_doc_path": str(primary_path) if ok_primary else None,
                "primary_doc_error": err_primary,
                "full_submission_path": str(full_txt_path) if (full_txt_path and ok_full) else None,
                "full_submission_error": err_full,
            }
            (form_dir / f"{base_name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            # Row for CSV summary
            summary_rows.append({
                "ticker": fm.ticker,
                "cik": fm.cik,
                "form": fm.form,
                "filingDate": fm.filingDate,
                "accessionNumber": fm.accessionNumber,
                "primary_doc_url": fm.primary_doc_url,
                "primary_doc_path": str(primary_path) if ok_primary else "",
                "full_submission_url": fm.full_submission_txt_url,
                "full_submission_path": str(full_txt_path) if (full_txt_path and ok_full) else "",
                "index_html_url": fm.index_html_url,
                "downloaded_at": _now(),
            })

            status_bits = []
            status_bits.append(f"primary={'OK' if ok_primary else ('ERR:'+str(err_primary))}")
            if args.full_submission.lower() == "true":
                status_bits.append(f"submission={'OK' if ok_full else ('ERR:'+str(err_full))}")
            print(f"[{_now()}] {tkr} {fm.form} {fm.filingDate} {fm.accessionNumber} -> " + "; ".join(status_bits))

    # Write a run summary CSV
    if summary_rows:
        csv_path = out_dir / "download_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[{_now()}] Wrote summary CSV: {csv_path} ({len(summary_rows)} rows)")

    print(f"[{_now()}] Done.")
    reorganize()
    return 0

if __name__ == "__main__":
    sys.exit(main())
