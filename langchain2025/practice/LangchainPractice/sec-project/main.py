from config_loader import *
from secfetcher import Sec10KFetcher
from secdownloader import SECFilerDownloader
from pdfutils import PdfUtils
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["G_MESSAGES_DEBUG"] = ""
os.environ["GIO_MODULE_DIR"] = "/nonexistent"

if __name__ == "__main__":
    tickers = ["AAPL", "MMM", "ABT","BRK-B","CVX","COST","DELL","WMT"]
    print(20*'#'," STEP1 ", 20*'#')
    fetcher = Sec10KFetcher(tickers)
    df_all = fetcher.run()
    if df_all.empty:
        raise ValueError("None of Ticker exists")
    print(20*'#'," STEP2 ", 20*'#')
    downloader = SECFilerDownloader(tickers)
    downloader.run()
    print(20*'#'," STEP3 ", 20*'#')
    utils = PdfUtils(download_dir="download")
    utils.htm_to_pdf()
    utils.pdf_to_json()