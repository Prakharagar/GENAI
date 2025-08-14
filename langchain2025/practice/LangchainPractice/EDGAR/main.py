from config_loader import *
from secfetcher import Sec10KFetcher
from secdownloader import SECFilerDownloader
from pdfutils import PdfUtils
import os

os.environ["G_MESSAGES_DEBUG"] = ""


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN"]
    print(20*'#'," STEP1 ", 20*'#')
    fetcher = Sec10KFetcher(tickers)
    df_all = fetcher.run()
    print(20*'#'," STEP2 ", 20*'#')
    downloader = SECFilerDownloader(tickers)
    downloader.run()
    utils = PdfUtils(download_dir="download")
    utils.htm_to_pdf()
    utils.pdf_to_json()