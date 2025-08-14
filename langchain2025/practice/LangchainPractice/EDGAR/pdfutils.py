from weasyprint import HTML
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import re, os, json

class PdfUtils:
    def __init__(self, download_dir="download"):
        self.download_dir=download_dir
        self.download_pdf_dir = Path(download_dir) / "10k_pdf"
        self.download_json_dir = Path(download_dir) / "10k_json"
        self.download_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.download_json_dir.mkdir(parents=True, exist_ok=True)

    def htm_to_pdf(self, html_folder=f"download\\10k_html"):
        html_folder = Path(html_folder)
        if not html_folder.exists():
            raise FileNotFoundError(f"HTML folder not found: {html_folder}")

        for html_file in html_folder.glob("*.htm"):
            pdf_file = self.download_pdf_dir / (html_file.stem + ".pdf")
            if pdf_file.exists():
                print(f"[SKIP] PDF already exists: {pdf_file}")
                continue
            try:
                HTML(str(html_file)).write_pdf(str(pdf_file))
                print(f"[OK] Converted: {html_file.name} -> {pdf_file.name}")
            except Exception as e:
                print(f"[ERR] Failed to convert {html_file.name}: {e}")

    def pdf_to_json(self, pdf_folder=f"download\\10k_pdf"):
        pdf_folder = Path(pdf_folder)
        if not pdf_folder.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

        for pdf_file in pdf_folder.glob("*.pdf"):
            json_file = self.download_json_dir / (pdf_file.stem + ".json")
            if json_file.exists():
                print(f"[SKIP] JSON already exists: {json_file}")
                continue
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                docs = loader.load()
                full_text = "\n".join(doc.page_content for doc in docs)

                pattern = re.compile(r'(?im)^\s*Item\s+(\d+[A-Za-z0-9]*)\s*\.?\s*(.+)', re.MULTILINE)
                matches = list(pattern.finditer(full_text))
                if not matches:
                    pattern = re.compile(r'(?m)^\s*ITEM\s+(\d+[A-Za-z0-9]*)\s*\.?\s*(.+)')
                    matches = list(pattern.finditer(full_text))

                headings = []
                for m in matches:
                    num = m.group(1).strip()
                    title_text = m.group(2).strip()
                    title_text = re.sub(r'\s+\d+\s*$', '', title_text).strip()
                    headings.append((f"Item {num}. {title_text}", m.start()))

                headings.sort(key=lambda x: x[1])

                sections = {}
                for i, (heading, start_pos) in enumerate(headings):
                    end_pos = headings[i + 1][1] if i + 1 < len(headings) else len(full_text)
                    section_text = full_text[start_pos:end_pos].strip()
                    key = re.sub(r'(?i)^\s*Item\s+\d+[A-Za-z0-9]*\s*\.?\s*', '', heading).strip(" .:")
                    sections[key] = section_text

                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(sections, f, indent=4, ensure_ascii=False)

                print(f"[OK] JSON created: {json_file}")
            except Exception as e:
                print(f"[ERR] Failed to convert {pdf_file.name}: {e}")

