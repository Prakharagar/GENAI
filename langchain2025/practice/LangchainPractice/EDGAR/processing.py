from config_loader import *
from datetime import datetime 
from langchain_community.document_loaders import DirectoryLoader,UnstructuredHTMLLoader
import os, re, json

ticker="AMZN"
latest_file=f"{ticker}-20250207.htm"
loader =DirectoryLoader(
    path=f"download/{ticker}",
    loader_cls=UnstructuredHTMLLoader,
)

# Utility: clean extracted text
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)  # remove page numbers
    return text.strip()

# Utility: split 10-K text into sections
def split_10k_sections(text: str) -> dict:
    parts = re.split(r'(ITEM\s+\d+[A-Z]?\..*?)(?=ITEM\s+\d+[A-Z]?\.|$)',
                     text, flags=re.IGNORECASE)
    section_dict = {}
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i+1].strip()
        section_dict[title] = content
    return section_dict

# Output folder for processed JSON
output_dir = f"download/processed_json/{ticker}"
os.makedirs(output_dir, exist_ok=True)

# Process each loaded document
start_time = datetime.now()
documents = loader.load()

for idx, doc in enumerate(documents, start=1):
    # Clean text
    text = clean_text(doc.page_content)

    # Split into sections
    sections = split_10k_sections(text)

    # Extract year from metadata or fallback
    file_path = doc.metadata.get("source", "")
    year_match = re.search(r"(\d{4})", file_path)
    year = year_match.group(1) if year_match else "Unknown"

    # Save as JSON
    json_path = os.path.join(output_dir, f"{ticker}_{year}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "company": ticker,
            "year": year,
            **sections
        }, f, indent=2, ensure_ascii=False)

    print(f"[{idx}] Processed → {json_path}")

print(f"✅ Done in {datetime.now() - start_time}")



