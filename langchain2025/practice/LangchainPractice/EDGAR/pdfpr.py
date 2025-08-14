from langchain_community.document_loaders import PyMuPDFLoader
import re
import json

# -------- CONFIG --------
pdf_path = "download\10k_pdf\MSFT-20250730.pdf"          # Input PDF
output_json_path = "pdf_sections_by_items3.json"  # Output JSON
# ------------------------

# Step 1: Load PDF as text using LangChain
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# Step 2: Merge all pages into one text block
full_text = "\n".join(doc.page_content for doc in docs)

# Step 3: Find "Item" headings
pattern = re.compile(r'(?im)^\s*Item\s+(\d+[A-Za-z0-9]*)\s*\.?\s*(.+)', re.MULTILINE)
matches = list(pattern.finditer(full_text))

if not matches:
    pattern2 = re.compile(r'(?m)^\s*ITEM\s+(\d+[A-Za-z0-9]*)\s*\.?\s*(.+)')
    matches = list(pattern2.finditer(full_text))

# Step 4: Store headings and positions
headings = []
for m in matches:
    num = m.group(1).strip()
    title_text = m.group(2).strip()
    title_text = re.sub(r'\s+\d+\s*$', '', title_text).strip()
    headings.append((f"Item {num}. {title_text}", m.start()))

headings.sort(key=lambda x: x[1])

# Step 5: Extract sections
sections = {}
for i, (heading, start_pos) in enumerate(headings):
    end_pos = headings[i+1][1] if i+1 < len(headings) else len(full_text)
    section_text = full_text[start_pos:end_pos].strip()
    key = re.sub(r'(?i)^\s*Item\s+\d+[A-Za-z0-9]*\s*\.?\s*', '', heading).strip(" .:")
    sections[key] = section_text

# Step 6: Save JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=4, ensure_ascii=False)

print(f"✅ Extracted {len(sections)} sections to {output_json_path}")
