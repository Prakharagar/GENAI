# Save as extract_sections.py and run: python extract_sections.py
from bs4 import BeautifulSoup, NavigableString, Tag
import re, json, os, sys

HTML_FILE = "./download/MSFT/MSFT-20250730.htm"   # change if needed
OUT_JSON = "sections.json"

# If you already have your index list (ordered markers), set it here to force regex-mode:
manual_markers = None  # e.g. ["Business","Risk Factors", ...]

def read_html(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except:
        return raw.decode("latin-1")

def normalize_marker_to_regex(marker):
    words = re.split(r"\s+", marker.strip())
    words = [w for w in words if w != ""]
    return r"\s+".join(re.escape(w) for w in words)

def extract_text_between_ids(soup, items):
    result = {}
    n = len(items)
    for i, (marker, frag) in enumerate(items):
        key = marker.strip()
        start_id = frag.lstrip("#")
        end_id = items[i+1][1].lstrip("#") if i+1 < n else None
        start_elem = soup.find(id=start_id) if start_id else None
        end_elem = soup.find(id=end_id) if end_id else None
        if start_elem is None:
            result[key] = ""
            continue
        texts = []
        for node in start_elem.next_siblings:
            if isinstance(node, Tag) and end_elem is not None and (node is end_elem or node.find(id=end_id) is not None):
                break
            if isinstance(node, Tag) and node.find(id=end_id) is not None:
                break
            if isinstance(node, NavigableString):
                s = str(node).strip()
                if s:
                    texts.append(s)
            elif isinstance(node, Tag):
                s = node.get_text(separator=" ", strip=True)
                if s:
                    texts.append(s)
        combined = "\n\n".join(t for t in texts if t)
        combined = re.sub(r"[ \t]+", " ", combined)
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        result[key] = combined.strip()
    return result

def extract_text_by_regex(full_text, markers):
    result = {}
    pos = 0
    n = len(markers)
    for i, marker in enumerate(markers):
        key = marker.strip()
        pat_start = re.compile(normalize_marker_to_regex(marker), flags=re.IGNORECASE)
        m_start = pat_start.search(full_text, pos)
        if not m_start:
            m_start = pat_start.search(full_text)
        if not m_start:
            result[key] = ""
            continue
        start_end = m_start.end()
        if i+1 < n:
            pat_next = re.compile(normalize_marker_to_regex(markers[i+1]), flags=re.IGNORECASE)
            m_next = pat_next.search(full_text, start_end)
            if not m_next:
                m_next = pat_next.search(full_text)
                if m_next and m_next.start() <= start_end:
                    m_next = None
            if m_next:
                section_text = full_text[start_end:m_next.start()]
                pos = m_next.start()
            else:
                section_text = full_text[start_end:]
                pos = len(full_text)
        else:
            section_text = full_text[start_end:]
            pos = len(full_text)
        section_text = re.sub(r"[ \t]+", " ", section_text)
        section_text = re.sub(r"\n{3,}", "\n\n", section_text)
        result[key] = section_text.strip()
    return result

def autodiscover_toc_markers(soup):
    toc_candidates = soup.find_all(string=re.compile(r"Table of Contents", re.I))
    for cand in toc_candidates:
        parent = cand.parent if hasattr(cand, "parent") else None
        if parent is None:
            continue
        toc_table = parent.find_next("table")
        if toc_table:
            anchors = []
            for a in toc_table.find_all("a", href=True):
                text = a.get_text(" ", strip=True)
                href = a["href"]
                if re.fullmatch(r"\d+", text):  # skip page-number anchors
                    continue
                if len(text) < 2:
                    continue
                if href.startswith("#"):
                    anchors.append((text, href))
            if anchors:
                seen = set()
                unique = []
                for t, h in anchors:
                    key = (t.strip(), h.strip())
                    if key not in seen:
                        seen.add(key)
                        unique.append((t, h))
                return unique
    return None

def main():
    if not os.path.exists(HTML_FILE):
        print("HTML file not found:", HTML_FILE, file=sys.stderr)
        return
    html = read_html(HTML_FILE)
    soup = BeautifulSoup(html, "html.parser")
    if manual_markers is not None:
        sections = extract_text_by_regex(soup.get_text("\n"), manual_markers)
    else:
        items = autodiscover_toc_markers(soup)
        if items:
            sections = extract_text_between_ids(soup, items)
        else:
            guess_markers = ["Business", "Risk Factors", "Legal Proceedings", "Management's Discussion and Analysis"]
            sections = extract_text_by_regex(soup.get_text("\n"), guess_markers)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(sections)} sections to {OUT_JSON}")

if __name__ == "__main__":
    main()
