#!/usr/bin/env python3
"""
SEC HTML -> JSON (Hierarchical Sections + LLM-friendly Tables, NO unit changes)

- Reads:  ./data/dbe_<TICKER>/*.htm|*.html
- Writes: ./json_data/dbe_<TICKER>/<same-filename>.json

What this script guarantees:
1) Proper hierarchy:
   sections (h1/heuristic) -> subsections (h2..h6) -> paragraphs/tables attached at the correct depth.
2) Tables are mapped by headers to values without modifying units or parsing numbers:
   - colspan/rowspan expanded to a rectangular grid
   - header rows detected from leading <tr> with <th>
   - composite column labels built by concatenating multiple header levels
   - data rows map column_key -> raw string cell
   - also provides a “records” list for easy row/column/value access

Dependencies: beautifulsoup4 (bs4)
    pip install beautifulsoup4
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString

# ---------- Text utils ----------

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[☐☒§•►■▪]", "", s)
    # keep only printable utf-8 characters
    s = ''.join(ch for ch in s if ch.isprintable())
    return re.sub(r"\s+", " ", s).strip()

def normalize_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "col"

# ---------- Heading heuristics ----------
ITEM_HEADING_RE = re.compile(r"^\s*item\s+\d+[a-z]?(?:\.\d+)?\b", re.IGNORECASE)

def is_heading_tag(tag: Tag) -> bool:
    return bool(tag.name) and tag.name.lower() in {"h1","h2","h3","h4","h5","h6"}

def heading_level(tag: Tag, text: str) -> int:
    """Return heading level from <h1>.. or fall back to heuristic 'Item X.' as h2."""
    if is_heading_tag(tag):
        try:
            return int(tag.name[1])
        except Exception:
            return 2
    # Heuristic: treat 'Item X.' lines as level 2
    if ITEM_HEADING_RE.match(text or ""):
        return 2
    return 0

# ---------- Table parsing with colspan/rowspan expansion (NO unit changes) ----------

def expand_table_to_grid(table: Tag) -> List[List[str]]:
    """
    Expand an HTML <table> into a rectangular grid of strings, respecting colspan/rowspan.
    Returns a list of rows; each row is a list of cell texts.
    """
    grid: List[List[str]] = []
    # span_map: col_idx -> (text, rows_remaining)
    span_map: Dict[int, Tuple[str, int]] = {}

    for tr in table.find_all("tr"):
        row: List[str] = []
        col_idx = 0

        # Fill cells carried from rowspans
        def fill_rowspan_cells():
            nonlocal col_idx
            while True:
                # if current col_idx is occupied by a carried cell, place it
                if col_idx in span_map and span_map[col_idx][1] > 0:
                    text, remaining = span_map[col_idx]
                    row.append(clean_text(text))
                    remaining -= 1
                    if remaining > 0:
                        span_map[col_idx] = (text, remaining)
                    else:
                        del span_map[col_idx]
                    col_idx += 1
                else:
                    # No carried cell at this column index
                    break

        cells = tr.find_all(["th","td"])
        for cell in cells:
            fill_rowspan_cells()
            text = clean_text(cell.get_text(" ", strip=True))
            colspan = int(cell.get("colspan", "1") or "1")
            rowspan = int(cell.get("rowspan", "1") or "1")

            # Place this cell text colspan times
            for _ in range(colspan):
                row.append(text)
                # Mark this column for future rows if rowspan > 1
                if rowspan > 1:
                    span_map[col_idx] = (text, rowspan - 1)
                col_idx += 1

        # After processing cells, we may still have trailing carried cells
        fill_rowspan_cells()

        grid.append(row)

    # Normalize row lengths (pad with empty strings so all rows have equal length)
    max_len = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < max_len:
            r.extend([""] * (max_len - len(r)))
    return grid

def detect_header_row_count(table: Tag) -> int:
    """
    Count how many leading rows are header rows.
    We treat a row as a header row if it contains at least one <th>.
    """
    count = 0
    for tr in table.find_all("tr"):
        has_th = any(c.name == "th" for c in tr.find_all(["th","td"]))
        # But we must be precise: row is header if any <th> present
        if any(c.name == "th" for c in tr.find_all(["th","td"])):
            count += 1
        else:
            break
    return count

def build_composite_column_labels(header_rows: List[List[str]]) -> List[Dict[str,str]]:
    """
    From expanded header rows, build composite column labels (one per final column).
    Each label is a ' | '-joined path from top header to bottom header for that column.
    Also returns a stable 'key' for programmatic use.
    """
    if not header_rows:
        # Unknown headers: synthesize generic columns
        return [{"key": f"col_{i}", "label": f"Column {i}"} for i in range(100)]

    max_cols = max(len(r) for r in header_rows)
    labels: List[Dict[str,str]] = []
    for c in range(max_cols):
        parts = []
        last = None
        for r in header_rows:
            cell = (r[c] if c < len(r) else "").strip()
            if cell and cell != last:
                parts.append(cell)
                last = cell
        label = " | ".join(parts).strip()
        key = normalize_key(label) if label else f"col_{c}"
        labels.append({"key": key, "label": label or f"Column {c}"})
    return labels

def extract_table_title(table: Tag) -> Optional[str]:
    """
    Prefer <caption>, otherwise look back a few siblings for a sentence-like title (e.g., ends with ':').
    """
    cap = table.find("caption")
    if cap:
        t = clean_text(cap.get_text(" ", strip=True))
        if t:
            return t
    # Look back up to ~4 sibling blocks for a title-ish line
    count = 0
    for sib in table.previous_siblings:
        if isinstance(sib, NavigableString):
            continue
        if not isinstance(sib, Tag):
            continue
        if sib.name == "table":
            break
        txt = clean_text(sib.get_text(" ", strip=True))
        if txt:
            count += 1
            if txt.endswith(":") or len(txt) <= 160:
                return txt
        if count >= 4:
            break
    return None

def table_to_json(table: Tag) -> Dict[str, Any]:
    """
    Convert <table> to a JSON object with:
      - title
      - header_rows (expanded grid slice)
      - columns (composite labels with keys)
      - data_rows -> list of {"label": <row label>, "cells": {<col_key>: "<raw>"}}
      - records -> list of {"row_label", "column_label", "column_key", "value"}
    No unit or numeric conversion is performed.
    """
    grid = expand_table_to_grid(table)
    if not grid:
        return {}

    header_count = detect_header_row_count(table)
    header_rows = grid[:header_count] if header_count > 0 else (grid[:1] if grid else [])
    data_rows = grid[header_count:] if header_count > 0 else grid[1:]

    columns = build_composite_column_labels(header_rows)
    max_cols = max((len(r) for r in grid), default=0)
    if len(columns) < max_cols:
        # pad columns if needed
        for c in range(len(columns), max_cols):
            columns.append({"key": f"col_{c}", "label": f"Column {c}"})

    # Assume first column is row label (stub)
    stub_index = 0

    rows_out: List[Dict[str, Any]] = []
    records: List[Dict[str, str]] = []

    for r in data_rows:
        if not r:
            continue
        # Normalize length
        if len(r) < len(columns):
            r = r + [""] * (len(columns) - len(r))

        row_label = r[stub_index] if stub_index < len(r) else ""
        row_label = clean_text(row_label)
        cells_map: Dict[str, str] = {}

        for c_idx, col_meta in enumerate(columns):
            if c_idx == stub_index:
                continue
            val = clean_text(r[c_idx]) if c_idx < len(r) else ""
            cells_map[col_meta["key"]] = val
            records.append({
                "row_label": row_label,
                "column_label": col_meta["label"],
                "column_key": col_meta["key"],
                "value": val
            })

        rows_out.append({
            "label": row_label,
            "cells": cells_map
        })

    return {
        "title": extract_table_title(table),
        "header_rows": header_rows,          # Expanded header grid (strings)
        "columns": columns,                  # Composite labels
        "rows": rows_out,                    # Mapping by column_key
        "records": records                   # Flat, row/column/value triplets
    }

# ---------- Hierarchical document parsing ----------

BLOCK_TAGS = {"h1","h2","h3","h4","h5","h6","p","div","li","table"}

def parse_document(html: str) -> List[Dict[str, Any]]:
    """
    Build a section tree:
      [{ "title": ..., "level": 1, "paragraphs": [...], "tables": [...], "subsections": [...] }, ...]
    Paragraphs and tables are attached to the **deepest active subsection**.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup

    sections: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    def push_section(title: str, level: int):
        nonlocal stack, sections
        node = {"title": title, "level": level, "paragraphs": [], "tables": [], "subsections": []}
        # Pop until we find a parent with lower level
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        if not stack:
            sections.append(node)
        else:
            stack[-1]["subsections"].append(node)
        stack.append(node)

    def current_node() -> Dict[str, Any]:
        if not stack:
            # Create an implicit top section if none encountered yet
            push_section("Document", 1)
        return stack[-1]

    # Walk DOM in order
    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        name = el.name.lower() if el.name else ""
        if name not in BLOCK_TAGS:
            continue

        if name == "table":
            tbl = table_to_json(el)
            if tbl:
                current_node()["tables"].append(tbl)
            continue

        text = clean_text(el.get_text(" ", strip=True))
        if not text:
            continue

        # Headings (native h1..h6) or Item X heuristic
        lvl = heading_level(el, text)
        if lvl > 0:
            push_section(text, lvl)
            continue

        # Content (paragraph-like)
        if name in {"p","div","li"}:
            current_node()["paragraphs"].append(text)

    # Prune empties (but keep parents with non-empty subsections)
    def prune(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        node["subsections"] = [p for p in (prune(s) for s in node.get("subsections", [])) if p]
        # Remove empty fields
        for k in ["paragraphs", "tables"]:
            if k in node and (not node[k]):
                del node[k]
        if not node.get("paragraphs") and not node.get("tables") and not node.get("subsections"):
            return None
        # Remove level if you want cleaner output; keep title/subsections/paragraphs/tables
        if "level" in node:
            # keep the numeric level; comment the next line if you prefer to strip it
            pass
        return node

    pruned = [p for p in (prune(s) for s in sections) if p]
    return pruned

# ---------- IO orchestration ----------

def process_root(input_root: str = "./data", output_root: str = "./json_data") -> None:
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for dbe_dir in sorted(in_root.glob("dbe_*")):
        if not dbe_dir.is_dir():
            continue
        ticker = dbe_dir.name.replace("dbe_", "", 1)
        out_dir = out_root / f"dbe_{ticker}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for html_file in dbe_dir.rglob("*.htm*"):
            html = html_file.read_text(encoding="utf-8", errors="ignore")
            sections = parse_document(html)

            out_path = out_dir / (html_file.stem + ".json")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "file": html_file.name,
                    "ticker": ticker,
                    "sections": sections
                }, f, indent=2, ensure_ascii=False)

            print(f"✓ {html_file} → {out_path}")

# ---------- Main ----------
if __name__ == "__main__":
    process_root()
