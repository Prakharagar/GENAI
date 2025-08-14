from config_loader import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# Import all templates
from template_combined import (
    AI_STRATEGY_PROMPT,
    FINANCIALS_PROMPT,
    ORG_MATURITY_PROMPT,
    STAKEHOLDER_PROMPT,
    EXTERNAL_PRESENCE_PROMPT,
    COMBINE_PROMPT,
    RETRIEVER_PROMPT
)

OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ------------------------------
# Helpers
# ------------------------------
def safe_json_parse(text, category_name):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed for {category_name}: {e}")
        return {"error": "Invalid JSON", "raw_output": cleaned}

def extract_metadata_from_filename(filename: str) -> Dict:
    match = re.match(r"^(.*?)[_-](\d{8})\.pdf$", os.path.basename(filename))
    if match:
        ticker, date_str = match.groups()
        return {"ticker": ticker, "date": date_str}
    else:
        return {"ticker": None, "date": None}

def create_or_load_vectorstore(docs=None, persist_dir: Path = None, collection_name: str = None):
    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    if not persist_dir.exists() and docs:
        return Chroma.from_documents(docs, emb, persist_directory=str(persist_dir), collection_name=collection_name)
    return Chroma(persist_directory=str(persist_dir), embedding_function=emb, collection_name=collection_name)

def build_chroma_from_dir(pdf_dir: str, persist_dir: str, collection_name: str) -> Chroma:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running.")
    files = sorted(Path(pdf_dir).glob("*.pdf"))
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for f in tqdm(files, desc="Loading PDFs"):
        loader = PyMuPDFLoader(str(f))
        pages = loader.load_and_split(text_splitter=splitter)
        meta = extract_metadata_from_filename(f.name)
        for d in pages:
            d.metadata.update({"source_file": str(f), **meta})
        all_docs.extend(pages)
    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return Chroma.from_documents(all_docs, emb, persist_directory=persist_dir, collection_name=collection_name)

# ------------------------------
# Multi-Prompt RAG Query
# ------------------------------
def chroma_rag_query(query, vectordb, ticker=None, k=3):
    retriever = vectordb.as_retriever(search_kwargs={"k": k, **({"filter": {"ticker": ticker}} if ticker else {})})
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

    # Expand query
    try:
        expansion = llm.invoke(RETRIEVER_PROMPT.format(question=query))
        expansion_json = safe_json_parse(expansion.content, "retriever_expansion")
        expanded_queries = expansion_json.get("rewrites", [query])
        print("🔍 Expanded queries:", expanded_queries)
    except Exception as e:
        print("⚠️ Expansion failed:", e)
        expanded_queries = [query]

    # Retrieve context
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(d.page_content for d in docs)

    # Category prompts
    categories = {
        "ai_strategy_and_executive_sentiment": AI_STRATEGY_PROMPT,
        "financial_performance_and_events": FINANCIALS_PROMPT,
        "org_maturity_and_structure": ORG_MATURITY_PROMPT,
        "stakeholders": STAKEHOLDER_PROMPT,
        "external_presence_and_customer_engagement": EXTERNAL_PRESENCE_PROMPT
    }

    partial_results = {}
    for key, tmpl in categories.items():
        resp = llm.invoke(tmpl.format(context=context_text, question=query))
        partial_results[key] = safe_json_parse(resp.content, key)

    # Combine step
    summaries_text = json.dumps(partial_results)
    combined_resp = llm.invoke(COMBINE_PROMPT.format(summaries=summaries_text, question=query))
    combined_json = safe_json_parse(combined_resp.content, "combine")

    print("\n===== FINAL STRUCTURED OUTPUT =====")
    print(json.dumps(combined_json, indent=2))
    return combined_json

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adobe Sales Intelligence RAG Pipeline")
    parser.add_argument("--pdf_dir", type=str, default="download/10k_pdf")
    parser.add_argument("--persist_dir", type=str, default="chroma_db_sec")
    parser.add_argument("--collection_name", type=str, default="10k")
    parser.add_argument("--query", type=str)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    print("\n===== Arguments =====")
    for arg, val in vars(args).items():
        print(f"{arg}: {val}")
    print("=====================\n")

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY.")
    
    query="Scan 10-Ks, earnings reports, and investor presentations for: “FTE growth,” “headcount,” “subscription volume,” “content output,” “guidance raised/beaten,” “divestiture,” “cost savings”"


    persist_dir = Path(args.persist_dir)
    vectordb = (
        build_chroma_from_dir(args.pdf_dir, str(persist_dir), args.collection_name)
        if args.rebuild or not persist_dir.exists()
        else create_or_load_vectorstore(None, persist_dir, args.collection_name)
    )

    chroma_rag_query(query, vectordb, ticker=args.ticker, k=args.k)

if __name__ == "__main__":
    main()

#python rag.py --query "What is the AI strategy sentiment for 2024?" --ticker ADBE --k 5            