from config_loader import *  # If you have any global configs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from template_sec import prompt,combine_prompt,refine_prompt,retriever_prompt

import os
import re
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

BATCH_SIZE = 50
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 500

# ------------------------------
# HELPERS
# ------------------------------
def extract_metadata_from_filename(filename: str) -> Dict:
    print("extract_metadata_from_filename")
    basename = os.path.basename(filename)
    print(f"filename: {basename}")
    match = re.match(r"^(.*?)[_-](\d{8})\.pdf$", basename)
    if match:
        ticker, date_str = match.groups()
        metadata_dict={"ticker": ticker, "date": date_str}
        print(f"Metadata: {metadata_dict}")
        return metadata_dict
    else:
        metadata_dict={"ticker": None, "date": None}
        return metadata_dict


def create_or_load_vectorstore(
    docs=None,
    persist_dir: Path = None,
    collection_name: str = None
):
    print("create_or_load_vectorstore")
    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    if not persist_dir.exists() and docs:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=emb,
            persist_directory=str(persist_dir),
            collection_name=collection_name
        )
    else:
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=emb,
            collection_name=collection_name
        )
    return vectordb


def build_chroma_from_dir(pdf_dir: str, persist_dir: str, collection_name: str) -> Chroma:
    print("build_chroma_from_dir")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before calling this function")

    pdf_dir = Path(pdf_dir)
    files = sorted([p for p in pdf_dir.glob("*.pdf")])
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )    

    for f in tqdm(files, desc="Loading PDFs"):
        print(f"file: {f}")
        loader = PyMuPDFLoader(str(f))
        pages = loader.load_and_split(text_splitter=splitter)
        metadata = extract_metadata_from_filename(f.name)
        for d in pages:
            d.metadata.update({
                "source_file": str(f),
                "ticker": metadata["ticker"],
                "date": metadata["date"]
            })
        all_docs.extend(pages)

    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=emb,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print(f"Collection '{collection_name}' created with {len(all_docs)} documents.")
    return vectordb


def chroma_rag_query(
    query: str,
    vectordb: Chroma,
    persist_dir: str,
    collection_name: str,
    ticker: Optional[str],
    retreiver_search_type: str = "mmr",
    chain_type: str = "map_reduce",
    k: int = 3
):
    print(f"chroma_rag_query: {ticker}")
    if ticker:
        retriever = vectordb.as_retriever(
            search_type=retreiver_search_type,
            search_kwargs={"k": k, "filter": {"ticker": ticker}}
        )
        question = f"{query}\n\nTicker={ticker}"
        
    else:
        retriever = vectordb.as_retriever(
            search_type=retreiver_search_type,
            search_kwargs={"k": k}
        )
        question = query

    if chain_type == "stuff":
        chain_kwargs = {"prompt": prompt}
    elif chain_type == "map_reduce":
        chain_kwargs = {"question_prompt": prompt, "combine_prompt": combine_prompt}
    elif chain_type == "refine":
        chain_kwargs = {"question_prompt": prompt, "refine_prompt": refine_prompt}
    else:
        raise ValueError(f"Invalid chain_type '{chain_type}'.")


    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    print("LLM MODEL INITIALISED")

    print(f'Based on chain type input: {chain_type}: \n {chain_kwargs} ')

    chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type=chain_type,
        llm=llm,
        chain_type_kwargs=chain_kwargs,
        return_source_documents=True,
        input_key="question"
    )
    result = chain.invoke({"question": question})

    print("Answer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f" - {doc.metadata.get('ticker')} ({doc.metadata.get('date')}): {doc.metadata.get('source_file')}")

    #print(result)

# ------------------------------
# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="SEC RAG Pipeline for Adobe-aligned challenges")
    parser.add_argument("--pdf_dir", type=str, default="download/10k_pdf", help="Path to PDF directory")
    parser.add_argument("--persist_dir", type=str, default="chroma_db_sec", help="Chroma DB persist directory")
    parser.add_argument("--collection_name", type=str, default="10k", help="Chroma collection name")
    parser.add_argument("--query", type=str, default=None, help="User query")
    parser.add_argument("--ticker", type=str, default=None, help="Optional ticker filter")
    parser.add_argument("--chain_type", type=str, default="map_reduce", choices=["stuff", "map_reduce", "refine"], help="Chain type")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma DB from PDFs")
    parser.add_argument("--k", type=int, default=10, help="Number of docs to retrieve")
    parser.add_argument("--search_type", type=str, default="mmr", choices=["similarity", "mmr"], help="Retriever search type")
    args = parser.parse_args()

    print(20*'#'," STEP1 ", 20*'#')

    print("\n===== Arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=====================\n")

    print(20*'#'," STEP2 ", 20*'#')
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    else:
        print("OPENAI_API_KEY Is Present in environment")

    print(20*'#'," STEP3 ", 20*'#')
    persist_dir = Path(args.persist_dir)

    if args.rebuild or not persist_dir.exists():
        print("Building Chroma DB...")
        vectordb = build_chroma_from_dir(args.pdf_dir, str(persist_dir), args.collection_name)
    else:
        print("Loading existing Chroma DB...")
        vectordb = create_or_load_vectorstore(
            docs=None,
            persist_dir=persist_dir,
            collection_name=args.collection_name
        )

    print(20*'#'," STEP4 ", 20*'#')

    query = args.query or (
        "From the context, extract up to 5 Adobe-relevant business challenges and return ONLY the markdown table."
    )
    
    if args.ticker:
        ticker=args.ticker
    else:
        ticker="APPL"
    print("executing chroma_rag_query...")
    chroma_rag_query(
        query=query,
        vectordb=vectordb,
        persist_dir=str(persist_dir),
        collection_name=args.collection_name,
        ticker=ticker,
        retreiver_search_type=args.search_type,
        chain_type=args.chain_type,
        k=args.k
    )

if __name__ == "__main__":
    main()
