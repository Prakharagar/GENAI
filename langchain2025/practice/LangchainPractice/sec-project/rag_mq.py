from config_loader import *  # If you have any global configs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from template import prompt,combine_prompt,refine_prompt,retriever_prompt
from typing import Optional, List
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
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 600

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
        chunk_overlap=CHUNK_OVERLAP
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
    print("chroma_rag_query")
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    print("LLM MODEL INITIALISED.....")

    
    if ticker:
        search_filter = {"ticker": ticker.upper()}  # or .lower() to match your stored metadata

        base_retriever = vectordb.as_retriever(
        search_type=retreiver_search_type,
        search_kwargs={"k": k, "filter": search_filter}
        )
    else:
        base_retriever = vectordb.as_retriever(
        search_type=retreiver_search_type,
        search_kwargs={"k": k}
        )
    retrieved_docs = base_retriever.invoke(query)

    #print(f"retrieved_docs: {retrieved_docs}")

    multi_retriever=MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=base_retriever,
    prompt=retriever_prompt,
    )

    generated_queries: List[str] = multi_retriever.llm_chain.invoke({"question": query})
    #print(generated_queries)
    generated_queries = [q.strip() for q in generated_queries if q.strip()]

    retrieved_docs = []
    for sub_q in generated_queries:
        docs = base_retriever.invoke(sub_q)
        retrieved_docs.extend(docs)

    unique_docs = {doc.page_content + str(doc.metadata): doc for doc in retrieved_docs}.values()


     
    if chain_type == "stuff":
        chain_kwargs = {"prompt": prompt}
    elif chain_type == "map_reduce":
        chain_kwargs = {"question_prompt": prompt, "combine_prompt": combine_prompt}
    elif chain_type == "refine":
        chain_kwargs = {"question_prompt": prompt, "refine_prompt": refine_prompt}
    else:
        raise ValueError(f"Invalid chain_type '{chain_type}'.")

   # print(f'Based on chain type input: {chain_type}: \n {chain_kwargs} ')

    chain = RetrievalQA.from_chain_type(
        retriever=multi_retriever,
        chain_type=chain_type,
        llm=llm,
        chain_type_kwargs=chain_kwargs,
        return_source_documents=True,
        input_key="question"
    )
    result = chain.invoke({"question": query})

       
    print("\nSources:")

    result["source_documents"] = unique_docs
    result["multi_queries"] = generated_queries

    #print(result)

    #print(f"TICKER: {ticker}")
    ##    print(f" - {doc.metadata.get('ticker')} ({doc.metadata.get('date')}): {doc.metadata.get('source_file')}")

    #print("retrieved docs",unique_docs)
    #print(result["multi_queries"])
    print("Answer:", result["result"])
# ------------------------------

# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Financial Document RAG Pipeline")
    parser.add_argument("--pdf_dir", type=str, default="download/10k_pdf", help="Path to PDF directory")
    parser.add_argument("--persist_dir", type=str, default="chroma_db_sec", help="Chroma DB persist directory")
    parser.add_argument("--collection_name", type=str, default="10k", help="Chroma collection name")
    parser.add_argument("--query", type=str, default=None, help="User query")
    parser.add_argument("--ticker", type=str, default=None, help="Optional ticker filter")
    parser.add_argument("--chain_type", type=str, default="map_reduce", choices=["stuff", "map_reduce", "refine"], help="Chain type")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma DB from PDFs")
    parser.add_argument("--k", type=int, default=8, help="Number of docs to retrieve")
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

    if args.query:
        query=args.query
    else:
        #query="Extract 10-Ks, earnings reports, and investor presentations for and give quantatitive numbers for: “FTE growth,” “headcount,” “subscription volume,” “content output,” “guidance raised/beaten,” “divestiture,” “cost savings”"
        #query="extract the ticker name and company for which filling take place, extract the filing date as well"
        query ="MMR"
    ticker="MMM"
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
