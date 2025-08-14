from config_loader import *
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader ,UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os ,re,json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from langchain.chains import RetrievalQA

import argparse

#Step 1 Indexing - document load, chuncking, store in vector store
#Step2 create a user query
#step3 retrever - Do the sementic/similarity search based on query 
#Step 4 Create prompt template having context and query
#step 5 Augumentation send the prompt template to llm

# ------------------------------
# CONFIG
# ------------------------------
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o-mini"   # You can use gpt-4o or gpt-4o-mini for faster responses

BATCH_SIZE = 50  # how many chunks per embedding call to OpenAI
CHUNK_SIZE = 1200  # characters per chunk (you can change to token-based splitting if desired)
CHUNK_OVERLAP = 200

# Step 2: Extract metadata from filename (ticker_yyyymmdd.pdf)
def extract_metadata_from_filename(filename: str) -> Dict:
    basename = os.path.basename(filename)
    match = re.match(r"^(.*?)_(\d{8})\.pdf$", basename)
    if match:
        ticker, date_str = match.groups()
        return {"ticker": ticker, "date": date_str}
    else:
        return {"ticker": None, "date": None}
    
# ==== STEP 2: Create or load Chroma store ====
def create_or_load_vectorstore(docs):
    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    if not persist_dir.exists():
        vectordb = Chroma.from_documents(
                                documents=docs,
                                embedding=emb,
                                persist_directory=str(persist_dir),
                                collection_name=collection_name
        )
        #vectordb.persist()
    else:
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=emb,
            collection_name=collection_name
        )
    return vectordb

def build_chroma_from_dir(pdf_dir: str, persist_dir: str) -> None:
    """Using LangChain loaders + Chroma vectorstore (stores metadata searchable).
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before calling this function")

    pdf_dir = Path(pdf_dir)
    files = sorted([p for p in pdf_dir.glob("*.pdf")])
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for f in tqdm(files, desc="Loading with LangChain"):
        loader = PyMuPDFLoader(str(f))
        pages = loader.load_and_split(text_splitter=splitter)
        ticker, date = extract_metadata_from_filename(f.name)
        for d in pages:
            d.metadata["source_file"] = str(f)
            d.metadata["ticker"] = ticker
            d.metadata["date"] = date
        all_docs.extend(pages)

    emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
   
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=emb,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print(f"✅ Collection '{collection_name}' created with {len(all_docs)} documents.")
    return vectordb

def chroma_rag_query(
        query: str, 
        vectordb: Chroma,
        persist_dir: str, 
        collection_name: str, 
        ticker: Optional[str],
        retreiver_search_type: str ="similarity",
        chain_type: str = "map_reduce",
        k:int = 3
        
        ):
    """
    Query an existing Chroma DB with a given collection name.
    Supports different chain types for flexibility.
    """
    if ticker:
        retriever = vectordb.as_retriever(
            search_type=retreiver_search_type,
            search_kwargs={'k':k,
                           "filter": {"ticker": ticker}
                           }
        )
            
    else:
        retriever = vectordb.as_retriever(
            search_type=retreiver_search_type,
            search_kwargs={'k':k}
        )

    template = """
    You are a financial analysis assistant.  
    Use ONLY the provided document excerpts to answer the question.  
    If the answer is not found in the context, clearly say "Not found in documents."  

    Context:
    {context}

    Question:
    {question}

    Answer (be concise, factual, and well-structured):

    If the answer is not present, say "Not found in documents."
    """
    combine_template ="""
    Combine the following partial answers into a final answer:
    {summaries}

    Question:
    {question}

    Final Answer:
    """
    refine_template ="""

            We have an existing answer:
            {existing_answer}

            We have more context to consider:
            {context}

            Refine the original answer if necessary:

    """
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
        )

    combine_prompt= PromptTemplate(
        template=combine_template, 
        input_variables=["summaries", "question"]
        )
    
    refine_prompt= PromptTemplate(
        template=refine_template, 
        input_variables=["existing_answer", "question"]
        )
    if chain_type == "stuff":
        chain_kwargs = {"prompt": prompt}
    elif chain_type == "map_reduce":
        chain_kwargs = {"map_prompt": prompt, "combine_prompt": combine_prompt}
    elif chain_type == "refine":
        chain_kwargs = {"question_prompt": prompt, "refine_prompt": refine_prompt}
    else:
        raise ValueError(f"Invalid chain_type '{chain_type}'. Must be one of: stuff, map_reduce, refine.")

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type=chain_type,   # stuff, map_reduce, refine
        llm=llm,
        chain_type_kwargs=chain_kwargs,
        return_source_documents=True
    )
    result = chain.invoke({"query": query})

    print("Answer:", result["result"])
    print("\n Sources:")
    for doc in result["source_documents"]:
        print(f" - {doc.metadata.get('ticker')} ({doc.metadata.get('date')}): {doc.metadata.get('source')}")

def main():
    parser = argparse.ArgumentParser(description="Financial Document RAG Pipeline")
    parser.add_argument("--pdf_dir", type=str, default="download/10k_pdf", help="Path to PDF directory")
    parser.add_argument("--persist_dir", type=str, default="chroma_db_sec", help="Chroma DB persist directory")
    parser.add_argument("--collection_name", type=str, default="10k", help="Chroma collection name")
    parser.add_argument("--query", type=str, required=True, help="User query for the RAG system")
    parser.add_argument("--ticker", type=str, default=None, help="Optional ticker filter for retrieval")
    parser.add_argument("--chain_type", type=str, default="map_reduce", choices=["stuff", "map_reduce", "refine"], help="Type of chain to run")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma DB from PDFs")
    parser.add_argument("--k", type=int, default=3, help="Number of top documents to retrieve")
    parser.add_argument("--search_type", type=str, default="similarity", choices=["similarity", "mmr"], help="Retriever search type")
    args = parser.parse_args()

    # Ensure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    persist_dir = Path(args.persist_dir)

    # Step 1: Build or load Chroma DB
    if args.rebuild or not persist_dir.exists():
        print("🔄 Building Chroma DB from PDFs...")
        vectordb = build_chroma_from_dir(args.pdf_dir, str(persist_dir))
    else:
        print("📂 Loading existing Chroma DB...")
        vectordb = create_or_load_vectorstore(docs=[], persist_dir=persist_dir, collection_name=args.collection_name)

    # Step 2: Query the system
    chroma_rag_query(
        query=args.query,
        vectordb=vectordb,
        persist_dir=str(persist_dir),
        collection_name=args.collection_name,
        ticker=args.ticker,
        retreiver_search_type=args.search_type,
        chain_type=args.chain_type,
        k=args.k
    )

if __name__ == "__main__":
    main()
