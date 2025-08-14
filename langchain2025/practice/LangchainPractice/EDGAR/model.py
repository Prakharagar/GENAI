import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ======================
# CONFIGURATION
# ======================

# Your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# Folder where PDFs are stored
pdf_folder = Path("pdfs")  # Example: pdfs/CompanyA.pdf, pdfs/CompanyB.pdf

# Folder for ChromaDB persistent storage
persist_dir = "chroma_store"

# ======================
# STEP 1: Load PDFs
# ======================

all_docs = []
for pdf_path in pdf_folder.glob("*.pdf"):
    print(f"Loading PDF: {pdf_path.name}")
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    # Optional: attach entity metadata
    for d in docs:
        d.metadata["entity"] = pdf_path.stem  # Use filename (without .pdf) as entity name

    all_docs.extend(docs)

# ======================
# STEP 2: Chunk documents
# ======================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # characters
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(all_docs)
print(f"Total chunks: {len(split_docs)}")

# ======================
# STEP 3: Embed & store in Chroma
# ======================

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_dir
)

vectorstore.persist()
print(f"✅ Data stored in Chroma at: {persist_dir}")

# ======================
# STEP 4: Create retriever & QA chain
# ======================

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ======================
# STEP 5: Example query
# ======================

query = "What are the main risk factors for CompanyA?"
result = qa_chain.invoke({"query": query})

print("\n=== Answer ===")
print(result["result"])

print("\n=== Sources ===")
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('entity')}: {doc.metadata.get('source')}")
