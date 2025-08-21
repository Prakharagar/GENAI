import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
#from config_loader import *

# LangChain v0.3 imports 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.document  import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from .rag_config import RAGConfig
from .prompt_manager import PromptManager
from .experiment_manager import ExperimentLogger
from .tfidf_wrapper import TfidfRetriever,TfidfRetrieverWrapper
from .config_loader import *

class RAGS:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
       
        self.llm = ChatOpenAI(model=self.cfg.MODEL_NAME, temperature=self.cfg.TEMPERATURE)
        self.embeddings = OpenAIEmbeddings(model=self.cfg.EMBEDDING_MODEL)

        
        self.prompt_manager = PromptManager(self.cfg.PROMPTS_DIR)
        self.experiment_logger = ExperimentLogger(self.cfg.EXPERIMENTS_DIR)

        
        self.parent_docs: List[LCDocument] = []   
        self.child_docs: List[LCDocument] = []    
        self.doc_id_map: Dict[str, LCDocument] = {}  

        self.child_vectorstore: Optional[Chroma] = None
        self.summaries_vectorstore: Optional[Chroma] = None
        self.multi_vector_retriever: Optional[MultiVectorRetriever] = None
        self.dense_retriever = None
        self.multiquery_retriever = None
        self.merger_retriever = None
        self.tfidf_retriever = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.CHUNK_SIZE,
            chunk_overlap=self.cfg.CHUNK_OVERLAP
        )

    def ingest_documents(self, json_root: Path):
        
        json_root = Path(json_root)
        if not json_root.exists():
            raise FileNotFoundError(f"{json_root} does not exist")

        self.parent_docs = []
        self.child_docs = []
        id_key = "doc_id"

        for dbe_dir in sorted(json_root.glob("dbe_*")):
            if not dbe_dir.is_dir():
                continue
            ticker = dbe_dir.name.replace("dbe_", "", 1)
            for jfile in sorted(dbe_dir.glob("*.json")):
                data = json.loads(jfile.read_text(encoding="utf-8"))
                file_label = data.get("file", jfile.name)
                
                parts = []
                sections = data.get("sections") or data.get("hierarchy") or []
                for sec in sections:
                    sec_title = sec.get("title") or sec.get("section") or ""
                    parts.append(f"SECTION: {sec_title}")
                    
                    for p in sec.get("paragraphs", []):
                        parts.append(p)
                    # subsections
                    for sub in sec.get("subsections", []):
                        sub_title = sub.get("title") or sub.get("subsection") or ""
                        parts.append(f"SUBSECTION: {sub_title}")
                        for p in sub.get("paragraphs", []):
                            parts.append(p)
                        
                        for tbl in sub.get("tables", []):
                            parts.append(self._flatten_table_text(tbl))
                
                    for tbl in sec.get("tables", []):
                        parts.append(self._flatten_table_text(tbl))

                full_text = "\n\n".join(parts).strip() or ""
            
                doc_id = str(uuid.uuid4())
                parent_md = {
                    "ticker": ticker,
                    "file": file_label,
                    "source_path": str(jfile),
                    "doc_id": doc_id
                }
                parent_doc = LCDocument(page_content=full_text, metadata=parent_md)
                self.parent_docs.append(parent_doc)
                self.doc_id_map[doc_id] = parent_doc

                idx_counter = 0
                for sec in sections:
                    sec_title = sec.get("title") or sec.get("section") or ""
                    # paragraphs at section
                    for p in sec.get("paragraphs", []):
                        child_md = dict(parent_md)
                        child_md.update({
                            id_key: doc_id,
                            "section_title": sec_title,
                            "subsection_title": None,
                            "is_table": False,
                            "child_index": idx_counter
                        })
                        idx_counter += 1
                        self.child_docs.append(LCDocument(page_content=p, metadata=child_md))
                    # subsections
                    for sub in sec.get("subsections", []):
                        sub_title = sub.get("title") or sub.get("subsection") or ""
                        for p in sub.get("paragraphs", []):
                            child_md = dict(parent_md)
                            child_md.update({
                                id_key: doc_id,
                                "section_title": sec_title,
                                "subsection_title": sub_title,
                                "is_table": False,
                                "child_index": idx_counter
                            })
                            idx_counter += 1
                            self.child_docs.append(LCDocument(page_content=p, metadata=child_md))
                        for t in sub.get("tables", []):
                            tbl_text = self._flatten_table_text(t)
                            child_md = dict(parent_md)
                            child_md.update({
                                id_key: doc_id,
                                "section_title": sec_title,
                                "subsection_title": sub_title,
                                "is_table": True,
                                "table_title": t.get("title"),
                                "child_index": idx_counter
                            })
                            idx_counter += 1
                            self.child_docs.append(LCDocument(page_content=tbl_text, metadata=child_md))
                    # tables at section level
                    for t in sec.get("tables", []):
                        tbl_text = self._flatten_table_text(t)
                        child_md = dict(parent_md)
                        child_md.update({
                            id_key: doc_id,
                            "section_title": sec_title,
                            "subsection_title": None,
                            "is_table": True,
                            "table_title": t.get("title"),
                            "child_index": idx_counter
                        })
                        idx_counter += 1
                        self.child_docs.append(LCDocument(page_content=tbl_text, metadata=child_md))

        logging.info(f"Ingested parent_docs={len(self.parent_docs)}, child_docs={len(self.child_docs)}")
        #use splitter
        self.parent_docs = self.splitter.split_documents(self.parent_docs)
        self.child_docs = self.splitter.split_documents(self.child_docs)

        logging.info(f"Ingested splitted parent_docs={len(self.parent_docs)}, child_docs={len(self.child_docs)}")


    def _flatten_table_text(self, table: Dict[str, Any]) -> str:
        
        lines = []
        title = table.get("title") or ""
        if title:
            lines.append(f"TABLE: {title}")

        columns = table.get("columns") or table.get("headers") or []
        if columns:
            
            col_labels = [c.get("label") if isinstance(c, dict) else str(c) for c in columns]
            lines.append("COLUMNS: " + " | ".join(col_labels))

        rows = table.get("rows") or table.get("data") or []
        for r in rows:
            if isinstance(r, dict):
                label = r.get("label") or r.get("row_header") or ""
                
                cells = r.get("cells") or r.get("values") or {}
                if isinstance(cells, dict):
                    kvs = [f"{k}: {v}" for k, v in cells.items()]
                    lines.append(f"{label} -> " + "; ".join(kvs))
                elif isinstance(cells, list):
                    lines.append(f"{label} -> " + " | ".join([str(x) for x in cells]))
                else:
                    
                    lines.append(json.dumps(r, ensure_ascii=False))
            elif isinstance(r, list):
               
                lines.append(" | ".join([str(x) for x in r]))
            else:
                lines.append(str(r))
        return "\n".join(lines)


    def build_indexes(self, persist: bool = False):
        """Builds the Chroma vectorstore for child docs and a MultiVectorRetriever mapping child vectors -> parent docs."""
        # sanity
        if not self.child_docs or not self.parent_docs:
            raise RuntimeError("No documents ingested. Run ingest_documents() first.")
        
        if persist:
            print("persist: ",persist)
            logging.info(f"Child embedding is building up....{len(self.child_docs)}")
           
            self.child_vectorstore = Chroma.from_documents(
            documents=self.child_docs,
            collection_name=self.cfg.CHROMA_COLLECTION_NAME,
            embedding=self.embeddings,
            persist_directory=str(self.cfg.CHROMA_PERSIST_DIR)
        )
        else:
            logging.info("Child embedding picked up from child vectorstore")
            self.child_vectorstore = Chroma(
            persist_directory=str(self.cfg.CHROMA_PERSIST_DIR),
            collection_name=self.cfg.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings
        )
        child_vs=self.child_vectorstore
        logging.info("Child vectorstore built and persisted.")

      
        self.dense_retriever = child_vs.as_retriever(search_kwargs={"k": self.cfg.K})
        logging.info("2) Dense retriever (simple as_retriever)")
       
        self.tfidf_retriever = TfidfRetriever(self.child_docs, k=self.cfg.K)
        self.wrapped_tfidf = TfidfRetrieverWrapper(self.tfidf_retriever)
        logging.info("3) TF-IDF retriever (sparse)")

        try:
            self.multiquery_retriever = MultiQueryRetriever.from_llm(
                retriever=child_vs.as_retriever(search_kwargs={"k": self.cfg.K}),
                llm=self.llm
                )
            logging.info(" 4) MultiQuery retriever (uses LLM to expand queries) done.")
        except Exception:
            
            logging.info("fallback: use dense retriever.")
            self.multiquery_retriever = self.dense_retriever

        self.merger_retriever = MergerRetriever(retrievers=[self.dense_retriever, self.wrapped_tfidf])
        logging.info("5) Merger retriever (merges dense + sparse)")
        """
        summaries = []
        doc_ids = []
        for parent in self.parent_docs:
            doc_id = parent.metadata.get("doc_id")
            doc_ids.append(doc_id)
            
            summarizer_template = (
                "Summarize the following filing in 2-3 short sentences focusing on key financial tables and amounts:\n\n"
                "{doc}"
            )
            prompt = ChatPromptTemplate.from_template(summarizer_template)
            chain = prompt | self.llm
            try:
                summary = chain.invoke({"doc": parent.page_content})
            except Exception:
                
                summary = parent.page_content[:1000]
            
            summary_text = str(summary).strip()
            summaries.append(LCDocument(page_content=summary_text, metadata={"doc_id": doc_id}))
   
        if persist:
            logging.info("Summary embedding is building up....")
            self.summaries_vectorstore = Chroma.from_documents(
            documents=summaries,
            collection_name=self.cfg.SUMMARIES_COLLECTION,
            embedding=self.embeddings,
            persist_directory=str(self.cfg.CHROMA_PERSIST_DIR)
        )
        else:
            logging.info("Summary embedding is picked up from vector stores..")
            self.summaries_vectorstore = Chroma(
            persist_directory=str(self.cfg.CHROMA_PERSIST_DIR),
            collection_name=self.cfg.SUMMARIES_COLLECTION,
            embedding_function=self.embeddings
        )
        summaries_vs=self.summaries_vectorstore

       
        byte_store = InMemoryByteStore()

        parent_pairs = [(p.metadata.get("doc_id"), p) for p in self.parent_docs]
        byte_store.mset(parent_pairs)

        mv_retriever = MultiVectorRetriever(
            vectorstore=summaries_vs,
            byte_store=byte_store,
            id_key="doc_id",
            search_kwargs={"k": self.cfg.K}
        )
        # note: we could add other vectors (hypothetical questions) to the same summaries_vs collection if desired
        self.multi_vector_retriever = mv_retriever
        """

        logging.info("Indexes and retrievers built: dense, multi-query, merger, multi-vector, tfidf (sparse).")

    
    def _hyde_retrieve(self, question: str, k: Optional[int] = None) -> List[LCDocument]:
        """HyDE: generate a hypothetical answer paragraph, embed it, then similarity search on child vectorstore."""
        k = k or self.cfg.K
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a short factual paragraph that directly answers the question (no sources). Question: {question}"
        )
        chain = hyde_prompt | self.llm
        hyde_text = chain.invoke({"question": question})
        hyde_text = str(hyde_text)
      
        vec = self.embeddings.embed_query(hyde_text)
        docs = self.child_vectorstore.similarity_search_by_vector(vec, k=k)
        return docs

    def _decomposition_retrieve(self, question: str, k_each: int = 4) -> List[LCDocument]:
        """Ask the LLM to decompose the question into sub-questions, retrieve for each, union results."""
        subq_prompt = ChatPromptTemplate.from_template(
            "Decompose the question into 3-5 focused sub-questions (one per line):\n\nQuestion: {question}"
        )
        chain = subq_prompt | self.llm
        subq_text = chain.invoke({"question": question})
        subq_text = str(subq_text)
        subqs = [s.strip("-. \t") for s in subq_text.splitlines() if s.strip()]
        all_docs = []
        seen = set()
        for sq in subqs:
            docs = self.dense_retriever.invoke(sq) if hasattr(self.dense_retriever, "get_relevant_documents") else self.dense_retriever(sq)
            for d in docs:
                key = (d.metadata.get("source_path"), d.page_content[:200])
                if key not in seen:
                    all_docs.append(d)
                    seen.add(key)
        return all_docs

    # ---------- Answer synthesis ----------
    def _synthesize_answer(self, docs: List[LCDocument], question: str, prompt_template: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Produce final answer with LLM and return (answer_text, provenance)."""
        # default prompt
        if prompt_template is None:
            prompt_template = (
                "You are an expert financial assistant. Use the context below to answer the question concisely and factually.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (short, cite sources inline as [Source X]):"
            )
        # build context string with provenance headers up to top-K
        context_lines = []
        for i, d in enumerate(docs[: self.cfg.K ]):
            md = d.metadata or {}
            src = md.get("source_path") or md.get("file") or md.get("ticker") or "unknown"
            header = f"[Source {i+1}] {src} (ticker={md.get('ticker')}, section={md.get('section_title')}, subsection={md.get('subsection_title')})"
            context_lines.append(header + "\n" + d.page_content)
        context_block = "\n\n---\n\n".join(context_lines) or "No context found."

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        try:
            out = chain.invoke({"context": context_block, "question": question})
            answer = str(out).strip()
        except Exception as ex:
            # fallback: make a simple call
            answer = f"(LLM error): {ex}"

        provenance = [{"source": (d.metadata.get("source_path") or d.metadata.get("file")), "metadata": d.metadata} for d in docs]
        return answer, provenance

    # ---------- Routing & Orchestration ----------
    def route_strategy(self, question: str, options: List[str]) -> str:
        """Ask LLM to pick a strategy name from options (deterministic intent, temp=0)."""
        prompt = (
            "You are a routing assistant. Choose exactly one best retrieval strategy (return only the exact name):\n\n"
            f"Options: {', '.join(options)}\n\nQuestion: {question}\n\nReturn one option name exactly."
        )
        chain = ChatPromptTemplate.from_template("{q}") | self.llm
        choice = chain.invoke({"q": prompt})
        choice = str(choice).strip()
        # validate
        if choice not in options:
            # attempt matching substring
            for o in options:
                if o.lower() in choice.lower():
                    return o
            return options[0]
        return choice

    def run_query(self, question: str, strategy: Optional[str] = None, prompt_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        High-level entry to run a single question.
        strategy options: ['naive','multi_query','fusion','hyde','decompose','step_back','raptor','multi_vector','auto']
        If prompt_id is provided, the prompt template from PromptManager is used for synthesis.
        """
        extra = extra or {}
        strategies = ['naive', 'multi_query', 'fusion', 'hyde', 'decompose', 'step_back', 'raptor', 'multi_vector']
        if strategy is None or strategy == "auto":
            chosen = self.route_strategy(question, strategies)
            logging.info(f"Router chose strategy: {chosen}")
            strategy = chosen

        prompt_template = None
        prompt_version = "default"
        if prompt_id:
            info = self.prompt_manager.load_prompt(prompt_id)
            prompt_template = info.get("template")
            prompt_version = prompt_id

        # retrieval stage
        docs = []
        if strategy == "naive":
            docs = self.dense_retriever.invoke(question)
        elif strategy == "multi_query":
            docs = self.multiquery_retriever.invoke(question)
        elif strategy == "fusion":
            # MergerRetriever: merges dense + tfidf
            docs = self.merger_retriever.invoke(question)
        elif strategy == "hyde":
            docs = self._hyde_retrieve(question)
        elif strategy == "decompose":
            docs = self._decomposition_retrieve(question)
        elif strategy == "step_back":
            # initial answer with naive, then follow-ups
            initial_docs = self.dense_retriever.invoke(question)
            answer0, _ = self._synthesize_answer(initial_docs, question, prompt_template)
            followup_prompt = ChatPromptTemplate.from_template(
                "You produced the following answer:\n\n{ans}\n\nGenerate up to 3 specific follow-up queries that would help obtain missing evidence (one per line)."
            )
            chain = followup_prompt | self.llm
            followup_text = chain.invoke({"ans": answer0})
            followups = [l.strip("-. \t") for l in str(followup_text).splitlines() if l.strip()]
            extra_docs = []
            for fq in followups:
                extra_docs.extend(self.dense_retriever.invoke(fq))
            # union & dedupe top
            seen = set(); merged=[]
            for d in (initial_docs + extra_docs):
                key = (d.metadata.get("source_path"), d.page_content[:200])
                if key not in seen:
                    merged.append(d); seen.add(key)
            docs = merged
        elif strategy == "raptor":
            top_parents = self.multi_vector_retriever.invoke(question)  # returns parent Documents
            
            parent_ids = [p.metadata.get("doc_id") for p in top_parents][:3]
           
            candidate_children = [c for c in self.child_docs if c.metadata.get("doc_id") in parent_ids]
            
            q_vec = self.embeddings.embed_query(question)
            
            raw_hits = self.child_vectorstore.similarity_search(question, k=self.cfg.K*3)
            filtered = [d for d in raw_hits if d.metadata.get("doc_id") in parent_ids]
            docs = filtered[: self.cfg.K]
        elif strategy == "multi_vector":
            
            parent_hits = self.multi_vector_retriever.invoke(question)
            
            docs = []
            for p in parent_hits[: self.cfg.K]:
                pid = p.metadata.get("doc_id")
                # get top child chunks for the parent (use child_vectorstore with filter)
                child_hits = self.child_vectorstore.similarity_search(question, k=4)
                # filter
                child_for_parent = [c for c in child_hits if c.metadata.get("doc_id") == pid]
                docs.extend(child_for_parent)
            if not docs:
                # fallback to dense retriever global
                docs = self.dense_retriever.invoke(question)
        else:
            # default naive
            docs = self.dense_retriever.invoke(question)

        # answer generation
        answer_text, provenance = self._synthesize_answer(docs, question, prompt_template)

        # log experiment
        log_entry = {
            "question": question,
            "strategy": strategy,
            "prompt_version": prompt_version,
            "model_name": self.cfg.MODEL_NAME,
            "k_retrieved": len(docs),
            "retrieved_docs": [{"source": d.metadata.get("source_path") or d.metadata.get("file"), "metadata": d.metadata} for d in docs],
            "answer": answer_text
        }
        run_id = self.experiment_logger.log(log_entry)
        return {"run_id": run_id, "answer": answer_text, "provenance": provenance, "retrieved_count": len(docs)}

    # ---------- Evaluation ----------
    @staticmethod
    def normalize_answer(s: str) -> str:
        s = s.lower().strip()
        import re
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    @staticmethod
    def exact_match(s1: str, s2: str) -> bool:
        return RAGS.normalize_answer(s1) == RAGS.normalize_answer(s2)

    @staticmethod
    def f1_score(pred: str, gold: str) -> float:
        p_tokens = RAGS.normalize_answer(pred).split()
        g_tokens = RAGS.normalize_answer(gold).split()
        if not p_tokens or not g_tokens:
            return 0.0
        common = set(p_tokens) & set(g_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(p_tokens)
        rec = len(common) / len(g_tokens)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def evaluate_dataset(self, qa_pairs: List[Dict[str, str]], strategy: str = "auto", prompt_id: Optional[str] = None) -> Dict[str, Any]:
        results = []
        for q in qa_pairs:
            out = self.run_query(q["question"], strategy=strategy, prompt_id=prompt_id)
            pred = out["answer"]
            em = 1 if self.exact_match(pred, q["answer"]) else 0
            f1 = self.f1_score(pred, q["answer"])
            results.append({"question": q["question"], "gold": q["answer"], "pred": pred, "em": em, "f1": f1, "run_id": out["run_id"]})
        avg_em = sum(r["em"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        return {"results": results, "avg_em": avg_em, "avg_f1": avg_f1}


