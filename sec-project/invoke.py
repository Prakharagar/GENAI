from sec_model.rag_config import RAGConfig
from sec_model.rag_utils import RAGS

def main():
    cfg = RAGConfig()
    rag = RAGS(cfg)

    rag.ingest_documents(cfg.JSON_INPUT_ROOT)

    # retrievers
    rag.build_indexes(persist=True)
    #rag.build_indexes()

    # register a prompt version
    default_prompt ="""
You are an internal Adobe financial and business intelligence analyst.
Your task is to carefully read the following document excerpt and extract only the 
information that is relevant to the question. 

The excerpt may contain:
- Textual explanations (e.g., market commentary, strategic plans)
- Data tables (e.g., revenue breakdown, customer growth metrics)
- Charts (converted to text in this format)

Guidelines:
- Focus on metrics, financial figures, percentages, growth rates, and trends.
- Keep “Challenge Context” and “Business Value Context” under 20 words each.
- Infer ticker and company name from the context (including any visible headers/footers/metadata). If still unclear, leave them blank.
- If a table is described, extract key numbers with their labels.
- If a chart is described, summarize the main trend or pattern.
- If no relevant information is found, return "No relevant information".
- Avoid speculation or outside knowledge — use ONLY the excerpt.
- Highlight numbers, dates, trends, and causes when present.
- Identify filing type (10-K, 10-Q, 8-K, DEF 14A, etc.) and filing date if present in the snippet text. If absent, leave blank.
- Keep sentences concise but informative.

---

Question:
{question}

Relevant Findings from this excerpt:
"""
    pid = rag.prompt_manager.register_prompt(
        "fin_v1", default_prompt,
        "Default financial response prompt"
        )

    q=" Please identify the following Ticker , Company Name ,headcount growth, Challenge Summary , Challenge Context , Business Value , Business Value Context , Department , Adobe Solution , Filing Type , Filing Date , Source Quote , Source Section ,"

    strategies = ['naive', 'multi_query', 'fusion', 'hyde', 'decompose', 'step_back']
    final_output=[]
    out={}
    for strtegy in strategies:
        print("="*30)
        print("strtegy:", strtegy)
        print("\n")
        
        out = rag.run_query(q, strategy=strtegy, prompt_id=pid)
        if "No relevant information." not in out["answer"]:
            final_output.append(out["answer"])
            out[strtegy]=out["answer"]

        print("Run ID:", out["run_id"])
        print("Answer:\n", out["answer"])
        print("Provenance (first 3):", out["provenance"][:3])
        print("="*30)
        print("\n"*2)


    print("FINAL ANSWER:\n", final_output)

if __name__ == "__main__":
    main()