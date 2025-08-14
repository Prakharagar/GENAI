from langchain_core.prompts import PromptTemplate

template="""
You are an internal Adobe financial and business intelligence analyst.
Your task is to carefully read the following document excerpt and extract only the 
information that is relevant to the question. 

The excerpt may contain:
- Textual explanations (e.g., market commentary, strategic plans)
- Data tables (e.g., revenue breakdown, customer growth metrics)
- Charts (converted to text in this format)

Guidelines:
- Focus on metrics, financial figures, percentages, growth rates, and trends.
- If a table is described, extract key numbers with their labels.
- If a chart is described, summarize the main trend or pattern.
- If no relevant information is found, return "No relevant information".
- Avoid speculation or outside knowledge — use ONLY the excerpt.
- Highlight numbers, dates, trends, and causes when present.
- Keep sentences concise but informative.

---
Document Excerpt:
{context}

Question:
{question}

Relevant Findings from this excerpt:
"""
combine_template="""
You are part of Adobe's internal insights team.
You are provided with relevant findings from multiple document excerpts.

Your goal:
- Combine these findings into a single, cohesive, and insightful response.
- Integrate insights from text, tables, and charts into a clear narrative.
- Remove redundancy but preserve all unique insights.
- Identify relationships between data points (e.g., revenue growth correlating with marketing spend).
- Clearly mention key figures, percentages, dates, and trends.
- If there are conflicting figures or trends, flag them for review.
- Do not add information not found in the findings.

---
Findings from documents:
{summaries}

Question:
{question}

Final Consolidated Answer (Adobe Insights Team Standard):
"""

refine_template = """
    We have an existing answer:
    {existing_answer}

    We have more context to consider:
    {context}

    Refine the original answer if necessary:
    """
retriever_prompt = PromptTemplate(
    template="Generate different versions of the following question to improve document retrieval:\n\n{question}",
    input_variables=["question"]
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
combine_prompt = PromptTemplate(template=combine_template, input_variables=["summaries", "question"])
refine_prompt = PromptTemplate(template=refine_template, input_variables=["existing_answer", "question"])
