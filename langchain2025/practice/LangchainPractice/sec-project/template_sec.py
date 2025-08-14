from langchain_core.prompts import PromptTemplate

# === Primary extraction prompt (must accept only {context} and {question}) ===
template = """
You are an expert analyst specializing in extracting business challenges from SEC filings and aligning them with Adobe's software solutions.

Your Task:
From the provided filing excerpts, identify up to **5 distinct business challenges** that Adobe's solutions could address — prioritize *Express* and *Firefly*, but include other Adobe products where applicable.

---
Question from user (may include a ticker hint like "Ticker=MMM"):
{question}

Document Context (snippets from SEC filings):
{context}
---

Where to Search in Filings (priority order):
1) Item 1A - Risk Factors (focus heavily here)
2) Management's Discussion & Analysis (MD&A)
3) Forward-Looking Statements
4) Strategy sections
5) Other narrative sections mentioning operational risks, market threats, or digital inefficiencies.

What to Look For:
- Tier 1 — Adobe Marketing & Creative Relevance:
  • Content creation or marketing bottlenecks
  • Difficulty adapting brand content across channels
  • Weak audience targeting/engagement
  • Slow or costly campaign production pipelines
  • Scaling creative operations across teams/regions
  • Remote/hybrid creative team collaboration issues
- Tier 2 — Digital Transformation & Innovation:
  • Customer experience/personalization gaps
  • AI/automation/generative content risks
  • Legacy system inefficiencies/digital modernization needs
  • E-commerce or omnichannel coordination issues
  • Collaboration platform disruptions
  • Innovation/product development/data integration challenges

Adobe Solutions to Prioritize:
- Express
- Firefly
- Also include Creative Cloud, Document Cloud, Acrobat, and e-signature workflows where relevant.

Rules for Extraction:
- Pull ONLY from the provided content (narrative, tables, charts).
- Use **direct quotes** in the “Source Quote” column — no paraphrasing there.
- Keep “Challenge Context” and “Business Value Context” under 20 words each.
- Allow inferred risks ONLY if clearly tied to evidence in the text.
- Avoid generic company descriptions.
- Identify filing type (10-K, 10-Q, 8-K, DEF 14A, etc.) and filing date if present in the snippet text. If absent, leave blank.
- Do NOT fabricate numbers, quotes, or facts.
- Max 5 rows.

Ticker/Company Handling:
- If the question mentions a ticker (e.g., "Ticker=MMM" or "for MMM"), use that ticker.
- Otherwise, infer ticker and company name from the context (including any visible headers/footers/metadata). If still unclear, leave them blank.

Output Format (return a clean markdown table ONLY; no commentary before/after):
| Ticker | Company Name | Challenge Summary | Challenge Context | Business Value | Business Value Context | Department | Adobe Solution | Filing Type | Filing Date | Source Quote | Source Section |
|--------|--------------|-------------------|-------------------|----------------|------------------------|------------|----------------|-------------|-------------|--------------|----------------|
"""

combine_template = """
You are part of Adobe's internal insights team.
You are provided with relevant findings from multiple document excerpts.

Goal:
- Combine these findings into a single, cohesive, and insightful response.
- Integrate insights from text, tables, and charts into a clear narrative.
- Remove redundancy but preserve all unique insights.
- Identify relationships between data points (e.g., revenue growth correlating with marketing spend).
- Clearly mention key figures, percentages, dates, and trends.
- If figures or trends conflict, flag them for review.
- Do not add information that does not appear in the findings.

---
Findings:
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

Refine the original answer as needed while preserving correct content and adding only evidence-backed details.
"""

# === Prompt objects with correct variable contracts ===
retriever_prompt = PromptTemplate(
    template="Generate several alternative phrasings of this question to improve document retrieval fidelity:\n\n{question}",
    input_variables=["question"],
)

# For RetrievalQA "stuff" and "map_reduce"/"refine", the map/question prompt must take ONLY these:
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
combine_prompt = PromptTemplate(template=combine_template, input_variables=["summaries", "question"])
# For refine chains, the refine prompt must accept the existing answer and new context
refine_prompt = PromptTemplate(template=refine_template, input_variables=["existing_answer", "context"])
