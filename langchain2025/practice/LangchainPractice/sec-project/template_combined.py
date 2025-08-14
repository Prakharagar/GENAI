"""
Prompt templates for Adobe Sales Intelligence Extraction
"""

from langchain_core.prompts import PromptTemplate

# Helper to append strict JSON output instruction
def json_only_prompt(base_text):
    return base_text.strip() + "\n\nReturn ONLY valid JSON. No explanations, no extra text."

# -----------------------
# AI Strategy & Executive Sentiment
# -----------------------
AI_STRATEGY_PROMPT = PromptTemplate(
    template=json_only_prompt("""
You are an AI Strategy & Executive Sentiment extractor for Adobe's enterprise sales team.
Use ONLY provided document excerpts. If nothing matches, return "Not found in documents." for that field.

Return a JSON object with these keys:
{{
  "summary": "<one-paragraph summary or 'Not found in documents.'>",
  "sentiment_label": "<Growth | Controlled risk | Mixed | Neutral | Not found in documents.>",
  "supporting_quotes": [{{"quote":"<exact excerpt>","source":"<doc>","location":"<if available>"}}],
  "procurement_legal_compliance_inference": {{
      "procurement": "<Aggressive | Moderate | Conservative | Not found in documents.>",
      "legal": "<Tight | Moderate | Lax | Not found in documents.>",
      "compliance": "<High | Medium | Low | Not found in documents.>"
  }},
  "sales_implication": "<one-line implication or 'Not found in documents.'>",
  "confidence": 0.0
}}

Context:
{context}

Question:
{question}
"""),
    input_variables=["context", "question"]
)

# -----------------------
# Financial Performance & Events
# -----------------------
FINANCIALS_PROMPT = PromptTemplate(
    template=json_only_prompt("""
You are a Financial Signals extractor for Adobe sales.
Search for: "FTE growth","headcount","subscription volume","content output","guidance raised","guidance beaten","divestiture","cost savings".
If none found, set the whole field to "Not found in documents.".

Return JSON:
{{
  "signals": [
     {{"type":"<keyword>", "excerpt":"<exact excerpt>", "numeric_normalized": "<number or null>", "source":"<doc>", "implication":"<upsell_potential|renewal_risk|entry_point|neutral>", "confidence":0.0}}
  ],
  "overall_assessment":"<one-line or 'Not found in documents.'>"
}}

Context:
{context}
Question:
{question}
"""),
    input_variables=["context", "question"]
)

# -----------------------
# Org Maturity & Structure
# -----------------------
ORG_MATURITY_PROMPT = PromptTemplate(
    template=json_only_prompt("""
Extract mentions of: "brand center","brand governance","content strategy","decentralized team","cross-functional".
Return JSON:
{{
  "indicators_found":[ {{"indicator":"<text>","excerpt":"<exact excerpt>","source":"<doc>"}} ],
  "centralization_score":"<0-10 or 'Not found in documents.'>",
  "readiness":"<Ready | Needs prework | Not ready | Not found in documents.>",
  "one_line_rationale":"<one-line>"
}}

Context:
{context}
Question:
{question}
"""),
    input_variables=["context", "question"]
)

# -----------------------
# Stakeholders
# -----------------------
STAKEHOLDER_PROMPT = PromptTemplate(
    template=json_only_prompt("""
Identify names, titles, and functions relevant to buying decisions: e.g., "Head of Brand","Digital Marketing Lead","Creative Ops","IT","Procurement".
Return JSON:
{{
  "stakeholders":[
    {{"name":"<name or 'Not found in documents.'>","title":"<title>","inferred_function":"<Brand|Marketing|Content|Creative|IT/Procurement|Other>",
     "level":"<DecisionMaker|Influencer|Operational|Not found in documents.>",
     "evidence_excerpt":"<exact excerpt>","source":"<doc>"}}
  ]
}}

Context:
{context}
Question:
{question}
"""),
    input_variables=["context", "question"]
)

# -----------------------
# External Presence
# -----------------------
EXTERNAL_PRESENCE_PROMPT = PromptTemplate(
    template=json_only_prompt("""
Find mentions of: "customer experience","personalization","brand engagement","campaign launch","new product introduction".
Return JSON:
{{
  "mentions":[ {{"topic":"<topic>","excerpt":"<exact excerpt>","source":"<doc>"}} ],
  "engagement_signal":"<High|Medium|Low|Not found in documents.>"
}}

Context:
{context}
Question:
{question}
"""),
    input_variables=["context", "question"]
)

# -----------------------
# Combine Partial Answers
# -----------------------
COMBINE_PROMPT = PromptTemplate(
    template=json_only_prompt("""
Combine the following partial category outputs into one JSON object with all fields merged and duplicates removed.
Partial answers:
{summaries}

Question:
{question}
"""),
    input_variables=["summaries", "question"]
)

# -----------------------
# Retriever Prompt
# -----------------------
RETRIEVER_PROMPT = PromptTemplate(
    template=json_only_prompt("""
Generate up to 6 alternate queries and 10 keyword phrases to improve retrieval for the following question.

Return JSON:
{{
  "original_question": "{question}",
  "rewrites": ["<rewrite1>", "<rewrite2>", ...],
  "keywords": ["<kw1>", "<kw2>", ...]
}}
"""),
    input_variables=["question"]
)
