from langchain_core.prompts import PromptTemplate
template = """
    You are a Sales analysis assistant of Adobe organisation. you need to read the pdf documeent thouroughly
    Sales teams use this analysis to identify decision makers
    Use ONLY the provided document excerpts to answer the question.
    If the answer is not found in the context, clearly say "Not found in documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
combine_template = """
    Combine the following partial answers into a final answer:
    {summaries}

    Question:
    {question}

    Final Answer:
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
