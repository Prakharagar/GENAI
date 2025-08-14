from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


'''
1. create model
2. design prompt
    2.a. give/extgract input which is required for design template
3. invoke prompt tempate
4. structure output to the model
4. invoke model
'''

#create model

model=ChatOpenAI(
    temperature=0,
    max_tokens=512
)
prompt ="""
You are a strict sentiment analysis model. Based only on the text below, 
complete the following with NO extra information or variation. Please proceed stet by step and understand the text completetly with chain of thoughts
There should be three explainations
Overall probablity should be calculated considering all three probablity into the account

Text: "{document_text}"
- Explanation 1: ...
  Sentiment: <Positive/Neutral/Negative>
  Probability (must add up to 100%):
    - Positive: <xx>%
    - Neutral: <xx>%
    - Negative: <xx>%

- Explanation 2: ...
  Sentiment: <Positive/Neutral/Negative>
  Probability (must add up to 100%):
    - Positive: <xx>%
    - Neutral: <xx>%
    - Negative: <xx>%

- Explanation 3: ...
  Sentiment: <Positive/Neutral/Negative>
  Probability (must add up to 100%):
    - Positive: <xx>%
    - Neutral: <xx>%
    - Negative: <xx>%

Overaall Explanation: <gives overall sentiments and also explain why the overall probablity is valid for this article>
Overall Sentiment: <Positive/Neutral/Negative>
Overall Probability: <xx>%
"""

document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks'''

prompt_template=PromptTemplate(
    template=prompt,
    input_variables=['document_text']
)
pt_result=prompt_template.invoke({'document_text':document_text})

from typing import TypedDict,Literal

class ExplainationDict(TypedDict):
    explanation: str
    sentiment: str
    probability: float

class OutputParser(TypedDict):
    explanations: list[ExplainationDict]
    overall_explaination: str
    overall_sentiments: Literal["Positive", "Negative", "Neutral"]
    overall_probablilty: float

structured_output=model.with_structured_output(OutputParser)
#print(structured_output)

result=structured_output.invoke(pt_result)
print(result)



