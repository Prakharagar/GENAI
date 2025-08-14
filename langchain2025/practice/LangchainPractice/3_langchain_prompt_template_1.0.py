from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

'''
1. create model
2. design template
    2.a. give/extgract input which is required for design template
3. invoke prompt tempate
4. invoke model
'''

model =ChatOpenAI(
    temperature=0,
    max_tokens=512
)

template ="""
You are a strict sentiment analysis model. Based only on the text below, 
complete the following with NO extra information or variation.

Text: "{document_text}"
- Explanation 1: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 2: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 3: ...
  Sentiment: <Positive/Neutral/Negative>

Probability (must add up to 100%):
- Positive: <xx>%
- Neutral: <xx>%
- Negative: <xx>%

Overall Sentiment: <Positive/Neutral/Negative>
Overall Probability: <xx>%
"""
document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks'''

prompttemplate=PromptTemplate(
    name="test", # so that in json prompt template will come
    template=template, 
    input_variables=['document_text'],
    validate_template=True
)

import json
prompttemplate.save('prompt_template_v1.0.json')

pt_result=prompttemplate.invoke(input={"document_text":document_text})
print(pt_result)

model_result=model.invoke(pt_result)
print(model_result.content)

