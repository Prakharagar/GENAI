from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt

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

document_text='''my day is not bad however i enjoy a lot in office 
but professionaly my day sucks'''

prompttemplate=load_prompt(
    path='prompt_template_v1.0.json')

print(prompttemplate)

pt_result=prompttemplate.invoke(input={"document_text":document_text})
print(pt_result)

model_result=model.invoke(pt_result)
print(model_result.content)

