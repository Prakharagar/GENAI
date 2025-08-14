from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config_loader import *

model =ChatOpenAI(max_tokens=56, temperature=0)
parser=JsonOutputParser()

template1="summarize me explanationon the topic of {text1}"
prompt1=PromptTemplate(template=template1,input_variables=['text1'])

template2="give me 3 headline for summary {text2} {output_instruction}"
prompt2=PromptTemplate(template=template2,input_variables=['text2'],
                       partial_variables={'output_instruction': parser.get_format_instructions()})



chain=prompt1|model|prompt2|model|parser



result= chain.invoke({'Data engineering'})
print(result)

chain.get_graph().print_ascii()

