from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

model_name=os.getenv('model_name')
token=os.getenv('token')
model=ChatOpenAI(model=model_name,api_key=token,temperature=0,max_tokens=100)


prompt_template = [
    ('system', "Translate the following into {language}:"),
    ('user', '{text}')
]

prompt=ChatPromptTemplate.from_messages(prompt_template)
parser=StrOutputParser()

chain=prompt|model|parser

input={'language':"Hindi","text":"i stay in agra near taj mahal"}

print(chain.invoke(input))