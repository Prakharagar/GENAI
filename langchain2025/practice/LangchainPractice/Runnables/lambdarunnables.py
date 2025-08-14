from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough,RunnableParallel
from config_loader import *
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model=ChatOpenAI(max_tokens=512,temperature=1)

query1 ='tell me breif jokes on topic {topic1}'
template1=PromptTemplate(template=query1,input_variables=['topic1'])

query2 ='tell me breif summary on topic {topic2}'
template2=PromptTemplate(template=query2,input_variables=['topic2'])

query3="merge {jokes} and summary {summary}"
template3=PromptTemplate(template=query3,input_variables=['jokes','summary'])

# output should be query 1 , count word + query 2 count words and query 3 count words

def countwords(text):
    return len(text.split(' '))

count_r=RunnableLambda(countwords)
parser=StrOutputParser()

joke_chain=template1|model|parser
jokes_count= joke_chain | RunnableLambda(countwords)

summary_chain=template2|model|parser
summary_count=summary_chain | RunnableLambda(countwords)

merge_chain =RunnableParallel({
    'jokes':joke_chain,   
    'summary':summary_chain    
}
)

chain=merge_chain|template3|model|parser

Final_chain =RunnableParallel (
    {
      'joke': joke_chain,
      'joke_count':jokes_count,
      'summary':summary_chain,
      'summary_count': summary_count,
      'final': chain,
      'final_count':chain|RunnableLambda(countwords)


    }
)
result=Final_chain.invoke({"topic1":"donald trump","topic2":"Rahul gandhi"})

print(result)

