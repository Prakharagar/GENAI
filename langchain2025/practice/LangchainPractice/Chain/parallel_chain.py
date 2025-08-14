from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from config_loader import *

model =ChatOpenAI(max_tokens=512, temperature=0)
parser=StrOutputParser()

template1="Give some advantage of this topic of {topic1}"
prompt1=PromptTemplate(template=template1,input_variables=['topic1'])

template2="Give some disadvantage of this topic of {topic1}"
prompt2=PromptTemplate(template=template2,input_variables=['topic1'])

template3= "give probablity to each advantages {text1} and disadvantages  on {text2}"
prompt3=PromptTemplate(
    template=template3,
    input_variables=['text1','text2']
)

parallelchain=RunnableParallel({
    'text1': prompt1 | model | parser,
    'text2': prompt2 | model | parser
}
)

chain=parallelchain|prompt3|model|parser


topic1={'topic1' :'Data engineering'}
result= chain.invoke(topic1)
print(result)

#chain.get_graph().print_ascii()

