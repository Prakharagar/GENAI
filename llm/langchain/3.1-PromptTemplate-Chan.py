from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch model name and token from environment variables
model_name = os.getenv('model_name')
token = os.getenv('token')

# Initialize the OpenAI model
model = OpenAI(model=model_name, api_key=token, temperature=0, max_tokens=100)

# Define the prompt template
template = [
    ('system', 'Translate the sentence in following language: {language}'),
    ('user', '{text}')
]

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template=template)
# Initialize the StrOutputParser
parser = StrOutputParser()

# Combine the prompt, model, and parser into a chain
# This is a placeholder for actual chaining; you may need to adapt based on the library's documentation
chain = prompt | model | parser

# Input for the translation
input_data = {'language': "Hindi", 'text': "Do you know how to do this?"}

# Invoke the chain and print the result
result = chain.invoke(input_data)
print(result)
