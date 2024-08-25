from langchain_huggingface import HuggingFaceEndpoint
import os
token=''

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]=token

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=model_id,max_length=128,temperature=0.7,token=token)

print(llm.invoke("extract the entity from this sentence. i went to goa by OLA"))