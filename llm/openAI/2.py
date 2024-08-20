from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
project_id='proj_XUmebccsti62Gor5BRpQS0C3'
org_id='org-8XwQuUkcdwOPFzPXvmJFEDhE'
model_name=os.getenv('model_name')
client=OpenAI(api_key=os.getenv('token'),
              project=os.getenv('project_id'),
              organization=os.getenv('org_id')
)
prompt=[
    { "role":"system",
     "content":"you act as a financial advisor of India."
    },
    {
        "role":"user",
        "content": "tell me what woill be the in hand monthly salary if your income is 60 lakh per annum"
    }
]
datas=client.chat.completions.create(
    
    model=model_name,
    messages=prompt,
    n=5,
    max_tokens=20,
    temperature=0.1,
    logprobs=True,
    

)
print(datas)

