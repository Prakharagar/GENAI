token='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'
#token='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'
import os
#os.environ["OPENAI_API_KEY"]=token
project_id='proj_XUmebccsti62Gor5BRpQS0C3'
org_id='org-8XwQuUkcdwOPFzPXvmJFEDhE'
model_name="gpt-3.5-turbo"

from openai import OpenAI
client = OpenAI(
                api_key=token,
                project=project_id,
                organization=org_id
                )

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)