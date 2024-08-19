import ollama
response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'extract the entity from this sentence. i went to goa by OLA',
  },
])
print(response['message']['content'])