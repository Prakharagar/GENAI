from config_loader import *
from langchain_openai import ChatOpenAI

model=ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    temperature=0,
    max_tokens=512,
    logprobs=False
    )

document_text='my day is not bad however i enjoy a lot in office but professionaly my day sucks'

prompt = f'''
You are a strict sentiment analysis model. Based only on the text below, 
complete the following with NO extra information or variation.

Text: "{document_text}"

- Explanation 1: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 2: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 3: ...
  Sentiment: <Positive/Neutral/Negative>

Probability (must add up to 100%):
- Positive: <xx>%
- Neutral: <xx>%
- Negative: <xx>%

Overall Sentiment: <Positive/Neutral/Negative>
Overall Probability: <xx>%

'''

result=model.invoke(prompt)
print(result.content)



'''
You're noticing inconsistent sentiment analysis results across multiple runs despite setting temperature=0 in ChatOpenAI. Let's break down why this is happening and how to make it deterministic.

✅ 1. Why is the result changing?
Even with temperature=0, which makes the model less random, the outputs can still vary due to the following reasons:

a. Non-deterministic backend (OpenAI)
OpenAI models like gpt-4, gpt-3.5-turbo, etc., can still show variability even at temperature=0 because:

They are run across multiple servers or infrastructures.

Internal randomness may exist in token sampling or response generation.

Responses with equal probabilities may differ.

b. Prompt ambiguity
Your prompt asks for multiple subjective judgments like:

“Give 3 brief explanations...”

“Probability scores in %”

“Overall sentiments”

The text itself includes mixed sentiments:

“not bad” → potentially neutral/positive

“enjoy a lot in office” → positive

“professionally my day sucks” → negative

So depending on how the model "interprets" emphasis or weighting, the output may shift.

c. No system-level control over determinism
ChatOpenAI does not expose a seed setting (like random seed in ML), so results can differ even at temperature 0.

🛑 2. Misconception: temperature=0 ≠ Fully deterministic
Setting temperature=0 does reduce randomness, but:

It does not guarantee the same exact output every time.

You’d need exact token sampling order, same backend, and same version of the model.

✅ 3. How to reduce randomness or get consistent output
🔒 Option 1: Cache the result (Best for production)
If you’re analyzing the same input text, cache the result:

python
Copy
Edit
import hashlib
import json
import os

def cache_result(prompt, model):
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = f"./cache/{cache_key}.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    else:
        result = model.invoke(prompt)
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"response": result.content}, f)
        return {"response": result.content}
Then call:

python
Copy
Edit
result = cache_result(prompt, model)
print(result["response"])
🧠 Option 2: Refactor the prompt for more deterministic structure
Give the model a clear evaluation format and fixed structure, e.g.:

python
Copy
Edit
prompt = f"""
You are a strict sentiment analysis model. Based only on the text below, complete the following with NO extra information or variation.

Text: "{document_text}"

- Explanation 1: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 2: ...
  Sentiment: <Positive/Neutral/Negative>

- Explanation 3: ...
  Sentiment: <Positive/Neutral/Negative>

Probability (must add up to 100%):
- Positive: <xx>%
- Neutral: <xx>%
- Negative: <xx>%

Overall Sentiment: <Positive/Neutral/Negative>
Overall Probability: <xx>%
"""
This reduces model creativity and forces it into a more rigid response pattern.

🧪 Option 3: Use a Fine-tuned or Sentiment-specific model
If you need consistent sentiment output, consider:

OpenAI fine-tuned models with static outputs.

Dedicated sentiment classifiers like:

textblob

VADER

HuggingFace transformers: distilbert-base-uncased-finetuned-sst-2-english

These are deterministic and trained for classification, not text generation.

✅ Summary
Issue	Solution
Responses vary	Prompt is ambiguous + OpenAI model backend is non-deterministic
temperature=0	Helps reduce randomness, but not fully deterministic
Want consistency	Use caching, simplify prompt, or switch to a classifier model

Let me know if you'd like help rewriting the prompt or using a classifier model.

'''
