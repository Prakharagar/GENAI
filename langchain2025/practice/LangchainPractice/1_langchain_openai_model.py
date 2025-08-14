import os
from config_loader import *
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
model_name=os.environ["OPENAI_MODEL"]
print(model_name)
model=ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    temperature=0,
    max_tokens=10,
    logprobs=False
    )
#result=model.invoke({"Hello"})
result=model.invoke([HumanMessage("POEM")])
print(result.content)
print(result.response_metadata)


"""
Absolutely! Let's go through ChatOpenAI(...) in detail — parameter by parameter — and understand:

✅ What each parameter does

📅 When to use it

⚙️ Why it's important

This class is from LangChain, and it wraps OpenAI's chat models (like gpt-3.5-turbo, gpt-4, etc.),
 giving you an easy way to integrate them into chains, tools, agents, etc.

 from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(
    openai_api_key="YOUR_API_KEY",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    streaming=False,
    request_timeout=60,
    n=1,
    logprobs=None,
    seed=None,
    openai_api_base=None,
    openai_organization=None,
    model_kwargs=None
)

3. temperature
What: Controls randomness of the output.
Range: 0 to 2 (default: 1).
Why: Lower = more deterministic, higher = more creative.
When to use:

0 → for factual, deterministic outputs.
> 1 → for brainstorming, creative tasks.

4. max_tokens
What: Max number of tokens in the response.
Why: Controls the output length.
When to use: When you want to limit costs or response size.

5. top_p
What: Nucleus sampling; another way to control randomness.
Why: Works with temperature. It limits token selection to a cumulative probability.
When to use:
Leave at 1 unless fine-tuning randomness with temperature.

6. frequency_penalty
What: Penalizes repeated tokens (on frequency)
Range: -2 to 2.
Why: Helps avoid repetition of words.
When to use: Set > 0 if the model keeps repeating.

7. presence_penalty
What: Penalizes tokens that have already appeared.
Range: -2 to 2.
Why: Encourages introducing new topics.
When to use: Set > 0 if the model is too focused on a single topic.

8. streaming
What: Enables streaming token-by-token responses.

Why: Useful for real-time applications (like chat UIs).

When to use: Set True for partial output as it’s being generated.

9. request_timeout
What: How long to wait for OpenAI’s API to respond (in seconds).

Why: Prevents hanging if API is slow or down.

When to use: Always recommended, especially in production.

10. n
What: Number of completions to generate per prompt.

Why: Allows multiple response options.

When to use: Useful for reranking or choosing from multiple outputs.

11. logprobs
What: (⚠️ Only for completion models) Log-probabilities for token analysis.

Why: Helps in debugging or getting model confidence.

When to use: Only if analyzing token-level confidence. Not supported by chat models directly.

12. seed
What: Random seed for reproducibility.

Why: Ensures the same output on repeated runs.

When to use: In testing or debugging, for consistent results.

13. openai_api_base
What: Custom API base URL.

Why: Use if you're proxying requests or using Azure OpenAI, etc.

When to use: Custom deployments or gateways.

14. openai_organization
What: Your OpenAI org ID (if using teams).

Why: Enables billing and org-scoped usage.

When to use: For enterprise or org-linked accounts.

15. model_kwargs
What: Dictionary of additional OpenAI parameters.

Why: To pass options not explicitly exposed by LangChain.

When to use: For rare or advanced settings like user, tools, response_format, etc.





OUTPUT OF print(result.response_metadata)

"response_metadata": {
    "logprobs": {
        "tokens": ["The", " quick", " brown", " fox"],
        "token_logprobs": [-0.1, -0.2, -0.05, -0.01],
        "top_logprobs": [
            {"The": -0.1, "A": -1.2, "It": -2.0},
            {" quick": -0.2, " slow": -1.1, " fast": -1.5},
            ...
        ],
        "text_offset": [0, 4, 10, 16]
    }
}
| Field            | Description                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| `tokens`         | List of tokens the model output (in order).                                                                    |
| `token_logprobs` | The log-probability the model assigned to **each generated token**. Higher (closer to 0) means more confident. |
| `top_logprobs`   | For each token position, a dictionary of the **top N alternative tokens** and their logprobs.                  |
| `text_offset`    | Character offset in the text where each token starts. Useful for aligning to original text.                    |


"""



