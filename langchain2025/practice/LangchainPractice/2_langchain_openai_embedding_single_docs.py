from config_loader import *
from langchain_openai import OpenAIEmbeddings
import os

print(os.environ["OPENAI_EMBEDED_MODEL"])
embed =OpenAIEmbeddings(
    model=os.environ["OPENAI_EMBEDED_MODEL"],
    dimensions=32,
    show_progress_bar=True

    )
query ="capital of india"

embs=embed.embed_query(query)
print(embs)

'''
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your-key",
    openai_api_base=None,
    openai_organization=None,
    chunk_size=1000,
    show_progress_bar=False,
    request_timeout=60,
    headers=None,
    dimensions=None
)

1. model
What: The name of the OpenAI embedding model.

Common values:

"text-embedding-3-small" (fast, cost-effective)

"text-embedding-3-large" (more accurate, higher dimensions)

(Older: "text-embedding-ada-002")

Why it matters:

Determines quality, speed, and embedding size.

Use text-embedding-3-large for better performance on long texts or semantic tasks.

When to use:

Use small for efficiency.

Use large for more precision (e.g., legal, technical documents).

Example:

python
Copy
Edit
OpenAIEmbeddings(model="text-embedding-3-large")
2. openai_api_key
What: Your API key from OpenAI.

When: Always required unless using environment variables.

Secure alternative: Set it via environment variable OPENAI_API_KEY.

3. openai_api_base
What: Custom API endpoint (e.g., if you're using Azure OpenAI or a proxy).

When to use:

If you're hosting your own proxy

If using Azure OpenAI

Example:

python
Copy
Edit
OpenAIEmbeddings(openai_api_base="https://your-azure-url.openai.azure.com/")
4. openai_organization
What: Your OpenAI organization ID.

When to use: If you're part of a team/org billing account.

Can also be set via: OPENAI_ORGANIZATION env variable.

5. chunk_size
What: Number of documents to send to the API in one batch.

Default: 1000

Why: Larger chunk sizes = better performance (fewer HTTP calls).

When to change it:

If you hit rate limits, reduce it.

If you want faster embedding, increase it (if API allows).

Example:

python
Copy
Edit
OpenAIEmbeddings(chunk_size=512)
6. show_progress_bar
What: Whether to show a progress bar while embedding batches.

Why: Helps track progress when embedding many docs.

When: Set True if running in notebooks or long tasks.

7. request_timeout
What: Timeout (in seconds) for API requests.

Why: Prevents the script from hanging if OpenAI is slow.

When to use: Set higher timeout if embedding long docs.

Example:

python
Copy
Edit
OpenAIEmbeddings(request_timeout=120)
8. headers
What: Custom headers to send with your request.

When: Only needed for advanced setups (e.g., internal proxies, extra auth).

Example:

python
Copy
Edit
OpenAIEmbeddings(headers={"X-My-Custom-Header": "token"})
9. dimensions
What: The number of dimensions to reduce the embedding to.

When: Only supported by text-embedding-3-* models.

Why: Useful for saving space in vector databases.


When and Why Use Embeddings?
Embeddings convert text into dense vector representations that capture meaning. You use them to:

| Use Case                             | Example                                           |
| ------------------------------------ | ------------------------------------------------- |
| Semantic Search                      | Find documents similar to a query                 |
| Clustering                           | Group similar support tickets or reviews          |
| RAG (Retrieval-Augmented Generation) | Provide context to GPT models                     |
| Classification                       | Build simple ML classifiers with vector distances |


'''