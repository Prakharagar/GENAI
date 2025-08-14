from config_loader import *
from langchain_openai import OpenAIEmbeddings
import os

print(os.environ["OPENAI_EMBEDED_MODEL"])
embed =OpenAIEmbeddings(
    model=os.environ["OPENAI_EMBEDED_MODEL"],
    dimensions=32,
    show_progress_bar=True

    )
docs =[
    "capital of india",
    "how are you",
    "manchester",
    "adobe excels in marketing",
    "president of united states",
    "prime minister of india"
]

doc_emb=embed.embed_documents(docs)
print(doc_emb)

from sklearn.metrics.pairwise import cosine_similarity

query="photoshop"

query_emb=embed.embed_query(query)

cs=cosine_similarity([query_emb],doc_emb)[0]
print(cs)
lst=list(enumerate(cs))
print(lst)

sorted_lst=sorted(lst,key=lambda x:x[1],reverse=True)
print(sorted_lst)
indx=sorted_lst[0][0]
print(docs[indx])


