from typing import List
from langchain.schema import BaseRetriever
from langchain.schema.document  import Document as LCDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfRetriever:
  
    def __init__(self, docs: List[LCDocument], k: int = 5):
        self.docs = docs
        self.k = k
        self.texts = [d.page_content for d in docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        if len(self.texts) == 0:
            self.mat = None
        else:
            self.mat = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query: str) -> List[LCDocument]:
        if self.mat is None:
            return []
        v = self.vectorizer.transform([query])
        scores = (self.mat @ v.T).toarray().ravel()
        idx = np.argsort(scores)[::-1][: self.k]
        # return top-k 
        return [self.docs[i] for i in idx]



class TfidfRetrieverWrapper(BaseRetriever):
    
    def __init__(self, tfidf_retriever):
        super().__init__()
        self._tfidf = tfidf_retriever   # existing class instance

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        return self._tfidf.get_relevant_documents(query)

    async def _aget_relevant_documents(self, query: str) -> List[LCDocument]:
        return self._get_relevant_documents(query)