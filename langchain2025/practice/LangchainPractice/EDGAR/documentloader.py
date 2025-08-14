from config_loader import *
from datetime import datetime 
from langchain_community.document_loaders import DirectoryLoader,UnstructuredHTMLLoader
tinker="AMZN"
latest_file=f"{tinker}-20250207.htm"
loader =DirectoryLoader(
    path=f"download/{tinker}",
    loader_cls=UnstructuredHTMLLoader,
)

start_time = datetime.now()
documents = loader.lazy_load()
# Print the loaded content
for doc in documents:
    #print(doc.page_content)
    print(doc.metadata)
    #print(doc.__pretty__)
    print(doc.page_content) 
end_time = datetime.now() 
time_spent = end_time - start_time
print("Total time spent:", time_spent)
print("Seconds spent:", time_spent.total_seconds())

# print(len(documents) )
# print(documents[-1].metadata)
# print(len(documents[-1].page_content))
    