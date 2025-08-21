from pathlib import Path

class RAGConfig:
    JSON_INPUT_ROOT: Path = Path("./json_data")         
    CHROMA_PERSIST_DIR: Path = Path("./chroma_store")   
    CHROMA_COLLECTION_NAME: str = "rag_child_chunks"
    SUMMARIES_COLLECTION: str = "rag_summaries"        
    PROMPTS_DIR: Path = Path("./prompt_versions")
    EXPERIMENTS_DIR: Path = Path("./experiments")
    MODEL_NAME: str = "gpt-4o-mini"                      
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    TEMPERATURE: float = 0.0
    K: int = 5                                          
    CHUNK_SIZE: int = 1500                             
    CHUNK_OVERLAP: int = 200

    def __init__(self):
        self.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)