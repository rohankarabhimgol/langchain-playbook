import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import load_all_documents

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class EmbeddingPipeline:
    def __init__(self,model_name:str = "all-MiniLM-L6-v2",chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self,documents:List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_chunks(self, chunks:List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated Embeddings with shape: {embeddings.shape}")
        return embeddings
    