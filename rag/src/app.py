from data_loader import load_all_documents
from vectorstore import FaissVectorStore
from search import RAGSearch
from embedding import EmbeddingPipeline

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("D:/Studies/Programming/LangChain/rag/data")
    # chunks = EmbeddingPipeline().chunk_documents(docs)
    # chunkvectors = EmbeddingPipeline().embed_chunks(chunks)
    # print(chunkvectors)
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    print(store.query("What is R-CNN?",top_k=3))

    rag_search = RAGSearch()
    query = "What is R-CNN?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)