from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_wikipedia_document(file_path: str):
    """
    Load the scraped Wikipedia text as a LangChain Document.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def chunk_documents(documents):
    """
    Split documents into smaller overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings():
    """
    Initialize the embedding model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_and_save_faiss_index(chunks, embeddings, index_path="vector_db/faiss_index"):
    """
    Create FAISS vector store and save it to disk.
    """
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store


def main():
    file_path = "data/wikipedia.txt"

    # Step 1: Load document
    documents = load_wikipedia_document(file_path)
    print(f"Loaded {len(documents)} document(s)")

    # Step 2: Chunk text
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} text chunks")

    # Step 3: Embeddings
    embeddings = create_embeddings()
    print("Embedding model loaded")

    # Step 4: Vector DB (FAISS)
    vector_store = create_and_save_faiss_index(chunks, embeddings)
    print("FAISS vector database created and saved")

    # Quick test: similarity search
    query = "What is artificial intelligence?"
    results = vector_store.similarity_search(query, k=2)

    print("\n--- Similarity Search Test ---\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\n{doc.page_content[:300]}\n")


if __name__ == "__main__":
    main()
