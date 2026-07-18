"""
================================================================
  STEP 2 — BUILD RAG KNOWLEDGE BASE
  Loads knowledge_base/*.txt → ChromaDB vector store
  Run this ONCE before running the agent.
================================================================
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

KNOWLEDGE_BASE_PATH = "knowledge_base/"
CHROMA_DB_PATH      = "rag/chroma_db/"


def build_rag():
    print("=" * 55)
    print("  STEP 2 — BUILDING RAG VECTOR STORE")
    print("=" * 55)

    # Free embeddings — no API key needed
    print("\n  Loading embedding model (MiniLM)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("  Embedding model loaded.")

    # Load knowledge base text files
    print("\n  Loading knowledge base documents...")
    documents = []
    kb_path = Path(KNOWLEDGE_BASE_PATH)

    for txt_file in sorted(kb_path.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = txt_file.name
        documents.extend(docs)
        print(f"    Loaded: {txt_file.name}")

    print(f"\n  Total documents loaded: {len(documents)}")

    # Split into chunks
    print("\n  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")

    # Build ChromaDB
    print("\n  Building ChromaDB vector store...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print(f"  Vector store saved to: {CHROMA_DB_PATH}")
    print(f"  Total vectors stored : {vectorstore._collection.count()}")

    # Quick test
    print("\n  Testing retrieval...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    test_docs = retriever.invoke("temperature too high what action to take")
    print(f"  Test query returned {len(test_docs)} documents.")
    print(f"  Sample: {test_docs[0].page_content[:100]}...")

    print("\n  RAG READY. Now run: python step3_main_agent.py")
    return vectorstore


def load_rag():
    """Load existing vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """Retrieve relevant knowledge for a query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    if not docs:
        return "No specific guidelines found."
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[Guideline {i} from {src}]\n{doc.page_content}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    build_rag()
