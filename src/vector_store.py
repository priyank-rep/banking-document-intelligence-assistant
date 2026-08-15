"""
Vector Store Module for Banking Document Intelligence Assistant.

Handles:
1. Local persistent vector storage using ChromaDB.
2. Embedding generation using OpenAI's text-embedding-3-small model.
3. Batch indexing with duplicate prevention via upsert.
4. Top-K semantic similarity retrieval with cosine distance and similarity scoring.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# OpenAI Client & Embeddings Generation
# ==============================================================================

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Initialize and return an OpenAI client instance.
    Uses the provided API key or resolves it securely from config.
    """
    key = api_key or config.get_openai_api_key()
    return OpenAI(api_key=key)


def generate_embeddings(
    texts: List[str],
    client: Optional[OpenAI] = None,
    model: str = config.EMBEDDING_MODEL,
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate vector embeddings for a list of text strings using OpenAI API.

    Args:
        texts: List of text strings to embed.
        client: Optional OpenAI client instance.
        model: Embedding model name (defaults to text-embedding-3-small).
        batch_size: Maximum number of texts to embed in a single API request.

    Returns:
        A list of embedding vectors (each vector is a list of floats, e.g. 1536 dimensions).
    """
    if not texts:
        return []

    openai_client = client or get_openai_client()
    all_embeddings = []

    # Process in batches to respect API payload limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace newlines with spaces as recommended by OpenAI for embedding inputs
        sanitized_batch = [t.replace("\n", " ") if t else " " for t in batch]

        response = openai_client.embeddings.create(
            input=sanitized_batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ==============================================================================
# ChromaDB Client & Collection Management
# ==============================================================================

def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """
    Initialize a local persistent ChromaDB client pointing to the data directory.
    """
    path = persist_directory or str(config.CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = config.CHROMA_COLLECTION_NAME
) -> chromadb.Collection:
    """
    Get or create a Chroma collection configured with cosine similarity distance space.
    OpenAI embeddings are normalized, making cosine space optimal for semantic matching.
    """
    chroma_client = client or get_chroma_client()
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ==============================================================================
# Indexing & Ingestion
# ==============================================================================

def index_chunks(
    chunks: List[Dict[str, Any]],
    collection: Optional[chromadb.Collection] = None,
    openai_client: Optional[OpenAI] = None
) -> int:
    """
    Index a list of chunk dictionaries into the persistent Chroma vector collection.

    Uses `upsert` rather than `add` to handle duplicate ingestion gracefully.
    If a chunk with the same chunk_id already exists, its text and embedding are updated
    rather than creating duplicate entries.

    Args:
        chunks: List of chunk dictionaries produced by src.chunker.chunk_pages
        collection: Optional Chroma collection instance
        openai_client: Optional OpenAI client instance

    Returns:
        The number of chunks indexed.
    """
    if not chunks:
        logger.warning("No chunks provided to index.")
        return 0

    target_collection = collection or get_or_create_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["chunk_text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk_id": chunk["chunk_id"],
            "token_count": int(chunk.get("token_count", 0))
        }
        for chunk in chunks
    ]

    logger.info(f"Generating embeddings for {len(documents)} chunks using {config.EMBEDDING_MODEL}...")
    embeddings = generate_embeddings(documents, client=openai_client)

    logger.info(f"Upserting {len(ids)} chunks into Chroma collection '{target_collection.name}'...")
    target_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    logger.info(f"Successfully indexed {len(ids)} chunks into vector store.")
    return len(ids)


# ==============================================================================
# Semantic Query & Retrieval
# ==============================================================================

def query_vector_store(
    query_text: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    collection: Optional[chromadb.Collection] = None,
    openai_client: Optional[OpenAI] = None
) -> List[Dict[str, Any]]:
    """
    Query the Chroma vector database using semantic similarity search.

    Args:
        query_text: The user's question or search query.
        top_k: Maximum number of relevant chunks to retrieve.
        collection: Optional Chroma collection instance.
        openai_client: Optional OpenAI client instance.

    Returns:
        A list of top-K retrieved chunk dictionaries sorted by relevance:
        [
            {
                "chunk_id": "apex_bank_fee_schedule_p1_c0",
                "source": "apex_bank_fee_schedule.pdf",
                "page": 1,
                "chunk_text": "...",
                "token_count": 318,
                "distance": 0.2145,           # Cosine distance (0.0 = identical, 2.0 = opposite)
                "similarity_score": 0.7855     # Cosine similarity = 1 - distance (higher is better)
            },
            ...
        ]
    """
    if not query_text or not query_text.strip():
        return []

    target_collection = collection or get_or_create_collection()

    # 1. Convert user query into an embedding vector
    query_embeddings = generate_embeddings([query_text.strip()], client=openai_client)
    if not query_embeddings:
        return []

    # 2. Query Chroma for the nearest neighbor chunks
    results = target_collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Format and flatten Chroma's nested list output
    retrieved_chunks = []
    if results and "ids" in results and results["ids"] and results["ids"][0]:
        ids_list = results["ids"][0]
        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        for i in range(len(ids_list)):
            cosine_dist = dists_list[i]
            # Cosine similarity is 1.0 - cosine_distance
            sim_score = max(0.0, min(1.0, 1.0 - cosine_dist))

            retrieved_chunks.append({
                "chunk_id": ids_list[i],
                "source": metas_list[i].get("source", "unknown"),
                "page": metas_list[i].get("page", 1),
                "chunk_text": docs_list[i],
                "token_count": metas_list[i].get("token_count", 0),
                "distance": round(float(cosine_dist), 4),
                "similarity_score": round(float(sim_score), 4)
            })

    return retrieved_chunks


# ==============================================================================
# Helper Utilities
# ==============================================================================

def get_collection_stats(collection: Optional[chromadb.Collection] = None) -> Dict[str, Any]:
    """
    Return summary statistics about the current Chroma collection.
    """
    target_collection = collection or get_or_create_collection()
    count = target_collection.count()
    return {
        "collection_name": target_collection.name,
        "total_chunks": count,
        "persist_dir": str(config.CHROMA_PERSIST_DIR)
    }


def clear_collection(collection: Optional[chromadb.Collection] = None) -> None:
    """
    Delete all chunks in the collection (useful for fresh resets).
    """
    target_collection = collection or get_or_create_collection()
    all_ids = target_collection.get()["ids"]
    if all_ids:
        target_collection.delete(ids=all_ids)
        logger.info(f"Cleared {len(all_ids)} chunks from collection '{target_collection.name}'.")
