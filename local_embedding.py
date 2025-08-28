# local_embedding.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any
import logging

# Set up logging
logger = logging.getLogger("local_embedding_service")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# 1. Choose lightweight models for embedding and reranking
# -----------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# --- Load Models ---
try:
    logger.info("Loading embedding model...")
    # Using 'cuda' for GPU acceleration if available, otherwise it defaults to CPU
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    logger.info(f"Embedding model '{EMBEDDING_MODEL}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embedding model: {e}")
    embedding_model = None
try:
    logger.info("Loading reranker model...")
    reranker_model = CrossEncoder(RERANKER_MODEL, device='cpu', max_length=512)
    logger.info(f"Reranker model '{RERANKER_MODEL}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading reranker model: {e}")
    reranker_model = None

# -----------------------------------------------------------------------------
# 2. Define API request and response models
# -----------------------------------------------------------------------------

# --- Embedding Models (emulating OpenAI) ---
class OpenAIEmbeddingRequest(BaseModel):
    input: List[str]
    model: str = Field(default=None) # Ignored, for API compatibility

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class UsageData(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = EMBEDDING_MODEL
    usage: UsageData

# --- Rerank Models (emulating Cohere) ---
class CohereRerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = Field(default=None)
    model: str = Field(default=None) # Ignored, for API compatibility

class RerankResult(BaseModel):
    index: int
    relevance_score: float

class CohereRerankResponse(BaseModel):
    results: List[RerankResult]

# -----------------------------------------------------------------------------
# 3. Create the FastAPI application and endpoints
# -----------------------------------------------------------------------------
app = FastAPI()

@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_embeddings(request: OpenAIEmbeddingRequest):
    """
    Generates embedding vectors for a list of input strings.
    """
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model is not available.")

    logger.info(f"Creating embeddings for {len(request.input)} documents...")
    embeddings = embedding_model.encode(request.input, show_progress_bar=False)
    logger.info("Embedding creation complete.")

    data = [
        EmbeddingData(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(embeddings)
    ]
    
    # You can implement actual token counting if needed, but for now, it's 0.
    usage = UsageData(prompt_tokens=0, total_tokens=0)

    logger.info(f"Created {len(data)} embeddings.")
    return OpenAIEmbeddingResponse(data=data, usage=usage)

@app.post("/v1/rerank", response_model=CohereRerankResponse)
async def rerank_documents(request: CohereRerankRequest):
    """
    Accepts a query and documents, returning them reranked by relevance.
    """
    if reranker_model is None:
        logger.error("Reranker model is not available.")
        raise HTTPException(status_code=500, detail="Reranker model is not available.")

    pairs = [[request.query, doc] for doc in request.documents]
    logger.info(f"Reranking {len(pairs)} documents...")
    scores = reranker_model.predict(pairs, show_progress_bar=False)
    logger.info("Reranking complete.")

    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    top_n = request.top_n if request.top_n is not None else len(sorted_scores)
    final_sorted_scores = sorted_scores[:top_n]

    results = [
        RerankResult(index=original_index, relevance_score=float(score))
        for original_index, score in final_sorted_scores
    ]

    logger.info(f"Reranked {len(results)} documents.")
    return CohereRerankResponse(results=results)

@app.get("/health")
async def health_check():
    """
    Health check for both models.
    """
    return {
        "status": "ok" if reranker_model and embedding_model else "error",
        "reranker_model_loaded": reranker_model is not None,
        "embedding_model_loaded": embedding_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server on a specific port, e.g., 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)