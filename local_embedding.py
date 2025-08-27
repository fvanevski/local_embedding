from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from typing import List

# -----------------------------------------------------------------------------
# 1. Choose a lightweight reranker model
# -----------------------------------------------------------------------------
# Models to consider (from largest to smallest):
# - "BAAI/bge-reranker-base" (~430MB) - Good quality, might fit.
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" (~90MB) - Excellent balance of size and performance.
# - "cross-encoder/ms-marco-TinyBERT-L-2-v2" (~50MB) - Very small and fast.

try:
    print("Loading reranker model...")
    # Using a smaller model to conserve VRAM
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# -----------------------------------------------------------------------------
# 2. Define API request and response models (to emulate Cohere's API)
# -----------------------------------------------------------------------------
class CohereRerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = Field(default=None)
    model: str = Field(default=None) # This will be ignored but is part of the Cohere API

class RerankResult(BaseModel):
    index: int
    relevance_score: float

class CohereRerankResponse(BaseModel):
    results: List[RerankResult]

# -----------------------------------------------------------------------------
# 3. Create the FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI()

@app.post("/v1/rerank", response_model=CohereRerankResponse)
async def rerank_documents(request: CohereRerankRequest):
    """
    Accepts a query and a list of documents, and returns them reranked by relevance.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Reranker model is not available.")

    # Create pairs of [query, document] for scoring
    pairs = [[request.query, doc] for doc in request.documents]

    # Predict scores
    print(f"Reranking {len(pairs)} documents...")
    scores = model.predict(pairs, show_progress_bar=False)
    print("Reranking complete.")

    # Combine original indices with scores
    indexed_scores = list(enumerate(scores))

    # Sort by score in descending order
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    # Optionally, limit to top_n results if specified
    top_n = request.top_n if request.top_n is not None else len(sorted_scores)
    final_sorted_scores = sorted_scores[:top_n]

    # Format the results in the Cohere API style
    results = [
        RerankResult(index=original_index, relevance_score=float(score))
        for original_index, score in final_sorted_scores
    ]

    return CohereRerankResponse(results=results)

@app.get("/health")
async def health_check():
    return {"status": "ok" if model is not None else "error", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    # Run the server on a specific port, e.g., 8004
    uvicorn.run(app, host="0.0.0.0", port=8004)
