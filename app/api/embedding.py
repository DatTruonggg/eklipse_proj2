from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch

from app.core.model_loader import embed_models
from app.configs import config

# Initialize router
router = APIRouter(
    prefix="/embeddings",
    tags=["Embedding"]
)

# =====================
# Pydantic Models
# =====================

class EmbedRequest(BaseModel):
    input: List[str] 
    model: str = config.embedding_model

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbedResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str

# =====================
# API Endpoints
# =====================

@router.get("/", response_model=dict)
async def health_check():
    """
    Health check to verify if Embedding model service is running.
    """
    if not embed_models.list_models():
        return {"status": "not started", "message": "Embedding model is not loaded yet."}

    return {
        "status": "running",
        "embedding_model_name": config.embedding_model,
        "embedding_service_url": config.llm_service
    }

@router.post("/", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for a list of input texts.
    """
    if not request.input or any(not text.strip() for text in request.input):
        raise HTTPException(status_code=400, detail="Input empty")

    model = embed_models.get_model(request.model)
    tokenizer = embed_models.get_tokenizer(request.model)

    embeddings = []

    for idx, text in enumerate(request.input):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Take the [CLS] token (first token) embedding
            embedding_vector = outputs.last_hidden_state[:, 0].squeeze().tolist()

        embeddings.append(EmbeddingObject(
            embedding=embedding_vector,
            index=idx
        ))

    response = EmbedResponse(
        object="list",
        data=embeddings,
        model=request.model
    )
    return response