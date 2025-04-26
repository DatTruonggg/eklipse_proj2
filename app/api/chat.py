# =====================
# app/api/chat.py (After Fixing Config Usage)
# =====================

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.model_loader import ml_models
from app.core.generate_response import generate_response
from app.configs import config 

# Initialize APIRouter
router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

# =====================
# Pydantic Models
# =====================
class ChatMessage(BaseModel):
    role: Optional[str] = "user"
    content: str

class ChatRequest(BaseModel):
    model: str = config.model_name 
    messages: List[ChatMessage]
    max_tokens: Optional[int] = config.max_token
    temperature: Optional[float] = config.temperature
    top_p: Optional[float] = config.top_p    
    frequency_penalty: Optional[float] = config.frequency_penalty
    stream: Optional[bool] = config.stream

class ChatResponse(BaseModel):
    id: str
    object: str
    created: float
    model: str
    choices: List

# =====================
# API Endpoints
# =====================

@router.get("/", response_model=dict)
async def health_check():
    """
    Health check to verify if LLM service is running.
    """
    if not ml_models.get("models") or not ml_models.get("tokenizers"):
        return {"status": "not started", "message": "You need to start the LLM first."}

    return {
        "status": "running",
        "llm_serving_url": config.llm_service, 
        "model_name": config.model_name,
        "max_tokens": config.max_token,
        "device": config.device,
        "do_sample": config.do_sample,
        "skip_special_tokens": config.skip_special_tokens,
    }

@router.post("/completions", response_model=ChatResponse)
async def complete_chat(request: ChatRequest):
    """
    Generate completion for the given chat prompt.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    prompt = request.messages

    if request.stream:
        stream = generate_response(prompt, model_name=request.model, stream=True)
        return StreamingResponse(stream, media_type="application/x-ndjson")

    generated_text = generate_response(prompt, model_name=request.model)

    return ChatResponse(
        id="completion-id",
        object="chat.completion",
        created=time.time(),
        model=request.model,
        choices=[{"message": {"role": "assistant", "content": generated_text}}]
    )