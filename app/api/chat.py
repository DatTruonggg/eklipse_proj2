import time
import psutil
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.model_loader import ml_models
from app.core.generate_response import generate_response
from app.configs import config

from prometheus_client import Summary, Gauge

# ===========
# Prometheus metrics
# ===========

REQUEST_LATENCY = Summary('llm_request_latency_seconds', 'Time for a single LLM request')
REQUEST_SUCCESS = Gauge('llm_request_success', 'Whether last request succeeded (1: success, 0: fail)')
RESPONSE_LENGTH = Gauge('llm_response_length', 'Number of characters in the response')
RESPONSE_TOKENS = Gauge('llm_response_tokens', 'Number of tokens in the response')

LLM_RAM_USAGE_MB = Gauge("llm_ram_usage_mb", "RAM usage of LLM process (MB)")
LLM_TOKENS_PER_SEC = Gauge("llm_tokens_per_second", "Token generation speed (tokens/sec)")
LLM_ACCURACY = Gauge("llm_eval_accuracy", "Offline benchmark accuracy or consistency")
LLM_LOGPROB_SUPPORT = Gauge("llm_logprob_support", "Whether model supports log probabilities")
LLM_NUM_PARAMS = Gauge("llm_num_parameters", "Total number of model parameters")

# ===========
# FastAPI Router
# ===========

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

# ===========
# Pydantic Models
# ===========

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

def compute_token_length(text: str) -> int:
    tokenizer = ml_models.get_tokenizer(config.model_name)
    return len(tokenizer.encode(text))

# ===========
# Init one-time metrics after model load
# ===========

def initialize_static_metrics():
    try:
        model = ml_models.get_model(config.model_name)
        config_obj = getattr(model, "config", None)

        # Only set once
        if not hasattr(initialize_static_metrics, "_initialized"):
            LLM_ACCURACY.set(0.71)  # Example static accuracy from benchmark
            LLM_NUM_PARAMS.set(sum(p.numel() for p in model.parameters()))
            if config_obj and getattr(config_obj, "output_scores", False):
                LLM_LOGPROB_SUPPORT.set(1)
            else:
                LLM_LOGPROB_SUPPORT.set(0)
            initialize_static_metrics._initialized = True
    except Exception as e:
        print(f"[WARN] Could not initialize static metrics: {e}")

# ===========
# API Endpoints
# ===========

@router.get("/", response_model=dict)
async def health_check():
    if not ml_models.get("models") or not ml_models.get("tokenizers"):
        return {"status": "not started", "message": "You need to start the LLM first."}

    initialize_static_metrics()

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
    if not request.messages:
        REQUEST_SUCCESS.set(0)
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    initialize_static_metrics()
    start_time = time.time()

    try:
        prompt = request.messages

        if request.stream:
            stream = generate_response(prompt, model_name=request.model, stream=True)
            REQUEST_SUCCESS.set(1)
            return StreamingResponse(stream, media_type="application/x-ndjson")

        generated_text = generate_response(prompt, model_name=request.model)
        num_tokens = compute_token_length(generated_text)
        RESPONSE_TOKENS.set(num_tokens)

        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        RESPONSE_LENGTH.set(len(generated_text))
        REQUEST_SUCCESS.set(1)

        # Dynamic metrics
        LLM_RAM_USAGE_MB.set(psutil.Process().memory_info().rss / 1024 ** 2)
        if duration > 0:
            LLM_TOKENS_PER_SEC.set(num_tokens / duration)

        return ChatResponse(
            id="completion-id",
            object="chat.completion",
            created=time.time(),
            model=request.model,
            choices=[{"message": {"role": "assistant", "content": generated_text}}]
        )
    except Exception as e:
        REQUEST_SUCCESS.set(0)
        raise e