# =====================
# app/main.py
# =====================

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import chat, embedding
from app.core.model_loader import load_model, load_embedder, ml_models, embed_models
from app.configs import config

from huggingface_hub import login
from prometheus_fastapi_instrumentator import Instrumentator
from app.logs import log
# =====================
# App Lifespan: Startup & Shutdown Hooks
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI app.
    """
    # Startup
    log.info("[INFO] Starting LLM service...")

    # 1. Login to HuggingFace Hub
    if config.huggingface_key and config.huggingface_key != "your_huggingface_token_here": # in config.yaml file
        login(token=config.huggingface_key)
        log.info("[INFO] Huggingface login successful.")

    # 2. Load main LLM model
    load_model(config.model_name)

    # 3. Load embedding model
    load_embedder(config.embedding_model)

    log.info("[INFO] Models loaded successfully.")
    yield

    # Shutdown
    ml_models.clear()
    embed_models.clear()
    log.info("[INFO] All models cleared. Service shutdown complete.")

# =====================
# Initialize FastAPI App
# =====================

app = FastAPI(
    title="Eklipse LLM Serving API",
    description="API server for chat completions and embeddings generation.",
    version="1.0.0",
    lifespan=lifespan
)

# =====================
# Include Routers
# =====================
Instrumentator().instrument(app).expose(app)

app.include_router(chat.router)
app.include_router(embedding.router)

# =====================
# Basic Root Endpoint
# =====================

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Eklipse LLM API server"}
