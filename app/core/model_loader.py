from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from app.configs import config
from app.logs import log

from ..core.model_registry import ModelRegistry

# =====================
# Initialize Registries
# =====================
ml_models = ModelRegistry()
embed_models = ModelRegistry()

# =====================
# Load Language Model (LLM)
# =====================

def load_model(model_name: str = config.model_name):
    """
    Load a language generation model and its tokenizer into memory.

    Args:
        model_name (str): Huggingface repo name.

    Returns:
        None
    """
    print(f"[INFO] Loading LLM model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    ml_models.register(model_name, model, tokenizer)

    log.info(f"[INFO] LLM model '{model_name}' loaded successfully.")

# =====================
# Load Embedding Model
# =====================

def load_embedder(embed_model_name: str = config.embedding_model):
    """
    Load an embedding model and its tokenizer into memory.

    Args:
        embed_model_name (str): Huggingface repo name.

    Returns:
        None
    """
    log.info(f"[INFO] Loading Embedding model: {embed_model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(embed_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        embed_model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    embed_models.register(embed_model_name, model, tokenizer)

    log.info(f"[INFO] Embedding model '{embed_model_name}' loaded successfully.")