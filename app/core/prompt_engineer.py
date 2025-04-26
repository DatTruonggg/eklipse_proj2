# =====================
# app/core/prompt_engineer.py
# =====================

from typing import List

def format_prompt(messages: List, model_name: str) -> str:
    """
    Format the user messages into a better prompt suitable for different model behaviors.

    Args:
        messages (List): List of ChatMessage (Pydantic model).
        model_name (str): Model name to adapt prompting style.

    Returns:
        str: Formatted input text to send into the model.
    """

    # Extract user message content
    user_prompt = messages[0].content.strip() if messages else ""

    if not user_prompt:
        raise ValueError("Prompt content cannot be empty.")

    # For BLOOM-like models
    if "bloom" in model_name.lower():
        formatted = f"You are a helpful assistant. Please answer the following question clearly:\n\n{user_prompt}\n\nAssistant:"
    else:
        # For other instruction-following models
        formatted = f"<|User|>\n{user_prompt}\n<|Assistant|>\n"

    return formatted
