# =====================
# app/core/generate_response.py
# =====================

from threading import Thread
from typing import Generator, List, Union
from transformers import TextIteratorStreamer

from app.core.model_loader import ml_models
from app.core.prompt_engineer import format_prompt
from app.configs import config
import time

def stream_response(streamer, model_name: str):
    """
    Wraps streamer output into JSON chunks for streaming.
    """
    for token in streamer:
        chunk = {
            "id": "completion-id",
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": model_name,
            "choices": [
                {
                    "delta": {"content": token},
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {chunk}\n\n"
    # End of stream
    yield "data: [DONE]\n\n"


def generate_response(
    prompt: List,
    model_name: str = config.model_name,
    stream: bool = config.stream
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a response for the given prompt using the specified model.

    Args:
        prompt (List): List of chat messages.
        model_name (str): Name of the model to use.
        stream (bool): If True, stream tokens as they are generated.

    Returns:
        str or Generator[str]: Generated response text or streaming tokens.
    """
    model = ml_models.get_model(model_name)
    tokenizer = ml_models.get_tokenizer(model_name)

    # Use Prompt Engineer to format the input nicely
    input_text = format_prompt(prompt, model_name)

    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": config.max_token,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "do_sample": config.do_sample,
    }

    if not stream:
        outputs = model.generate(**generation_kwargs)
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=config.skip_special_tokens
        )
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:", 1)[1].strip()

        return generated_text
    else:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=config.skip_special_tokens
        )
        generation_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        return stream_response(streamer, model_name)