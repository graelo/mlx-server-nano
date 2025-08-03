import time
import threading
from typing import Optional, List, Tuple
from pathlib import Path
import os

from mlx_lm import load, generate
from .schemas import Tool, ToolCall
from .tool_calling import get_tool_parser
from .chat_templates import format_messages_for_model

MODEL_IDLE_TIMEOUT = 300  # seconds
MODEL_CACHE_DIR = Path(os.environ.get("MLX_MODEL_CACHE_DIR", "models"))

_loaded_model = None
_model_name = None
_last_used_time = 0
_unload_timer = None
_lock = threading.Lock()


def get_current_time():
    return time.time()


def unload_model():
    global _loaded_model, _model_name
    print(f"[MODEL] Unloading model '{_model_name}' due to inactivity.")
    _loaded_model = None
    _model_name = None


def _schedule_unload():
    global _unload_timer
    if _unload_timer:
        _unload_timer.cancel()

    def unload_later():
        time.sleep(MODEL_IDLE_TIMEOUT)
        with _lock:
            if get_current_time() - _last_used_time >= MODEL_IDLE_TIMEOUT:
                unload_model()

    _unload_timer = threading.Thread(target=unload_later, daemon=True)
    _unload_timer.start()


def load_model(name: str):
    global _loaded_model, _model_name, _last_used_time

    with _lock:
        _last_used_time = get_current_time()

        if _model_name == name and _loaded_model:
            _schedule_unload()
            return _loaded_model

        print(f"[MODEL] Loading model '{name}'...")
        model_path = MODEL_CACHE_DIR / name

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        try:
            tokenizer, model = load(model_path.as_posix())
        except Exception as e:
            print(f"[MODEL] Failed to load model '{name}': {e}")
            raise RuntimeError(f"Failed to load model '{name}': {e}")

        _loaded_model = (tokenizer, model)
        _model_name = name

        _schedule_unload()
        return _loaded_model


def generate_response_with_tools(
    model_name: str, messages: List, tools: Optional[List[Tool]] = None, **kwargs
) -> Tuple[Optional[str], List[ToolCall]]:
    """Generate response with tool calling support"""

    tokenizer, model = load_model(model_name)

    # Format the prompt with tools
    prompt = format_messages_for_model(messages, model_name, tools)

    # Set up generation parameters
    generation_kwargs = {
        "max_tokens": kwargs.get("max_tokens", 512),
        "temperature": kwargs.get("temperature", 0.7),
    }

    # Add stop strings for Qwen3 models
    if "qwen" in model_name.lower():
        generation_kwargs["stop_strings"] = ["✿RESULT✿:", "✿RETURN✿:", "<|im_end|>"]

    # Generate response
    response = generate(
        prompt=prompt, tokenizer=tokenizer, model=model, **generation_kwargs
    )

    # Parse tool calls from response
    parser = get_tool_parser(model_name)
    content, tool_calls = parser.parse_tool_calls(response)

    return content, tool_calls
