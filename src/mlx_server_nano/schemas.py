"""
Pydantic schemas for OpenAI-compatible API requests and responses

Defines the data models used for chat completion requests, tool calling,
and response formatting. All schemas are designed to be compatible with
the OpenAI API specification.

Models:
- Message: Individual conversation messages with role and content
- Tool/Function: Tool definition and calling structures
- ChatCompletionRequest: Main request schema for chat completions
- ChatCompletionResponse: Response schema for chat completions
"""

from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Optional, Union


class Message(BaseModel):
    """Individual message in a conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = (
        None  # Always check for None before using
    )
    tool_call_id: Optional[str] = None


class Function(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    """Tool definition with function specification."""

    type: str = "function"
    function: Function


class ToolChoice(BaseModel):
    """Tool choice specification for requests."""

    type: str
    function: Optional[Dict[str, str]] = None


class ChatCompletionRequest(BaseModel):
    """Request schema for chat completion endpoint."""

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Tool]] = None  # Always check for None before using
    tool_choice: Optional[Union[str, ToolChoice]] = "auto"
    # Additional OpenAI-compatible parameters
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(
        cls, v: Optional[Union[str, List[str]]]
    ) -> Optional[Union[str, List[str]]]:
        if v is not None:
            if isinstance(v, list) and len(v) > 4:
                raise ValueError("stop can contain at most 4 sequences")
        return v


class ToolCall(BaseModel):
    """Tool call result from model response."""

    id: str
    type: str = "function"
    function: Dict[str, Any]

    @property
    def name(self) -> str:
        """Get the function name for backward compatibility."""
        return self.function.get("name", "")

    @property
    def arguments(self) -> Union[str, Dict[str, Any]]:
        """Get the function arguments for backward compatibility."""
        return self.function.get("arguments", {})


class ChatCompletionResponse(BaseModel):
    """Response schema for chat completion endpoint."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


def tool_to_openai_format(tool: Tool) -> Dict[str, Any]:
    """
    Convert our Tool schema to OpenAI format for MLX-LM.

    Args:
        tool: Our internal Tool schema

    Returns:
        Dictionary in OpenAI tool format for MLX-LM
    """
    return {
        "type": "function",
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
        },
    }


def tools_to_openai_format(
    tools: Optional[List[Tool]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert a list of tools to OpenAI format for MLX-LM.

    Args:
        tools: Optional list of our internal Tool schemas

    Returns:
        Optional list of dictionaries in OpenAI tool format, or None if no tools
    """
    if not tools:
        return None
    return [tool_to_openai_format(tool) for tool in tools]
