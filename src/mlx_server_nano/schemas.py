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

from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union


class Message(BaseModel):
    """Individual message in a conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
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
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = "auto"


class ToolCall(BaseModel):
    """Tool call result from model response."""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionResponse(BaseModel):
    """Response schema for chat completion endpoint."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
