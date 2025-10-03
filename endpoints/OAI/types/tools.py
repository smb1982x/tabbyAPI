from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
from uuid import uuid4


class Function(BaseModel):
    """Represents a description of a tool function."""

    name: str
    description: str
    parameters: Dict[str, object]


class ToolSpec(BaseModel):
    """Wrapper for an inner tool function."""

    function: Function
    type: Literal["function"]


class Tool(BaseModel):
    """Represents an OAI tool description."""

    name: str

    # Makes more sense to be a dict, but OAI knows best
    arguments: str


class ToolCall(BaseModel):
    """Represents an OAI tool description."""

    id: str = Field(default_factory=lambda: str(uuid4()).replace("-", "")[:9])
    function: Tool
    type: Literal["function"] = "function"


class ExtractedToolCallInformation(BaseModel):
    """Information extracted from tool call parsing.

    This class represents the result of parsing tool calls from model output,
    including whether tools were called, the parsed tool calls themselves, and
    any non-tool-call content.

    Attributes:
        tools_called: Whether any tools were detected in the output
        tool_calls: List of parsed tool calls
        content: Non-tool-call content from the response
    """

    tools_called: bool = Field(description="Whether any tools were called")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="List of parsed tool calls"
    )
    content: Optional[str] = Field(
        default=None, description="Non-tool-call content from the response"
    )
