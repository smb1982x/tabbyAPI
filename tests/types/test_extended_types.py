# SPDX-License-Identifier: Apache-2.0
# Unit tests for extended type definitions
# Tests ChatCompletionMessage with reasoning_content and ExtractedToolCallInformation

import json

import pytest
from pydantic import ValidationError

from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.OAI.types.tools import ExtractedToolCallInformation, Function, ToolCall

# ========================================================================
# ChatCompletionMessage Tests
# ========================================================================


def test_chat_completion_message_with_reasoning():
    """Test ChatCompletionMessage with reasoning_content field."""
    message = ChatCompletionMessage(
        role="assistant",
        content="The answer is 42.",
        reasoning_content="Let me think about this carefully step by step.",
    )

    assert message.role == "assistant"
    assert message.content == "The answer is 42."
    assert (
        message.reasoning_content == "Let me think about this carefully step by step."
    )


def test_chat_completion_message_reasoning_optional():
    """Test that reasoning_content is optional (backwards compatibility)."""
    # Without reasoning_content - should work
    message = ChatCompletionMessage(role="assistant", content="Hello world")

    assert message.role == "assistant"
    assert message.content == "Hello world"
    assert message.reasoning_content is None


def test_chat_completion_message_serialization():
    """Test Pydantic serialization with reasoning_content."""
    message = ChatCompletionMessage(
        role="assistant",
        content="The final answer",
        reasoning_content="Analyzing the problem deeply",
    )

    # Serialize to dict
    message_dict = message.model_dump()
    assert "reasoning_content" in message_dict
    assert message_dict["reasoning_content"] == "Analyzing the problem deeply"
    assert message_dict["content"] == "The final answer"

    # Serialize to JSON
    message_json = message.model_dump_json()
    parsed = json.loads(message_json)
    assert parsed["reasoning_content"] == "Analyzing the problem deeply"
    assert parsed["role"] == "assistant"


def test_chat_completion_message_deserialization():
    """Test Pydantic deserialization with reasoning_content."""
    data = {
        "role": "assistant",
        "content": "Answer text",
        "reasoning_content": "Thinking process",
    }

    message = ChatCompletionMessage(**data)
    assert message.reasoning_content == "Thinking process"
    assert message.content == "Answer text"


def test_chat_completion_message_with_tool_calls():
    """Test ChatCompletionMessage with both reasoning and tool calls."""
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=Function(name="test_func", arguments="{}"),
    )

    message = ChatCompletionMessage(
        role="assistant",
        content="Let me check that.",
        reasoning_content="User needs function call",
        tool_calls=[tool_call],
    )

    assert message.reasoning_content == "User needs function call"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "test_func"


# ========================================================================
# ExtractedToolCallInformation Tests
# ========================================================================


def test_extracted_tool_call_information_validation():
    """Test ExtractedToolCallInformation validation with tool calls."""
    tool_call = ToolCall(
        id="call_456",
        type="function",
        function=Function(name="get_weather", arguments='{"location": "Paris"}'),
    )

    info = ExtractedToolCallInformation(
        tools_called=True, tool_calls=[tool_call], content="Some remaining text"
    )

    assert info.tools_called is True
    assert len(info.tool_calls) == 1
    assert info.tool_calls[0].function.name == "get_weather"
    assert info.content == "Some remaining text"


def test_extracted_tool_call_information_defaults():
    """Test ExtractedToolCallInformation default values."""
    # Only required field (tools_called)
    info = ExtractedToolCallInformation(tools_called=False)

    assert info.tools_called is False
    assert info.tool_calls == []  # Default empty list
    assert info.content is None  # Default None


def test_extracted_tool_call_information_no_tools():
    """Test ExtractedToolCallInformation when no tools are called."""
    info = ExtractedToolCallInformation(
        tools_called=False,
        tool_calls=[],
        content="Regular response without any tool calls",
    )

    assert info.tools_called is False
    assert len(info.tool_calls) == 0
    assert info.content == "Regular response without any tool calls"


def test_extracted_tool_call_information_serialization():
    """Test Pydantic serialization of ExtractedToolCallInformation."""
    tool_call = ToolCall(
        id="call_789",
        type="function",
        function=Function(name="calculate", arguments='{"x": 10, "y": 20}'),
    )

    info = ExtractedToolCallInformation(
        tools_called=True, tool_calls=[tool_call], content="Calculating..."
    )

    # Serialize to dict
    info_dict = info.model_dump()
    assert info_dict["tools_called"] is True
    assert len(info_dict["tool_calls"]) == 1
    assert info_dict["tool_calls"][0]["function"]["name"] == "calculate"

    # Serialize to JSON
    info_json = info.model_dump_json()
    parsed = json.loads(info_json)
    assert parsed["tools_called"] is True
    assert parsed["content"] == "Calculating..."


# ========================================================================
# Backwards Compatibility Tests
# ========================================================================


def test_backwards_compatibility():
    """Test backwards compatibility with existing code."""
    # Old code without reasoning_content should still work
    message = ChatCompletionMessage(
        role="user", content="Test message without reasoning"
    )

    # New field should exist but be None
    assert hasattr(message, "reasoning_content")
    assert message.reasoning_content is None

    # Serialization should not break
    message_dict = message.model_dump()
    assert "content" in message_dict
    assert "role" in message_dict

    # Can deserialize old format
    old_data = {"role": "user", "content": "Old format message"}
    old_message = ChatCompletionMessage(**old_data)
    assert old_message.content == "Old format message"
    assert old_message.reasoning_content is None


# ========================================================================
# Validation Error Tests
# ========================================================================


def test_pydantic_validation_errors():
    """Test that Pydantic validation catches errors."""
    # Missing required field (tools_called)
    with pytest.raises(ValidationError):
        ExtractedToolCallInformation()

    # Invalid type for tools_called (should be bool)
    with pytest.raises(ValidationError):
        ExtractedToolCallInformation(tools_called="not a boolean")

    # Invalid type for tool_calls (should be list)
    with pytest.raises(ValidationError):
        ExtractedToolCallInformation(tools_called=True, tool_calls="not a list")


# ========================================================================
# Complex Structure Tests
# ========================================================================


def test_tool_call_nested_structure():
    """Test nested structure of tool calls in ExtractedToolCallInformation."""
    tool_call = ToolCall(
        id="call_complex",
        type="function",
        function=Function(
            name="complex_function",
            arguments='{"nested": {"key": "value", "number": 42}, "array": [1, 2, 3]}',
        ),
    )

    info = ExtractedToolCallInformation(
        tools_called=True, tool_calls=[tool_call], content=None
    )

    # Verify nested structure is preserved in serialization
    args = json.loads(info.tool_calls[0].function.arguments)
    assert args["nested"]["key"] == "value"
    assert args["nested"]["number"] == 42
    assert args["array"] == [1, 2, 3]

    # Verify serialization preserves structure
    info_dict = info.model_dump()
    tool_call_dict = info_dict["tool_calls"][0]
    args_from_dict = json.loads(tool_call_dict["function"]["arguments"])
    assert args_from_dict["nested"]["key"] == "value"


def test_multiple_tool_calls_in_extracted_info():
    """Test multiple tool calls in ExtractedToolCallInformation."""
    tool_calls = [
        ToolCall(
            id=f"call_{i}",
            type="function",
            function=Function(
                name=f"function_{i}", arguments='{"index": ' + str(i) + "}"
            ),
        )
        for i in range(3)
    ]

    info = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=tool_calls,
        content="Multiple function calls detected",
    )

    assert len(info.tool_calls) == 3
    assert all(tc.type == "function" for tc in info.tool_calls)
    assert info.tool_calls[0].function.name == "function_0"
    assert info.tool_calls[1].function.name == "function_1"
    assert info.tool_calls[2].function.name == "function_2"

    # Verify each has unique ID
    ids = [tc.id for tc in info.tool_calls]
    assert len(ids) == len(set(ids))  # All IDs unique


def test_empty_content_serialization():
    """Test serialization with None and empty string content."""
    # None content
    info1 = ExtractedToolCallInformation(tools_called=True, tool_calls=[], content=None)
    dict1 = info1.model_dump()
    assert dict1["content"] is None

    # Empty string content
    info2 = ExtractedToolCallInformation(tools_called=False, tool_calls=[], content="")
    dict2 = info2.model_dump()
    assert dict2["content"] == ""
