# SPDX-License-Identifier: Apache-2.0
# Integration tests for parser integration with non-streaming generation pipeline
# Tests full request/response cycle with tool calling and reasoning extraction

import json
import pytest
from unittest.mock import MagicMock

from endpoints.OAI.utils.chat_completion import _parse_generation_output
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, Function
from backends.exllamav3.model import ExllamaV3Container
from common.parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from common.parsers.glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser


# ========================================================================
# Test Fixtures
# ========================================================================


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for parser initialization."""
    class MockTokenizer:
        def get_vocab(self):
            return {
                "<think>": 30996,
                "</think>": 30997,
                "<|assistant|>": 151336,
                "[gMASK]": 151329,
                "<sop>": 151336,
                "<|system|>": 151331,
                "<|user|>": 151333,
            }
    return MockTokenizer()


@pytest.fixture
def mock_container_with_parsers(mock_tokenizer):
    """Create mock container with both parsers initialized."""
    container = MagicMock(spec=ExllamaV3Container)
    container.tool_parser = Glm4MoeModelToolParser(mock_tokenizer)
    container.reasoning_parser = Glm4MoeModelReasoningParser(mock_tokenizer)
    return container


@pytest.fixture
def mock_container_without_parsers():
    """Create mock container without parsers for backward compatibility tests."""
    container = MagicMock(spec=ExllamaV3Container)
    container.tool_parser = None
    container.reasoning_parser = None
    return container


# ========================================================================
# Integration Tests
# ========================================================================


def test_tool_calling_integration(mock_container_with_parsers):
    """Test tool calling through full generation pipeline."""
    generation = {
        "text": "<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
        "prompt_tokens": 50,
        "gen_tokens": 20
    }

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get current weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                )
            )
        ]
    )

    # Parse generation output
    parsed = _parse_generation_output(generation, request, mock_container_with_parsers)

    # Verify tool calls extracted correctly
    assert parsed["tool_calls"] is not None
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0].function.name == "get_weather"

    # Verify arguments are valid JSON with correct content
    args = json.loads(parsed["tool_calls"][0].function.arguments)
    assert args["location"] == "Paris"

    # Verify content is empty (all consumed by tool call)
    assert parsed["content"] == "" or parsed["content"] is None


def test_reasoning_extraction_integration(mock_container_with_parsers):
    """Test reasoning extraction through generation pipeline."""
    generation = {
        "text": "<think>Let me analyze the user's request step by step.</think>The answer is 42.",
        "prompt_tokens": 30,
        "gen_tokens": 15
    }

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air"
    )

    # Parse generation output
    parsed = _parse_generation_output(generation, request, mock_container_with_parsers)

    # Verify reasoning extracted
    assert parsed["reasoning_content"] == "Let me analyze the user's request step by step."
    assert parsed["content"] == "The answer is 42."

    # Verify no tool calls (none in output)
    assert parsed["tool_calls"] is None


def test_combined_tool_and_reasoning(mock_container_with_parsers):
    """Test combined tool calling and reasoning extraction."""
    generation = {
        "text": "<think>User needs weather information for Tokyo.</think>Let me check that for you.<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>",
        "prompt_tokens": 40,
        "gen_tokens": 25
    }

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather info",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                )
            )
        ]
    )

    # Parse generation output
    parsed = _parse_generation_output(generation, request, mock_container_with_parsers)

    # Verify both reasoning and tool calls extracted
    assert parsed["reasoning_content"] == "User needs weather information for Tokyo."
    assert parsed["tool_calls"] is not None
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0].function.name == "get_weather"

    # Verify content (non-tool-call text)
    assert "Let me check that for you." in parsed["content"]

    # Verify tool arguments
    args = json.loads(parsed["tool_calls"][0].function.arguments)
    assert args["location"] == "Tokyo"


def test_parser_error_fallback(mock_container_with_parsers):
    """Test graceful fallback when parser encounters errors."""
    # Create container with broken reasoning parser
    container = MagicMock(spec=ExllamaV3Container)

    def broken_reasoning_parser(*args, **kwargs):
        raise Exception("Simulated parser error")

    container.reasoning_parser = MagicMock()
    container.reasoning_parser.extract_reasoning_content = broken_reasoning_parser
    container.tool_parser = None

    generation = {
        "text": "Hello world, this is a normal response.",
        "prompt_tokens": 10,
        "gen_tokens": 5
    }

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air"
    )

    # Should not crash - should fallback to raw text
    parsed = _parse_generation_output(generation, request, container)

    # Verify fallback behavior
    assert parsed["content"] == "Hello world, this is a normal response."
    assert parsed["reasoning_content"] is None
    assert parsed["tool_calls"] is None


def test_response_structure_matches_openai(mock_container_with_parsers):
    """Verify parsed output structure matches OpenAI specification."""
    generation = {
        "text": "<tool_call>test_function\n<arg_key>arg1</arg_key>\n<arg_value>value1</arg_value>\n<arg_key>arg2</arg_key>\n<arg_value>value2</arg_value>\n</tool_call>",
        "prompt_tokens": 20,
        "gen_tokens": 10
    }

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="test_function",
                    description="Test function",
                    parameters={
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string"},
                            "arg2": {"type": "string"}
                        }
                    }
                )
            )
        ]
    )

    parsed = _parse_generation_output(generation, request, mock_container_with_parsers)

    # Verify parsed data structure has required fields
    assert "content" in parsed
    assert "reasoning_content" in parsed
    assert "tool_calls" in parsed

    # Verify tool call structure matches OpenAI spec
    if parsed["tool_calls"]:
        tool_call = parsed["tool_calls"][0]
        assert hasattr(tool_call, "id")
        assert hasattr(tool_call, "type")
        assert hasattr(tool_call, "function")
        assert tool_call.type == "function"
        assert hasattr(tool_call.function, "name")
        assert hasattr(tool_call.function, "arguments")

        # Verify arguments are valid JSON
        args = json.loads(tool_call.function.arguments)
        assert isinstance(args, dict)
        assert "arg1" in args
        assert "arg2" in args
        assert args["arg1"] == "value1"
        assert args["arg2"] == "value2"


def test_without_parsers_backward_compatibility(mock_container_without_parsers):
    """Test backward compatibility when parsers are not available."""
    generation = {
        "text": "Hello world, this is a standard response without any special formatting.",
        "prompt_tokens": 10,
        "gen_tokens": 5
    }

    request = ChatCompletionRequest(
        messages=[],
        model="standard-model"
    )

    # Should work without parsers - no errors
    parsed = _parse_generation_output(generation, request, mock_container_without_parsers)

    # Verify basic functionality preserved
    assert parsed["content"] == "Hello world, this is a standard response without any special formatting."
    assert parsed["reasoning_content"] is None
    assert parsed["tool_calls"] is None
