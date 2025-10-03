# SPDX-License-Identifier: Apache-2.0
# Unit tests for parser manager infrastructure
# Tests registration and retrieval mechanisms for
# ToolParserManager and ReasoningParserManager

import pytest

from common.parsers.abstract_reasoning_parser import (
    ReasoningParser,
    ReasoningParserManager,
)
from common.parsers.abstract_tool_parser import ToolParser, ToolParserManager
from endpoints.OAI.types.tools import ExtractedToolCallInformation


class MockTokenizer:
    """Mock tokenizer for testing parser instantiation."""

    def get_vocab(self):
        return {}


def test_tool_parser_manager_registration():
    """Test ToolParserManager registration mechanism."""

    @ToolParserManager.register_module("test_tool_parser")
    class TestToolParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Verify parser registered
    assert "test_tool_parser" in ToolParserManager.tool_parsers
    retrieved = ToolParserManager.get_tool_parser("test_tool_parser")
    assert retrieved == TestToolParser


def test_reasoning_parser_manager_registration():
    """Test ReasoningParserManager registration mechanism."""

    @ReasoningParserManager.register_module("test_reasoning_parser")
    class TestReasoningParser(ReasoningParser):
        def is_reasoning_end(self, input_ids):
            return False

        def extract_content_ids(self, input_ids):
            return input_ids

        def extract_reasoning_content(self, model_output, request):
            return (None, model_output)

        def extract_reasoning_content_streaming(self, *args, **kwargs):
            return None

    # Verify parser registered
    assert "test_reasoning_parser" in ReasoningParserManager.reasoning_parsers
    retrieved = ReasoningParserManager.get_reasoning_parser("test_reasoning_parser")
    assert retrieved == TestReasoningParser


def test_parser_retrieval_by_name():
    """Test retrieving parsers by name."""

    # Register test parser
    @ToolParserManager.register_module("retrieval_test")
    class RetrievalTestParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Test retrieval
    parser_class = ToolParserManager.get_tool_parser("retrieval_test")
    assert parser_class == RetrievalTestParser

    # Test instantiation
    instance = parser_class(MockTokenizer())
    assert isinstance(instance, ToolParser)
    assert hasattr(instance, "extract_tool_calls")
    assert hasattr(instance, "extract_tool_calls_streaming")


def test_decorator_pattern():
    """Test decorator registration pattern without explicit name."""

    # Test decorator without explicit name (uses class name)
    @ToolParserManager.register_module()
    class DecoratorTestParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Should use class name as registration name
    assert "DecoratorTestParser" in ToolParserManager.tool_parsers
    retrieved = ToolParserManager.get_tool_parser("DecoratorTestParser")
    assert retrieved == DecoratorTestParser


def test_error_on_unknown_parser():
    """Test error when requesting unknown parser."""

    with pytest.raises(KeyError) as exc_info:
        ToolParserManager.get_tool_parser("nonexistent_parser")
    assert "not found in tool_parsers" in str(exc_info.value)

    with pytest.raises(KeyError) as exc_info:
        ReasoningParserManager.get_reasoning_parser("nonexistent_parser")
    assert "not found in reasoning_parsers" in str(exc_info.value)


def test_force_override_registration():
    """Test force override of existing parser registration."""

    @ToolParserManager.register_module("override_test", force=True)
    class FirstParser(ToolParser):
        marker = "first"

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Verify first parser registered
    first_retrieved = ToolParserManager.get_tool_parser("override_test")
    assert first_retrieved == FirstParser
    assert first_retrieved.marker == "first"

    # Override with force=True
    @ToolParserManager.register_module("override_test", force=True)
    class SecondParser(ToolParser):
        marker = "second"

        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Should get the second parser
    second_retrieved = ToolParserManager.get_tool_parser("override_test")
    assert second_retrieved == SecondParser
    assert second_retrieved.marker == "second"
    assert second_retrieved != FirstParser


def test_no_force_override_error():
    """Test error when trying to override without force=True."""

    # Register first parser
    @ToolParserManager.register_module("no_force_test", force=True)
    class FirstParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Try to override with force=False should raise KeyError
    with pytest.raises(KeyError) as exc_info:

        @ToolParserManager.register_module("no_force_test", force=False)
        class SecondParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=[], content=model_output
                )

            def extract_tool_calls_streaming(self, *args, **kwargs):
                return None

    assert "already registered" in str(exc_info.value)


def test_glm45_parsers_registered():
    """Test that GLM-4.5 parsers are registered."""

    # Check that glm45 tool parser is registered
    assert "glm45" in ToolParserManager.tool_parsers
    tool_parser_class = ToolParserManager.get_tool_parser("glm45")
    assert tool_parser_class is not None

    # Check that glm45 reasoning parser is registered
    assert "glm45" in ReasoningParserManager.reasoning_parsers
    reasoning_parser_class = ReasoningParserManager.get_reasoning_parser("glm45")
    assert reasoning_parser_class is not None

    # Test instantiation
    tokenizer = MockTokenizer()
    tool_parser = tool_parser_class(tokenizer)
    reasoning_parser = reasoning_parser_class(tokenizer)

    assert isinstance(tool_parser, ToolParser)
    assert isinstance(reasoning_parser, ReasoningParser)


def test_multiple_name_registration():
    """Test registering a parser with multiple names."""

    @ToolParserManager.register_module(["multi_name_1", "multi_name_2"])
    class MultiNameParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        def extract_tool_calls_streaming(self, *args, **kwargs):
            return None

    # Both names should retrieve the same class
    parser1 = ToolParserManager.get_tool_parser("multi_name_1")
    parser2 = ToolParserManager.get_tool_parser("multi_name_2")

    assert parser1 == MultiNameParser
    assert parser2 == MultiNameParser
    assert parser1 == parser2


def test_reasoning_parser_multiple_names():
    """Test reasoning parser registration with multiple names."""

    @ReasoningParserManager.register_module(["reasoning_a", "reasoning_b"])
    class MultiNameReasoningParser(ReasoningParser):
        def is_reasoning_end(self, input_ids):
            return False

        def extract_content_ids(self, input_ids):
            return input_ids

        def extract_reasoning_content(self, model_output, request):
            return (None, model_output)

        def extract_reasoning_content_streaming(self, *args, **kwargs):
            return None

    # Both names should work
    parser_a = ReasoningParserManager.get_reasoning_parser("reasoning_a")
    parser_b = ReasoningParserManager.get_reasoning_parser("reasoning_b")

    assert parser_a == MultiNameReasoningParser
    assert parser_b == MultiNameReasoningParser
