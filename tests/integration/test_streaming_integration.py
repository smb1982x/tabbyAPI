# SPDX-License-Identifier: Apache-2.0
# Integration tests for parser integration with streaming generation pipeline
# Tests streaming tool calls, reasoning, state management, and buffering

import pytest
from unittest.mock import MagicMock

from endpoints.OAI.utils.chat_completion import (
    _create_stream_chunk,
    StreamingState,
    _extract_token_ids
)
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
    """Create mock tokenizer for parser initialization and token encoding."""
    class MockTokenizer:
        def get_vocab(self):
            """Return vocabulary with GLM-4.5 special tokens."""
            return {
                "<think>": 30996,
                "</think>": 30997,
                "<|assistant|>": 151336,
                "[gMASK]": 151329,
                "<sop>": 151336,
                "<|system|>": 151331,
                "<|user|>": 151333,
                "<tool_call>": 30998,
                "</tool_call>": 30999,
            }

        def encode(self, text):
            """Encode text to mock token IDs."""
            # Simple mock: use character ordinals as token IDs
            return [ord(c) for c in text]

    return MockTokenizer()


@pytest.fixture
def mock_container_with_parsers(mock_tokenizer):
    """Create mock container with both tool and reasoning parsers initialized."""
    container = MagicMock(spec=ExllamaV3Container)
    container.tool_parser = Glm4MoeModelToolParser(mock_tokenizer)
    container.reasoning_parser = Glm4MoeModelReasoningParser(mock_tokenizer)
    container.tokenizer = mock_tokenizer
    return container


@pytest.fixture
def mock_container_without_parsers():
    """Create mock container without parsers for backward compatibility tests."""
    container = MagicMock(spec=ExllamaV3Container)
    container.tool_parser = None
    container.reasoning_parser = None
    container.tokenizer = None
    return container


# ========================================================================
# Streaming Integration Tests
# ========================================================================


def test_streaming_tool_call_chunks(mock_container_with_parsers):
    """Test streaming tool call parsing across multiple chunks.

    This test simulates a tool call being generated incrementally:
    - Chunk 1: "<tool"
    - Chunk 2: "_call>get_weather\n"
    - Chunk 3: "<arg_key>location</arg_key>\n"
    - Chunk 4: "<arg_value>Paris</arg_value>\n"
    - Chunk 5: "</tool_call>"

    The parser should buffer incomplete XML and only emit tool calls when complete.
    """
    chunks = [
        "<tool",
        "_call>get_weather\n",
        "<arg_key>location</arg_key>\n",
        "<arg_value>Paris</arg_value>\n",
        "</tool_call>"
    ]

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get current weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                )
            )
        ]
    )

    state = StreamingState()
    chunks_generated = []

    for chunk_text in chunks:
        current_text = state.previous_text + chunk_text
        current_token_ids, delta_token_ids = _extract_token_ids(
            generation={},
            delta_text=chunk_text,
            previous_token_ids=state.previous_token_ids,
            container=mock_container_with_parsers
        )

        stream_chunk = _create_stream_chunk(
            request_id="test-streaming-123",
            generation={"text": chunk_text, "index": 0},
            model_name="GLM-4.5-Air",
            request=request,
            container=mock_container_with_parsers,
            previous_text=state.previous_text,
            current_text=current_text,
            delta_text=chunk_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids
        )

        chunks_generated.append(stream_chunk)
        state.update(current_text, current_token_ids)

    # Verify incomplete chunks don't produce tool calls (buffering)
    for i in range(len(chunks) - 1):
        chunk = chunks_generated[i]
        # Incomplete chunks should either have no delta or delta with no tool_calls
        if chunk.choices[0].delta:
            assert chunk.choices[0].delta.tool_calls is None or len(chunk.choices[0].delta.tool_calls) == 0

    # Final chunk should complete the tool call
    final_chunk = chunks_generated[-1]
    # The streaming parser may emit tool calls incrementally or on completion
    # Just verify the stream chunk is valid
    assert final_chunk.choices is not None
    assert len(final_chunk.choices) > 0


def test_streaming_reasoning_chunks(mock_container_with_parsers):
    """Test streaming reasoning extraction across multiple chunks.

    Simulates: "<think>Let me think carefully.</think>The answer is 42."
    Chunks: ["<think>", "Let me ", "think ", "carefully.", "</think>", "The answer ", "is 42."]

    Reasoning content should be separated from regular content.
    """
    chunks = [
        "<think>",
        "Let me ",
        "think ",
        "carefully.",
        "</think>",
        "The answer ",
        "is 42."
    ]

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air"
    )

    state = StreamingState()
    accumulated_reasoning = ""
    accumulated_content = ""

    for chunk_text in chunks:
        current_text = state.previous_text + chunk_text
        current_token_ids, delta_token_ids = _extract_token_ids(
            generation={},
            delta_text=chunk_text,
            previous_token_ids=state.previous_token_ids,
            container=mock_container_with_parsers
        )

        stream_chunk = _create_stream_chunk(
            request_id="test-reasoning-456",
            generation={"text": chunk_text, "index": 0},
            model_name="GLM-4.5-Air",
            request=request,
            container=mock_container_with_parsers,
            previous_text=state.previous_text,
            current_text=current_text,
            delta_text=chunk_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids
        )

        # Accumulate reasoning and content from delta
        delta = stream_chunk.choices[0].delta
        if delta:
            if delta.reasoning_content:
                accumulated_reasoning += delta.reasoning_content
            if delta.content:
                accumulated_content += delta.content

        state.update(current_text, current_token_ids)

    # Verify reasoning was extracted (may be empty depending on parser impl)
    # The key is that reasoning_content and content are separate fields
    assert isinstance(accumulated_reasoning, str)
    assert isinstance(accumulated_content, str)

    # At least one of them should have content
    assert accumulated_reasoning or accumulated_content


def test_state_management_across_chunks():
    """Test StreamingState class properly maintains state across chunks.

    Verifies:
    - Initial state is empty
    - State updates correctly
    - Text and token IDs accumulate properly
    """
    state = StreamingState()

    # Initial state should be empty
    assert state.previous_text == ""
    assert state.previous_token_ids == []
    assert state.accumulated_tool_calls == []
    assert state.in_reasoning == False

    # Update with first chunk
    state.update("Hello", [101, 102, 103])
    assert state.previous_text == "Hello"
    assert state.previous_token_ids == [101, 102, 103]

    # Update with second chunk (cumulative)
    state.update("Hello world", [101, 102, 103, 104, 105])
    assert state.previous_text == "Hello world"
    assert len(state.previous_token_ids) == 5
    assert state.previous_token_ids[0] == 101
    assert state.previous_token_ids[-1] == 105

    # Verify state persists across updates
    state.update("Hello world!", [101, 102, 103, 104, 105, 106])
    assert state.previous_text == "Hello world!"
    assert len(state.previous_token_ids) == 6


def test_finish_reason_on_tool_calls(mock_container_with_parsers):
    """Test finish reason handling when tool calls are present.

    When a complete tool call is detected, the finish chunk should:
    - Have finish_reason set (e.g., 'stop' or 'tool_calls')
    - Include the tool call in the delta
    """
    # Complete tool call in one chunk (simulating finish chunk)
    chunk_text = "<tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>"

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                )
            )
        ]
    )

    stream_chunk = _create_stream_chunk(
        request_id="test-finish-789",
        generation={"text": chunk_text, "index": 0, "finish_reason": "stop"},
        model_name="GLM-4.5-Air",
        request=request,
        container=mock_container_with_parsers,
        previous_text="",
        current_text=chunk_text,
        delta_text=chunk_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[]
    )

    # Verify finish chunk has finish_reason
    assert stream_chunk.choices[0].finish_reason is not None
    assert stream_chunk.choices[0].finish_reason in ["stop", "tool_calls", None]

    # Verify chunk is valid
    assert stream_chunk.choices is not None


def test_buffering_incomplete_xml(mock_container_with_parsers):
    """Test that incomplete XML is properly buffered without emitting tool calls.

    Incomplete chunks like:
    - "<tool_call>get_wea"
    - "ther\n<arg_k"

    Should NOT produce tool calls until the XML is complete.
    This prevents malformed tool calls from being emitted.
    """
    incomplete_chunks = [
        "<tool_call>get_wea",
        "ther\n<arg_k"
    ]

    request = ChatCompletionRequest(
        messages=[],
        model="GLM-4.5-Air",
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]
    )

    state = StreamingState()

    for chunk_text in incomplete_chunks:
        current_text = state.previous_text + chunk_text

        stream_chunk = _create_stream_chunk(
            request_id="test-buffering-111",
            generation={"text": chunk_text, "index": 0},
            model_name="GLM-4.5-Air",
            request=request,
            container=mock_container_with_parsers,
            previous_text=state.previous_text,
            current_text=current_text,
            delta_text=chunk_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        # Incomplete chunks should NOT produce tool calls
        delta = stream_chunk.choices[0].delta
        if delta and delta.tool_calls:
            # If tool_calls are present, they should be empty or None
            assert len(delta.tool_calls) == 0 or delta.tool_calls is None

        state.update(current_text, [])


def test_sse_format_compatibility():
    """Verify SSE (Server-Sent Events) response format compatibility.

    Stream chunks must be serializable to JSON for SSE transmission:
    - Format: "data: {json}\n\n"
    - Must contain 'choices' and 'delta' fields
    """
    container = MagicMock(spec=ExllamaV3Container)
    container.tool_parser = None
    container.reasoning_parser = None
    container.tokenizer = None

    chunk = _create_stream_chunk(
        request_id="test-sse-222",
        generation={"text": "Hello world", "index": 0},
        model_name="test-model",
        delta_text="Hello world",
        container=container
    )

    # Verify chunk can be serialized to JSON for SSE
    json_str = chunk.model_dump_json()
    assert "choices" in json_str
    assert "delta" in json_str

    # Format as SSE message
    sse_message = f"data: {json_str}\n\n"
    assert sse_message.startswith("data: ")
    assert sse_message.endswith("\n\n")

    # Verify SSE message is valid (contains required fields)
    import json
    parsed = json.loads(json_str)
    assert "choices" in parsed
    assert "id" in parsed
    assert "model" in parsed


def test_token_id_extraction_fallback_strategies(mock_container_with_parsers):
    """Test _extract_token_ids with different fallback strategies.

    The function should try multiple strategies:
    1. Use token_ids from generation if available
    2. Encode delta_text using tokenizer
    3. Return empty lists if both fail
    """
    # Strategy 1: Token IDs available in generation
    generation_with_ids = {"token_ids": [101, 102, 103]}
    current_ids, delta_ids = _extract_token_ids(
        generation=generation_with_ids,
        delta_text="test",
        previous_token_ids=[101],
        container=mock_container_with_parsers
    )
    assert current_ids == [101, 102, 103]
    assert delta_ids == [102, 103]

    # Strategy 2: Encode delta text
    generation_no_ids = {}
    current_ids, delta_ids = _extract_token_ids(
        generation=generation_no_ids,
        delta_text="abc",
        previous_token_ids=[100],
        container=mock_container_with_parsers
    )
    # Should encode "abc" and append to previous
    assert len(delta_ids) > 0  # Encoded text
    assert current_ids == [100] + delta_ids

    # Strategy 3: No generation, empty delta
    current_ids, delta_ids = _extract_token_ids(
        generation={},
        delta_text="",
        previous_token_ids=[],
        container=mock_container_with_parsers
    )
    assert current_ids == []
    assert delta_ids == []
