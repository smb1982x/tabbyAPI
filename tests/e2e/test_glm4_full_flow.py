"""End-to-end tests for GLM-4.5 model with TabbyAPI server.

These tests validate the complete integration of GLM-4.5 parsers
with the TabbyAPI server through HTTP requests.

Requirements:
- TabbyAPI server running on http://10.1.1.10:8080
- GLM-4.5-Air model loaded
- Server configured with ExllamaV3 backend
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import requests


# ========================================================================
# Configuration and Helpers
# ========================================================================

API_BASE_URL = "http://10.1.1.10:8080/v1"
MODEL_NAME = "GLM-4.5-Air"
REQUEST_TIMEOUT = 120  # 2 minutes for generation


def is_server_ready() -> bool:
    """Check if TabbyAPI server is ready to accept requests."""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_server(max_wait: int = 30) -> bool:
    """Wait for server to become ready.

    Args:
        max_wait: Maximum seconds to wait

    Returns:
        True if server is ready, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if is_server_ready():
            return True
        time.sleep(1)
    return False


def has_glm_model() -> bool:
    """Check if GLM-4.5 model is available."""
    model_paths = [
        Path("/opt/tabbyAPI/models/GLM-4.5-Air"),
    ]
    return any(p.exists() for p in model_paths) or os.getenv("GLM_MODEL_PATH")


@pytest.fixture(scope="module", autouse=True)
def ensure_server_ready():
    """Ensure server is ready before running tests."""
    if not is_server_ready():
        pytest.skip("TabbyAPI server is not running or not ready")


# ========================================================================
# Basic API Tests
# ========================================================================


def test_server_health_check():
    """Test that server is responding to health checks."""
    response = requests.get(f"{API_BASE_URL}/models", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)


def test_model_loaded():
    """Test that GLM-4.5 model is loaded and available."""
    response = requests.get(f"{API_BASE_URL}/models", timeout=5)
    assert response.status_code == 200
    data = response.json()

    # Check if any model is loaded
    models = data.get("data", [])
    assert len(models) > 0, "No models loaded"

    # Verify model info
    model_info = models[0]
    assert "id" in model_info
    print(f"Loaded model: {model_info['id']}")


# ========================================================================
# Tool Calling Tests
# ========================================================================


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_tool_calling_weather_example():
    """Test tool calling with weather function (non-streaming)."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200, f"Request failed: {response.text}"
    data = response.json()

    # Verify response structure
    assert "choices" in data
    assert len(data["choices"]) > 0

    message = data["choices"][0]["message"]
    assert "role" in message
    assert message["role"] == "assistant"

    # Log response for debugging
    print(f"\n=== Tool Calling Response ===")
    print(f"Content: {message.get('content')}")
    print(f"Tool calls: {message.get('tool_calls')}")
    print(f"Reasoning: {message.get('reasoning_content')}")

    # Check for tool calls - model may or may not use them depending on prompt
    if message.get("tool_calls"):
        tool_call = message["tool_calls"][0]
        assert "id" in tool_call
        assert "type" in tool_call
        assert tool_call["type"] == "function"
        assert "function" in tool_call
        assert "name" in tool_call["function"]
        assert "arguments" in tool_call["function"]

        # Verify arguments are valid JSON
        arguments = json.loads(tool_call["function"]["arguments"])
        assert "location" in arguments
        print(f"✓ Tool call detected: {tool_call['function']['name']}({arguments})")
    else:
        print("! Model did not use tool call (may have answered directly)")


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_tool_calling_multiple_functions():
    """Test tool calling with multiple available functions."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Check the weather in Paris and get the current time there"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"}
                        },
                        "required": ["timezone"]
                    }
                }
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]

    print(f"\n=== Multiple Functions Response ===")
    print(f"Content: {message.get('content')}")
    print(f"Tool calls: {message.get('tool_calls')}")

    # Model may call one or both functions
    if message.get("tool_calls"):
        assert isinstance(message["tool_calls"], list)
        print(f"✓ Detected {len(message['tool_calls'])} tool call(s)")

        for tool_call in message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            print(f"  - {function_name}: {arguments}")


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_tool_calling_with_conversation_context():
    """Test tool calling with multi-turn conversation."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "I'm planning a trip to London"},
            {"role": "assistant", "content": "That sounds exciting! How can I help you plan your trip?"},
            {"role": "user", "content": "What's the weather like there?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "max_tokens": 512
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]

    print(f"\n=== Context-Aware Tool Calling ===")
    print(f"Message: {message}")

    # Should understand "there" refers to London from context
    if message.get("tool_calls"):
        arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        print(f"✓ Tool call with context: {arguments}")


# ========================================================================
# Reasoning Extraction Tests
# ========================================================================


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_reasoning_extraction_basic():
    """Test reasoning content extraction (non-streaming)."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Explain step by step: why is the sky blue?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]

    print(f"\n=== Reasoning Extraction ===")
    print(f"Content: {message.get('content')}")
    print(f"Reasoning: {message.get('reasoning_content')}")

    # Reasoning content is optional - model may or may not use <think> tags
    if message.get("reasoning_content"):
        assert isinstance(message["reasoning_content"], str)
        assert len(message["reasoning_content"]) > 0
        print(f"✓ Reasoning content extracted ({len(message['reasoning_content'])} chars)")
    else:
        print("! No reasoning tags detected in response")

    # Content should always be present
    assert message.get("content") is not None


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_reasoning_with_math_problem():
    """Test reasoning extraction with a math problem."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Solve this problem step by step: If a train travels 120km in 2 hours, what's its average speed?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.5
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]

    print(f"\n=== Math Problem Reasoning ===")
    print(f"Reasoning: {message.get('reasoning_content')}")
    print(f"Answer: {message.get('content')}")

    # Verify response structure
    assert "content" in message


# ========================================================================
# Streaming Tests
# ========================================================================


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_streaming_basic():
    """Test basic streaming generation."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        "max_tokens": 256,
        "stream": True
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT,
        stream=True
    )

    assert response.status_code == 200

    chunks_received = []
    full_content = ""

    print(f"\n=== Streaming Response ===")

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode('utf-8')
        if not line_str.startswith("data: "):
            continue

        data_str = line_str[6:]  # Remove "data: " prefix

        if data_str == "[DONE]":
            print("\n✓ Stream completed")
            break

        try:
            chunk_data = json.loads(data_str)
            chunks_received.append(chunk_data)

            # Extract content delta
            if chunk_data.get("choices"):
                delta = chunk_data["choices"][0].get("delta", {})
                if delta.get("content"):
                    full_content += delta["content"]
                    print(delta["content"], end="", flush=True)

        except json.JSONDecodeError as e:
            print(f"\n! JSON decode error: {e}")
            print(f"  Data: {data_str[:100]}")

    assert len(chunks_received) > 0, "No chunks received"
    print(f"\n✓ Received {len(chunks_received)} chunks")
    print(f"✓ Full content: {full_content}")


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_streaming_tool_calls():
    """Test streaming with tool calls."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "What's the weather in Berlin?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "max_tokens": 512,
        "stream": True
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT,
        stream=True
    )

    assert response.status_code == 200

    chunks_with_tool_calls = []
    chunks_with_content = []
    chunks_with_reasoning = []

    print(f"\n=== Streaming Tool Calls ===")

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode('utf-8')
        if not line_str.startswith("data: "):
            continue

        data_str = line_str[6:]
        if data_str == "[DONE]":
            break

        try:
            chunk_data = json.loads(data_str)
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})

            if delta.get("tool_calls"):
                chunks_with_tool_calls.append(chunk_data)
                print(f"[TOOL] {delta['tool_calls']}")

            if delta.get("content"):
                chunks_with_content.append(chunk_data)
                print(f"[CONTENT] {delta['content']}")

            if delta.get("reasoning_content"):
                chunks_with_reasoning.append(chunk_data)
                print(f"[REASONING] {delta['reasoning_content']}")

        except json.JSONDecodeError:
            pass

    print(f"\n✓ Chunks with tool calls: {len(chunks_with_tool_calls)}")
    print(f"✓ Chunks with content: {len(chunks_with_content)}")
    print(f"✓ Chunks with reasoning: {len(chunks_with_reasoning)}")


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_streaming_reasoning():
    """Test streaming with reasoning content."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Think carefully and explain: what is 15 * 24?"}
        ],
        "max_tokens": 1024,
        "stream": True,
        "temperature": 0.7
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT,
        stream=True
    )

    assert response.status_code == 200

    reasoning_chunks = []
    content_chunks = []

    print(f"\n=== Streaming Reasoning ===")

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode('utf-8')
        if not line_str.startswith("data: "):
            continue

        data_str = line_str[6:]
        if data_str == "[DONE]":
            break

        try:
            chunk_data = json.loads(data_str)
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})

            if delta.get("reasoning_content"):
                reasoning_chunks.append(delta["reasoning_content"])
                print(f"[THINK] {delta['reasoning_content']}", end="")

            if delta.get("content"):
                content_chunks.append(delta["content"])
                print(f"[ANSWER] {delta['content']}", end="")

        except json.JSONDecodeError:
            pass

    print(f"\n✓ Reasoning chunks: {len(reasoning_chunks)}")
    print(f"✓ Content chunks: {len(content_chunks)}")

    full_reasoning = "".join(reasoning_chunks)
    full_content = "".join(content_chunks)

    if full_reasoning:
        print(f"✓ Full reasoning: {full_reasoning[:200]}...")
    if full_content:
        print(f"✓ Full content: {full_content[:200]}...")


# ========================================================================
# Combined Features Tests
# ========================================================================


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_combined_reasoning_and_tools():
    """Test both reasoning and tool calling in same request."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Think about what information you need, then check the weather in New York"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]

    print(f"\n=== Combined Reasoning + Tools ===")
    print(f"Reasoning: {message.get('reasoning_content')}")
    print(f"Content: {message.get('content')}")
    print(f"Tool calls: {message.get('tool_calls')}")

    # At least one feature should be present
    has_reasoning = message.get("reasoning_content") is not None
    has_tool_calls = message.get("tool_calls") is not None
    has_content = message.get("content") is not None

    assert has_reasoning or has_tool_calls or has_content, "No response generated"

    print(f"✓ Has reasoning: {has_reasoning}")
    print(f"✓ Has tool calls: {has_tool_calls}")
    print(f"✓ Has content: {has_content}")


# ========================================================================
# OpenAI SDK Compatibility Tests
# ========================================================================


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_openai_sdk_compatibility():
    """Test compatibility with OpenAI Python SDK."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("OpenAI SDK not installed")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key="dummy"  # Auth is disabled
    )

    # Test basic completion
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Say 'Hello, World!' and nothing else"}
        ],
        max_tokens=50
    )

    assert response.choices is not None
    assert len(response.choices) > 0
    message = response.choices[0].message
    assert message.content is not None

    print(f"\n=== OpenAI SDK Compatibility ===")
    print(f"Response: {message.content}")
    print(f"✓ OpenAI SDK works correctly")


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_openai_sdk_tool_calling():
    """Test OpenAI SDK with tool calling."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("OpenAI SDK not installed")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key="dummy"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "What's the weather in Sydney?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ],
        max_tokens=512
    )

    message = response.choices[0].message

    print(f"\n=== OpenAI SDK Tool Calling ===")
    print(f"Content: {message.content}")

    # Check if tool_calls attribute exists (may be None)
    has_tool_calls_attr = hasattr(message, "tool_calls")
    print(f"✓ Has tool_calls attribute: {has_tool_calls_attr}")

    if has_tool_calls_attr and message.tool_calls:
        print(f"✓ Tool calls detected via SDK: {len(message.tool_calls)}")
        for tool_call in message.tool_calls:
            print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")


# ========================================================================
# Error Handling and Edge Cases
# ========================================================================


def test_invalid_tool_schema():
    """Test handling of invalid tool schema."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Test"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "invalid_tool",
                    # Missing required 'parameters' field
                }
            }
        ],
        "max_tokens": 100
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    # Should either reject with error or handle gracefully
    print(f"\n=== Invalid Tool Schema ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")


def test_empty_messages():
    """Test handling of empty messages array."""
    request_data = {
        "model": MODEL_NAME,
        "messages": [],
        "max_tokens": 100
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    print(f"\n=== Empty Messages ===")
    print(f"Status: {response.status_code}")
    # Should return error for empty messages


def test_very_long_context():
    """Test handling of long conversation context."""
    # Create a long conversation
    long_messages = [
        {"role": "user", "content": f"Message {i}"}
        for i in range(20)
    ]

    request_data = {
        "model": MODEL_NAME,
        "messages": long_messages,
        "max_tokens": 100
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        json=request_data,
        timeout=REQUEST_TIMEOUT
    )

    print(f"\n=== Long Context ===")
    print(f"Status: {response.status_code}")
    # Should handle without crashing
