# TESTING: Create End-to-End Tests

---

## Summary
Create end-to-end tests that validate the complete flow with a real GLM-4.5 model. They verify the entire system works correctly with actual model inference.

## Implementation Approach

### Steps

**Load tabbyAPI And monitor its output**
```
1. cd /opt/GLM_tabbyAPI
2. source venv/bin/activate
3. python start.py
```

**NOW WAIT FOR tabbyAPI to finish loading**
```
INFO:     Started server process [$pid$]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://10.1.1.10:8080
(Press CTRL+C to quit)
```
**ONLY WHEN THAT APPEARS IN OUTPUT, THEN CONTINUE!**


1. Create `test_glm4_full_flow.py` in `tabbyAPI/tests/e2e/`
2. Add pytest skip decorator for missing model
3. Test full tool calling with real model
4. Test reasoning extraction with real model
5. Test streaming with real model
6. Verify OpenAI SDK compatibility
7. Mark tests to skip if model unavailable

### Test Structure

**File: `tabbyAPI/tests/e2e/test_glm4_full_flow.py`**

```python
import pytest
import os
from pathlib import Path
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.tools import ToolSpec, Function
from backends.exllamav3.model import ExllamaV3Container


def has_glm_model():
    """Check if GLM-4.5 model is available."""
    # Check for GLM model in common locations
    model_paths = [
        Path("/opt/tabbyAPI/models/GLM-4.5-Air"),
    ]
    return any(p.exists() for p in model_paths) or os.getenv("GLM_MODEL_PATH")


def get_glm_model_path():
    """Get path to GLM model."""
    if os.getenv("GLM_MODEL_PATH"):
        return Path(os.getenv("GLM_MODEL_PATH"))

    model_paths = [
        Path("/opt/tabbyAPI/models/GLM-4.5-Air"),
    ]

    for p in model_paths:
        if p.exists():
            return p

    return None


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
@pytest.mark.asyncio
async def test_glm4_model_loads_with_parsers():
    """Test that GLM-4.5 model loads with parsers initialized."""
    model_path = get_glm_model_path()
    assert model_path is not None

    # Create container
    container = await ExllamaV3Container.create(
        model_directory=model_path,
        hf_model=None  # provide actual HFModel
    )

    # Verify parsers initialized
    assert container.tool_parser is not None, "Tool parser should be initialized for GLM model"
    assert container.reasoning_parser is not None, "Reasoning parser should be initialized for GLM model"


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
@pytest.mark.asyncio
async def test_full_tool_calling_flow():
    """Test complete tool calling flow with real GLM-4.5 model."""
    model_path = get_glm_model_path()
    container = await ExllamaV3Container.create(
        model_directory=model_path,
        hf_model=None
    )

    # Create request with tools
    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get current weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                )
            )
        ]
    )

    # Generate response
    response = await container.generate(...)  # Provide actual generate call

    # Verify response structure
    assert "text" in response
    output = response["text"]

    # Parse with tool parser
    tool_info = container.tool_parser.extract_tool_calls(output, request)

    # Should detect tool call
    # Note: Actual assertion depends on model behavior
    if "<tool_call>" in output:
        assert tool_info.tools_called == True
        assert len(tool_info.tool_calls) > 0
        assert tool_info.tool_calls[0].function.name == "get_weather"


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
@pytest.mark.asyncio
async def test_reasoning_extraction_with_real_model():
    """Test reasoning extraction with real GLM-4.5 model."""
    model_path = get_glm_model_path()
    container = await ExllamaV3Container.create(
        model_directory=model_path,
        hf_model=None
    )

    # Create request asking for reasoning
    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "Explain step by step why the sky is blue."}
        ]
    )

    # Generate response
    response = await container.generate(...)

    output = response["text"]

    # Parse with reasoning parser
    reasoning, content = container.reasoning_parser.extract_reasoning_content(output, request)

    # Verify reasoning extracted if model used <think> tags
    if "<think>" in output:
        assert reasoning is not None
        assert content is not None
        assert reasoning != content


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
@pytest.mark.asyncio
async def test_streaming_with_real_model():
    """Test streaming generation with real GLM-4.5 model."""
    model_path = get_glm_model_path()
    container = await ExllamaV3Container.create(
        model_directory=model_path,
        hf_model=None
    )

    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ]
    )

    chunks_received = []
    async for chunk in container.stream_generate(...):
        chunks_received.append(chunk)

    # Verify streaming worked
    assert len(chunks_received) > 0
    assert any("text" in chunk for chunk in chunks_received)


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
def test_openai_sdk_compatibility():
    """Test compatibility with OpenAI Python SDK."""
    pytest.importorskip("openai")

    from openai import OpenAI

    # Create client pointing to TabbyAPI
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="dummy"
    )

    # Test tool calling
    response = client.chat.completions.create(
        model="GLM-4.5-Air",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
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
        ]
    )

    # Verify response structure
    assert response.choices is not None
    assert len(response.choices) > 0

    message = response.choices[0].message

    # Check for new fields
    if hasattr(message, "reasoning_content"):
        # reasoning_content field exists
        assert message.reasoning_content is None or isinstance(message.reasoning_content, str)

    if hasattr(message, "tool_calls"):
        # tool_calls field exists
        assert message.tool_calls is None or isinstance(message.tool_calls, list)


@pytest.mark.skipif(not has_glm_model(), reason="GLM model not available")
@pytest.mark.asyncio
async def test_combined_tool_and_reasoning():
    """Test that both tool calls and reasoning can be extracted together."""
    model_path = get_glm_model_path()
    container = await ExllamaV3Container.create(
        model_directory=model_path,
        hf_model=None
    )

    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "Check the weather in London and explain why."}
        ],
        tools=[
            ToolSpec(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={"type": "object", "properties": {"location": {"type": "string"}}}
                )
            )
        ]
    )

    response = await container.generate(...)
    output = response["text"]

    # Parse reasoning
    reasoning, content = container.reasoning_parser.extract_reasoning_content(output, request)

    # Parse tools
    tool_info = container.tool_parser.extract_tool_calls(content or output, request)

    # If model used both features, verify they work together
    if "<think>" in output and "<tool_call>" in output:
        assert reasoning is not None
        assert tool_info.tools_called == True
```

## Mandatory Reading
- tabbyAPI/backends/exllamav3/model.py
- OpenAI Python SDK documentation

## Input/Process/Output

### Expected Input
- GLM-4.5 model (optional, may not be available)
- Full TabbyAPI server setup
- OpenAI Python SDK (optional)

### Process
1. Check for GLM model availability
2. Skip tests if model not found
3. Test with real model inference
4. Verify full integration
5. Test OpenAI SDK compatibility

### Expected Output
- Test marked with `@pytest.mark.skipif` for missing model
- Test: Full tool calling with real model
- Test: Reasoning extraction with real model
- Test: Streaming with real model
- Verify OpenAI SDK compatibility
- Tests passing when model available

## Tests

### Run Tests
```bash
cd /opt/GLM_tabbyAPI/tabbyAPI

# Run e2e tests (will skip if no model)
pytest tests/e2e/test_glm4_full_flow.py -v

# Run with model path
GLM_MODEL_PATH=/opt/tabbyAPI/models/GLM-4.5-Air pytest tests/e2e/test_glm4_full_flow.py -v

# Show skipped tests
pytest tests/e2e/test_glm4_full_flow.py -v -rs
```

### Success Criteria
- ✅ Test marked with `@pytest.mark.skipif` for missing model
- ✅ Test: Full tool calling with real model
- ✅ Test: Reasoning extraction with real model
- ✅ Test: Streaming with real model
- ✅ Verify OpenAI SDK compatibility
- ✅ Tests passing when model available

### FINAL ACTIONS:

Run all tests, multiple times with different data, to get enough information to create an Indepth report about 
failures, include logs and details about failures, and write out to /opt/GLM_tabbyAPI/TEST_RESULTS.md

<End>
