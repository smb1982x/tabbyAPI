# GLM-4 MoE Parser Support

## Overview

TabbyAPI includes native support for GLM-4.5 family models' advanced features:
- **Tool/Function Calling**: Converts GLM's XML-based tool calls to OpenAI-compatible JSON format
- **Reasoning Extraction**: Separates reasoning content (`<think>` tags) from final responses

This support is **automatic** - no configuration required. Parsers initialize when a GLM-4.5 model is loaded.

**Backend Requirement**: ExLlamaV3 only (ExLlamaV2 not supported)

## Supported Models

- GLM-4.5
- GLM-4.5-Air
- GLM-4.5V

Models are detected automatically based on directory name.

## Features

### Tool/Function Calling

GLM-4.5 models can call functions by outputting XML like:
```xml
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>Paris</arg_value>
</tool_call>
```

TabbyAPI's parser automatically converts this to OpenAI-compatible format:
```json
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"Paris\"}"
    }
  }]
}
```

### Reasoning Extraction

GLM-4.5 models can show their reasoning process using `<think>` tags:
```xml
<think>User wants weather information. I should use the get_weather function with Paris as location.</think>
I'll check the weather for you.
```

TabbyAPI separates reasoning from the final response:
```json
{
  "message": {
    "content": "I'll check the weather for you.",
    "reasoning_content": "User wants weather information. I should use the get_weather function with Paris as location."
  }
}
```

## Usage Examples

### Tool Calling with Python Requests

```python
import requests

response = requests.post("http://localhost:5000/v1/chat/completions", json={
    "model": "GLM-4.5-Air",
    "messages": [
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    "tools": [{
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
                    }
                },
                "required": ["location"]
            }
        }
    }]
})

data = response.json()
message = data["choices"][0]["message"]

# Access tool calls
if message.get("tool_calls"):
    for tool_call in message["tool_calls"]:
        print(f"Function: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
```

### Tool Calling with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="GLM-4.5-Air",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=[{
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
    }]
)

message = response.choices[0].message

# OpenAI SDK automatically handles tool calls
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Reasoning Extraction

```python
import requests

response = requests.post("http://localhost:5000/v1/chat/completions", json={
    "model": "GLM-4.5-Air",
    "messages": [
        {"role": "user", "content": "Explain quantum entanglement"}
    ]
})

message = response.json()["choices"][0]["message"]

# Access reasoning and final answer separately
if message.get("reasoning_content"):
    print("Reasoning:", message["reasoning_content"])

print("Answer:", message["content"])
```

### Streaming Example

```python
import requests

response = requests.post(
    "http://localhost:5000/v1/chat/completions",
    json={
        "model": "GLM-4.5-Air",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith("data: "):
            data = line[6:]
            if data != "[DONE]":
                import json
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]

                # Stream can include reasoning_content or tool_calls
                if delta.get("reasoning_content"):
                    print(f"[Reasoning] {delta['reasoning_content']}", end="")
                if delta.get("content"):
                    print(delta["content"], end="")
                if delta.get("tool_calls"):
                    print(f"[Tool] {delta['tool_calls']}")
```

## Configuration

### Auto-Detection

Parsers **automatically initialize** for GLM-4.5 models. No configuration needed.

Detection logic:
- Model directory name contains "glm" (case-insensitive)
- AND contains "4.5", "4-5", or "45"

Examples of detected names:
- `GLM-4.5-Air`
- `glm-4.5`
- `GLM-4-5-Chat`

### Manual Verification

Check if parsers initialized:
```python
# In TabbyAPI code
from backends.exllamav3.model import ExllamaV3Container

container = ExllamaV3Container.create(...)

print(f"Tool parser: {container.tool_parser}")
print(f"Reasoning parser: {container.reasoning_parser}")
```

Parsers are `None` for non-GLM models.

## Troubleshooting

### Tool Calls Not Detected

**Issue**: Model outputs tool call XML but response doesn't include `tool_calls` field

**Solutions**:
1. Verify GLM-4.5 model loaded (check server logs for "Initialized GLM-4.5 parsers")
2. Ensure `tools` parameter included in request
3. Check model actually output `<tool_call>` tags in raw response
4. Verify ExLlamaV3 backend (not ExLlamaV2)

### Reasoning Content Not Extracted

**Issue**: Model uses `<think>` tags but `reasoning_content` is None

**Solutions**:
1. Verify GLM-4.5 model loaded
2. Check model actually output `<think>` tags
3. Ensure complete `</think>` closing tag present
4. Check server logs for parser initialization errors

### Parser Initialization Failed

**Issue**: Server logs show "Failed to initialize GLM-4.5 parsers"

**Solutions**:
1. Verify regex package installed: `pip install regex>=2024.0.0`
2. Check TabbyAPI version includes parser support
3. Review full error message in logs
4. Verify model directory name matches detection pattern

### Streaming Incomplete

**Issue**: Streaming stops mid-response or chunks missing

**Solutions**:
1. Check for parser errors in server logs
2. Verify network connection stable
3. Ensure client handles SSE format correctly
4. Try non-streaming mode to isolate issue

## Implementation Notes

### ExLlamaV3 Only

Parser support is **only available for ExLlamaV3 backend**. ExLlamaV2 is not supported.

Reason: Parser implementation requires features specific to ExLlamaV3.

### Performance Impact

- Parser overhead: <5ms per request
- Memory increase: <10MB per parser instance
- No impact on tokens/second throughput

### OpenAI Compatibility

Responses follow OpenAI API specification:
- `tool_calls` format matches OpenAI
- `reasoning_content` is an extension (not in OpenAI spec)
- Compatible with OpenAI Python SDK

## Advanced Topics

### Multiple Tool Calls

GLM-4.5 can call multiple tools in one response:
```xml
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>Paris</arg_value>
</tool_call><tool_call>get_time
<arg_key>timezone</arg_key>
<arg_value>Europe/Paris</arg_value>
</tool_call>
```

Both are extracted and returned in `tool_calls` array.

### Type Preservation

Argument types are automatically detected and preserved:
```python
# GLM output: <arg_value>42</arg_value>
# Parsed as: {"arg": 42} (int, not string)

# GLM output: <arg_value>true</arg_value>
# Parsed as: {"arg": true} (bool)
```

### Partial Content

If tool call includes preceding text:
```xml
Let me check that.<tool_call>get_weather...
```

The text "Let me check that." is available in the `content` field.

## Support

For issues or questions:
- GitHub Issues: [TabbyAPI repository]
- Documentation: This file
- Logs: Check TabbyAPI server logs for parser initialization messages
