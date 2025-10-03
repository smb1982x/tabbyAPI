# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TabbyAPI is a FastAPI-based inference server for large language models (LLMs) using ExllamaV2 and ExllamaV3 backends. This is a **fork** with added GLM-4.5 model support, specifically for parsing tool calls and reasoning content.

### Key Features Added in This Fork
- **GLM-4.5 Parser Support**: Automatic conversion of GLM's XML-based tool calls to OpenAI-compatible JSON format
- **Reasoning Extraction**: Separates reasoning content (`<think>` tags) from final responses
- **ExllamaV3 Only**: Parser functionality requires ExllamaV3 backend

## Common Commands

### Development
```bash
# Run linting and formatting
./formatting.sh              # Linux
formatting.bat               # Windows

# Black formatter
python -m black .

# Ruff linter
python -m ruff check .
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test suites
python -m pytest tests/parsers/                    # Parser tests
python -m pytest tests/integration/                # Integration tests
python -m pytest tests/parsers/test_glm4_tool_parser.py -v  # Single file
```

### Running the Server
```bash
# Start with automatic setup
python start.py

# Start directly
python main.py

# Start with custom config
python main.py --config custom_config.yml

# Update dependencies
./update_scripts/update_deps.sh    # Linux
update_scripts\update_deps.bat     # Windows
```

### Model Management
```bash
# Load model on startup - edit config.yml
model:
  model_name: "GLM-4.5-Air"
  model_dir: "models"
  backend: "exllamav3"  # Required for GLM-4.5
```

## Architecture

### Directory Structure
- **`backends/`**: Model container implementations (ExllamaV2/V3)
  - `backends/exllamav2/`: V2 backend (no GLM parser support)
  - `backends/exllamav3/`: V3 backend with GLM parser integration
- **`common/`**: Shared utilities and core logic
  - `common/parsers/`: **Parser implementations** for tool/reasoning extraction
    - Abstract base classes: `abstract_tool_parser.py`, `abstract_reasoning_parser.py`
    - GLM-4.5 parsers: `glm4_moe_tool_parser.py`, `glm4_moe_reasoning_parser.py`
- **`endpoints/`**: API endpoint definitions
  - `endpoints/OAI/`: OpenAI-compatible API
  - `endpoints/Kobold/`: KoboldAI-compatible API
- **`tests/`**: Test suites
  - `tests/parsers/`: Parser unit tests
  - `tests/integration/`: Integration and streaming tests

### Parser Integration Flow

1. **Model Loading** (`backends/exllamav3/model.py:ExllamaV3Container.create()`):
   - Detects GLM-4.5 models by directory name pattern
   - Auto-initializes `tool_parser` and `reasoning_parser` for GLM models
   - Parsers remain `None` for non-GLM models

2. **Response Processing** (`endpoints/OAI/utils/chat_completion.py`):
   - Passes raw model output through parsers (if present)
   - Converts XML tool calls → OpenAI JSON format
   - Extracts `<think>` content → `reasoning_content` field

3. **Streaming Support** (`endpoints/OAI/utils/chat_completion.py`):
   - Parsers maintain state across chunks
   - Emits `delta` updates for tool calls and reasoning content

### GLM-4.5 Model Detection

Models are auto-detected if directory name contains:
- "glm" (case-insensitive) AND
- "4.5" OR "4-5" OR "45"

Examples: `GLM-4.5-Air`, `glm-4.5`, `GLM-4-5-Chat`

### Parser Manager Pattern

Both tool and reasoning parsers use a manager pattern:
- **`ToolParserManager`**: Tracks partial tool call state across streaming chunks
- **`ReasoningParserManager`**: Tracks partial reasoning content across chunks

Managers handle:
- Buffering incomplete XML tags
- Emitting deltas when complete tags are found
- Finalizing content at stream end

## Important Development Notes

### Parser Modifications

When editing parsers:
1. **Update unit tests** in `tests/parsers/` to match behavior changes
2. **Update integration tests** in `tests/integration/` if streaming behavior changes
3. **Regex patterns** use `regex` library (not `re`) for advanced features
4. Tool argument types are auto-detected: `"42"` → `42`, `"true"` → `true`

### Backend-Specific Code

- **ExllamaV3 only**: Parser code only exists in `backends/exllamav3/`
- **Do NOT** add parser support to ExllamaV2 (not compatible)
- Check `container.tool_parser` for `None` before using (non-GLM models)

### OpenAI Compatibility

- `tool_calls` format must match OpenAI spec exactly
- `reasoning_content` is a **custom extension** (not in OpenAI spec)
- Tool call IDs use format: `call_` + random hex

### Configuration

- **`config.yml`**: Main config (created from `config_sample.yml` on first run)
- Backend must be set to `exllamav3` for GLM-4.5 models
- No special parser config required - auto-initialized

### Testing Strategy

- **Unit tests**: Test parsers in isolation with mock tokenizers
- **Integration tests**: Test full request/response cycle with real model containers
- **Streaming tests**: Verify chunk-by-chunk parsing and delta emission

### Code Style

- Follow Ruff linting rules (see `pyproject.toml`)
- Black formatter with 88-character line length
- Type hints required for public functions
- Docstrings for classes and complex functions

## Modified Files (Fork-Specific)

Core changes from upstream TabbyAPI:
```
backends/exllamav3/model.py              # Parser initialization
endpoints/OAI/router.py                  # Parser integration
endpoints/OAI/types/chat_completion.py   # Response type extensions
endpoints/OAI/types/common.py            # Common type definitions
endpoints/OAI/types/tools.py             # Tool call types
endpoints/OAI/utils/chat_completion.py   # Parser usage
endpoints/OAI/utils/tools.py             # Tool handling
```

New files:
```
common/parsers/abstract_reasoning_parser.py
common/parsers/abstract_tool_parser.py
common/parsers/glm4_moe_reasoning_parser.py
common/parsers/glm4_moe_tool_parser.py
common/parsers/__init__.py
docs/glm4_parsers.md
tests/parsers/test_glm4_tool_parser.py
tests/parsers/test_glm4_reasoning_parser.py
tests/parsers/test_parser_managers.py
tests/integration/test_parser_integration.py
tests/integration/test_streaming_integration.py
```

## Documentation

- **User Documentation**: `docs/glm4_parsers.md` - Full GLM-4.5 parser guide with examples
- **API Docs**: `https://theroyallab.github.io/tabbyAPI`
- **Wiki**: `https://github.com/theroyallab/tabbyAPI/wiki`
