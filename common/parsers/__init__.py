"""Parser module for tool calling and reasoning extraction.

This module provides abstract base classes and concrete implementations
for parsing model output to extract tool calls and reasoning content.

Supported Models:
    - GLM-4.5 family (GLM-4.5, GLM-4.5-Air, GLM-4.5V)

Usage:
    from common.parsers import Glm4MoeModelToolParser, Glm4MoeModelReasoningParser
    from exllamav3 import Tokenizer

    tool_parser = Glm4MoeModelToolParser(tokenizer)
    reasoning_parser = Glm4MoeModelReasoningParser(tokenizer)
"""

from .abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
    DeltaMessage,
)
from .abstract_reasoning_parser import (
    ReasoningParser,
    ReasoningParserManager,
)
from .glm4_moe_tool_parser import Glm4MoeModelToolParser
from .glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser

# Exports will be added as parsers are implemented
__all__ = [
    "ToolParser",
    "ToolParserManager",
    "ReasoningParser",
    "ReasoningParserManager",
    "DeltaMessage",
    "Glm4MoeModelToolParser",
    "Glm4MoeModelReasoningParser",
]
