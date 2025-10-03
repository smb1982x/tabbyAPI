# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM project (vllm/tests/reasoning/test_glm4_moe_reasoning_parser.py)
# Adapted for TabbyAPI with modified imports and inline test utilities
# ruff: noqa: E501

from typing import Optional

import pytest

from common.parsers.glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser
from endpoints.OAI.types.chat_completion import ChatCompletionRequest, DeltaMessage

# ========================================================================
# Test Utilities (Inline) - Ported from vllm/tests/reasoning/utils.py
# ========================================================================


class StreamingReasoningReconstructor:
    """Accumulates streaming deltas for testing."""

    def __init__(self):
        self.reasoning_content = None
        self.other_content = None

    def append_delta(self, delta: DeltaMessage):
        # content and the reasoning content should not be present
        # at the same time
        assert delta.content is None or delta.reasoning_content is None, (
            "Both content and reasoning content are present in the " "delta message"
        )
        if delta.content is not None:
            if self.other_content is None:
                self.other_content = delta.content
            else:
                self.other_content += delta.content
        else:
            if self.reasoning_content is None:
                self.reasoning_content = delta.reasoning_content
            else:
                self.reasoning_content += delta.reasoning_content


def run_reasoning_extraction_nonstreaming(
    reasoning_parser: Glm4MoeModelReasoningParser,
    model_output: list[str],
    request: Optional[ChatCompletionRequest] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Helper to run non-streaming reasoning extraction."""
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return reasoning_parser.extract_reasoning_content(
        model_output="".join(model_output), request=request
    )


def run_reasoning_extraction_streaming(
    reasoning_parser: Glm4MoeModelReasoningParser,
    model_deltas: list[str],
    request: Optional[ChatCompletionRequest] = None,
) -> StreamingReasoningReconstructor:
    """Helper to run streaming reasoning extraction."""
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for delta in model_deltas:
        # Get token IDs for this delta
        token_delta = [
            reasoning_parser.vocab.get(token)
            for token in reasoning_parser.model_tokenizer.tokenize(delta)
            if token in reasoning_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = reasoning_parser.extract_reasoning_content_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


def run_reasoning_extraction(
    reasoning_parser: Glm4MoeModelReasoningParser,
    model_output: list[str],
    request: Optional[ChatCompletionRequest] = None,
    streaming: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """Run reasoning extraction in streaming or non-streaming mode."""
    if streaming:
        reconstructor = run_reasoning_extraction_streaming(
            reasoning_parser,
            model_output,
            request,
        )
        return (
            reconstructor.reasoning_content,
            reconstructor.other_content or None,
        )
    else:
        reasoning, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser, model_output, request
        )
        return reasoning, content


# ========================================================================
# Test Fixtures
# ========================================================================


@pytest.fixture
def glm45_tokenizer():
    """Create mock GLM-4.5 tokenizer for tests."""

    class MockTokenizer:
        def get_vocab(self):
            # Return a minimal vocab with special tokens
            return {
                "<think>": 30996,
                "</think>": 30997,
                "<|assistant|>": 151336,
                "[gMASK]": 151329,
                "<sop>": 151336,
                "<|system|>": 151331,
                "<|user|>": 151333,
            }

        def tokenize(self, text: str) -> list[str]:
            """Simple tokenization by characters for testing."""
            # Split by special tokens first
            tokens = []
            special_tokens = [
                "<think>",
                "</think>",
                "<|assistant|>",
                "[gMASK]",
                "<sop>",
                "<|system|>",
                "<|user|>",
            ]
            remaining = text
            while remaining:
                found = False
                for special in special_tokens:
                    if remaining.startswith(special):
                        tokens.append(special)
                        remaining = remaining[len(special) :]
                        found = True
                        break
                if not found:
                    tokens.append(remaining[0])
                    remaining = remaining[1:]
            return tokens

        def convert_tokens_to_string(self, tokens: list[str]) -> str:
            """Convert tokens back to string."""
            return "".join(tokens)

        def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
            """Convert tokens to IDs."""
            vocab = self.get_vocab()
            return [
                vocab.get(token, ord(token[0]) if len(token) == 1 else 0)
                for token in tokens
            ]

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            """Convert IDs to tokens."""
            vocab = self.get_vocab()
            reverse_vocab = {v: k for k, v in vocab.items()}
            return [reverse_vocab.get(id, chr(id)) for id in ids]

    return MockTokenizer()


# ========================================================================
# Test Data - Ported from vLLM
# ========================================================================


WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}

WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": False,
}

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}

MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning_content": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}

ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning_content": None,
    "content": "<think>This is a reasoning section",
    "is_reasoning_end": False,
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}

TEST_CASES = [
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
]

STILL_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think>The user is asking for the capital of"""

DONE_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think>The user is asking for the capital of France.</think>
The capital of France is Paris."""

MULTI_TURN_STILL_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think></think>
The capital of France is Paris.<|user|>
What about Chile?<|assistant|>
<think>The user is asking for the capital of"""

MULTI_TURN_DONE_REASONING_PROMPT = """[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
What is the capital of France?<|assistant|>
<think></think>
The capital of France is Paris.<|user|>
What about Chile?<|assistant|>
<think>The user is asking for the capital of Chile.</think>
The capital of Chile is Santiago."""

REASONING_END_TEST_CASES = [
    pytest.param(STILL_REASONING_PROMPT, False, id="still_reasoning"),
    pytest.param(DONE_REASONING_PROMPT, True, id="done_reasoning"),
    pytest.param(
        MULTI_TURN_STILL_REASONING_PROMPT, False, id="multi_turn_still_reasoning"
    ),
    pytest.param(
        MULTI_TURN_DONE_REASONING_PROMPT, True, id="multi_turn_done_reasoning"
    ),
]


# ========================================================================
# Tests
# ========================================================================


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    glm45_tokenizer,
):
    """Test reasoning extraction in both streaming and non-streaming modes."""
    output = glm45_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        glm45_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser = Glm4MoeModelReasoningParser(glm45_tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]

    output_ids = glm45_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]


@pytest.mark.parametrize("prompt, is_reasoning_end", REASONING_END_TEST_CASES)
def test_is_reasoning_end_full_prompt(
    prompt: str, is_reasoning_end: bool, glm45_tokenizer
):
    """Test is_reasoning_end with full prompts including multi-turn conversations."""
    parser = Glm4MoeModelReasoningParser(glm45_tokenizer)
    tokens = glm45_tokenizer.tokenize(prompt)
    token_ids = glm45_tokenizer.convert_tokens_to_ids(tokens)
    check_is_reasoning_end = parser.is_reasoning_end(token_ids)
    assert check_is_reasoning_end == is_reasoning_end
